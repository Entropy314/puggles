use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crate::core::{EvalFn, Problem, Solution};
use crate::genetic_operators::mutation::MutationManager;
use crate::genetic_operators::crossover::CrossoverManager;
use crate::genetic_operators::selectors::CrowdingTournamentSelector;
use crate::dominance::{ParetoDominance, fast_non_dominated_sort, crowding_distance};

/// Controls how fitness evaluations and population operations are parallelized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    /// Single-threaded execution. Best for cheap objective functions or small populations.
    Sequential,
    /// Multi-threaded via rayon. Automatically scales to available CPU cores.
    /// Good default for most workloads.
    MultiThreaded,
    /// GPU-accelerated evaluation (requires objective function compiled to GPU kernel).
    /// Falls back to MultiThreaded if GPU is unavailable.
    GPU,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::MultiThreaded
    }
}

pub trait GeneticAlgorithm {
    fn initialize(&mut self);
    fn iterate(&mut self);
    fn evaluate_population(&mut self, population: &mut Vec<Solution>);
    fn run(&mut self, nfe: usize);
}

pub struct NSGAII {
    pub problem: Arc<Problem>,
    pub population_size: usize,
    pub population: Vec<Solution>,
    pub nfe: AtomicUsize,
    pub execution_mode: ExecutionMode,
    pub selector: CrowdingTournamentSelector,
    pub dominance: ParetoDominance,
    pub mutation_manager: MutationManager,
    pub crossover_manager: CrossoverManager,
    pub archive: Vec<Solution>,
    // Per-individual rank and crowding distance (updated each iteration)
    ranks: Vec<usize>,
    crowding_distances: Vec<f64>,
    /// Optional GPU evaluator. When set and execution_mode == GPU, population evaluation
    /// is offloaded to the GPU via wgpu compute shaders.
    #[cfg(feature = "gpu")]
    pub gpu_evaluator: Option<crate::gpu_evaluator::GpuEvaluator>,
}

impl NSGAII {
    pub fn new(
        problem: Arc<Problem>,
        population_size: usize,
        execution_mode: ExecutionMode,
    ) -> Self {
        Self {
            problem,
            population_size,
            population: Vec::with_capacity(population_size),
            nfe: AtomicUsize::new(0),
            execution_mode,
            selector: CrowdingTournamentSelector::new(2, None),
            dominance: ParetoDominance,
            mutation_manager: MutationManager::new(),
            crossover_manager: CrossoverManager::new(),
            archive: Vec::new(),
            ranks: Vec::new(),
            crowding_distances: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu_evaluator: None,
        }
    }

    /// Attach a GPU evaluator for use when `execution_mode == GPU`.
    /// Builder method — call before `initialize()`.
    #[cfg(feature = "gpu")]
    pub fn with_gpu_evaluator(mut self, evaluator: crate::gpu_evaluator::GpuEvaluator) -> Self {
        self.gpu_evaluator = Some(evaluator);
        self
    }

    /// Assign ranks and crowding distances for the current population.
    fn assign_fitness(&mut self) {
        let n = self.population.len();
        self.ranks.resize(n, 0);
        self.crowding_distances.resize(n, 0.0);

        let fronts = fast_non_dominated_sort(&self.population, &self.dominance);

        for (rank, front) in fronts.iter().enumerate() {
            let cd = crowding_distance(&self.population, front);
            for (local_idx, &global_idx) in front.iter().enumerate() {
                self.ranks[global_idx] = rank;
                self.crowding_distances[global_idx] = cd[local_idx];
            }
        }
    }

    /// NSGA-II environmental selection: from a combined population of size 2N,
    /// select the best N by front rank then crowding distance.
    /// Operates on indices to avoid unnecessary cloning until the final selection.
    fn environmental_selection(&self, combined: &[Solution], target_size: usize) -> Vec<Solution> {
        let fronts = fast_non_dominated_sort(combined, &self.dominance);
        let mut selected: Vec<Solution> = Vec::with_capacity(target_size);

        for front in &fronts {
            if selected.len() + front.len() <= target_size {
                // Entire front fits
                for &idx in front {
                    selected.push(combined[idx].clone());
                }
            } else {
                // Partial front: pick by crowding distance (higher = better diversity)
                let remaining = target_size - selected.len();
                let cd = crowding_distance(combined, front);

                // Create (local_index, crowding_distance) pairs and sort descending by cd
                let mut indexed_cd: Vec<(usize, f64)> = cd.into_iter().enumerate().collect();
                indexed_cd.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for i in 0..remaining {
                    let global_idx = front[indexed_cd[i].0];
                    selected.push(combined[global_idx].clone());
                }
                break;
            }
        }

        selected
    }

    /// Archive non-dominated feasible solutions (Pareto front).
    pub fn update_archive(&mut self) {
        let feasible: Vec<Solution> = self.population.iter()
            .filter(|s| s.feasible && s.evaluated)
            .cloned()
            .collect();

        if feasible.is_empty() {
            return;
        }

        // Merge with existing archive
        let mut combined = std::mem::take(&mut self.archive);
        combined.extend(feasible);

        // Keep only non-dominated solutions
        let fronts = fast_non_dominated_sort(&combined, &self.dominance);
        self.archive = if !fronts.is_empty() {
            fronts[0].iter().map(|&idx| combined[idx].clone()).collect()
        } else {
            Vec::new()
        };
    }

    pub fn get_archive(&self) -> &[Solution] {
        &self.archive
    }

    pub fn get_nfe(&self) -> usize {
        self.nfe.load(Ordering::Relaxed)
    }

    /// Run one generation, creating at most `max_offspring` new solutions.
    /// Passing `self.population_size` is equivalent to a full iteration.
    /// Used by `run()` to honour the NFE budget without overshoot.
    pub fn iterate_n(&mut self, max_offspring: usize) {
        self.assign_fitness();

        let parent_indices = self.selector.select_indices(
            self.population_size,
            &self.ranks,
            &self.crowding_distances,
        );

        let selected_parents: Vec<Solution> = parent_indices.iter()
            .map(|&idx| self.population[idx].clone())
            .collect();

        let mut offspring: Vec<Solution> = Vec::with_capacity(max_offspring);

        let mut i = 0;
        while offspring.len() < max_offspring {
            let p1 = &selected_parents[i % selected_parents.len()];
            let p2 = &selected_parents[(i + 1) % selected_parents.len()];
            i += 2;

            let children = self.crossover_manager.perform_crossover(p1, p2);
            for child in children {
                let mutated = self.mutation_manager.mutate(&child);
                if offspring.len() < max_offspring {
                    offspring.push(mutated);
                }
            }
        }

        self.evaluate_population(&mut offspring);

        let mut combined = std::mem::take(&mut self.population);
        combined.extend(offspring);
        self.population = self.environmental_selection(&combined, self.population_size);
    }

    /// Run for at most `max_nfe` evaluations OR until `time_limit` elapses, whichever comes first.
    pub fn run_timed(&mut self, max_nfe: usize, time_limit: std::time::Duration) {
        if self.population.is_empty() {
            self.initialize();
        }
        let mut pop = std::mem::take(&mut self.population);
        self.evaluate_population(&mut pop);
        self.population = pop;
        let start = std::time::Instant::now();
        while self.nfe.load(Ordering::Relaxed) < max_nfe && start.elapsed() < time_limit {
            let remaining = max_nfe - self.nfe.load(Ordering::Relaxed);
            self.iterate_n(remaining.min(self.population_size));
            self.update_archive();
        }
    }
}

impl GeneticAlgorithm for NSGAII {
    fn initialize(&mut self) {
        self.population = match self.execution_mode {
            ExecutionMode::Sequential => {
                (0..self.population_size)
                    .map(|_| {
                        let mut sol = Solution::new(Arc::clone(&self.problem));
                        sol.solution = self.problem.generate_solution();
                        sol
                    })
                    .collect()
            }
            ExecutionMode::MultiThreaded | ExecutionMode::GPU => {
                let problem = Arc::clone(&self.problem);
                (0..self.population_size)
                    .into_par_iter()
                    .map(|_| {
                        let mut sol = Solution::new(Arc::clone(&problem));
                        sol.solution = problem.generate_solution();
                        sol
                    })
                    .collect()
            }
        };
    }

    fn evaluate_population(&mut self, population: &mut Vec<Solution>) {
        // Batch path: call the batch objective function once with all unevaluated solutions.
        if let EvalFn::Batch(batch_fn) = self.problem.eval_fn {
            let unevaluated: Vec<usize> = population.iter().enumerate()
                .filter(|(_, s)| !s.evaluated)
                .map(|(i, _)| i)
                .collect();
            if !unevaluated.is_empty() {
                let inputs: Vec<Vec<f64>> = unevaluated.iter()
                    .map(|&i| population[i].solution.clone())
                    .collect();
                let outputs = batch_fn(&inputs);
                for (local_i, &global_i) in unevaluated.iter().enumerate() {
                    population[global_i].objective_fitness_values = outputs[local_i].clone();
                    population[global_i].evaluated = true;
                    let cv = population[global_i].evaluate_constraints();
                    population[global_i].constraint_values = cv;
                    let viol = population[global_i].calculate_constraint_violation();
                    population[global_i].constraint_violation = viol;
                    let feas = population[global_i].is_feasible();
                    population[global_i].feasible = feas;
                }
                self.nfe.fetch_add(unevaluated.len(), Ordering::Relaxed);
            }
            return;
        }

        // GPU path: batch-evaluate via wgpu compute shader.
        #[cfg(feature = "gpu")]
        if self.execution_mode == ExecutionMode::GPU {
            if let Some(ref gpu) = self.gpu_evaluator {
                let unevaluated: Vec<usize> = population.iter().enumerate()
                    .filter(|(_, s)| !s.evaluated)
                    .map(|(i, _)| i)
                    .collect();
                if !unevaluated.is_empty() {
                    let inputs: Vec<Vec<f64>> = unevaluated.iter()
                        .map(|&i| population[i].solution.clone())
                        .collect();
                    let outputs = gpu.evaluate_batch(&inputs);
                    for (local_i, &global_i) in unevaluated.iter().enumerate() {
                        population[global_i].objective_fitness_values = outputs[local_i].clone();
                        population[global_i].evaluated = true;
                        let cv = population[global_i].evaluate_constraints();
                        population[global_i].constraint_values = cv;
                        let viol = population[global_i].calculate_constraint_violation();
                        population[global_i].constraint_violation = viol;
                        let feas = population[global_i].is_feasible();
                        population[global_i].feasible = feas;
                    }
                    self.nfe.fetch_add(unevaluated.len(), Ordering::Relaxed);
                }
                return;
            } else {
                eprintln!(
                    "rustypus warning: ExecutionMode::GPU selected but no GpuEvaluator attached. \
                     Falling back to MultiThreaded. Use NSGAII::with_gpu_evaluator() to enable GPU."
                );
            }
        }

        // Single-evaluation path (Sequential or Rayon-parallel).
        let new_evals: usize = match self.execution_mode {
            ExecutionMode::Sequential => {
                population.iter_mut()
                    .filter(|s| !s.evaluated)
                    .map(|s| { s.evaluate(); 1 })
                    .sum()
            }
            ExecutionMode::MultiThreaded | ExecutionMode::GPU => {
                population.par_iter_mut()
                    .filter(|s| !s.evaluated)
                    .map(|s| { s.evaluate(); 1 })
                    .sum()
            }
        };
        self.nfe.fetch_add(new_evals, Ordering::Relaxed);
    }

    fn iterate(&mut self) {
        self.iterate_n(self.population_size);
    }

    fn run(&mut self, max_nfe: usize) {
        // Issue 6: auto-initialize if population is empty
        if self.population.is_empty() {
            self.initialize();
        }

        let mut pop = std::mem::take(&mut self.population);
        self.evaluate_population(&mut pop);
        self.population = pop;

        while self.nfe.load(Ordering::Relaxed) < max_nfe {
            let remaining = max_nfe - self.nfe.load(Ordering::Relaxed);
            self.iterate_n(remaining.min(self.population_size));
            self.update_archive();
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EvalFn, Problem};
    use crate::gatypes::SolutionDataTypes;
    use crate::gatypes::{BitBinary, Integer, Real};
    use crate::benchmark_objective_functions::parabloid_5_loc;

    fn setup_single_objective_problem() -> Problem {
        Problem {
            solution_length: 5,
            number_of_objectives: 1,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1]), // minimize
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(-10.0), Some(10.0))),
                SolutionDataTypes::Real(Real::new(Some(-10.0), Some(10.0))),
                SolutionDataTypes::Real(Real::new(Some(-10.0), Some(10.0))),
                SolutionDataTypes::Real(Real::new(Some(-10.0), Some(10.0))),
                SolutionDataTypes::Real(Real::new(Some(-10.0), Some(10.0))),
            ],
            eval_fn: EvalFn::Single(parabloid_5_loc),
        }
    }

    fn setup_multi_objective_problem() -> Problem {
        Problem {
            solution_length: 3,
            number_of_objectives: 2,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1]),
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(-10.0), Some(10.0))),
                SolutionDataTypes::Integer(Integer::new(Some(-10), Some(10))),
                SolutionDataTypes::BitBinary(BitBinary::new()),
            ],
            eval_fn: EvalFn::Single(|x| vec![
                x[0] * x[0] + x[1] as f64,
                (x[0] - 2.0).powi(2) + x[2],
            ]),
        }
    }

    fn setup_mixed_type_problem() -> Problem {
        Problem {
            solution_length: 5,
            number_of_objectives: 1,
            objective_constraint: Some(vec![Some(10.0)]),
            objective_constraint_operands: Some(vec![Some("<".to_string())]),
            direction: Some(vec![1]),
            solution_data_types: vec![
                SolutionDataTypes::BitBinary(BitBinary::new()),
                SolutionDataTypes::Integer(Integer::new(Some(10), Some(2000))),
                SolutionDataTypes::Real(Real::new(Some(10.0), Some(1000.0))),
                SolutionDataTypes::Real(Real::new(Some(10.0), Some(1000.0))),
                SolutionDataTypes::Real(Real::new(Some(10.0), Some(1000.0))),
            ],
            eval_fn: EvalFn::Single(|x| vec![x.iter().sum()]),
        }
    }

    #[test]
    fn test_nsgaii_initialize_sequential() {
        let problem = Arc::new(setup_single_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential);
        ga.initialize();
        assert_eq!(ga.population.len(), 20);
    }

    #[test]
    fn test_nsgaii_initialize_multithreaded() {
        let problem = Arc::new(setup_single_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 50, ExecutionMode::MultiThreaded);
        ga.initialize();
        assert_eq!(ga.population.len(), 50);
    }

    #[test]
    fn test_nsgaii_single_objective_run() {
        let problem = Arc::new(setup_single_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 30, ExecutionMode::MultiThreaded);
        ga.initialize();
        ga.run(500);

        assert!(ga.get_nfe() >= 500);
        assert_eq!(ga.population.len(), 30);

        // All solutions should be evaluated
        for sol in &ga.population {
            assert!(sol.evaluated);
        }
    }

    #[test]
    fn test_nsgaii_multi_objective_run() {
        let problem = Arc::new(setup_multi_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 40, ExecutionMode::MultiThreaded);
        ga.initialize();
        ga.run(1000);

        assert!(ga.get_nfe() >= 1000);
        assert_eq!(ga.population.len(), 40);

        // Check archive has non-dominated solutions
        for sol in &ga.archive {
            assert!(sol.evaluated);
        }
    }

    #[test]
    fn test_nsgaii_mixed_types() {
        let problem = Arc::new(setup_mixed_type_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential);
        ga.initialize();
        ga.run(200);

        assert!(ga.get_nfe() >= 200);
        assert_eq!(ga.population.len(), 20);
    }

    #[test]
    fn test_nsgaii_iterate_preserves_population_size() {
        let problem = Arc::new(setup_multi_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential);
        // run() auto-initializes and evaluates; then check iterate keeps size constant
        ga.run(20); // evaluate one generation worth
        for _ in 0..5 {
            ga.iterate();
            assert_eq!(ga.population.len(), 20, "Population size must stay constant across iterations");
        }
    }

    #[test]
    fn test_execution_mode_default() {
        assert_eq!(ExecutionMode::default(), ExecutionMode::MultiThreaded);
    }

    #[test]
    fn test_nsgaii_gpu_fallback_to_multithreaded() {
        // No GpuEvaluator attached — should fall back to MultiThreaded silently.
        let problem = Arc::new(setup_multi_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 40, ExecutionMode::GPU);
        ga.initialize();
        ga.run(1000);

        assert!(ga.get_nfe() >= 1000);
        assert_eq!(ga.population.len(), 40);
        for sol in &ga.population {
            assert!(sol.evaluated);
        }
    }

    #[test]
    fn test_nsgaii_multithreaded_archive_nonempty() {
        let problem = Arc::new(setup_multi_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 40, ExecutionMode::MultiThreaded);
        ga.initialize();
        ga.run(1000);

        assert!(ga.get_nfe() >= 1000);
        assert_eq!(ga.population.len(), 40);
        assert!(!ga.archive.is_empty(), "archive should contain non-dominated solutions");
        for sol in &ga.population {
            assert!(sol.evaluated);
        }
    }

    #[test]
    fn test_nsgaii_auto_initialize_on_run() {
        // Issue 6: run() should auto-initialize if population is empty
        let problem = Arc::new(setup_single_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential);
        // Do NOT call ga.initialize() — run() should do it automatically
        ga.run(100);
        assert!(ga.get_nfe() >= 100);
        assert_eq!(ga.population.len(), 20);
    }

    #[test]
    fn test_nsgaii_run_timed() {
        let problem = Arc::new(setup_multi_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential);
        // run_timed with a generous time limit — should complete by NFE
        ga.run_timed(200, std::time::Duration::from_secs(60));
        assert!(ga.get_nfe() >= 200);
        assert_eq!(ga.population.len(), 20);
    }
}
