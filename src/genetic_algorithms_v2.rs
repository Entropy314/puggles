use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::core::{Problem, Solution};
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

pub trait GeneticAlgorithm<'a> {
    fn initialize(&mut self);
    fn iterate(&mut self);
    fn evaluate_population(&mut self, population: &mut Vec<Solution<'a>>);
    fn run(&mut self, nfe: usize);
}

pub struct NSGAII<'a> {
    pub problem: &'a Problem,
    pub population_size: usize,
    pub population: Vec<Solution<'a>>,
    pub nfe: AtomicUsize,
    pub execution_mode: ExecutionMode,
    pub selector: CrowdingTournamentSelector,
    pub dominance: ParetoDominance,
    pub mutation_manager: MutationManager<'a>,
    pub crossover_manager: CrossoverManager<'a>,
    pub archive: Vec<Solution<'a>>,
    // Per-individual rank and crowding distance (updated each iteration)
    ranks: Vec<usize>,
    crowding_distances: Vec<f64>,
    /// Optional GPU evaluator. When set and execution_mode == GPU, population evaluation
    /// is offloaded to the GPU via wgpu compute shaders.
    #[cfg(feature = "gpu")]
    pub gpu_evaluator: Option<crate::gpu_evaluator::GpuEvaluator>,
}

impl<'a> NSGAII<'a> {
    pub fn new(
        problem: &'a Problem,
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
    fn environmental_selection(&self, combined: &[Solution<'a>], target_size: usize) -> Vec<Solution<'a>> {
        let fronts = fast_non_dominated_sort(combined, &self.dominance);
        let mut selected: Vec<Solution<'a>> = Vec::with_capacity(target_size);

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
        let feasible: Vec<Solution<'a>> = self.population.iter()
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

    pub fn get_archive(&self) -> &[Solution<'a>] {
        &self.archive
    }

    pub fn get_nfe(&self) -> usize {
        self.nfe.load(Ordering::Relaxed)
    }
}

impl<'a> GeneticAlgorithm<'a> for NSGAII<'a> {
    fn initialize(&mut self) {
        self.population = match self.execution_mode {
            ExecutionMode::Sequential => {
                (0..self.population_size)
                    .map(|_| {
                        let mut sol = Solution::new(self.problem);
                        sol.solution = self.problem.generate_solution();
                        sol
                    })
                    .collect()
            }
            ExecutionMode::MultiThreaded | ExecutionMode::GPU => {
                (0..self.population_size)
                    .into_par_iter()
                    .map(|_| {
                        let mut sol = Solution::new(self.problem);
                        sol.solution = self.problem.generate_solution();
                        sol
                    })
                    .collect()
            }
        };
    }

    fn evaluate_population(&mut self, population: &mut Vec<Solution<'a>>) {
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
                        population[global_i].feasible = true;
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
        // 1. Assign rank and crowding distance to current population
        self.assign_fitness();

        // 2. Binary tournament selection using crowding comparison to pick parents
        let parent_indices = self.selector.select_indices(
            self.population_size,
            &self.ranks,
            &self.crowding_distances,
        );

        // 3. Clone selected parents to break the borrow on self.population
        let selected_parents: Vec<Solution<'a>> = parent_indices.iter()
            .map(|&idx| self.population[idx].clone())
            .collect();

        // 4. Create offspring via crossover + mutation (pairs of parents)
        let mut offspring: Vec<Solution<'a>> = Vec::with_capacity(self.population_size);

        let mut i = 0;
        while offspring.len() < self.population_size {
            let p1 = &selected_parents[i % selected_parents.len()];
            let p2 = &selected_parents[(i + 1) % selected_parents.len()];
            i += 2;

            let children = self.crossover_manager.perform_crossover(p1, p2);
            for child in children {
                if offspring.len() < self.population_size {
                    offspring.push(child);
                }
            }
        }

        // 5. Evaluate offspring
        self.evaluate_population(&mut offspring);

        // 6. Environmental selection: combine parent + offspring, select best N
        let mut combined = std::mem::take(&mut self.population);
        combined.extend(offspring);

        self.population = self.environmental_selection(&combined, self.population_size);
    }

    fn run(&mut self, max_nfe: usize) {
        // Evaluate initial population using the existing evaluate_population path,
        // avoiding the borrow conflict by temporarily taking ownership.
        let mut pop = std::mem::take(&mut self.population);
        self.evaluate_population(&mut pop);
        self.population = pop;

        while self.nfe.load(Ordering::Relaxed) < max_nfe {
            self.iterate();
            self.update_archive();
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Problem, Solution};
    use crate::gatypes::SolutionDataTypes;
    use crate::gatypes::{BitBinary, Integer, Real};
    use crate::benchmark_objective_functions::{simple_objective, parabloid_5_loc};

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
            objective_function: parabloid_5_loc,
            batch_objective_function: None,
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
            objective_function: |x| vec![
                x[0] * x[0] + x[1] as f64,
                (x[0] - 2.0).powi(2) + x[2],
            ],
            batch_objective_function: None,
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
            objective_function: |x| vec![x.iter().sum()],
            batch_objective_function: None,
        }
    }

    #[test]
    fn test_nsgaii_initialize_sequential() {
        let problem = setup_single_objective_problem();
        let mut ga = NSGAII::new(&problem, 20, ExecutionMode::Sequential);
        ga.initialize();
        assert_eq!(ga.population.len(), 20);
    }

    #[test]
    fn test_nsgaii_initialize_multithreaded() {
        let problem = setup_single_objective_problem();
        let mut ga = NSGAII::new(&problem, 50, ExecutionMode::MultiThreaded);
        ga.initialize();
        assert_eq!(ga.population.len(), 50);
    }

    #[test]
    fn test_nsgaii_single_objective_run() {
        let problem = setup_single_objective_problem();
        let mut ga = NSGAII::new(&problem, 30, ExecutionMode::MultiThreaded);
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
        let problem = setup_multi_objective_problem();
        let mut ga = NSGAII::new(&problem, 40, ExecutionMode::MultiThreaded);
        ga.initialize();
        ga.run(1000);

        assert!(ga.get_nfe() >= 1000);
        assert_eq!(ga.population.len(), 40);

        // Check archive has non-dominated solutions
        println!("Archive size: {}", ga.archive.len());
        for sol in &ga.archive {
            assert!(sol.evaluated);
            // println!("  objectives: {:?}", sol.objective_fitness_values);
        }
    }

    #[test]
    fn test_nsgaii_mixed_types() {
        let problem = setup_mixed_type_problem();
        let mut ga = NSGAII::new(&problem, 20, ExecutionMode::Sequential);
        ga.initialize();
        ga.run(200);

        assert!(ga.get_nfe() >= 200);
        assert_eq!(ga.population.len(), 20);
    }

    #[test]
    fn test_nsgaii_iterate_preserves_population_size() {
        let problem = setup_multi_objective_problem();
        let mut ga = NSGAII::new(&problem, 20, ExecutionMode::Sequential);
        ga.initialize();

        // Evaluate initial population
        {
            let pop = &mut ga.population;
            for s in pop.iter_mut() {
                s.evaluate();
            }
            ga.nfe.fetch_add(20, Ordering::Relaxed);
        }

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
    fn test_nsgaii_gpu_print_solutions() {
        let problem = setup_multi_objective_problem();
        let mut ga = NSGAII::new(&problem, 40, ExecutionMode::GPU);
        ga.initialize();
        ga.run(10000);

        println!("=== NSGA-II GPU Mode Results ===");
        println!("NFE: {}", ga.get_nfe());
        println!("Population size: {}", ga.population.len());
        println!("Archive size: {}", ga.archive.len());

        println!("\n-- Population --");
        for (i, sol) in ga.population.iter().enumerate() {
            println!(
                "  [{}] solution: {:?}  objectives: {:?}",
                i, sol.solution, sol.objective_fitness_values
            );
        }

        println!("\n-- Archive (Pareto Front) --");
        for (i, sol) in ga.archive.iter().enumerate() {
            println!(
                "  [{}] solution: {:?}  objectives: {:?}",
                i, sol.solution, sol.objective_fitness_values
            );
        }

        assert!(ga.get_nfe() >= 1000);
        assert_eq!(ga.population.len(), 40);
        for sol in &ga.population {
            assert!(sol.evaluated);
        }
    }

    #[test]
    fn test_nsgaii_multithreaded_print_solutions() {
        let problem = setup_multi_objective_problem();
        let mut ga = NSGAII::new(&problem, 40, ExecutionMode::MultiThreaded);
        ga.initialize();
        ga.run(10000);

        println!("=== NSGA-II MultiThreaded Mode Results ===");
        println!("NFE: {}", ga.get_nfe());
        println!("Population size: {}", ga.population.len());
        println!("Archive size: {}", ga.archive.len());

        println!("\n-- Population --");
        for (i, sol) in ga.population.iter().enumerate() {
            println!(
                "  [{}] solution: {:?}  objectives: {:?}",
                i, sol.solution, sol.objective_fitness_values
            );
        }

        println!("\n-- Archive (Pareto Front) --");
        for (i, sol) in ga.archive.iter().enumerate() {
            println!(
                "  [{}] solution: {:?}  objectives: {:?}",
                i, sol.solution, sol.objective_fitness_values
            );
        }

        assert!(ga.get_nfe() >= 1000);
        assert_eq!(ga.population.len(), 40);
        for sol in &ga.population {
            assert!(sol.evaluated);
        }
    }
}
