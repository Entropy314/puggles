use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crate::core::{EvalFn, Problem, Solution};
use crate::genetic_operators::mutation::{MutationManager, PolynomialMutation};
use crate::genetic_operators::crossover::CrossoverManager;
use crate::genetic_operators::selectors::CrowdingTournamentSelector;
use crate::dominance::{ParetoDominance, fast_non_dominated_sort, crowding_distance};
use rand::rngs::SmallRng;
use rand::SeedableRng;

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
    /// Master RNG for initial-population generation and the (sequential) variation loop.
    /// Seed it with [`with_seed`](NSGAII::with_seed) for reproducible runs.
    rng: SmallRng,
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
        let n = problem.solution_length;
        let mut mutation_manager = MutationManager::new();
        mutation_manager.set_default_real_mutation(Arc::new(PolynomialMutation::new(
            Some(1.0 / n as f64),
            Some(20.0),
        )));
        Self {
            problem,
            population_size,
            population: Vec::with_capacity(population_size),
            nfe: AtomicUsize::new(0),
            execution_mode,
            selector: CrowdingTournamentSelector::new(2, None),
            dominance: ParetoDominance,
            mutation_manager,
            crossover_manager: CrossoverManager::new(),
            archive: Vec::new(),
            rng: SmallRng::from_entropy(),
            ranks: Vec::new(),
            crowding_distances: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu_evaluator: None,
        }
    }

    /// Seed every source of randomness (initial population, crossover, mutation, and
    /// tournament selection) so the run is reproducible. Builder method — call before
    /// `initialize()`/`run()`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self.selector = CrowdingTournamentSelector::new(2, Some(seed));
        self
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

    /// NSGA-II environmental selection: from a combined population of size 2N, select the best
    /// N by front rank then crowding distance. Takes `combined` by value and **moves** the
    /// survivors out (no cloning), and records their rank + crowding distance (from the combined
    /// sort) into `self.ranks`/`self.crowding_distances` so the next generation's parent
    /// selection needs no extra non-dominated sort.
    fn environmental_selection(&mut self, combined: Vec<Solution>, target_size: usize) -> Vec<Solution> {
        let fronts = fast_non_dominated_sort(&combined, &self.dominance);
        let n = combined.len();
        let mut keep = vec![false; n];
        let mut rank_of = vec![0usize; n];
        let mut crowd_of = vec![0.0f64; n];
        let mut count = 0;

        for (rank, front) in fronts.iter().enumerate() {
            if count + front.len() <= target_size {
                // Entire front fits.
                let cd = crowding_distance(&combined, front);
                for (local_idx, &global_idx) in front.iter().enumerate() {
                    keep[global_idx] = true;
                    rank_of[global_idx] = rank;
                    crowd_of[global_idx] = cd[local_idx];
                }
                count += front.len();
            } else {
                // Partial front: pick the least-crowded (highest crowding distance).
                let remaining = target_size - count;
                let cd = crowding_distance(&combined, front);
                let mut indexed_cd: Vec<(usize, f64)> = cd.into_iter().enumerate().collect();
                indexed_cd.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for &(local_idx, cdist) in indexed_cd.iter().take(remaining) {
                    let global_idx = front[local_idx];
                    keep[global_idx] = true;
                    rank_of[global_idx] = rank;
                    crowd_of[global_idx] = cdist;
                }
                break;
            }
        }

        // Move survivors out (no clone) and record their rank/crowding aligned to the result order.
        self.ranks.clear();
        self.crowding_distances.clear();
        let mut selected: Vec<Solution> = Vec::with_capacity(target_size);
        for (i, sol) in combined.into_iter().enumerate() {
            if keep[i] {
                self.ranks.push(rank_of[i]);
                self.crowding_distances.push(crowd_of[i]);
                selected.push(sol);
            }
        }
        selected
    }

    /// Archive non-dominated feasible solutions, capped at population_size.
    /// When the non-dominated set exceeds the cap, prune by crowding distance
    /// (drop the most crowded solutions) to keep diversity.
    pub fn update_archive(&mut self) {
        // Only population front-0 can enter the archive: any feasible non-front-0 solution is
        // dominated by a feasible rank-0 one (a feasible solution always dominates an infeasible
        // one on constraints), so nondominated(archive ∪ feasible) == nondominated(archive ∪
        // front0-feasible). Exact — same archive — but clones only the candidates, not every
        // feasible member. Falls back to all-feasible if ranks are stale (called outside the loop).
        let ranks_valid = self.ranks.len() == self.population.len();
        let ranks = &self.ranks;
        let feasible: Vec<Solution> = self
            .population
            .iter()
            .enumerate()
            .filter(|(i, s)| s.feasible && s.evaluated && (!ranks_valid || ranks[*i] == 0))
            .map(|(_, s)| s.clone())
            .collect();

        if feasible.is_empty() {
            return;
        }

        let mut combined = std::mem::take(&mut self.archive);
        combined.extend(feasible);

        let fronts = fast_non_dominated_sort(&combined, &self.dominance);
        let front0: Vec<Solution> = if !fronts.is_empty() {
            fronts[0].iter().map(|&idx| combined[idx].clone()).collect()
        } else {
            Vec::new()
        };

        self.archive = if front0.len() <= self.population_size {
            front0
        } else {
            // Trim to population_size by dropping the most crowded solutions.
            let indices: Vec<usize> = (0..front0.len()).collect();
            let cd = crowding_distance(&front0, &indices);
            let mut indexed: Vec<(usize, f64)> = cd.into_iter().enumerate().collect();
            // Sort descending by crowding distance (most spread-out first).
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed[..self.population_size]
                .iter()
                .map(|&(i, _)| front0[i].clone())
                .collect()
        };
    }

    pub fn get_archive(&self) -> &[Solution] {
        &self.archive
    }

    pub fn get_nfe(&self) -> usize {
        self.nfe.load(Ordering::Relaxed)
    }

    /// Snapshot population, archive, and NFE for checkpointing (see [`crate::checkpoint`]).
    pub fn save_state(&self) -> crate::checkpoint::GaState {
        use crate::checkpoint::{GaState, SolutionRecord};
        GaState {
            population: self.population.iter().map(SolutionRecord::from_solution).collect(),
            archive: self.archive.iter().map(SolutionRecord::from_solution).collect(),
            nfe: self.get_nfe(),
        }
    }

    /// Restore a checkpoint, rehydrating each solution against this GA's problem.
    /// `run()` afterwards resumes from the restored NFE (already-evaluated solutions are skipped).
    pub fn load_state(&mut self, state: crate::checkpoint::GaState) {
        let problem = &self.problem;
        self.population = state.population.into_iter().map(|r| r.into_solution(Arc::clone(problem))).collect();
        self.archive = state.archive.into_iter().map(|r| r.into_solution(Arc::clone(problem))).collect();
        self.nfe = AtomicUsize::new(state.nfe);
    }

    /// The execution path population evaluation will *actually* take, which can
    /// differ from `execution_mode`:
    /// - A batch objective (`EvalFn::Batch`) bypasses per-solution parallelism —
    ///   the batch fn is called once on the calling thread, reported as `Sequential`.
    /// - `GPU` downgrades to `MultiThreaded` unless built with `--features gpu`
    ///   *and* a `GpuEvaluator` is attached via `with_gpu_evaluator()`.
    pub fn effective_mode(&self) -> ExecutionMode {
        if matches!(self.problem.eval_fn, EvalFn::Batch(_)) {
            return ExecutionMode::Sequential;
        }
        if self.execution_mode == ExecutionMode::GPU {
            #[cfg(feature = "gpu")]
            let gpu_ready = self.gpu_evaluator.is_some();
            #[cfg(not(feature = "gpu"))]
            let gpu_ready = false;
            return if gpu_ready { ExecutionMode::GPU } else { ExecutionMode::MultiThreaded };
        }
        self.execution_mode
    }

    /// Run one generation, creating at most `max_offspring` new solutions.
    /// Passing `self.population_size` is equivalent to a full iteration.
    /// Used by `run()` to honour the NFE budget without overshoot.
    pub fn iterate_n(&mut self, max_offspring: usize) {
        // Ranks/crowding are refreshed by environmental_selection each generation (standard
        // NSGA-II: parent selection reuses the combined-sort ranks). Only sort here when
        // they're stale — i.e. the first iteration after initialize/load.
        if self.ranks.len() != self.population.len() {
            self.assign_fitness();
        }

        let parent_indices = self.selector.select_indices(
            self.population_size,
            &self.ranks,
            &self.crowding_distances,
        );

        // Index parents directly instead of cloning them all into a scratch Vec.
        let np = parent_indices.len();
        let mut offspring: Vec<Solution> = Vec::with_capacity(max_offspring);
        let mut i = 0;
        while offspring.len() < max_offspring {
            let p1 = &self.population[parent_indices[i % np]];
            let p2 = &self.population[parent_indices[(i + 1) % np]];
            i += 2;

            let children = self.crossover_manager.perform_crossover(p1, p2, &mut self.rng);
            for child in children {
                let mutated = self.mutation_manager.mutate(&child, &mut self.rng);
                if offspring.len() < max_offspring {
                    offspring.push(mutated);
                }
            }
        }

        self.evaluate_population(&mut offspring);

        let mut combined = std::mem::take(&mut self.population);
        combined.extend(offspring);
        self.population = self.environmental_selection(combined, self.population_size);
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

    /// Run until the NFE budget is spent OR the archive stops improving.
    /// Convergence signal: the best value of each objective (respecting `direction`)
    /// hasn't improved by more than `epsilon` for `patience` consecutive generations.
    /// Returns the number of generations run.
    ///
    /// ponytail: per-objective-best plateau — cheap and reference-free, but it tracks
    /// front *extent*, not spread. Swap in a hypervolume plateau (see `metrics`) if you
    /// need diversity-aware stopping.
    pub fn run_until_converged(&mut self, max_nfe: usize, patience: usize, epsilon: f64) -> usize {
        if self.population.is_empty() {
            self.initialize();
        }
        let mut pop = std::mem::take(&mut self.population);
        self.evaluate_population(&mut pop);
        self.population = pop;

        let n_obj = self.problem.number_of_objectives;
        let dirs = self.problem.direction.clone().unwrap_or_else(|| vec![-1; n_obj]);
        let mut best = vec![f64::INFINITY; n_obj]; // in "lower is better" space
        let mut stale = 0usize;
        let mut gens = 0usize;

        while self.nfe.load(Ordering::Relaxed) < max_nfe {
            let remaining = max_nfe - self.nfe.load(Ordering::Relaxed);
            self.iterate_n(remaining.min(self.population_size));
            self.update_archive();
            gens += 1;

            let src = if self.archive.is_empty() { &self.population } else { &self.archive };
            let mut improved = false;
            for (i, d) in dirs.iter().enumerate() {
                // Map to "lower is better": negate maximized objectives.
                let cur = src
                    .iter()
                    .map(|s| if *d == 1 { -s.objective_fitness_values[i] } else { s.objective_fitness_values[i] })
                    .fold(f64::INFINITY, f64::min);
                if cur < best[i] - epsilon {
                    best[i] = cur;
                    improved = true;
                }
            }
            if improved {
                stale = 0;
            } else {
                stale += 1;
                if stale >= patience {
                    break;
                }
            }
        }
        gens
    }
}

impl NSGAII {
    /// Generate the initial population from the master RNG. Kept sequential so the
    /// initial population is deterministic under a fixed seed (generation is cheap —
    /// a few hundred random vectors — so parallelizing it wasn't buying anything).
    pub fn initialize(&mut self) {
        let problem = Arc::clone(&self.problem);
        let mut pop = Vec::with_capacity(self.population_size);
        for _ in 0..self.population_size {
            let solution = problem.generate_solution(&mut self.rng);
            pop.push(Solution {
                problem: Arc::clone(&problem),
                solution,
                objective_fitness_values: Default::default(),
                constraint_values: Default::default(),
                evaluated: false,
                constraint_violation: 0,
                feasible: false,
            });
        }
        self.population = pop;
    }

    pub fn evaluate_population(&mut self, population: &mut Vec<Solution>) {
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
                    population[global_i].objective_fitness_values = outputs[local_i].clone().into();
                    population[global_i].evaluated = true;
                    let cv = population[global_i].evaluate_constraints();
                    population[global_i].constraint_values = cv.into();
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
                        population[global_i].objective_fitness_values = outputs[local_i].clone().into();
                        population[global_i].evaluated = true;
                        let cv = population[global_i].evaluate_constraints();
                        population[global_i].constraint_values = cv.into();
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
                    "puggles warning: ExecutionMode::GPU selected but no GpuEvaluator attached. \
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

    pub fn iterate(&mut self) {
        self.iterate_n(self.population_size);
    }

    pub fn run(&mut self, max_nfe: usize) {
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
            variable_constraints: None,
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
            variable_constraints: None,
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
            variable_constraints: None,
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
    fn test_effective_mode() {
        let problem = Arc::new(setup_single_objective_problem());

        // Sequential / MultiThreaded pass through unchanged.
        let ga = NSGAII::new(Arc::clone(&problem), 10, ExecutionMode::Sequential);
        assert_eq!(ga.effective_mode(), ExecutionMode::Sequential);

        // GPU requested with no evaluator attached (and no `gpu` feature) → CPU fallback.
        let ga = NSGAII::new(Arc::clone(&problem), 10, ExecutionMode::GPU);
        assert_eq!(ga.effective_mode(), ExecutionMode::MultiThreaded);

        // A batch objective bypasses per-solution parallelism regardless of mode.
        let batch_problem = Arc::new(Problem {
            solution_length: 3,
            number_of_objectives: 1,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1]),
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(-1.0), Some(1.0))),
                SolutionDataTypes::Real(Real::new(Some(-1.0), Some(1.0))),
                SolutionDataTypes::Real(Real::new(Some(-1.0), Some(1.0))),
            ],
            variable_constraints: None,
            eval_fn: EvalFn::Batch(|batch| batch.iter().map(|x| vec![x.iter().sum::<f64>()]).collect()),
        });
        let ga = NSGAII::new(batch_problem, 10, ExecutionMode::MultiThreaded);
        assert_eq!(ga.effective_mode(), ExecutionMode::Sequential);
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
    fn test_checkpoint_roundtrip() {
        let problem = Arc::new(setup_multi_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential).with_seed(7);
        ga.run(400);
        let nfe = ga.get_nfe();
        let before: Vec<_> = ga.get_archive().iter().map(|s| s.objective_fitness_values.clone()).collect();

        let state = ga.save_state();
        let mut ga2 = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential);
        ga2.load_state(state);

        assert_eq!(ga2.get_nfe(), nfe);
        let after: Vec<_> = ga2.get_archive().iter().map(|s| s.objective_fitness_values.clone()).collect();
        assert_eq!(before, after, "restored archive must match");
        // Resume continues from the restored NFE.
        ga2.run(800);
        assert!(ga2.get_nfe() >= 800);
    }

    #[test]
    fn test_seeded_runs_are_reproducible() {
        let problem = Arc::new(setup_multi_objective_problem());
        let run = || {
            let mut ga = NSGAII::new(Arc::clone(&problem), 30, ExecutionMode::Sequential).with_seed(123);
            ga.run(600);
            ga.get_archive()
                .iter()
                .map(|s| s.objective_fitness_values.clone())
                .collect::<Vec<_>>()
        };
        let a = run();
        let b = run();
        assert!(!a.is_empty());
        assert_eq!(a, b, "same seed must produce an identical archive");
    }

    #[test]
    fn test_run_until_converged_stops_early() {
        // Sphere converges fast; with a big NFE budget and small patience the
        // plateau check should stop well before exhausting the budget.
        let problem = Arc::new(setup_single_objective_problem());
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential);
        let gens = ga.run_until_converged(1_000_000, 10, 1e-6);
        assert!(ga.get_nfe() < 1_000_000, "should stop on plateau, not exhaust NFE");
        assert!(gens > 0);
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
