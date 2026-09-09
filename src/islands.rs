//! Island-model parallelism: run several independent populations, then merge their archives.
//!
//! One large population explores from a single gene pool, so it can converge onto one region
//! of the front and stay there. N smaller populations with different seeds explore independently
//! and are pooled at the end, which trades some per-island depth for coverage.
//!
//! Islands are embarrassingly parallel, so this is a Rayon `par_iter` over whole GA runs —
//! coarser and cheaper to coordinate than the per-evaluation parallelism inside a single run.
//!
//! ```ignore
//! let front = run_islands(problem, IslandConfig { islands: 8, population_size: 50, ..Default::default() });
//! ```

use crate::core::{Problem, Solution};
use crate::dominance::{fast_non_dominated_sort, ParetoDominance};
use crate::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use rayon::prelude::*;
use std::sync::Arc;

/// Settings for an island-model run.
#[derive(Debug, Clone)]
pub struct IslandConfig {
    /// Number of independent populations.
    pub islands: usize,
    /// Population size *per island*.
    pub population_size: usize,
    /// Evaluation budget *per island*. Total work is `islands * max_nfe_per_island`.
    pub max_nfe_per_island: usize,
    /// Base seed. Island `i` runs with `seed + i`, so the whole run is reproducible while the
    /// islands still differ from one another. `None` draws fresh entropy per island.
    pub seed: Option<u64>,
    /// Execution mode *within* each island. Islands already run in parallel with each other, so
    /// `Sequential` is usually right — nested Rayon parallelism just adds contention.
    pub execution_mode: ExecutionMode,
}

impl Default for IslandConfig {
    fn default() -> Self {
        Self {
            islands: 4,
            population_size: 50,
            max_nfe_per_island: 10_000,
            seed: Some(0),
            // ponytail: sequential inside each island — islands supply the parallelism.
            // Switch to MultiThreaded only if you run very few islands on many cores.
            execution_mode: ExecutionMode::Sequential,
        }
    }
}

/// Run `config.islands` independent NSGA-II populations in parallel and return the
/// non-dominated set of their pooled archives.
///
/// Returns the merged Pareto front. Each island is seeded distinctly, so with `seed: Some(k)`
/// the whole run reproduces exactly.
pub fn run_islands(problem: Arc<Problem>, config: IslandConfig) -> Vec<Solution> {
    assert!(config.islands > 0, "islands must be > 0");

    let archives: Vec<Vec<Solution>> = (0..config.islands)
        .into_par_iter()
        .map(|i| {
            let mut ga = NSGAII::new(
                Arc::clone(&problem),
                config.population_size,
                config.execution_mode,
            );
            if let Some(seed) = config.seed {
                // Distinct seed per island: same base seed reproduces, islands still differ.
                ga = ga.with_seed(seed.wrapping_add(i as u64));
            }
            ga.run(config.max_nfe_per_island);
            ga.get_archive().to_vec()
        })
        .collect();

    merge_fronts(archives)
}

/// Pool several archives and keep only the globally non-dominated solutions.
pub fn merge_fronts(archives: Vec<Vec<Solution>>) -> Vec<Solution> {
    let pooled: Vec<Solution> = archives.into_iter().flatten().collect();
    if pooled.is_empty() {
        return Vec::new();
    }
    let fronts = fast_non_dominated_sort(&pooled, &ParetoDominance);
    match fronts.first() {
        Some(front) => front.iter().map(|&i| pooled[i].clone()).collect(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EvalFn, Encoding, Problem};
    use crate::gatypes::{Real, SolutionDataTypes};

    fn zdt1_like() -> Problem {
        Problem {
            solution_length: 4,
            number_of_objectives: 2,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1]),
            solution_data_types: (0..4)
                .map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))))
                .collect(),
            variable_constraints: None,
            encoding: Encoding::PerGene,
            eval_fn: EvalFn::Single(|x| {
                let f1 = x[0];
                let g = 1.0 + 9.0 / 3.0 * x[1..].iter().sum::<f64>();
                vec![f1, g * (1.0 - (f1 / g).sqrt())]
            }),
        }
    }

    #[test]
    fn test_run_islands_returns_nondominated_front() {
        let problem = Arc::new(zdt1_like());
        let front = run_islands(
            Arc::clone(&problem),
            IslandConfig { islands: 4, population_size: 20, max_nfe_per_island: 800, seed: Some(7), ..Default::default() },
        );
        assert!(!front.is_empty(), "island run produced no solutions");
        assert!(front.iter().all(|s| s.evaluated));
        // The merged result must itself be a single non-dominated front.
        let fronts = fast_non_dominated_sort(&front, &ParetoDominance);
        assert_eq!(fronts.len(), 1, "merged front still contains dominated solutions");
    }

    #[test]
    fn test_run_islands_is_reproducible() {
        let problem = Arc::new(zdt1_like());
        let run = || {
            let cfg = IslandConfig { islands: 3, population_size: 16, max_nfe_per_island: 600, seed: Some(42), ..Default::default() };
            let mut objs: Vec<Vec<f64>> = run_islands(Arc::clone(&problem), cfg)
                .iter()
                .map(|s| s.objective_fitness_values.to_vec())
                .collect();
            // Islands finish in nondeterministic order, so compare as a set.
            objs.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap()));
            objs
        };
        assert_eq!(run(), run());
    }

    /// More islands must not lose the front: pooling can only add candidates.
    #[test]
    fn test_islands_cover_at_least_single_island() {
        let problem = Arc::new(zdt1_like());
        let one = run_islands(
            Arc::clone(&problem),
            IslandConfig { islands: 1, population_size: 20, max_nfe_per_island: 800, seed: Some(3), ..Default::default() },
        );
        let many = run_islands(
            Arc::clone(&problem),
            IslandConfig { islands: 4, population_size: 20, max_nfe_per_island: 800, seed: Some(3), ..Default::default() },
        );
        assert!(!one.is_empty() && !many.is_empty());
        // Island 0 is seeded identically in both runs, so its solutions are candidates in the
        // pooled run; the merged front must be at least as large.
        assert!(many.len() >= one.len(), "pooling islands shrank the front");
    }
}
