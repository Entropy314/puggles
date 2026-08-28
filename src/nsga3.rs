//! NSGA-III — reference-point-based many-objective optimizer (Deb & Jain, 2014).
//!
//! Same variation machinery as [`crate::genetic_algorithms_v2::NSGAII`], but survival
//! selection replaces crowding distance with **structured reference points**: solutions
//! are associated to reference directions and the least-represented directions are
//! preferred. This keeps diversity in problems with >3 objectives, where crowding
//! distance loses discriminating power.
//!
//! ponytail: normalization uses ideal-point translation + per-objective range (a common
//! simplification), not the full achievement-scalarizing hyperplane-intercept method.
//! Good enough for typical use; swap in exact intercepts if a benchmark needs it.

use crate::core::{Problem, Solution};
use crate::dominance::{crowding_distance, fast_non_dominated_sort, ParetoDominance};
use crate::genetic_algorithms_v2::ExecutionMode;
use crate::genetic_operators::crossover::CrossoverManager;
use crate::genetic_operators::mutation::{MutationManager, PolynomialMutation};
use crate::genetic_operators::selectors::CrowdingTournamentSelector;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Structured reference points on the unit simplex (Das & Dennis): every point has
/// non-negative coordinates summing to 1, on a grid of `divisions` steps. For `m`
/// objectives the count is C(m + divisions - 1, divisions).
pub fn das_dennis(m: usize, divisions: usize) -> Vec<Vec<f64>> {
    fn rec(m: usize, idx: usize, left: usize, div: usize, point: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if idx == m - 1 {
            point[idx] = left;
            out.push(point.clone());
            return;
        }
        for i in 0..=left {
            point[idx] = i;
            rec(m, idx + 1, left - i, div, point, out);
        }
    }
    let mut out = Vec::new();
    let mut point = vec![0usize; m];
    rec(m, 0, divisions, divisions, &mut point, &mut out);
    out.into_iter()
        .map(|p| p.iter().map(|&v| v as f64 / divisions as f64).collect())
        .collect()
}

pub struct NSGAIII {
    pub problem: Arc<Problem>,
    pub population_size: usize,
    pub population: Vec<Solution>,
    pub archive: Vec<Solution>,
    nfe: AtomicUsize,
    execution_mode: ExecutionMode,
    selector: CrowdingTournamentSelector,
    dominance: ParetoDominance,
    mutation_manager: MutationManager,
    crossover_manager: CrossoverManager,
    rng: SmallRng,
    reference_points: Vec<Vec<f64>>,
    ranks: Vec<usize>,
    crowding: Vec<f64>,
}

impl NSGAIII {
    /// `divisions` controls reference-point density (e.g. 12 for 3 objectives → 91 points).
    /// The population size is taken as the reference-point count rounded up to a multiple
    /// of 4 if you pass 0, otherwise your value is used as-is.
    pub fn new(problem: Arc<Problem>, population_size: usize, divisions: usize, execution_mode: ExecutionMode) -> Self {
        let n = problem.solution_length;
        let m = problem.number_of_objectives;
        let mut mutation_manager = MutationManager::new();
        mutation_manager.set_default_real_mutation(Arc::new(PolynomialMutation::new(Some(1.0 / n as f64), Some(20.0))));
        let reference_points = das_dennis(m, divisions);
        let population_size = if population_size == 0 {
            (reference_points.len() + 3) / 4 * 4
        } else {
            population_size
        };
        Self {
            problem,
            population_size,
            population: Vec::with_capacity(population_size),
            archive: Vec::new(),
            nfe: AtomicUsize::new(0),
            execution_mode,
            selector: CrowdingTournamentSelector::new(2, None),
            dominance: ParetoDominance,
            mutation_manager,
            crossover_manager: CrossoverManager::new(),
            rng: SmallRng::from_entropy(),
            reference_points,
            ranks: Vec::new(),
            crowding: Vec::new(),
        }
    }

    /// Seed all randomness for a reproducible run. Builder-style; call before `run`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self.selector = CrowdingTournamentSelector::new(2, Some(seed));
        self
    }

    pub fn get_archive(&self) -> &[Solution] {
        &self.archive
    }
    pub fn get_nfe(&self) -> usize {
        self.nfe.load(Ordering::Relaxed)
    }
    pub fn reference_points(&self) -> &[Vec<f64>] {
        &self.reference_points
    }

    fn initialize(&mut self) {
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

    fn evaluate(&self, population: &mut Vec<Solution>) -> usize {
        let n: usize = match self.execution_mode {
            ExecutionMode::Sequential => population.iter_mut().filter(|s| !s.evaluated).map(|s| { s.evaluate(); 1 }).sum(),
            _ => population.par_iter_mut().filter(|s| !s.evaluated).map(|s| { s.evaluate(); 1 }).sum(),
        };
        self.nfe.fetch_add(n, Ordering::Relaxed);
        n
    }

    /// Run until the NFE budget is spent. Auto-initializes.
    pub fn run(&mut self, max_nfe: usize) {
        if self.population.is_empty() {
            self.initialize();
        }
        let mut pop = std::mem::take(&mut self.population);
        self.evaluate(&mut pop);
        self.population = pop;

        while self.nfe.load(Ordering::Relaxed) < max_nfe {
            self.iterate();
        }
        self.archive = self.nondominated(&self.population);
    }

    fn iterate(&mut self) {
        // Parent selection by rank + crowding tournament (crowding is cheap and harmless).
        self.assign_fitness();
        let parent_indices = self.selector.select_indices(self.population_size, &self.ranks, &self.crowding);
        let parents: Vec<Solution> = parent_indices.iter().map(|&i| self.population[i].clone()).collect();

        let mut offspring: Vec<Solution> = Vec::with_capacity(self.population_size);
        let mut i = 0;
        while offspring.len() < self.population_size {
            let p1 = &parents[i % parents.len()];
            let p2 = &parents[(i + 1) % parents.len()];
            i += 2;
            for child in self.crossover_manager.perform_crossover(p1, p2, &mut self.rng) {
                if offspring.len() < self.population_size {
                    offspring.push(self.mutation_manager.mutate(&child, &mut self.rng));
                }
            }
        }
        self.evaluate(&mut offspring);

        let mut combined = std::mem::take(&mut self.population);
        combined.extend(offspring);
        self.population = self.environmental_selection(&combined);
    }

    fn assign_fitness(&mut self) {
        let n = self.population.len();
        self.ranks.resize(n, 0);
        self.crowding.resize(n, 0.0);
        let fronts = fast_non_dominated_sort(&self.population, &self.dominance);
        for (rank, front) in fronts.iter().enumerate() {
            let cd = crowding_distance(&self.population, front);
            for (local, &global) in front.iter().enumerate() {
                self.ranks[global] = rank;
                self.crowding[global] = cd[local];
            }
        }
    }

    fn nondominated(&self, pop: &[Solution]) -> Vec<Solution> {
        let fronts = fast_non_dominated_sort(pop, &self.dominance);
        fronts.first().map(|f| f.iter().map(|&i| pop[i].clone()).collect()).unwrap_or_default()
    }

    /// NSGA-III survival: fill by front, then niche the splitting front against reference points.
    fn environmental_selection(&self, combined: &[Solution]) -> Vec<Solution> {
        let target = self.population_size;
        let fronts = fast_non_dominated_sort(combined, &self.dominance);

        let mut selected: Vec<usize> = Vec::new();
        let mut splitting: Vec<usize> = Vec::new();
        for front in &fronts {
            if selected.len() + front.len() <= target {
                selected.extend(front.iter().copied());
            } else {
                splitting = front.clone();
                break;
            }
        }
        if selected.len() == target || splitting.is_empty() {
            selected.truncate(target);
            return selected.iter().map(|&i| combined[i].clone()).collect();
        }

        // Normalize objectives (minimization space) over selected ∪ splitting.
        let m = self.problem.number_of_objectives;
        let dirs = self.problem.direction.clone().unwrap_or_else(|| vec![-1; m]);
        let mut pool: Vec<usize> = selected.clone();
        pool.extend(splitting.iter().copied());
        let norm = normalize(combined, &pool, m, &dirs);

        // Associate each pool member with its nearest reference point (perpendicular distance).
        let (assoc, dist): (Vec<usize>, Vec<f64>) = norm.iter().map(|p| associate(p, &self.reference_points)).unzip();

        // Niche counts from the already-selected members (pool[0..selected.len()]).
        let mut niche = vec![0usize; self.reference_points.len()];
        for k in 0..selected.len() {
            niche[assoc[k]] += 1;
        }

        // Candidate splitting members = pool indices selected.len()..pool.len().
        let mut avail: Vec<usize> = (selected.len()..pool.len()).collect();
        let need = target - selected.len();
        let mut chosen: Vec<usize> = Vec::new();

        while chosen.len() < need {
            // Reference point(s) with the smallest niche count that still have candidates.
            let refs_with_candidates: Vec<usize> = avail.iter().map(|&k| assoc[k]).collect();
            let min_niche = refs_with_candidates.iter().map(|&r| niche[r]).min().unwrap();
            let cand_refs: Vec<usize> = refs_with_candidates.into_iter().filter(|&r| niche[r] == min_niche).collect();
            // Pick one such reference point (deterministic: smallest index).
            let j = *cand_refs.iter().min().unwrap();

            // Candidates associated with j.
            let members: Vec<usize> = avail.iter().copied().filter(|&k| assoc[k] == j).collect();
            // If this niche is empty, take the closest candidate; otherwise any (take closest too).
            let pick = *members.iter().min_by(|&&a, &&b| dist[a].partial_cmp(&dist[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap();

            chosen.push(pool[pick]);
            niche[j] += 1;
            avail.retain(|&k| k != pick);
        }

        selected.extend(chosen);
        selected.iter().map(|&i| combined[i].clone()).collect()
    }
}

/// Translate by the ideal point and divide by the per-objective range, in minimization space.
fn normalize(pop: &[Solution], idx: &[usize], m: usize, dirs: &[i8]) -> Vec<Vec<f64>> {
    let val = |i: usize, o: usize| {
        let v = pop[i].objective_fitness_values[o];
        if dirs[o] == 1 { -v } else { v } // maximize → negate into minimization space
    };
    let mut ideal = vec![f64::INFINITY; m];
    let mut nadir = vec![f64::NEG_INFINITY; m];
    for &i in idx {
        for o in 0..m {
            let v = val(i, o);
            ideal[o] = ideal[o].min(v);
            nadir[o] = nadir[o].max(v);
        }
    }
    idx.iter()
        .map(|&i| {
            (0..m)
                .map(|o| {
                    let range = nadir[o] - ideal[o];
                    if range > f64::EPSILON { (val(i, o) - ideal[o]) / range } else { 0.0 }
                })
                .collect()
        })
        .collect()
}

/// Nearest reference point by perpendicular distance from `point` to the line through the origin
/// and the reference point. Returns (reference index, distance).
fn associate(point: &[f64], refs: &[Vec<f64>]) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (j, r) in refs.iter().enumerate() {
        let rr: f64 = r.iter().map(|x| x * x).sum();
        if rr < f64::EPSILON {
            continue;
        }
        let dot: f64 = point.iter().zip(r).map(|(p, x)| p * x).sum();
        let t = dot / rr;
        let perp: f64 = point.iter().zip(r).map(|(p, x)| (p - t * x).powi(2)).sum::<f64>().sqrt();
        if perp < best.1 {
            best = (j, perp);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EvalFn, Problem};
    use crate::gatypes::{Real, SolutionDataTypes};

    fn dtlz_like() -> Problem {
        // 3-objective, 6-variable toy problem, minimize all.
        Problem {
            solution_length: 6,
            number_of_objectives: 3,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1, -1]),
            solution_data_types: (0..6).map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))).collect(),
            variable_constraints: None,
            eval_fn: EvalFn::Single(|x| {
                let g: f64 = x[2..].iter().map(|v| (v - 0.5).powi(2)).sum();
                vec![(1.0 + g) * x[0], (1.0 + g) * (1.0 - x[0]) * x[1], (1.0 + g) * (1.0 - x[0]) * (1.0 - x[1])]
            }),
        }
    }

    #[test]
    fn test_das_dennis_count() {
        // 3 objectives, 4 divisions → C(3+4-1, 4) = C(6,4) = 15 points, all summing to 1.
        let pts = das_dennis(3, 4);
        assert_eq!(pts.len(), 15);
        for p in &pts {
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_nsga3_runs_and_keeps_size() {
        let problem = Arc::new(dtlz_like());
        let mut ga = NSGAIII::new(Arc::clone(&problem), 24, 4, ExecutionMode::Sequential).with_seed(1);
        ga.run(2_000);
        assert!(ga.get_nfe() >= 2_000);
        assert_eq!(ga.population.len(), 24);
        assert!(!ga.get_archive().is_empty());
        for s in ga.get_archive() {
            assert!(s.evaluated);
        }
    }

    #[test]
    fn test_nsga3_reproducible() {
        let problem = Arc::new(dtlz_like());
        let run = || {
            let mut ga = NSGAIII::new(Arc::clone(&problem), 24, 4, ExecutionMode::Sequential).with_seed(42);
            ga.run(1_200);
            ga.get_archive().iter().map(|s| s.objective_fitness_values.clone()).collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
