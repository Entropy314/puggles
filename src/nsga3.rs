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
use crate::genetic_algorithms_v2::{ExecutionInfo, ExecutionMode, GenerationTimes};
use crate::genetic_operators::crossover::CrossoverManager;
use crate::genetic_operators::mutation::MutationManager;
use crate::genetic_operators::selectors::CrowdingTournamentSelector;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
    /// The uniform Das–Dennis grid, kept so a focus can be cleared.
    base_reference_points: Vec<Vec<f64>>,
    ranks: Vec<usize>,
    crowding: Vec<f64>,
    /// Each population member's front, as survival found it — exact for the population, since
    /// it is whole fronts 0..k-1 plus part of front k — so the next generation's parent
    /// selection and the archive need no second sort. `None` once the population may have been
    /// changed from outside (an observer's `&mut Self`, a new `run`).
    survivor_ranks: Option<Vec<usize>>,
    generation_times: GenerationTimes,
}

impl NSGAIII {
    /// `divisions` controls reference-point density (e.g. 12 for 3 objectives → 91 points).
    /// The population size is taken as the reference-point count rounded up to a multiple
    /// of 4 if you pass 0, otherwise your value is used as-is.
    pub fn new(problem: Arc<Problem>, population_size: usize, divisions: usize, execution_mode: ExecutionMode) -> Self {
        let n = problem.solution_length;
        let m = problem.number_of_objectives;
        let mutation_manager = MutationManager::new(n);
        let reference_points = das_dennis(m, divisions);
        let population_size = if population_size == 0 {
            (reference_points.len() + 3) / 4 * 4
        } else {
            population_size
        };
        assert!(
            population_size > 0,
            "population_size must be > 0 (a zero population produces no offspring, so a run \
             can never spend its evaluation budget)"
        );
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
            base_reference_points: reference_points.clone(),
            reference_points,
            ranks: Vec::new(),
            crowding: Vec::new(),
            survivor_ranks: None,
            generation_times: GenerationTimes::default(),
        }
    }

    /// Seed all randomness for a reproducible run. Builder-style; call before `run`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self.selector = CrowdingTournamentSelector::new(2, Some(seed));
        self
    }

    /// Move the archive out (leaving it empty), e.g. when the optimizer is about to be dropped.
    pub fn take_archive(&mut self) -> Vec<Solution> {
        std::mem::take(&mut self.archive)
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

    /// Concentrate the search near `aspiration` (one target value per objective, in objective
    /// space; `NaN` = no preference, use the best value found). The reference grid is shrunk
    /// around the aspiration's direction from the current ideal point (as in R-NSGA-III);
    /// `spread` in (0, 1] sets how tightly (1 = the full front). Takes effect from the next
    /// generation; safe to call mid-run from `run_with`.
    pub fn focus_on(&mut self, aspiration: &[f64], spread: f64) {
        let m = self.problem.number_of_objectives;
        if self.population.is_empty() || aspiration.len() != m {
            return;
        }
        let dirs = self.problem.direction.clone().unwrap_or_else(|| vec![-1; m]);
        let idx: Vec<usize> = (0..self.population.len()).collect();
        let val = |i: usize, o: usize| {
            let v = self.population[i].objective_fitness_values[o];
            if dirs[o] == 1 { -v } else { v }
        };
        let (mut ideal, mut nadir) = (vec![f64::INFINITY; m], vec![f64::NEG_INFINITY; m]);
        for &i in &idx {
            for o in 0..m {
                ideal[o] = ideal[o].min(val(i, o));
                nadir[o] = nadir[o].max(val(i, o));
            }
        }
        // The aspiration in the same normalized minimization space the niching uses.
        let mut d: Vec<f64> = (0..m)
            .map(|o| {
                let z = aspiration[o];
                if !z.is_finite() {
                    return 0.0;
                }
                let z = if dirs[o] == 1 { -z } else { z };
                let range = nadir[o] - ideal[o];
                if range > f64::EPSILON { ((z - ideal[o]) / range).max(0.0) } else { 0.0 }
            })
            .collect();
        let sum: f64 = d.iter().sum();
        if sum > f64::EPSILON {
            d.iter_mut().for_each(|v| *v /= sum);
        } else {
            d = vec![1.0 / m as f64; m];
        }
        let spread = spread.clamp(0.01, 1.0);
        let centroid = 1.0 / m as f64;
        self.reference_points = self
            .base_reference_points
            .iter()
            .map(|p| {
                let mut q: Vec<f64> = p.iter().zip(&d).map(|(pi, di)| (di + spread * (pi - centroid)).max(0.0)).collect();
                let s: f64 = q.iter().sum();
                if s > f64::EPSILON {
                    q.iter_mut().for_each(|v| *v /= s);
                }
                q
            })
            .collect();
    }

    /// Back to the uniform reference grid (the whole front).
    pub fn clear_focus(&mut self) {
        self.reference_points = self.base_reference_points.clone();
    }

    /// Where evaluations run. NSGA-III has no GPU path: `GPU` runs on CPU threads.
    pub fn execution_info(&self) -> ExecutionInfo {
        match self.execution_mode {
            ExecutionMode::Sequential => ExecutionInfo::cpu(ExecutionMode::Sequential, ExecutionMode::Sequential, None),
            ExecutionMode::MultiThreaded => {
                ExecutionInfo::cpu(ExecutionMode::MultiThreaded, ExecutionMode::MultiThreaded, None)
            }
            ExecutionMode::GPU => ExecutionInfo::cpu(
                ExecutionMode::GPU,
                ExecutionMode::MultiThreaded,
                Some("NSGA-III has no GPU evaluation path; evaluating on CPU threads (use NSGAII for GPU)".to_string()),
            ),
        }
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
        self.run_inner(max_nfe, false, |_| true);
    }

    /// `run`, calling `on_generation` after every generation with the archive refreshed to the
    /// current non-dominated front. Return `false` from it to stop early. Used for live
    /// progress / dashboards.
    ///
    /// The closure gets `&mut Self`, so it can steer between generations (e.g. [`focus_on`](Self::focus_on)).
    pub fn run_with(&mut self, max_nfe: usize, on_generation: impl FnMut(&mut Self) -> bool) {
        self.run_inner(max_nfe, true, on_generation);
    }

    /// `run` skips the per-generation archive refresh (a full non-dominated sort) that only
    /// an observer needs, so plain runs cost exactly what they did before `run_with` existed.
    fn run_inner(&mut self, max_nfe: usize, refresh_archive: bool, mut on_generation: impl FnMut(&mut Self) -> bool) {
        self.survivor_ranks = None;
        if self.population.is_empty() {
            self.initialize();
        }
        let mut pop = std::mem::take(&mut self.population);
        self.evaluate(&mut pop);
        self.population = pop;

        self.archive = self.nondominated(&self.population);
        while self.nfe.load(Ordering::Relaxed) < max_nfe {
            let remaining = max_nfe - self.nfe.load(Ordering::Relaxed);
            self.iterate_n(remaining.min(self.population_size));
            if refresh_archive {
                let started = Instant::now();
                self.archive = self.population_front0();
                self.generation_times.archive = started.elapsed().as_secs_f64();
            }
            if !on_generation(self) {
                break;
            }
            if refresh_archive {
                // An observer gets `&mut Self` and may have changed the population.
                self.survivor_ranks = None;
            }
        }
        self.archive = self.population_front0();
    }

    /// Run one generation, creating at most `max_offspring` new solutions, so `run` can honour
    /// its NFE budget exactly instead of overshooting by up to a full population.
    fn iterate_n(&mut self, max_offspring: usize) {
        let started = Instant::now();
        // Parent selection by rank + crowding tournament (crowding is cheap and harmless).
        self.assign_fitness();
        let parent_indices = self.selector.select_indices(self.population_size, &self.ranks, &self.crowding);
        // Index parents directly instead of cloning them all into a scratch Vec.
        let np = parent_indices.len();
        let mut offspring: Vec<Solution> = Vec::with_capacity(max_offspring);
        let mut i = 0;
        while offspring.len() < max_offspring {
            let p1 = &self.population[parent_indices[i % np]];
            let p2 = &self.population[parent_indices[(i + 1) % np]];
            i += 2;
            let (c1, c2) = self.crossover_manager.perform_crossover_pair(p1, p2, &mut self.rng);
            for child in [c1, c2] {
                if offspring.len() < max_offspring {
                    offspring.push(self.mutation_manager.mutate_owned(child, &mut self.rng));
                }
            }
        }
        let varied = Instant::now();
        self.evaluate(&mut offspring);
        let evaluated = Instant::now();

        let mut combined = std::mem::take(&mut self.population);
        combined.extend(offspring);
        let (survivors, ranks) = self.environmental_selection(combined);
        self.population = survivors;
        self.survivor_ranks = Some(ranks);

        let t = &mut self.generation_times;
        t.variation = (varied - started).as_secs_f64();
        t.evaluation = (evaluated - varied).as_secs_f64();
        t.selection = evaluated.elapsed().as_secs_f64();
    }

    /// Time spent in each phase of the most recent generation. `archive` is only measured
    /// under `run_with` (plain `run` skips the per-generation archive refresh).
    pub fn last_generation_times(&self) -> GenerationTimes {
        self.generation_times
    }

    fn assign_fitness(&mut self) {
        let n = self.population.len();
        self.ranks.resize(n, 0);
        self.crowding.resize(n, 0.0);
        let fronts = match self.survivor_ranks.take() {
            Some(ranks) if ranks.len() == n => fronts_from_ranks(&ranks),
            _ => fast_non_dominated_sort(&self.population, &self.dominance),
        };
        for (rank, front) in fronts.iter().enumerate() {
            let cd = crowding_by_index(&self.population, front);
            for (local, &global) in front.iter().enumerate() {
                self.ranks[global] = rank;
                self.crowding[global] = cd[local];
            }
        }
    }

    /// The population's first front, from the survival ranks when they're current.
    fn population_front0(&self) -> Vec<Solution> {
        match &self.survivor_ranks {
            Some(ranks) if ranks.len() == self.population.len() => {
                let front0: Vec<usize> = (0..ranks.len()).filter(|&i| ranks[i] == 0).collect();
                let pop = &self.population;
                crate::core::first_of_each_genome(pop, &front0).into_iter().map(|i| pop[i].clone()).collect()
            }
            _ => self.nondominated(&self.population),
        }
    }

    /// The first front, without repeated genomes (identical copies never dominate each other).
    fn nondominated(&self, pop: &[Solution]) -> Vec<Solution> {
        let fronts = fast_non_dominated_sort(pop, &self.dominance);
        match fronts.into_iter().next() {
            Some(mut f) => {
                f.sort_unstable(); // population order, as `population_front0` gives it
                crate::core::first_of_each_genome(pop, &f).into_iter().map(|i| pop[i].clone()).collect()
            }
            None => Vec::new(),
        }
    }

    /// NSGA-III survival: fill by front, then niche the splitting front against reference points.
    /// Returns the survivors and each one's front.
    fn environmental_selection(&self, combined: Vec<Solution>) -> (Vec<Solution>, Vec<usize>) {
        let target = self.population_size;
        let fronts = fast_non_dominated_sort(&combined, &self.dominance);

        let mut selected: Vec<usize> = Vec::new();
        let mut ranks: Vec<usize> = Vec::with_capacity(target);
        let mut splitting: Vec<usize> = Vec::new();
        for (rank, front) in fronts.iter().enumerate() {
            if selected.len() + front.len() <= target {
                selected.extend(front.iter().copied());
                ranks.resize(selected.len(), rank);
            } else {
                splitting = front.clone();
                break;
            }
        }
        if selected.len() == target || splitting.is_empty() {
            return (take_in_order(combined, &selected), ranks);
        }
        let splitting_rank = ranks.last().map_or(0, |r| r + 1);

        // Normalize objectives (minimization space) over selected ∪ splitting.
        let m = self.problem.number_of_objectives;
        let dirs = self.problem.direction.clone().unwrap_or_else(|| vec![-1; m]);
        let mut pool: Vec<usize> = selected.clone();
        pool.extend(splitting.iter().copied());
        let norm = normalize(&combined, &pool, m, &dirs);

        // Associate each pool member with its nearest reference point (perpendicular distance).
        let ref_norms: Vec<f64> = self.reference_points.iter().map(|r| r.iter().map(|x| x * x).sum()).collect();
        let (assoc, dist): (Vec<usize>, Vec<f64>) =
            norm.iter().map(|p| associate(p, &self.reference_points, &ref_norms)).unzip();

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
            // The reference point with the smallest niche count that still has candidates
            // (ties: smallest index), found in one pass with no scratch Vecs.
            let j = avail.iter().map(|&k| (niche[assoc[k]], assoc[k])).min().expect("candidates left").1;

            // Its closest candidate (ties: the first in `avail`).
            let mut pos = None::<usize>;
            for (p, &k) in avail.iter().enumerate() {
                if assoc[k] != j {
                    continue;
                }
                match pos {
                    Some(b) if dist[avail[b]].partial_cmp(&dist[k]) != Some(std::cmp::Ordering::Greater) => {}
                    _ => pos = Some(p),
                }
            }
            let pick = avail.remove(pos.expect("reference point has a candidate"));

            chosen.push(pool[pick]);
            niche[j] += 1;
        }

        ranks.resize(selected.len() + chosen.len(), splitting_rank);
        selected.extend(chosen);
        (take_in_order(combined, &selected), ranks)
    }
}

/// Crowding distance with ties broken by population index, so it doesn't depend on the order of
/// `front` — a front from survival ranks and one from a fresh sort give the same values, which
/// keeps `run_with` (which must re-sort after its observer) identical to `run`.
fn crowding_by_index(pop: &[Solution], front: &[usize]) -> Vec<f64> {
    let mut sorted = front.to_vec();
    sorted.sort_unstable();
    let by_index = crowding_distance(pop, &sorted);
    front.iter().map(|i| by_index[sorted.binary_search(i).expect("member of front")]).collect()
}

/// Fronts (lists of indices, in order) from each member's front number.
fn fronts_from_ranks(ranks: &[usize]) -> Vec<Vec<usize>> {
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new(); ranks.iter().max().map_or(0, |r| r + 1)];
    for (i, &r) in ranks.iter().enumerate() {
        fronts[r].push(i);
    }
    fronts
}

/// Move `pop[i]` for each (distinct) `i` in `idx` out of `pop`, in that order — no clones.
fn take_in_order(pop: Vec<Solution>, idx: &[usize]) -> Vec<Solution> {
    let mut slots: Vec<Option<Solution>> = pop.into_iter().map(Some).collect();
    idx.iter().map(|&i| slots[i].take().expect("distinct survivor indices")).collect()
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
/// `ref_norms[j]` is the squared length of `refs[j]`.
fn associate(point: &[f64], refs: &[Vec<f64>], ref_norms: &[f64]) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (j, (r, &rr)) in refs.iter().zip(ref_norms).enumerate() {
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

    #[test]
    fn test_nsga3_run_with_matches_run_and_stops_early() {
        let problem = Arc::new(dtlz_like());
        let front = |ga: &NSGAIII| {
            ga.get_archive().iter().map(|s| s.objective_fitness_values.clone()).collect::<Vec<_>>()
        };
        let mut plain = NSGAIII::new(Arc::clone(&problem), 24, 4, ExecutionMode::Sequential).with_seed(3);
        plain.run(1_200);
        let mut observed = NSGAIII::new(Arc::clone(&problem), 24, 4, ExecutionMode::Sequential).with_seed(3);
        observed.run_with(1_200, |ga| {
            assert!(!ga.get_archive().is_empty(), "archive must be current during the run");
            true
        });
        assert_eq!(front(&plain), front(&observed), "observing a run must not change it");

        let mut stopped = NSGAIII::new(Arc::clone(&problem), 24, 4, ExecutionMode::Sequential);
        stopped.run_with(1_000_000, |ga| ga.get_nfe() < 200);
        assert!(stopped.get_nfe() < 1_000, "returning false must stop the run");
    }

    #[test]
    fn test_nsga3_execution_info_has_no_gpu_path() {
        let problem = Arc::new(dtlz_like());
        let info = NSGAIII::new(problem, 24, 4, ExecutionMode::GPU).execution_info();
        assert_eq!(info.effective, ExecutionMode::MultiThreaded);
        assert!(info.note.unwrap().contains("no GPU"));
    }

    #[test]
    fn test_focus_concentrates_the_front_near_the_aspiration() {
        let problem = Arc::new(dtlz_like());
        let spread_of = |focus: bool| {
            let mut ga = NSGAIII::new(Arc::clone(&problem), 48, 6, ExecutionMode::Sequential).with_seed(5);
            ga.run_with(6_000, |ga| {
                if focus && ga.get_nfe() == 48 * 2 {
                    ga.focus_on(&[0.1, 0.1, 0.8], 0.15);
                }
                true
            });
            // Mean distance of the front from the aspiration (in objective space).
            let f = ga.get_archive();
            f.iter().map(|s| {
                let v = &s.objective_fitness_values;
                ((v[0] - 0.1).powi(2) + (v[1] - 0.1).powi(2) + (v[2] - 0.8).powi(2)).sqrt()
            }).sum::<f64>() / f.len() as f64
        };
        let (uniform, focused) = (spread_of(false), spread_of(true));
        assert!(focused < uniform * 0.7, "focused front should crowd the aspiration: {focused} vs {uniform}");
    }

    #[test]
    fn test_archive_has_no_repeated_genomes() {
        let problem = Arc::new(dtlz_like());
        let mut ga = NSGAIII::new(Arc::clone(&problem), 24, 4, ExecutionMode::Sequential).with_seed(2);
        ga.run(2_000);
        let keys: std::collections::HashSet<_> = ga.get_archive().iter().map(|s| crate::core::GenomeKey(&s.solution)).collect();
        assert_eq!(keys.len(), ga.get_archive().len());
    }
}
