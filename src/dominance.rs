use crate::core::{Problem, Solution};


pub trait Dominance: Send + Sync {
    fn compare_solutions(&self, solution_1: &Solution, solution_2: &Solution) -> i32;
}

#[derive(Debug)]
pub struct ParetoDominance;

impl Dominance for ParetoDominance {
    /// Returns -1 if solution_1 dominates, 1 if solution_2 dominates, 0 if non-dominated
    #[inline]
    fn compare_solutions(&self, solution_1: &Solution, solution_2: &Solution) -> i32 {
        let problem: &Problem = &solution_1.problem;
        let n_objectives = problem.number_of_objectives;

        // Constraint-based dominance: fewer violations wins
        if let Some(ref constraints) = &problem.objective_constraint {
            if !constraints.is_empty() && solution_1.constraint_violation != solution_2.constraint_violation {
                return match (solution_1.constraint_violation, solution_2.constraint_violation) {
                    (0, _) => -1,
                    (_, 0) => 1,
                    (v1, v2) if v1 < v2 => -1,
                    (v1, v2) if v1 > v2 => 1,
                    _ => 0,
                };
            }
        }

        let mut is_solution_1_better = false;
        let mut is_solution_2_better = false;

        for i in 0..n_objectives {
            let mut obj_1 = solution_1.objective_fitness_values[i];
            let mut obj_2 = solution_2.objective_fitness_values[i];

            if let Some(direction) = &problem.direction {
                // For minimization (direction == -1), negate so "lower is better" becomes "higher is better"
                if direction[i] == -1 {
                    obj_1 = -obj_1;
                    obj_2 = -obj_2;
                }
            }

            if obj_1 < obj_2 {
                is_solution_2_better = true;
                if is_solution_1_better {
                    return 0; // non-dominated
                }
            } else if obj_1 > obj_2 {
                is_solution_1_better = true;
                if is_solution_2_better {
                    return 0; // non-dominated
                }
            }
        }

        if is_solution_1_better == is_solution_2_better {
            0 // equal on all objectives
        } else if is_solution_1_better {
            -1
        } else {
            1
        }
    }
}

/// Single-objective specialization of [`fast_non_dominated_sort`] (O(N log N) instead of O(N²)).
/// Reproduces `ParetoDominance`'s M=1 ordering: constraint violations first (only when
/// `objective_constraint` is present and non-empty), then the direction-adjusted objective.
// ponytail: assumes Pareto-style dominance, the only `Dominance` impl. If a non-Pareto
// `Dominance` is ever added, guard the M=1 branch in `fast_non_dominated_sort` on it.
fn single_objective_fronts(population: &[Solution]) -> Vec<Vec<usize>> {
    let problem = &population[0].problem;
    let use_constraints = problem
        .objective_constraint
        .as_ref()
        .map_or(false, |c| !c.is_empty());
    // ParetoDominance only negates the objective when direction[0] == -1 (minimize); when
    // direction is None it does not negate (treats it as maximize). Mirror that exactly.
    let minimize = problem.direction.as_ref().map_or(false, |d| d[0] == -1);

    // Sort key: smaller `primary` (fewer violations) is better; higher `adj` is better.
    let key = |i: usize| -> (usize, f64) {
        let primary = if use_constraints { population[i].constraint_violation } else { 0 };
        let obj = population[i].objective_fitness_values[0];
        (primary, if minimize { -obj } else { obj })
    };

    let mut order: Vec<usize> = (0..population.len()).collect();
    order.sort_by(|&a, &b| {
        let (pa, aa) = key(a);
        let (pb, ab) = key(b);
        pa.cmp(&pb)
            .then(ab.partial_cmp(&aa).unwrap_or(std::cmp::Ordering::Equal)) // adj descending
            .then(a.cmp(&b)) // deterministic tie-break by index
    });

    // Consecutive equal keys are mutually non-dominated → same front; a strictly better key
    // dominates → an earlier front. Same exact-f64 tie handling as the general path.
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut prev: Option<(usize, f64)> = None;
    for &i in &order {
        let k = key(i);
        if prev.map_or(false, |p| p == k) {
            current.push(i);
        } else {
            if !current.is_empty() {
                fronts.push(std::mem::take(&mut current));
            }
            current.push(i);
            prev = Some(k);
        }
    }
    if !current.is_empty() {
        fronts.push(current);
    }
    fronts
}

/// Fast non-dominated sorting (Deb et al., 2002).
/// Returns a vector of fronts, where each front is a vector of indices into `population`.
/// Front 0 is the Pareto-optimal set.
pub fn fast_non_dominated_sort<D: Dominance + ?Sized>(
    population: &[Solution],
    dominance: &D,
) -> Vec<Vec<usize>> {
    if population.is_empty() {
        return Vec::new();
    }
    // Single-objective fast path: O(N log N), skipping the pairwise machinery entirely.
    if population[0].problem.number_of_objectives == 1 {
        return single_objective_fronts(population);
    }
    // Constraint dominance (feasibility beats objectives) breaks Best-Order-Sort's
    // objective-sorted invariant, so constrained problems use ENS-SS (which folds constraint
    // violations into its sort key). The common unconstrained case uses Best-Order-Sort.
    let use_constraints = population[0]
        .problem
        .objective_constraint
        .as_ref()
        .map_or(false, |c| !c.is_empty());
    // Best-Order-Sort only pays off once the population is large enough — below ~N=200 its
    // per-objective sorted-list overhead makes ENS-SS faster (it's ~0.8× at N=100), while it
    // grows to ~1.4× at N=500 and ~2.3× at N=4000. Route small sorts to ENS so there is never a
    // regression. // ponytail: threshold from the `bench_sort_bos_vs_ens` micro-benchmark.
    if use_constraints || population.len() < 200 {
        ens_ss_fronts(population, dominance)
    } else {
        best_order_sort(population, dominance)
    }
}

/// Efficient Non-dominated Sort, sequential search. Exact fronts; handles constraint dominance.
/// Replaces the classic O(MN²) all-pairs Deb-2002 loop — a solution is only compared against
/// already-placed ones, no O(N)-Vec-per-item domination bookkeeping.
fn ens_ss_fronts<D: Dominance + ?Sized>(population: &[Solution], dominance: &D) -> Vec<Vec<usize>> {
    let n = population.len();
    let problem = &population[0].problem;
    let use_constraints = problem
        .objective_constraint
        .as_ref()
        .map_or(false, |c| !c.is_empty());
    let n_obj = problem.number_of_objectives;
    let signs: Vec<f64> = (0..n_obj)
        .map(|m| match &problem.direction {
            Some(d) if d[m] == -1 => -1.0,
            _ => 1.0,
        })
        .collect();
    let primary = |i: usize| -> usize {
        if use_constraints {
            population[i].constraint_violation
        } else {
            0
        }
    };

    // Best-first order: fewer constraint violations, then adjusted objectives descending
    // (lexicographic), then index. Guarantees a solution can only be dominated by one sorted
    // before it — the invariant ENS relies on.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        primary(a)
            .cmp(&primary(b))
            .then_with(|| {
                for m in 0..n_obj {
                    let aa = population[a].objective_fitness_values[m] * signs[m];
                    let ab = population[b].objective_fitness_values[m] * signs[m];
                    match ab.partial_cmp(&aa).unwrap_or(std::cmp::Ordering::Equal) {
                        std::cmp::Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                std::cmp::Ordering::Equal
            })
            .then(a.cmp(&b))
    });

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    for &p in &order {
        let mut target = 0usize;
        for k in (0..fronts.len()).rev() {
            if fronts[k]
                .iter()
                .any(|&q| dominance.compare_solutions(&population[q], &population[p]) == -1)
            {
                target = k + 1;
                break;
            }
        }
        if target == fronts.len() {
            fronts.push(vec![p]);
        } else {
            fronts[target].push(p);
        }
    }
    fronts
}

/// Best-Order-Sort (Roy, Islam, Deb 2016) for unconstrained problems. Same exact fronts as ENS,
/// but each solution is only compared within per-objective sorted lists, so the comparison count
/// grows more slowly with population size. // ponytail: unconstrained only — the objective-sorted
/// invariant doesn't hold under constraint dominance; `fast_non_dominated_sort` routes constrained
/// problems to `ens_ss_fronts`.
fn best_order_sort<D: Dominance + ?Sized>(population: &[Solution], dominance: &D) -> Vec<Vec<usize>> {
    let n = population.len();
    let problem = &population[0].problem;
    let m = problem.number_of_objectives;
    // Adjusted objective with "smaller = better" so each list sorts ascending (best first).
    let signs: Vec<f64> = (0..m)
        .map(|k| match &problem.direction {
            Some(d) if d[k] == -1 => 1.0, // minimize: smaller objective is better
            _ => -1.0,                    // maximize: negate so smaller adjusted is better
        })
        .collect();
    let adj = |s: usize, k: usize| population[s].objective_fitness_values[k] * signs[k];

    // One sorted list per objective (ascending adjusted; lexicographic + index tie-break → a
    // deterministic total order).
    let q: Vec<Vec<usize>> = (0..m)
        .map(|k| {
            let mut v: Vec<usize> = (0..n).collect();
            v.sort_by(|&a, &b| {
                adj(a, k)
                    .partial_cmp(&adj(b, k))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        for kk in 0..m {
                            match adj(a, kk).partial_cmp(&adj(b, kk)).unwrap_or(std::cmp::Ordering::Equal) {
                                std::cmp::Ordering::Equal => continue,
                                o => return o,
                            }
                        }
                        std::cmp::Ordering::Equal
                    })
                    .then(a.cmp(&b))
            });
            v
        })
        .collect();

    let mut rank = vec![usize::MAX; n];
    let mut is_ranked = vec![false; n];
    // comp[k][f] = solutions already ranked into front f that have appeared in objective k so far.
    let mut comp: Vec<Vec<Vec<usize>>> = vec![Vec::new(); m];
    let mut ranked_count = 0usize;

    'scan: for i in 0..n {
        for k in 0..m {
            let s = q[k][i];
            if is_ranked[s] {
                let r = rank[s];
                if comp[k].len() <= r {
                    comp[k].resize(r + 1, Vec::new());
                }
                comp[k][r].push(s);
                continue;
            }
            // Lowest front in objective k's comparison sets with no solution dominating s.
            let mut target = None;
            for f in 0..comp[k].len() {
                let dominated = comp[k][f]
                    .iter()
                    .any(|&t| dominance.compare_solutions(&population[t], &population[s]) == -1);
                if !dominated {
                    target = Some(f);
                    break;
                }
            }
            let f = match target {
                Some(f) => f,
                None => {
                    comp[k].push(Vec::new());
                    comp[k].len() - 1
                }
            };
            rank[s] = f;
            is_ranked[s] = true;
            comp[k][f].push(s);
            ranked_count += 1;
            if ranked_count == n {
                break 'scan;
            }
        }
    }

    let max_rank = rank.iter().copied().filter(|&r| r != usize::MAX).max().unwrap_or(0);
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
    for s in 0..n {
        fronts[rank[s]].push(s);
    }
    fronts
}

/// Crowding distance assignment for a single front.
/// Returns a vector of crowding distances indexed by position in `front_indices`.
pub fn crowding_distance(population: &[Solution], front_indices: &[usize]) -> Vec<f64> {
    let n = front_indices.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let mut distances = vec![0.0f64; n];
    let n_objectives = population[front_indices[0]].problem.number_of_objectives;

    for m in 0..n_objectives {
        // Sort front by objective m
        let mut sorted_local: Vec<usize> = (0..n).collect();
        sorted_local.sort_by(|&a, &b| {
            let va = population[front_indices[a]].objective_fitness_values[m];
            let vb = population[front_indices[b]].objective_fitness_values[m];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let f_min = population[front_indices[sorted_local[0]]].objective_fitness_values[m];
        let f_max = population[front_indices[sorted_local[n - 1]]].objective_fitness_values[m];
        let range = f_max - f_min;

        // Boundary solutions get infinite distance
        distances[sorted_local[0]] = f64::INFINITY;
        distances[sorted_local[n - 1]] = f64::INFINITY;

        if range > f64::EPSILON {
            for i in 1..(n - 1) {
                let prev = population[front_indices[sorted_local[i - 1]]].objective_fitness_values[m];
                let next = population[front_indices[sorted_local[i + 1]]].objective_fitness_values[m];
                distances[sorted_local[i]] += (next - prev) / range;
            }
        }
    }

    distances
}


#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;
    use std::sync::Arc;
    use crate::core::{EvalFn, Problem, Solution};
    use crate::gatypes::{SolutionDataTypes, Real};
    use crate::benchmark_objective_functions::parabloid_5;

    #[test]
    fn test_pareto_dominance_sol1_dominates() {
        let problem = Arc::new(Problem {
            solution_length: 5,
            number_of_objectives: 1,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![1]), // maximize
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            variable_constraints: None,
            eval_fn: EvalFn::Single(parabloid_5),
        });

        let mut sol1 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![10.0, 10.0, 10.0, 10.0, 10.0],
            objective_fitness_values: smallvec![],
            constraint_values: smallvec![],
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let mut sol2 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            objective_fitness_values: smallvec![],
            constraint_values: smallvec![],
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };

        sol1.evaluate();
        sol2.evaluate();

        let result = ParetoDominance.compare_solutions(&sol1, &sol2);
        assert_eq!(result, -1); // sol1 has higher value, maximizing => sol1 dominates
    }

    #[test]
    fn test_pareto_dominance_sol2_dominates() {
        let problem = Arc::new(Problem {
            solution_length: 5,
            number_of_objectives: 1,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1]), // minimize
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            variable_constraints: None,
            eval_fn: EvalFn::Single(parabloid_5),
        });

        let mut sol1 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![10.0, 10.0, 10.0, 10.0, 10.0],
            objective_fitness_values: smallvec![],
            constraint_values: smallvec![],
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let mut sol2 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            objective_fitness_values: smallvec![],
            constraint_values: smallvec![],
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };

        sol1.evaluate();
        sol2.evaluate();

        let result = ParetoDominance.compare_solutions(&sol1, &sol2);
        assert_eq!(result, 1); // sol2 has lower value, minimizing => sol2 dominates
    }

    #[test]
    fn test_pareto_dominance_non_dominated() {
        let problem = Arc::new(Problem {
            solution_length: 2,
            number_of_objectives: 2,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1]), // minimize both
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            variable_constraints: None,
            eval_fn: EvalFn::Single(|x| vec![x[0], x[1]]),
        });

        // sol1 is better on obj0, sol2 is better on obj1 => non-dominated
        let sol1 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![1.0, 10.0],
            objective_fitness_values: smallvec![1.0, 10.0],
            constraint_values: smallvec![],
            constraint_violation: 0,
            feasible: true,
            evaluated: true,
        };
        let sol2 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![10.0, 1.0],
            objective_fitness_values: smallvec![10.0, 1.0],
            constraint_values: smallvec![],
            constraint_violation: 0,
            feasible: true,
            evaluated: true,
        };

        let result = ParetoDominance.compare_solutions(&sol1, &sol2);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_fast_non_dominated_sort() {
        let problem = Arc::new(Problem {
            solution_length: 2,
            number_of_objectives: 2,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1]),
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            variable_constraints: None,
            eval_fn: EvalFn::Single(|x| vec![x[0], x[1]]),
        });

        let population = vec![
            // Front 0: Pareto-optimal
            Solution { problem: Arc::clone(&problem), solution: vec![1.0, 5.0], objective_fitness_values: smallvec![1.0, 5.0], constraint_values: smallvec![], constraint_violation: 0, feasible: true, evaluated: true },
            Solution { problem: Arc::clone(&problem), solution: vec![5.0, 1.0], objective_fitness_values: smallvec![5.0, 1.0], constraint_values: smallvec![], constraint_violation: 0, feasible: true, evaluated: true },
            // Front 1: dominated by front 0
            Solution { problem: Arc::clone(&problem), solution: vec![3.0, 6.0], objective_fitness_values: smallvec![3.0, 6.0], constraint_values: smallvec![], constraint_violation: 0, feasible: true, evaluated: true },
            // Front 1: dominated by front 0
            Solution { problem: Arc::clone(&problem), solution: vec![6.0, 3.0], objective_fitness_values: smallvec![6.0, 3.0], constraint_values: smallvec![], constraint_violation: 0, feasible: true, evaluated: true },
            // Front 2: dominated by front 1
            Solution { problem: Arc::clone(&problem), solution: vec![10.0, 10.0], objective_fitness_values: smallvec![10.0, 10.0], constraint_values: smallvec![], constraint_violation: 0, feasible: true, evaluated: true },
        ];

        let fronts = fast_non_dominated_sort(&population, &ParetoDominance);

        assert_eq!(fronts.len(), 3);
        assert_eq!(fronts[0].len(), 2); // front 0: indices 0,1
        assert_eq!(fronts[1].len(), 2); // front 1: indices 2,3
        assert_eq!(fronts[2].len(), 1); // front 2: index 4
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let problem = Arc::new(Problem {
            solution_length: 2,
            number_of_objectives: 2,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1]),
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            variable_constraints: None,
            eval_fn: EvalFn::Single(|x| vec![x[0], x[1]]),
        });

        let population = vec![
            Solution { problem: Arc::clone(&problem), solution: vec![1.0, 5.0], objective_fitness_values: smallvec![1.0, 5.0], constraint_values: smallvec![], constraint_violation: 0, feasible: true, evaluated: true },
            Solution { problem: Arc::clone(&problem), solution: vec![3.0, 3.0], objective_fitness_values: smallvec![3.0, 3.0], constraint_values: smallvec![], constraint_violation: 0, feasible: true, evaluated: true },
            Solution { problem: Arc::clone(&problem), solution: vec![5.0, 1.0], objective_fitness_values: smallvec![5.0, 1.0], constraint_values: smallvec![], constraint_violation: 0, feasible: true, evaluated: true },
        ];

        let front = vec![0, 1, 2];
        let distances = crowding_distance(&population, &front);

        // boundary solutions should have infinite distance
        assert_eq!(distances[0], f64::INFINITY);
        assert_eq!(distances[2], f64::INFINITY);
        // middle solution should have finite distance
        assert!(distances[1].is_finite());
        assert!(distances[1] > 0.0);
    }

    // Reference O(N³) non-dominated sort by repeated peeling — obviously correct, used to
    // cross-validate Best-Order-Sort.
    fn naive_reference_fronts(pop: &[Solution], dom: &ParetoDominance) -> Vec<usize> {
        let n = pop.len();
        let mut rank = vec![usize::MAX; n];
        let mut assigned = 0;
        let mut f = 0;
        while assigned < n {
            let mut this = Vec::new();
            for s in 0..n {
                if rank[s] != usize::MAX {
                    continue;
                }
                let dominated = (0..n).any(|t| {
                    t != s && rank[t] == usize::MAX && dom.compare_solutions(&pop[t], &pop[s]) == -1
                });
                if !dominated {
                    this.push(s);
                }
            }
            for &s in &this {
                rank[s] = f;
                assigned += 1;
            }
            f += 1;
        }
        rank
    }

    fn fronts_to_rank(fronts: &[Vec<usize>], n: usize) -> Vec<usize> {
        let mut r = vec![usize::MAX; n];
        for (f, front) in fronts.iter().enumerate() {
            for &s in front {
                r[s] = f;
            }
        }
        r
    }

    #[test]
    fn test_best_order_sort_matches_naive() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(42);
        // Small objective range → lots of ties and multiple fronts, the stressful case.
        for &(n, m) in &[(20usize, 2usize), (30, 3), (50, 3), (40, 4), (60, 2)] {
            let problem = Arc::new(Problem {
                solution_length: 1,
                number_of_objectives: m,
                objective_constraint: None,
                objective_constraint_operands: None,
                direction: Some(vec![-1; m]),
                solution_data_types: vec![SolutionDataTypes::Real(Real::new(Some(0.), Some(1.)))],
                variable_constraints: None,
                eval_fn: EvalFn::Single(|x| x.clone()),
            });
            let pop: Vec<Solution> = (0..n)
                .map(|_| {
                    let objs: Vec<f64> = (0..m).map(|_| (rng.gen::<u32>() % 5) as f64).collect();
                    Solution {
                        problem: Arc::clone(&problem),
                        solution: vec![0.0],
                        objective_fitness_values: objs.into(),
                        constraint_values: smallvec![],
                        evaluated: true,
                        constraint_violation: 0,
                        feasible: true,
                    }
                })
                .collect();

            let bos = fronts_to_rank(&best_order_sort(&pop, &ParetoDominance), n);
            let naive = naive_reference_fronts(&pop, &ParetoDominance);
            assert_eq!(bos, naive, "Best-Order-Sort ranks differ from naive for n={n} m={m}");
        }
    }

    // Timing only: `cargo test bench_sort -- --ignored --nocapture`. Compares Best-Order-Sort
    // vs ENS-SS on random M=3 populations across sizes, to see where BOS's edge appears.
    #[test]
    #[ignore]
    fn bench_sort_bos_vs_ens() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        use std::time::Instant;
        let m = 3;
        for &n in &[100usize, 200, 500, 1000, 2000, 4000] {
            let mut rng = SmallRng::seed_from_u64(7);
            let problem = Arc::new(Problem {
                solution_length: 1,
                number_of_objectives: m,
                objective_constraint: None,
                objective_constraint_operands: None,
                direction: Some(vec![-1; m]),
                solution_data_types: vec![SolutionDataTypes::Real(Real::new(Some(0.), Some(1.)))],
                variable_constraints: None,
                eval_fn: EvalFn::Single(|x| x.clone()),
            });
            let pop: Vec<Solution> = (0..n)
                .map(|_| Solution {
                    problem: Arc::clone(&problem),
                    solution: vec![0.0],
                    objective_fitness_values: (0..m).map(|_| rng.gen::<f64>()).collect(),
                    constraint_values: smallvec![],
                    evaluated: true,
                    constraint_violation: 0,
                    feasible: true,
                })
                .collect();
            let reps = 50;
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(best_order_sort(&pop, &ParetoDominance));
            }
            let bos_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(ens_ss_fronts(&pop, &ParetoDominance));
            }
            let ens_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            println!(
                "N={n:>5}  BOS {bos_us:>9.1} µs   ENS {ens_us:>9.1} µs   speedup {:.2}×",
                ens_us / bos_us
            );
        }
    }
}
