//! Guards for the bug fixes and features added after the 2026-09-08 audit.
//!
//! One test per defect or capability, each the smallest thing that fails if it regresses.

use puggles::checkpoint::GaState;
use puggles::core::{ConfigError, Encoding, EvalFn, Problem, Solution};
use puggles::dominance::{fast_non_dominated_sort, Dominance, ParetoDominance};
use puggles::gatypes::{Integer, Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use puggles::genetic_operators::crossover::{Crossover, OrderCrossover};
use puggles::genetic_operators::mutation::swap_mutation;
use puggles::islands::{run_islands, IslandConfig};
use puggles::nsga3::NSGAIII;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Bug 1 — NSGA-III silently ignored GPU mode (and could not take a batch objective)
// ---------------------------------------------------------------------------

fn sum_batch(pop: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    pop.iter()
        .map(|x| vec![x.iter().sum::<f64>(), x.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>()])
        .collect()
}

fn batch_problem(n_obj: usize) -> Problem {
    let mut p = Problem::new(
        3,
        n_obj,
        None,
        None,
        Some(vec![-1; n_obj]),
        (0..3).map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(2.0)))).collect(),
        |x| vec![x.iter().sum(), 0.0],
    );
    p.eval_fn = EvalFn::Batch(sum_batch);
    p
}

/// NSGA-III used to panic on a batch objective: its evaluate() called Solution::evaluate(),
/// which rejects EvalFn::Batch. It now routes through the shared population evaluator.
#[test]
fn nsga3_supports_batch_objective() {
    let problem = Arc::new(batch_problem(2));
    let mut ga = NSGAIII::new(Arc::clone(&problem), 16, 4, ExecutionMode::Sequential).with_seed(1);
    ga.run(500);
    assert!(ga.get_nfe() >= 500, "batch run spent no budget");
    assert!(!ga.get_archive().is_empty());
    assert!(ga.get_archive().iter().all(|s| s.evaluated));
}

/// GPU mode without an attached evaluator must still produce a correct run (falling back),
/// not silently diverge from the CPU result. Without the gpu feature there is no evaluator to
/// attach, so this pins the fallback path.
#[test]
fn nsga3_gpu_mode_falls_back_and_still_runs() {
    let problem = Arc::new(batch_problem(2));
    let mut ga = NSGAIII::new(Arc::clone(&problem), 16, 4, ExecutionMode::GPU).with_seed(3);
    ga.run(400);
    assert!(ga.get_nfe() >= 400);
    assert!(ga.get_archive().iter().all(|s| s.evaluated));
}

// ---------------------------------------------------------------------------
// Bug 2 — Integer bounds were exclusive in generation but inclusive everywhere else
// ---------------------------------------------------------------------------

/// The upper bound must be reachable by initial generation, since mutation and crossover both
/// clamp into the closed interval. Previously generation could never produce it.
#[test]
fn integer_upper_bound_is_generatable() {
    let problem = Problem::new(
        1,
        1,
        None,
        None,
        Some(vec![-1]),
        vec![SolutionDataTypes::Integer(Integer::new(Some(0), Some(4)))],
        |x| vec![x[0]],
    );
    let mut rng = SmallRng::seed_from_u64(11);
    let seen: Vec<f64> = (0..2000).map(|_| problem.generate_solution(&mut rng)[0]).collect();
    assert!(seen.contains(&4.0), "closed upper bound never generated");
    assert!(seen.contains(&0.0), "closed lower bound never generated");
    assert!(seen.iter().all(|&v| (0.0..=4.0).contains(&v)), "generated out of bounds");
}

// ---------------------------------------------------------------------------
// Bug 3 — equally-infeasible solutions were ranked on objectives, not violation magnitude
// ---------------------------------------------------------------------------

fn constrained_problem() -> Arc<Problem> {
    // g(x) = x[0] - 1 <= 0, so x[0] > 1 is infeasible by exactly x[0] - 1.
    Arc::new(
        Problem::new(
            1,
            1,
            None,
            None,
            Some(vec![-1]),
            vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(100.0)))],
            |x| vec![x[0]],
        )
        .with_variable_constraints(vec![|x: &Vec<f64>| x[0] - 1.0]),
    )
}

/// Two solutions each break exactly one constraint, so the violation *count* ties. The one
/// closer to feasibility must win. Before the fix this fell through to comparing objectives,
/// which ranked the wildly-infeasible solution as better because its objective was smaller.
#[test]
fn equally_infeasible_solutions_rank_by_violation_magnitude() {
    let problem = constrained_problem();

    let mut mild = Solution::new(Arc::clone(&problem));
    mild.solution = vec![2.0]; // violates by 1.0, objective 2.0
    mild.evaluate();

    let mut severe = Solution::new(Arc::clone(&problem));
    severe.solution = vec![90.0]; // violates by 89.0, objective 90.0
    severe.evaluate();

    assert_eq!(mild.constraint_violation, severe.constraint_violation, "counts must tie");
    assert!(mild.constraint_violation_magnitude < severe.constraint_violation_magnitude);
    assert_eq!(
        ParetoDominance.compare_solutions(&mild, &severe),
        -1,
        "the less-violating solution must dominate"
    );
    assert_eq!(ParetoDominance.compare_solutions(&severe, &mild), 1);
}

/// The magnitude tiebreak must also drive the sort, or the fronts would disagree with the
/// dominance relation they are built from.
#[test]
fn violation_magnitude_orders_fronts() {
    let problem = constrained_problem();
    let mut pop = Vec::new();
    for x in [50.0, 2.0, 20.0] {
        let mut s = Solution::new(Arc::clone(&problem));
        s.solution = vec![x];
        s.evaluate();
        pop.push(s);
    }
    let fronts = fast_non_dominated_sort(&pop, &ParetoDominance);
    // All infeasible with one violation each → strictly ordered by magnitude, one per front.
    assert_eq!(fronts.len(), 3, "equally-infeasible solutions should not share a front");
    assert_eq!(fronts[0], vec![1], "x=2.0 (smallest violation) belongs in front 0");
}

/// A feasible solution still beats an infeasible one regardless of magnitude.
#[test]
fn feasible_still_beats_infeasible() {
    let problem = constrained_problem();
    let mut feasible = Solution::new(Arc::clone(&problem));
    feasible.solution = vec![0.9];
    feasible.evaluate();
    let mut infeasible = Solution::new(Arc::clone(&problem));
    infeasible.solution = vec![1.001]; // barely violating
    infeasible.evaluate();

    assert!(feasible.feasible && !infeasible.feasible);
    assert_eq!(ParetoDominance.compare_solutions(&feasible, &infeasible), -1);
}

// ---------------------------------------------------------------------------
// Bug 4 — invalid bounds slipped past validation and panicked inside `rand`
// ---------------------------------------------------------------------------

#[test]
fn one_sided_bounds_are_validated_at_construction() {
    // Effective bounds collapse to a single point; used to panic later inside rand.
    assert!(Integer::try_new(Some(i64::MAX), None).is_err());
    assert!(Integer::try_new(None, Some(i64::MIN)).is_err());
}

#[test]
fn unbounded_real_is_rejected_not_deferred_to_a_rand_panic() {
    assert!(Real::try_new(None, None).is_err());
    assert!(Real::try_new(Some(0.0), None).is_err());
    // Finite but overflowing span.
    assert!(Real::try_new(Some(f64::MIN), Some(f64::MAX)).is_err());
    // A normal range is unaffected.
    let r = Real::try_new(Some(-1.0), Some(1.0)).unwrap();
    let mut rng = SmallRng::seed_from_u64(0);
    assert!((-1.0..=1.0).contains(&r.generate_value(&mut rng).unwrap()));
}

// ---------------------------------------------------------------------------
// Feature — Result-returning constructors
// ---------------------------------------------------------------------------

#[test]
fn try_new_reports_configuration_errors_instead_of_panicking() {
    let types = vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))];
    // Length mismatch.
    let err: ConfigError = Problem::try_new(2, 1, None, None, None, types.clone(), |x| vec![x[0]])
        .unwrap_err();
    assert!(err.to_string().contains("solution_length"));

    // Half-configured constraints.
    let err = Problem::try_new(1, 1, Some(vec![Some(1.0)]), None, None, types.clone(), |x| vec![x[0]])
        .unwrap_err();
    assert!(err.to_string().contains("supply both or neither"));

    // Unknown operand — previously only discovered at evaluation time.
    let err = Problem::try_new(
        1,
        1,
        Some(vec![Some(1.0)]),
        Some(vec![Some("=<".into())]),
        None,
        types.clone(),
        |x| vec![x[0]],
    )
    .unwrap_err();
    assert!(err.to_string().contains("Invalid operand"));

    // Bad direction value.
    let err = Problem::try_new(1, 1, None, None, Some(vec![0]), types.clone(), |x| vec![x[0]])
        .unwrap_err();
    assert!(err.to_string().contains("direction"));

    // The valid case still builds.
    assert!(Problem::try_new(1, 1, None, None, None, types, |x| vec![x[0]]).is_ok());
}

// ---------------------------------------------------------------------------
// Feature — permutation encoding
// ---------------------------------------------------------------------------

fn is_permutation(v: &[f64]) -> bool {
    let mut seen = vec![false; v.len()];
    for &x in v {
        let i = x as usize;
        if x < 0.0 || i >= v.len() || (x - x.round()).abs() > 1e-9 || seen[i] {
            return false;
        }
        seen[i] = true;
    }
    true
}

/// Total tour length for a fixed set of points on a line — optimal order is sorted.
fn tour_length(x: &Vec<f64>) -> Vec<f64> {
    let pos: [f64; 8] = [0.0, 9.0, 3.0, 7.0, 1.0, 5.0, 8.0, 2.0];
    let mut total = 0.0;
    for w in x.windows(2) {
        total += (pos[w[0] as usize] - pos[w[1] as usize]).abs();
    }
    vec![total]
}

fn tsp_problem() -> Arc<Problem> {
    Arc::new(
        Problem::new(
            8,
            1,
            None,
            None,
            Some(vec![-1]),
            (0..8).map(|_| SolutionDataTypes::Integer(Integer::new(Some(0), Some(7)))).collect(),
            tour_length,
        )
        .with_permutation_encoding(),
    )
}

#[test]
fn permutation_encoding_generates_permutations() {
    let problem = tsp_problem();
    assert_eq!(problem.encoding, Encoding::Permutation);
    let mut rng = SmallRng::seed_from_u64(5);
    for _ in 0..50 {
        assert!(is_permutation(&problem.generate_solution(&mut rng)));
    }
}

#[test]
fn order_crossover_preserves_permutations() {
    let problem = tsp_problem();
    let mut rng = SmallRng::seed_from_u64(2);
    let op = OrderCrossover { probability: 1.0 };
    for _ in 0..100 {
        let mut p1 = Solution::new(Arc::clone(&problem));
        let mut p2 = Solution::new(Arc::clone(&problem));
        p1.solution = problem.generate_solution(&mut rng);
        p2.solution = problem.generate_solution(&mut rng);
        let (c1, c2) = op.crossover(&p1, &p2, &mut rng);
        assert!(is_permutation(&c1.solution), "OX produced {:?}", c1.solution);
        assert!(is_permutation(&c2.solution), "OX produced {:?}", c2.solution);
    }
}

#[test]
fn swap_mutation_preserves_permutations() {
    let mut rng = SmallRng::seed_from_u64(4);
    for _ in 0..100 {
        let mut v: Vec<f64> = (0..8).map(|i| i as f64).collect();
        swap_mutation(&mut v, 0.5, &mut rng);
        assert!(is_permutation(&v), "swap produced {:?}", v);
    }
}

/// End to end: the GA must beat random permutation sampling on an ordering problem, and every
/// solution it returns must still be a valid permutation.
#[test]
fn permutation_ga_beats_random_search_and_stays_valid() {
    let problem = tsp_problem();
    let mut rng = SmallRng::seed_from_u64(77);
    let baseline = (0..5_000)
        .map(|_| tour_length(&problem.generate_solution(&mut rng))[0])
        .fold(f64::INFINITY, f64::min);

    let mut ga = NSGAII::new(Arc::clone(&problem), 40, ExecutionMode::Sequential).with_seed(9);
    ga.run(5_000);
    let best = ga
        .get_archive()
        .iter()
        .map(|s| s.objective_fitness_values[0])
        .fold(f64::INFINITY, f64::min);

    for s in ga.get_archive() {
        assert!(is_permutation(&s.solution), "GA returned a non-permutation: {:?}", s.solution);
    }
    // Optimal tour on a line is the sorted order, length = span = 9.0.
    assert!(best <= baseline, "GA ({best}) did not match random search ({baseline})");
    assert!(best < 12.0, "GA got stuck at {best}, expected near the optimum of 9.0");
}

// ---------------------------------------------------------------------------
// Feature — hypervolume-based convergence
// ---------------------------------------------------------------------------

fn zdt1() -> Arc<Problem> {
    Arc::new(Problem::new(
        4,
        2,
        None,
        None,
        Some(vec![-1, -1]),
        (0..4).map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))).collect(),
        |x| {
            let f1 = x[0];
            let g = 1.0 + 9.0 / 3.0 * x[1..].iter().sum::<f64>();
            vec![f1, g * (1.0 - (f1 / g).sqrt())]
        },
    ))
}

#[test]
fn hv_convergence_stops_early_and_reports_hypervolume() {
    let mut ga = NSGAII::new(zdt1(), 40, ExecutionMode::Sequential).with_seed(1);
    // A huge budget with tight patience: it must stop on the plateau, not spend the budget.
    let gens = ga.run_until_hv_converged(1_000_000, 8, 1e-6, [11.0, 11.0]);
    assert!(gens > 0);
    assert!(ga.get_nfe() < 1_000_000, "did not stop early: spent {} evals", ga.get_nfe());
    let hv = ga.archive_hypervolume([11.0, 11.0]);
    assert!(hv > 0.0, "hypervolume should be positive on a real front");
}

#[test]
#[should_panic(expected = "exactly 2 objectives")]
fn hv_convergence_rejects_non_two_objective_problems() {
    let problem = Arc::new(Problem::new(
        3,
        3,
        None,
        None,
        Some(vec![-1, -1, -1]),
        (0..3).map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))).collect(),
        |x| vec![x[0], x[1], x[2]],
    ));
    let mut ga = NSGAII::new(problem, 8, ExecutionMode::Sequential);
    ga.run_until_hv_converged(100, 2, 1e-6, [2.0, 2.0]);
}

// ---------------------------------------------------------------------------
// Feature — island model
// ---------------------------------------------------------------------------

#[test]
fn islands_produce_a_valid_front() {
    let front = run_islands(
        zdt1(),
        IslandConfig { islands: 4, population_size: 20, max_nfe_per_island: 1_000, seed: Some(1), ..Default::default() },
    );
    assert!(!front.is_empty());
    assert!(front.iter().all(|s| s.evaluated && s.feasible));
    assert_eq!(fast_non_dominated_sort(&front, &ParetoDominance).len(), 1);
}

// ---------------------------------------------------------------------------
// Feature — checkpoint round-trip keeps the new field
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_roundtrip_preserves_violation_magnitude() {
    let problem = constrained_problem();
    let mut ga = NSGAII::new(Arc::clone(&problem), 12, ExecutionMode::Sequential).with_seed(2);
    ga.run(200);
    let state = ga.save_state();
    let json = serde_json::to_string(&state).expect("state serializes");
    let restored: GaState = serde_json::from_str(&json).expect("state deserializes");

    let mut ga2 = NSGAII::new(Arc::clone(&problem), 12, ExecutionMode::Sequential);
    ga2.load_state(restored);
    assert_eq!(ga2.get_nfe(), ga.get_nfe());
    for (a, b) in ga.population.iter().zip(ga2.population.iter()) {
        assert_eq!(a.constraint_violation, b.constraint_violation);
        assert_eq!(a.constraint_violation_magnitude, b.constraint_violation_magnitude);
    }
}

/// Checkpoints written before the magnitude field existed must still load.
#[test]
fn checkpoint_without_magnitude_field_still_loads() {
    let legacy = r#"{"population":[{"solution":[0.5],"objective_fitness_values":[0.5],
        "constraint_values":[],"constraint_violation":0,"feasible":true,"evaluated":true}],
        "archive":[],"nfe":7}"#;
    let state: GaState = serde_json::from_str(legacy).expect("legacy checkpoint should load");
    assert_eq!(state.nfe, 7);
    assert_eq!(state.population[0].constraint_violation_magnitude, 0.0);
}
