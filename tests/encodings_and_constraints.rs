//! Regression guards for the defects found in the 2026-08-31 audit. Each test is the smallest
//! thing that fails if one of those defects comes back.
//!
//! The common shape: run the GA against a random-search baseline on the *same* problem with the
//! *same* evaluation budget. A GA that isn't recombining or is mutating every gene scores about
//! the same as random sampling; a working one is orders of magnitude better.

use puggles::core::{Problem, Solution};
use puggles::dominance::{Dominance, ParetoDominance};
use puggles::gatypes::{BitBinary, Integer, Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use puggles::genetic_operators::crossover::{ArithmeticCrossover, Crossover, UniformCrossover};
use puggles::genetic_operators::mutation::{GaussianMutation, Mutation};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

const NFE: usize = 20_000;

fn offset_sphere(x: &Vec<f64>) -> Vec<f64> {
    vec![x.iter().map(|v| (v - 50.0).powi(2)).sum()]
}
fn neg_ones(x: &Vec<f64>) -> Vec<f64> {
    vec![-x.iter().sum::<f64>()]
}

/// Best objective found by drawing `NFE` random solutions — the bar a real GA must clear.
fn random_search_baseline(problem: &Problem, f: fn(&Vec<f64>) -> Vec<f64>) -> f64 {
    let mut rng = SmallRng::seed_from_u64(99);
    (0..NFE)
        .map(|_| f(&problem.generate_solution(&mut rng))[0])
        .fold(f64::INFINITY, f64::min)
}

fn best_of(problem: Arc<Problem>) -> f64 {
    let mut ga = NSGAII::new(problem, 100, ExecutionMode::Sequential).with_seed(1);
    ga.run(NFE);
    ga.get_archive()[0].objective_fitness_values[0]
}

#[test]
fn integer_encoding_beats_random_search() {
    let problem = Arc::new(Problem::new(
        10, 1, None, None, Some(vec![-1]),
        vec![SolutionDataTypes::Integer(Integer::new(Some(0), Some(101))); 10],
        offset_sphere,
    ));
    let baseline = random_search_baseline(&problem, offset_sphere);
    let best = best_of(Arc::clone(&problem));
    println!("integer: GA {best:.1} vs random {baseline:.1}");
    // Before the fix: 1061 vs 1132 — the GA was random search with extra steps.
    assert!(best < baseline / 20.0, "GA {best:.1} vs random search {baseline:.1}");
}

#[test]
fn binary_encoding_solves_onemax() {
    let problem = Arc::new(Problem::new(
        40, 1, None, None, Some(vec![-1]),
        vec![SolutionDataTypes::BitBinary(BitBinary::new()); 40],
        neg_ones,
    ));
    let ones = -best_of(problem);
    println!("binary: OneMax {ones:.0}/40");
    // Before the fix: 33/40. OneMax is the easiest GA benchmark there is.
    assert!(ones >= 39.0, "OneMax reached only {ones:.0}/40");
}

#[test]
fn real_encoding_still_converges() {
    let problem = Arc::new(Problem::new(
        10, 1, None, None, Some(vec![-1]),
        vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(101.0))); 10],
        offset_sphere,
    ));
    let best = best_of(problem);
    println!("real: GA {best:.4}");
    assert!(best < 5.0, "real-encoded convergence regressed: {best}");
}

#[test]
fn variable_constraints_drive_dominance() {
    // g(x) = 1 - sum(x) <= 0, so the unconstrained optimum (all zeros) is infeasible.
    let problem = Arc::new(
        Problem::new(
            2, 1, None, None, Some(vec![-1]),
            vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(10.0))); 2],
            |x: &Vec<f64>| vec![x.iter().map(|v| v * v).sum()],
        )
        .with_variable_constraints(vec![|x: &Vec<f64>| 1.0 - x.iter().sum::<f64>()]),
    );

    let mut feasible = Solution::new(Arc::clone(&problem));
    feasible.solution = vec![1.0, 1.0]; // sum 2 >= 1 -> feasible, objective 2
    feasible.evaluate();
    let mut infeasible = Solution::new(Arc::clone(&problem));
    infeasible.solution = vec![0.0, 0.0]; // sum 0 -> infeasible, objective 0 (better!)
    infeasible.evaluate();

    assert_eq!(infeasible.constraint_violation, 1);
    // Before the fix this returned 1: constraint dominance was gated on `objective_constraint`,
    // so a decision-variable constraint had no effect on selection at all.
    assert_eq!(
        ParetoDominance.compare_solutions(&feasible, &infeasible),
        -1,
        "a feasible solution must dominate an infeasible one even with a worse objective"
    );

    // ...and the GA must actually respect it end to end.
    let mut ga = NSGAII::new(Arc::clone(&problem), 40, ExecutionMode::Sequential).with_seed(3);
    ga.run(4_000);
    assert!(ga.get_archive().iter().all(|s| s.feasible), "archive holds infeasible solutions");
}

#[test]
fn uniform_crossover_recombines_rather_than_swapping() {
    let problem = Arc::new(Problem::new(
        6, 1, None, None, Some(vec![-1]),
        vec![SolutionDataTypes::Integer(Integer::new(Some(0), Some(100))); 6],
        offset_sphere,
    ));
    let mut p1 = Solution::new(Arc::clone(&problem));
    p1.solution = vec![10.0; 6];
    let mut p2 = Solution::new(Arc::clone(&problem));
    p2.solution = vec![90.0; 6];

    let op = UniformCrossover { probability: 0.5 };
    let mut rng = SmallRng::seed_from_u64(5);
    let mut mixed = 0;
    for _ in 0..50 {
        let (c1, _) = op.crossover(&p1, &p2, &mut rng);
        // A genuine mix has genes from both parents; the old per-bit loop always returned p2.
        if c1.solution.iter().any(|&v| v == 10.0) && c1.solution.iter().any(|&v| v == 90.0) {
            mixed += 1;
        }
    }
    assert!(mixed > 40, "only {mixed}/50 children mixed both parents");
}

#[test]
fn arithmetic_crossover_produces_distinct_children() {
    let problem = Arc::new(Problem::new(
        4, 1, None, None, Some(vec![-1]),
        vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(100.0))); 4],
        offset_sphere,
    ));
    let mut p1 = Solution::new(Arc::clone(&problem));
    p1.solution = vec![10.0; 4];
    let mut p2 = Solution::new(Arc::clone(&problem));
    p2.solution = vec![90.0; 4];

    let op = ArithmeticCrossover { probability: 1.0 };
    let mut rng = SmallRng::seed_from_u64(6);
    let (c1, c2) = op.crossover(&p1, &p2, &mut rng);
    // Before the fix both children were the identical midpoint, and no rng was consumed.
    assert_ne!(c1.solution, c2.solution, "children are identical — diversity collapses");
    let (d1, _) = op.crossover(&p1, &p2, &mut rng);
    assert_ne!(c1.solution, d1.solution, "operator is deterministic");
}

#[test]
fn gaussian_mutation_is_two_sided() {
    let problem = Arc::new(Problem::new(
        1, 1, None, None, Some(vec![-1]),
        vec![SolutionDataTypes::Real(Real::new(Some(-10.0), Some(10.0)))],
        offset_sphere,
    ));
    let mut parent = Solution::new(Arc::clone(&problem));
    parent.solution = vec![0.0];

    let op = GaussianMutation::new(Some(1.0), Some(1.0));
    let mut rng = SmallRng::seed_from_u64(7);
    let samples: Vec<f64> = (0..500).map(|_| op.mutate(&parent, 0, &mut rng)).collect();
    let below = samples.iter().filter(|&&v| v < 0.0).count();
    // Before the fix this drew a uniform [0,1): strictly positive, so `below` was 0.
    assert!((150..350).contains(&below), "{below}/500 samples below the parent — not symmetric");
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    assert!(mean.abs() < 0.2, "mean shift {mean:.3} — mutation is biased");
}

#[test]
fn budget_smaller_than_population_still_archives() {
    let problem = Arc::new(Problem::new(
        5, 1, None, None, Some(vec![-1]),
        vec![SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))); 5],
        offset_sphere,
    ));
    let mut ga = NSGAII::new(problem, 100, ExecutionMode::Sequential).with_seed(9);
    ga.run(50); // budget below one population — previously returned an empty archive
    assert!(!ga.get_archive().is_empty(), "archive empty after a sub-population budget");
}

#[test]
#[should_panic(expected = "population_size must be > 0")]
fn zero_population_is_rejected_instead_of_hanging() {
    let problem = Arc::new(Problem::new(
        2, 1, None, None, Some(vec![-1]),
        vec![SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))); 2],
        offset_sphere,
    ));
    NSGAII::new(problem, 0, ExecutionMode::Sequential);
}

#[test]
#[should_panic(expected = "supply both or neither")]
fn half_configured_objective_constraints_are_rejected() {
    // Operands without bounds used to make every constraint vanish silently.
    let problem = Arc::new(Problem {
        solution_length: 1,
        number_of_objectives: 1,
        objective_constraint: None,
        objective_constraint_operands: Some(vec![Some("<".to_string())]),
        direction: Some(vec![-1]),
        solution_data_types: vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))],
        eval_fn: puggles::core::EvalFn::Single(offset_sphere),
        variable_constraints: None,
    });
    Solution::new(problem).evaluate();
}

#[test]
fn optional_objective_constraint_elements_are_skipped() {
    // A `None` element means "no constraint on this objective" — it used to panic on unwrap.
    let problem = Arc::new(Problem::new(
        2, 2, Some(vec![None, Some(1.0)]), Some(vec![None, Some("<".to_string())]),
        Some(vec![-1, -1]),
        vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(10.0))); 2],
        |x: &Vec<f64>| vec![x[0], x[1]],
    ));
    let mut s = Solution::new(Arc::clone(&problem));
    s.solution = vec![7.0, 0.5]; // objective 0 unconstrained; objective 1 satisfies < 1
    s.evaluate();
    assert_eq!(s.constraint_violation, 0);
    assert!(s.feasible);
}
