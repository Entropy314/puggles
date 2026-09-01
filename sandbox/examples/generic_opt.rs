//! ZDT1 bi-objective benchmark — the canonical NSGA-II test problem.
//!
//! Decision variables: x ∈ [0, 1]^30
//!   f1 = x1
//!   g  = 1 + 9/(n-1) * Σ(x2..xn)
//!   f2 = g * (1 - sqrt(f1/g))
//!
//! Known Pareto front: x2=…=xn=0  →  f2 = 1 - sqrt(f1), f1 ∈ [0,1]
//! Run: cargo run --example generic_opt

use puggles::core::{EvalFn, Problem};
use puggles::gatypes::{Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use std::sync::Arc;

// ZDT1 with N=10 converges well within 20k NFEs.
const N: usize = 10;

fn zdt1(x: &Vec<f64>) -> Vec<f64> {
    let f1 = x[0];
    let g = 1.0 + 9.0 / (N as f64 - 1.0) * x[1..].iter().sum::<f64>();
    let f2 = g * (1.0 - (f1 / g).sqrt());
    vec![f1, f2]
}

fn main() {
    let types: Vec<SolutionDataTypes> = (0..N)
        .map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))))
        .collect();

    let problem = Arc::new(Problem {
        solution_length: N,
        number_of_objectives: 2,
        objective_constraint: None,
        objective_constraint_operands: None,
        direction: Some(vec![-1, -1]),
        solution_data_types: types,
        variable_constraints: None,
        eval_fn: EvalFn::Single(zdt1),
    });

    let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded);
    ga.run(20_000);

    println!("NFE: {}", ga.get_nfe());

    let mut archive = ga.get_archive().to_vec();
    archive.sort_by(|a, b| {
        a.objective_fitness_values[0]
            .partial_cmp(&b.objective_fitness_values[0])
            .unwrap()
    });

    // Deduplicate by rounding objectives to 3 decimal places
    let mut seen: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
    let unique: Vec<_> = archive
        .iter()
        .filter(|s| {
            let k = (
                (s.objective_fitness_values[0] * 1000.0).round() as u64,
                (s.objective_fitness_values[1] * 1000.0).round() as u64,
            );
            seen.insert(k)
        })
        .collect();

    println!("Pareto front: {} unique points\n", unique.len());
    println!("{:<10} {:<10} {:<12}", "f1", "f2", "f2_ideal");
    for sol in &unique {
        let f1 = sol.objective_fitness_values[0];
        let f2 = sol.objective_fitness_values[1];
        let ideal = 1.0 - f1.sqrt();
        println!("{:<10.4} {:<10.4} {:<12.4}", f1, f2, ideal);
    }
}
