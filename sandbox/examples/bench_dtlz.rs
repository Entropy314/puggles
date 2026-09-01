//! DTLZ2 (3-objective) NSGA-II benchmark — native puggles.
//!
//! Run in release mode:  cargo run --release --example bench_dtlz
//!
//! Uses the STANDARD DTLZ2 formulation (not the library's built-in `dtlz2`, whose
//! objective count is coupled to n) so the front is comparable across libraries.
//! Emits machine-readable lines the Python harness (`bench_dtlz.py`) parses:
//!   RESULT<TAB>seq<TAB><mean_ms><TAB><std_ms>
//!   RESULT<TAB>par<TAB><mean_ms><TAB><std_ms>
//!   PT<TAB>f1<TAB>f2<TAB>f3    (final sequential archive front — for HV/IGD)

use puggles::core::{EvalFn, Problem};
use puggles::gatypes::{Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

const N: usize = 12; // n_var
const M: usize = 3; // n_obj
const POP: usize = 100;
const NFE: usize = 20_000;
const RUNS: usize = 5;

/// Standard DTLZ2: first M-1 vars are angles, remaining k = n-(M-1) drive g.
/// Pareto front is the unit sphere first octant (Σ fᵢ² = 1).
fn dtlz2(x: &Vec<f64>) -> Vec<f64> {
    let g: f64 = x[(M - 1)..].iter().map(|&xi| (xi - 0.5).powi(2)).sum();
    let mut f = vec![0.0; M];
    for i in 0..M {
        let mut val = 1.0 + g;
        for j in 0..(M - 1 - i) {
            val *= (x[j] * PI / 2.0).cos();
        }
        if i > 0 {
            val *= (x[M - 1 - i] * PI / 2.0).sin();
        }
        f[i] = val;
    }
    f
}

fn stats(xs: &[f64]) -> (f64, f64) {
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let std = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64).sqrt();
    (mean, std)
}

fn bench(problem: &Arc<Problem>, mode: ExecutionMode) -> Vec<f64> {
    // Warm-up
    NSGAII::new(Arc::clone(problem), POP, mode).run(NFE);
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let mut ga = NSGAII::new(Arc::clone(problem), POP, mode);
        let t0 = Instant::now();
        ga.run(NFE);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times
}

fn main() {
    let types: Vec<SolutionDataTypes> = (0..N)
        .map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))))
        .collect();
    let problem = Arc::new(Problem {
        solution_length: N,
        number_of_objectives: M,
        objective_constraint: None,
        objective_constraint_operands: None,
        direction: Some(vec![-1; M]),
        solution_data_types: types,
        variable_constraints: None,
        eval_fn: EvalFn::Single(dtlz2),
    });

    let seq = bench(&problem, ExecutionMode::Sequential);
    let par = bench(&problem, ExecutionMode::MultiThreaded);
    let (ms, ss) = stats(&seq);
    let (mp, sp) = stats(&par);

    eprintln!("# DTLZ2  n={N}  m={M}  pop={POP}  NFE={NFE}  {RUNS} runs");
    println!("RESULT\tseq\t{ms:.1}\t{ss:.1}");
    println!("RESULT\tpar\t{mp:.1}\t{sp:.1}");

    // One more sequential run: dump its archive front for HV/IGD in the harness.
    let mut ga = NSGAII::new(Arc::clone(&problem), POP, ExecutionMode::Sequential);
    ga.run(NFE);
    for s in ga.get_archive() {
        let o = &s.objective_fitness_values;
        println!("PT\t{}\t{}\t{}", o[0], o[1], o[2]);
    }
}
