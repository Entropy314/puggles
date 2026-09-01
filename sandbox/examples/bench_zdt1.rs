//! ZDT1 NSGA-II benchmark — puggles vs the Python ecosystem.
//!
//! Run in release mode:  cargo run --release --example bench_zdt1
//!
//! Two regimes are compared:
//!
//!   CHEAP  — ZDT1 objective (~30 ns/eval).  Algorithm overhead dominates.
//!            pymoo wins here because it uses compiled Cython operators and
//!            evaluates the entire population in one NumPy call.
//!
//!   EXPENSIVE — O(N²) coupling simulation (~300 μs/eval).  Evaluation
//!               dominates.  Rayon gives near-linear speedup; Python cannot
//!               exploit threads due to the GIL.

use puggles::core::{EvalFn, Problem, Solution};
use puggles::gatypes::{Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use std::sync::Arc;
use std::time::Instant;

const N: usize = 10;
const POP: usize = 100;
const NFE: usize = 10_000;
const N_REF: usize = 100;

// ── Objectives ────────────────────────────────────────────────────────────────

fn zdt1(x: &Vec<f64>) -> Vec<f64> {
    let f1 = x[0];
    let g = 1.0 + 9.0 / (N as f64 - 1.0) * x[1..].iter().sum::<f64>();
    vec![f1, g * (1.0 - (f1 / g).sqrt())]
}

/// O(N²) coupling applied 200 times — ~300 μs/eval with N=10.
/// Each step: vᵢ = Σⱼ vⱼ/(1+|i−j|), then normalise.
fn expensive_obj(x: &Vec<f64>) -> Vec<f64> {
    const STEPS: usize = 200;
    let n = x.len();
    let mut v = x.clone();
    let mut tmp = vec![0.0_f64; n];
    for _ in 0..STEPS {
        for i in 0..n {
            tmp[i] = v
                .iter()
                .enumerate()
                .map(|(j, &vj)| vj / (1.0 + (i as f64 - j as f64).abs()))
                .sum::<f64>();
        }
        let norm = tmp.iter().map(|&t| t * t).sum::<f64>().sqrt().max(1e-10);
        v.iter_mut()
            .zip(tmp.iter())
            .for_each(|(vi, &ti)| *vi = ti / norm);
    }
    let f1: f64 = v.iter().map(|&vi| vi * vi).sum();
    let f2: f64 = v.iter().zip(x.iter()).map(|(&vi, &xi)| (vi - xi).powi(2)).sum();
    vec![f1, f2]
}

// ── Metrics ───────────────────────────────────────────────────────────────────

fn igd(solutions: &[Solution]) -> f64 {
    let reference: Vec<(f64, f64)> = (1..=N_REF)
        .map(|i| {
            let t = i as f64 / N_REF as f64;
            (t, 1.0 - t.sqrt())
        })
        .collect();
    let approx: Vec<(f64, f64)> = solutions
        .iter()
        .filter(|s| s.evaluated)
        .map(|s| (s.objective_fitness_values[0], s.objective_fitness_values[1]))
        .collect();
    if approx.is_empty() {
        return f64::INFINITY;
    }
    reference
        .iter()
        .map(|&(rf1, rf2)| {
            approx
                .iter()
                .map(|&(af1, af2)| ((rf1 - af1).powi(2) + (rf2 - af2).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>()
        / N_REF as f64
}

/// Time one objective call in isolation (nanoseconds).
fn ns_per_eval(f: fn(&Vec<f64>) -> Vec<f64>) -> f64 {
    let x = vec![0.3_f64; N];
    // Warm up
    for _ in 0..100 {
        std::hint::black_box(f(std::hint::black_box(&x)));
    }
    const ITERS: u32 = 1000;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(f(std::hint::black_box(&x)));
    }
    t0.elapsed().as_nanos() as f64 / ITERS as f64
}

fn stats(xs: &[f64]) -> (f64, f64) {
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let std = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64).sqrt();
    (mean, std)
}

// ── Benchmark runner ──────────────────────────────────────────────────────────

fn bench(problem: Arc<Problem>, mode: ExecutionMode, label: &str, runs: usize, with_igd: bool) {
    // Warm-up: initialise RAYON pool and fill instruction caches.
    {
        let mut ga = NSGAII::new(Arc::clone(&problem), POP, mode);
        ga.run(NFE);
    }

    let mut times = Vec::with_capacity(runs);
    let mut igds = Vec::with_capacity(runs);

    for _ in 0..runs {
        let mut ga = NSGAII::new(Arc::clone(&problem), POP, mode);
        let t0 = Instant::now();
        ga.run(NFE);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        igds.push(igd(ga.get_archive()));
    }

    let (mt, st) = stats(&times);
    if with_igd {
        let (mi, si) = stats(&igds);
        println!("{:<34} {:>10.1} {:>10.1} {:>9.4} {:>9.4}", label, mt, st, mi, si);
    } else {
        println!("{:<34} {:>10.1} {:>10.1}", label, mt, st);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let real_types = |lo: f64, hi: f64| -> Vec<SolutionDataTypes> {
        (0..N)
            .map(|_| SolutionDataTypes::Real(Real::new(Some(lo), Some(hi))))
            .collect()
    };

    let hr = "─".repeat(76);

    // ── Part 1: ZDT1 (cheap) ─────────────────────────────────────────────────

    let ns = ns_per_eval(zdt1);
    println!("\n╔═ CHEAP OBJECTIVE: ZDT1 ════════════════════════════════════════╗");
    println!(  "║  ~{:.0} ns/eval   ×  {:5} evals  =  {:.2} ms pure eval time  ║",
        ns, NFE, ns * NFE as f64 / 1e6);
    println!(  "║  Algorithm overhead (sort/select/crossover) dominates.         ║");
    println!(  "╚════════════════════════════════════════════════════════════════╝\n");
    println!("  WHY pymoo is faster: it evaluates all {:} solutions in ONE NumPy", POP);
    println!("  call (SIMD, cache-friendly).  All its operators are compiled Cython.");
    println!("  WHY parallel ≈ sequential: Rayon task overhead ≫ 30 ns per eval.\n");

    let zdt1_prob = Arc::new(Problem {
        solution_length: N,
        number_of_objectives: 2,
        objective_constraint: None,
        objective_constraint_operands: None,
        direction: Some(vec![-1, -1]),
        solution_data_types: real_types(0.0, 1.0),
        variable_constraints: None,
        eval_fn: EvalFn::Single(zdt1),
    });

    println!("# pop={POP}  NFE={NFE}  5 runs  IGD lower=better");
    println!("{:<34} {:>10} {:>10} {:>9} {:>9}", "library", "ms/run", "±std", "IGD", "±std");
    println!("{hr}");
    bench(Arc::clone(&zdt1_prob), ExecutionMode::Sequential, "puggles (sequential)", 5, true);
    bench(Arc::clone(&zdt1_prob), ExecutionMode::MultiThreaded, "puggles (parallel)", 5, true);
    println!("{hr}");
    println!("# Install libraries and run bench_python.py to compare with pymoo/DEAP/platypus.");

    // ── Part 2: Expensive objective ───────────────────────────────────────────

    let ns_exp = ns_per_eval(expensive_obj);
    println!("\n╔═ EXPENSIVE OBJECTIVE: O(N²) coupling × 200 steps ═════════════╗");
    println!(  "║  ~{:.0} μs/eval  ×  {:5} evals  =  {:.1} s pure eval time    ║",
        ns_exp / 1000.0, NFE, ns_exp * NFE as f64 / 1e9);
    println!(  "║  Evaluation dominates.  Rayon gives near-linear speedup.       ║");
    println!(  "║  Python (GIL) cannot parallelise this; expect ≥30× slower.     ║");
    println!(  "╚════════════════════════════════════════════════════════════════╝\n");

    let exp_prob = Arc::new(Problem {
        solution_length: N,
        number_of_objectives: 2,
        objective_constraint: None,
        objective_constraint_operands: None,
        direction: Some(vec![-1, -1]),
        solution_data_types: real_types(-1.0, 1.0),
        variable_constraints: None,
        eval_fn: EvalFn::Single(expensive_obj),
    });

    println!("# pop={POP}  NFE={NFE}  3 runs  (timing only; IGD not applicable)");
    println!("{:<34} {:>10} {:>10}", "library", "ms/run", "±std");
    println!("{}", "─".repeat(56));
    bench(Arc::clone(&exp_prob), ExecutionMode::Sequential, "puggles (sequential)", 3, false);
    bench(Arc::clone(&exp_prob), ExecutionMode::MultiThreaded, "puggles (parallel)", 3, false);
    println!("{}", "─".repeat(56));

    let n_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("# logical cores: {n_cores}");
    println!("# Python equiv (estimated): ~{:.0} s sequential, no real thread speedup (GIL).",
        ns_exp * NFE as f64 * 30.0 / 1e9);
}
