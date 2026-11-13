//! The dataset benchmark's portfolio problem on an NVIDIA GPU via `CudaEvaluator`.
//! Same CLI and output lines as sandbox/examples/bench_datasets.rs, so
//! sandbox/bench_datasets.py can run it as the `puggles-cuda` row:
//!   cargo run --release --example bench_cuda -- ../sandbox/data/real_returns_1gb.csv multi puggles-cuda 2000 1

use puggles::core::{EvalFn, Problem};
use puggles::gatypes::{Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use puggles_cuda::CudaEvaluator;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

const N: usize = 20;
const POP: usize = 100;

static DATA: OnceLock<Vec<f64>> = OnceLock::new();

fn rows() -> usize {
    DATA.get().unwrap().len() / N
}

/// Row-major returns from a `date,[scenario,]A,B,...` CSV.
fn load(path: &str) -> f64 {
    let t0 = Instant::now();
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let keep: Vec<usize> = (0..header.len()).filter(|&i| header[i] != "date" && header[i] != "scenario").collect();
    assert_eq!(keep.len(), N);
    let mut data = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        data.extend(keep.iter().map(|&i| f[i].parse::<f64>().unwrap()));
    }
    DATA.set(data).unwrap();
    t0.elapsed().as_secs_f64() * 1000.0
}

fn finish_multi(s: &[f64]) -> Vec<f64> {
    let t = rows() as f64;
    let m = s[0] / t;
    vec![(s[1] / t - m * m).max(0.0), -m]
}

fn finish_single(s: &[f64]) -> Vec<f64> {
    let t = rows() as f64;
    let (m, v) = (s[0] / t, (s[1] / t - (s[0] / t).powi(2)).max(0.0));
    vec![if v > 0.0 { -m / v.sqrt() } else { 0.0 }]
}

fn port_stats(x: &[f64]) -> (f64, f64) {
    let total: f64 = x.iter().sum();
    let (mut s, mut s2) = (0.0, 0.0);
    for row in DATA.get().unwrap().chunks_exact(N) {
        let r: f64 = row.iter().zip(x).map(|(a, b)| a * b / total).sum();
        s += r;
        s2 += r * r;
    }
    let m = s / rows() as f64;
    (m, (s2 / rows() as f64 - m * m).max(0.0))
}

fn unused_objective(_: &Vec<f64>) -> Vec<f64> {
    unreachable!("evaluation runs on the CUDA evaluator")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, multi) = (&a[1], a[2] == "multi");
    let (nfe, seed): (usize, u64) = (a[4].parse().unwrap(), a[5].parse().unwrap());
    let load_ms = load(path);

    let t0 = Instant::now(); // device init + upload count as optimizer time, like the wgpu row
    let eval = CudaEvaluator::new(DATA.get().unwrap(), N, if multi { finish_multi } else { finish_single })
        .expect("CUDA device");
    let n_obj = if multi { 2 } else { 1 };
    let problem = Arc::new(Problem {
        solution_length: N,
        number_of_objectives: n_obj,
        objective_constraint: None,
        objective_constraint_operands: None,
        direction: Some(vec![-1; n_obj]),
        solution_data_types: (0..N).map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))).collect(),
        variable_constraints: None,
        encoding: puggles::core::Encoding::PerGene,
        eval_fn: EvalFn::Single(unused_objective),
    });
    let mut ga = NSGAII::new(problem, POP, ExecutionMode::GPU).with_seed(seed).with_batch_evaluator(eval);
    ga.run(nfe);
    let opt_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("RESULT\t{load_ms:.1}\t{opt_ms:.1}\t{}", ga.get_nfe());
    if multi {
        for s in ga.get_archive() {
            let (m, v) = port_stats(&s.solution);
            println!("FRONT\t{v:e}\t{m:e}");
        }
    } else {
        let best = ga.get_archive().iter().chain(ga.population.iter())
            .min_by(|a, b| a.objective_fitness_values[0].total_cmp(&b.objective_fitness_values[0]))
            .unwrap();
        let (m, v) = port_stats(&best.solution);
        println!("BEST\t{:.6}", if v > 0.0 { m / v.sqrt() } else { 0.0 });
    }
}
