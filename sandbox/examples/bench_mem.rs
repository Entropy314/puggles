//! Isolated optimizer memory + time: puggles on a synthetic 20-asset portfolio.
//!
//! No CSV load — a synthetic 20x20 covariance is built in-process, so peak RSS reflects
//! only the Rust runtime + optimizer working set (a clean comparison vs the Python libs,
//! each run in its own process by bench_mem.py).
//!
//! Run: cargo run --release --example bench_mem

use puggles::core::{EvalFn, Problem};
use puggles::gatypes::{Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use std::sync::{atomic::{AtomicU64, Ordering}, Arc, RwLock};
use std::time::Instant;
use sysinfo::{Pid, System};

const N: usize = 20;
const POP: usize = 200;
const NFE: usize = 10_000;

struct Data { means: Vec<f64>, cov: Vec<Vec<f64>> }
static DATA: RwLock<Option<Data>> = RwLock::new(None);

fn portfolio(x: &Vec<f64>) -> Vec<f64> {
    let g = DATA.read().unwrap();
    let d = g.as_ref().unwrap();
    let total: f64 = x.iter().sum();
    let w: Vec<f64> = if total > 1e-12 { x.iter().map(|v| v / total).collect() } else { vec![1.0 / N as f64; N] };
    let ret: f64 = w.iter().zip(&d.means).map(|(wi, mi)| wi * mi).sum();
    let var: f64 = (0..N).map(|i| (0..N).map(|j| w[i] * w[j] * d.cov[i][j]).sum::<f64>()).sum();
    vec![var, -ret]
}

fn peak_rss<F: FnOnce()>(f: F) -> (f64, f64) {
    let peak = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicU64::new(0));
    let (pk, st, pid) = (Arc::clone(&peak), Arc::clone(&stop), Pid::from(std::process::id() as usize));
    let h = std::thread::spawn(move || {
        let mut sys = System::new();
        while st.load(Ordering::Relaxed) == 0 {
            sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
            if let Some(p) = sys.process(pid) { pk.fetch_max(p.memory() / 1024, Ordering::Relaxed); }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
    let t0 = Instant::now();
    f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    stop.store(1, Ordering::Relaxed);
    let _ = h.join();
    (ms, peak.load(Ordering::Relaxed) as f64 / 1024.0)
}

fn main() {
    // Synthetic PSD covariance (A^T A) + small means; values don't affect memory.
    let mut a = vec![vec![0.0f64; N]; 64];
    let mut s: u64 = 0x9e3779b97f4a7c15;
    let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
    for row in a.iter_mut() { for v in row.iter_mut() { *v = rnd() * 0.02; } }
    let mut cov = vec![vec![0.0; N]; N];
    for i in 0..N { for j in 0..N { cov[i][j] = (0..64).map(|k| a[k][i] * a[k][j]).sum::<f64>() / 64.0; } }
    let means: Vec<f64> = (0..N).map(|_| rnd() * 0.01).collect();
    *DATA.write().unwrap() = Some(Data { means, cov });

    let types: Vec<SolutionDataTypes> = (0..N).map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))).collect();
    let problem = Arc::new(Problem {
        solution_length: N, number_of_objectives: 2,
        objective_constraint: None, objective_constraint_operands: None,
        direction: Some(vec![-1, -1]), solution_data_types: types,
        variable_constraints: None, eval_fn: EvalFn::Single(portfolio),
    });

    for (label, mode) in [("puggles-seq", ExecutionMode::Sequential), ("puggles-par", ExecutionMode::MultiThreaded)] {
        NSGAII::new(Arc::clone(&problem), POP, mode).run(NFE); // warm-up
        let (ms, mb) = peak_rss(|| { NSGAII::new(Arc::clone(&problem), POP, mode).run(NFE); });
        println!("MEM\t{label}\t{ms:.1}\t{mb:.0}");
    }
}
