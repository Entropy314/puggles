//! Large-dataset portfolio benchmark: load → covariance → NSGA-II.
//! Reports timing AND peak RSS memory per stage.
//!
//! Run: cargo run --release --example bench_large_data

use polars::prelude::*;
use rustypus::core::{EvalFn, Problem};
use rustypus::gatypes::{Real, SolutionDataTypes};
use rustypus::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use std::sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}};
use std::time::Instant;
use sysinfo::{Pid, System};

const N_ASSETS: usize = 20;
const POP: usize = 200;
const NFE: usize = 10_000;
const OPT_RUNS: usize = 3;

struct PortfolioData {
    means: Vec<f64>,
    cov: Vec<Vec<f64>>,
}

static DATA: RwLock<Option<PortfolioData>> = RwLock::new(None);

fn portfolio_objective(x: &Vec<f64>) -> Vec<f64> {
    let guard = DATA.read().unwrap();
    let data = guard.as_ref().unwrap();
    let n = data.means.len();
    let total: f64 = x.iter().sum();
    let w: Vec<f64> = if total > 1e-12 {
        x.iter().map(|xi| xi / total).collect()
    } else {
        vec![1.0 / n as f64; n]
    };
    let ret: f64 = w.iter().zip(&data.means).map(|(wi, mi)| wi * mi).sum();
    let var: f64 = (0..n)
        .map(|i| (0..n).map(|j| w[i] * w[j] * data.cov[i][j]).sum::<f64>())
        .sum();
    vec![var, -ret]
}

// ── Peak-RSS sampler ──────────────────────────────────────────────────────────

/// Spawn a background thread that polls this process's RSS every 10 ms.
/// Returns a handle; call `.stop()` to stop sampling and get peak MB.
struct PeakRss {
    peak_kb: Arc<AtomicU64>,
    stop:    Arc<AtomicU64>,  // 0 = running, 1 = stop requested
    handle:  Option<std::thread::JoinHandle<()>>,
}

impl PeakRss {
    fn start() -> Self {
        let peak_kb = Arc::new(AtomicU64::new(0));
        let stop    = Arc::new(AtomicU64::new(0));
        let pk2     = Arc::clone(&peak_kb);
        let st2     = Arc::clone(&stop);
        let pid     = Pid::from(std::process::id() as usize);

        let handle = std::thread::spawn(move || {
            let mut sys = System::new();
            while st2.load(Ordering::Relaxed) == 0 {
                sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(proc) = sys.process(pid) {
                    let rss = proc.memory(); // bytes
                    pk2.fetch_max(rss / 1024, Ordering::Relaxed); // store as KB
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });

        Self { peak_kb, stop, handle: Some(handle) }
    }

    fn stop(mut self) -> f64 {
        self.stop.store(1, Ordering::Relaxed);
        if let Some(h) = self.handle.take() { let _ = h.join(); }
        self.peak_kb.load(Ordering::Relaxed) as f64 / 1024.0  // KB → MB
    }
}

// ── Pipeline stages ───────────────────────────────────────────────────────────

fn time_load(path: &str) -> (DataFrame, f64, f64) {
    let sampler = PeakRss::start();
    let t0 = Instant::now();
    let df = LazyCsvReader::new(path)
        .with_has_header(true)
        .finish().unwrap().collect().unwrap();
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
    let peak_mb = sampler.stop();
    (df, elapsed, peak_mb)
}

fn time_cov(df: &DataFrame) -> (PortfolioData, f64, f64) {
    let sampler = PeakRss::start();
    let t0 = Instant::now();

    let tickers: Vec<String> = df.get_column_names().iter()
        .filter(|n| **n != "date").map(|s| s.to_string()).collect();
    let n = tickers.len();
    let mut series: Vec<Vec<f64>> = Vec::with_capacity(n);
    for ticker in &tickers {
        series.push(df.column(ticker).unwrap().f64().unwrap()
            .into_no_null_iter().collect());
    }
    let nrows = series[0].len() as f64;
    let means: Vec<f64> = series.iter().map(|s| s.iter().sum::<f64>() / nrows).collect();
    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let c = series[i].iter().zip(&series[j])
                .map(|(xi, xj)| (xi - means[i]) * (xj - means[j]))
                .sum::<f64>() / (nrows - 1.0);
            cov[i][j] = c; cov[j][i] = c;
        }
    }
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
    let peak_mb = sampler.stop();
    (PortfolioData { means, cov }, elapsed, peak_mb)
}

/// Returns (mean_ms, peak_rss_mb) over OPT_RUNS measured runs.
fn time_opt(problem: Arc<Problem>, mode: ExecutionMode) -> (f64, f64) {
    { NSGAII::new(Arc::clone(&problem), POP, mode).run(NFE); } // warm-up

    let mut times   = Vec::with_capacity(OPT_RUNS);
    let mut peaks   = Vec::with_capacity(OPT_RUNS);

    for _ in 0..OPT_RUNS {
        let sampler = PeakRss::start();
        let t0 = Instant::now();
        NSGAII::new(Arc::clone(&problem), POP, mode).run(NFE);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        peaks.push(sampler.stop());
    }

    let mean_ms  = times.iter().sum::<f64>() / times.len() as f64;
    let peak_mb  = peaks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean_ms, peak_mb)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let datasets = [
        ("10kb",  "data/returns_10kb.csv"),
        ("1mb",   "data/returns_1mb.csv"),
        ("10mb",  "data/returns_10mb.csv"),
        ("100mb", "data/returns_100mb.csv"),
        ("1gb",   "data/returns_1gb.csv"),
    ];

    let types: Vec<SolutionDataTypes> = (0..N_ASSETS)
        .map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))).collect();

    let problem = Arc::new(Problem {
        solution_length: N_ASSETS,
        number_of_objectives: 2,
        objective_constraint: None,
        objective_constraint_operands: None,
        direction: Some(vec![-1, -1]),
        solution_data_types: types,
        variable_constraints: None,
        eval_fn: EvalFn::Single(portfolio_objective),
    });

    let hr = "─".repeat(100);
    println!("\n╔═ Portfolio Pipeline Benchmark ══════════════════════════════════════════════════════════════╗");
    println!(  "║  N_ASSETS={N_ASSETS}  pop={POP}  NFE={NFE}  {OPT_RUNS} opt runs  │  timing + peak RSS per stage                  ║");
    println!(  "╚═════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    println!("{:<8} {:>7} {:>10} {:>9} {:>10} {:>9} {:>13} {:>12} {:>13} {:>12}",
        "size", "rows",
        "load(ms)", "mem(MB)",
        "cov(ms)",  "mem(MB)",
        "seq-opt(ms)", "mem(MB)",
        "par-opt(ms)", "mem(MB)");
    println!("{hr}");

    for (label, path) in &datasets {
        if !std::path::Path::new(path).exists() {
            eprintln!("# skipping {label}: {path} not found");
            continue;
        }
        let (df, t_load, m_load)  = time_load(path);
        let n_rows = df.height();
        let (port_data, t_cov, m_cov) = time_cov(&df);
        *DATA.write().unwrap() = Some(port_data);

        let (t_seq, m_seq) = time_opt(Arc::clone(&problem), ExecutionMode::Sequential);
        let (t_par, m_par) = time_opt(Arc::clone(&problem), ExecutionMode::MultiThreaded);

        println!("{label:>6}  {:>7}  {:>8.1}  {:>7.1}  {:>8.1}  {:>7.1}  {:>11.1}  {:>10.1}  {:>11.1}  {:>10.1}",
            n_rows, t_load, m_load, t_cov, m_cov, t_seq, m_seq, t_par, m_par);
        // Machine-readable line for the Python harness to parse.
        println!("RESULT\t{label}\t{n_rows}\t{t_load:.1}\t{m_load:.1}\t{t_cov:.1}\t{m_cov:.1}\t{t_seq:.1}\t{m_seq:.1}\t{t_par:.1}\t{m_par:.1}");
    }

    println!("{hr}");
    let n_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("# {n_cores} logical cores  |  mem = peak RSS of this process during each stage");
    println!("# RSS includes polars/rayon internal buffers — opt memory ≈ POP×N×8×2 ≈ {:.0} KB theoretical",
        (POP * N_ASSETS * 8 * 2) as f64 / 1024.0);
}
