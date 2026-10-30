//! Interactive plots and the dashboard from Rust.
//!
//!   cargo run --release --features plot --example plot_pareto            # write HTML files
//!   cargo run --release --features plot --example plot_pareto -- --live    # watch a run live
//!
//! Live options:
//!   --problem NAME   zdt1 (default) | knapsack | knapsack3 | dtlz2-5
//!   --gpu            evaluate on the GPU (zdt1, knapsack); build with --features plot,gpu
//!   --pace-ms N      sleep N ms per generation so the run is watchable (default 10; 0 = full speed)
//!
//! e.g.  cargo run --release --features plot,gpu --example plot_pareto -- --live --problem knapsack --gpu
//!
//! Files go to `target/plots/`. Set PUGGLES_NO_BROWSER=1 to skip opening them.

use puggles::core::Problem;
use puggles::gatypes::{BitBinary, Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use puggles::nsga3::NSGAIII;
use puggles::plot::{aes, ggplot, open_in_browser, reference, Dashboard, Field::*, PlotError, RunHistory, RunMeta};
use std::f64::consts::FRAC_PI_2;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

fn zdt1(x: &Vec<f64>) -> Vec<f64> {
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
    vec![x[0], g * (1.0 - (x[0] / g).sqrt())]
}

/// DTLZ2 with M objectives = 12 - k + 1, k = 10 → M = 3.
fn dtlz2_3(x: &Vec<f64>) -> Vec<f64> {
    dtlz2(x, 3)
}
fn dtlz2_5(x: &Vec<f64>) -> Vec<f64> {
    dtlz2(x, 5)
}
fn dtlz2(x: &[f64], m: usize) -> Vec<f64> {
    let g: f64 = x[m - 1..].iter().map(|v| (v - 0.5).powi(2)).sum();
    (0..m)
        .map(|i| {
            let mut f = 1.0 + g;
            for v in &x[..m - 1 - i] {
                f *= (v * FRAC_PI_2).cos();
            }
            if i > 0 {
                f *= (x[m - 1 - i] * FRAC_PI_2).sin();
            }
            f
        })
        .collect()
}

fn reals(n: usize) -> Vec<SolutionDataTypes> {
    vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))); n]
}

// -- multi-objective 0/1 knapsack ---------------------------------------------------------
//
// Items each have a profit and a weight per knapsack; choose items (one bit each) to maximize
// every knapsack's profit while each knapsack's weight stays within its capacity. Zitzler &
// Thiele's benchmark uses half the total weight; 40% here makes random solutions almost always
// infeasible, so the dashboard shows the run finding feasibility. Data is generated once.

const ITEMS: usize = 250;

struct Knapsack {
    profit: [Vec<f64>; 3],
    weight: [Vec<f64>; 3],
    capacity: [f64; 3],
}

fn knapsack_data() -> &'static Knapsack {
    static DATA: OnceLock<Knapsack> = OnceLock::new();
    DATA.get_or_init(|| {
        let mut state: u64 = 0x5eed_cafe;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            10.0 + ((state >> 33) % 91) as f64 // integer in 10..=100
        };
        let mut col = || (0..ITEMS).map(|_| next()).collect::<Vec<f64>>();
        let profit = [col(), col(), col()];
        let weight = [col(), col(), col()];
        let capacity = [0, 1, 2].map(|k| weight[k].iter().sum::<f64>() * 0.4);
        Knapsack { profit, weight, capacity }
    })
}

fn dot(x: &[f64], v: &[f64]) -> f64 {
    x.iter().zip(v).map(|(a, b)| a * b).sum()
}
fn knapsack2(x: &Vec<f64>) -> Vec<f64> {
    let d = knapsack_data();
    vec![dot(x, &d.profit[0]), dot(x, &d.profit[1])]
}
fn knapsack3(x: &Vec<f64>) -> Vec<f64> {
    let d = knapsack_data();
    vec![dot(x, &d.profit[0]), dot(x, &d.profit[1]), dot(x, &d.profit[2])]
}
// Capacity constraints g(x) <= 0.
fn over0(x: &Vec<f64>) -> f64 {
    let d = knapsack_data();
    dot(x, &d.weight[0]) - d.capacity[0]
}
fn over1(x: &Vec<f64>) -> f64 {
    let d = knapsack_data();
    dot(x, &d.weight[1]) - d.capacity[1]
}
fn over2(x: &Vec<f64>) -> f64 {
    let d = knapsack_data();
    dot(x, &d.weight[2]) - d.capacity[2]
}

fn knapsack_problem(m: usize) -> Problem {
    let f = if m == 2 { knapsack2 } else { knapsack3 };
    let g: Vec<fn(&Vec<f64>) -> f64> = [over0, over1, over2][..m].to_vec();
    Problem::new(ITEMS, m, None, None, Some(vec![1; m]), vec![SolutionDataTypes::BitBinary(BitBinary::new()); ITEMS], f)
        .with_variable_constraints(g)
}

// -- GPU shaders (same objectives, evaluated in WGSL) ------------------------------------------

#[cfg(feature = "gpu")]
const ZDT1_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> solutions: array<f32>;
@group(0) @binding(1) var<storage, read_write> objectives: array<f32>;
struct Params { solution_length: u32, num_objectives: u32, pop_size: u32 };
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if (s >= params.pop_size) { return; }
    let n = params.solution_length;
    let off = s * n;
    var rest = 0.0;
    for (var i = 1u; i < n; i = i + 1u) { rest = rest + solutions[off + i]; }
    let x0 = solutions[off];
    let g = 1.0 + 9.0 * rest / f32(n - 1u);
    objectives[s * 2u] = x0;
    objectives[s * 2u + 1u] = g * (1.0 - sqrt(x0 / g));
}
"#;

/// Profits are uploaded once as `data` (knapsack-major); each invocation scores one solution.
#[cfg(feature = "gpu")]
const KNAPSACK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> solutions: array<f32>;
@group(0) @binding(1) var<storage, read_write> objectives: array<f32>;
struct Params { solution_length: u32, num_objectives: u32, pop_size: u32 };
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if (s >= params.pop_size) { return; }
    let n = params.solution_length;
    let m = params.num_objectives;
    for (var k = 0u; k < m; k = k + 1u) {
        var total = 0.0;
        for (var i = 0u; i < n; i = i + 1u) { total = total + solutions[s * n + i] * data[k * n + i]; }
        objectives[s * m + k] = total;
    }
}
"#;

/// An NSGA-II that evaluates on the GPU (built with `--features gpu`).
#[cfg(feature = "gpu")]
fn gpu_nsgaii(problem: Arc<Problem>, pop: usize, problem_name: &str) -> NSGAII {
    use puggles::gpu_evaluator::GpuEvaluator;
    let (n, m) = (problem.solution_length, problem.number_of_objectives);
    let evaluator = if problem_name == "zdt1" {
        GpuEvaluator::new_blocking(ZDT1_WGSL, n, m)
    } else {
        let d = knapsack_data();
        let profits: Vec<f32> = d.profit[..m].iter().flatten().map(|&v| v as f32).collect();
        GpuEvaluator::new_blocking_with_data(KNAPSACK_WGSL, n, m, &profits)
    };
    NSGAII::new(problem, pop, ExecutionMode::GPU).with_gpu_evaluator(evaluator)
}

/// Without the `gpu` feature, GPU is still *requested*, so the dashboard shows the fallback and why.
#[cfg(not(feature = "gpu"))]
fn gpu_nsgaii(problem: Arc<Problem>, pop: usize, _problem_name: &str) -> NSGAII {
    eprintln!("note: built without the `gpu` feature; rebuild with --features plot,gpu to use the GPU");
    NSGAII::new(problem, pop, ExecutionMode::GPU)
}

// -- live dashboard ----------------------------------------------------------------------------

fn live(args: &[String]) -> Result<(), PlotError> {
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let problem_name = value("--problem").unwrap_or_else(|| "zdt1".into());
    let gpu = args.iter().any(|a| a == "--gpu");
    let pace_ms: u64 = value("--pace-ms").and_then(|v| v.parse().ok()).unwrap_or(10);
    let mode = ExecutionMode::MultiThreaded;

    let (problem, title, names, nfe): (Arc<Problem>, String, Vec<String>, usize) = match problem_name.as_str() {
        "zdt1" => (
            Arc::new(Problem::new(30, 2, None, None, Some(vec![-1, -1]), reals(30), zdt1)),
            "ZDT1 · 30 variables".into(), vec!["f1".into(), "f2".into()], 300_000,
        ),
        "knapsack" | "knapsack3" => {
            let m = if problem_name == "knapsack" { 2 } else { 3 };
            (
                Arc::new(knapsack_problem(m)),
                format!("0/1 knapsack · {ITEMS} items × {m} knapsacks"),
                (1..=m).map(|k| format!("profit {k}")).collect(), 400_000,
            )
        }
        "dtlz2-5" => (
            Arc::new(Problem::new(14, 5, None, None, None, reals(14), dtlz2_5)),
            "DTLZ2 · 5 objectives".into(), (1..=5).map(|k| format!("f{k}")).collect(), 200_000,
        ),
        other => {
            eprintln!("unknown --problem {other:?}; choose zdt1, knapsack, knapsack3 or dtlz2-5");
            std::process::exit(2);
        }
    };
    let title = if pace_ms > 0 { format!("{title} (paced {pace_ms} ms/gen)") } else { title };
    let mut meta = RunMeta::from_problem(&problem).title(title).objective_names(names);
    // Known true fronts: drawn faintly, and IGD against them becomes a convergence metric.
    match problem_name.as_str() {
        "zdt1" => meta = meta.reference_front(reference::zdt1(200)),
        "dtlz2-5" => meta = meta.reference_front(reference::dtlz2(5, 6)),
        _ => {}
    }
    let dash = Dashboard::start(meta, "127.0.0.1", 0)?;
    dash.open_browser();

    let pace = || std::thread::sleep(Duration::from_millis(pace_ms));
    // NSGA-III for 3+ objectives; it has no GPU path, so --gpu there shows the CPU fallback.
    if problem.number_of_objectives >= 3 {
        let divisions = if problem.number_of_objectives == 3 { 12 } else { 4 };
        let mut ga = NSGAIII::new(Arc::clone(&problem), 0, divisions, if gpu { ExecutionMode::GPU } else { mode });
        ga.run_with(nfe, |ga| {
            pace();
            dash.apply_focus(ga); // goals set in the page can steer NSGA-III
            dash.record(ga)
        });
    } else {
        let mut ga = if gpu { gpu_nsgaii(Arc::clone(&problem), 100, &problem_name) } else { NSGAII::new(Arc::clone(&problem), 100, mode) };
        ga.run_with(nfe, |ga| {
            pace();
            dash.record(ga)
        });
    }
    dash.finish();
    dash.wait();
    Ok(())
}

fn main() -> Result<(), PlotError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--live") {
        return live(&args);
    }
    let zdt = Arc::new(Problem::new(10, 2, None, None, Some(vec![-1, -1]), reals(10), zdt1));

    let out = std::path::Path::new("target/plots");
    std::fs::create_dir_all(out)?;

    // 2-objective front, recorded generation by generation.
    let mut ga = NSGAII::new(Arc::clone(&zdt), 100, ExecutionMode::MultiThreaded).with_seed(42);
    let mut history = RunHistory::new(RunMeta::from_problem(&zdt).title("ZDT1").reference_front(reference::zdt1(200)));
    ga.run_with(20_000, |ga| {
        history.record(ga);
        true
    });
    let front = ggplot(ga.get_archive())
        .aes(aes().x(Obj(0)).y(Obj(1)))
        .geom_line().alpha(0.35)
        .geom_point().layer_aes(aes().color(Var(1)))
        .title("ZDT1 Pareto front")
        .save(out.join("zdt1_front.html"))?;
    let convergence = ggplot(&history)
        .aes(aes().x(Nfe).y(Obj(1)))
        .geom_line()
        .title("Best f2 on the front")
        .xlab("evaluations")
        .ylab("f2")
        .save(out.join("zdt1_convergence.html"))?;
    let replay = history.save_html(out.join("zdt1_replay.html"))?;
    history.save_json(out.join("zdt1.pgrun.json"))?;

    // 3 objectives: a rotatable 3-D scatter, colored by front rank across the population.
    let dtlz3 = Arc::new(Problem::new(12, 3, None, None, None, reals(12), dtlz2_3));
    let mut ga3 = NSGAIII::new(Arc::clone(&dtlz3), 0, 12, ExecutionMode::MultiThreaded).with_seed(1);
    ga3.run(30_000);
    let three = ggplot(&ga3.population)
        .aes(aes().x(Obj(0)).y(Obj(1)).z(Obj(2)).color(Rank))
        .geom_point()
        .scale_color_categorical()
        .title("DTLZ2, 3 objectives (NSGA-III)")
        .save(out.join("dtlz2_3d.html"))?;

    // 5 objectives: parallel coordinates.
    let dtlz5 = Arc::new(Problem::new(14, 5, None, None, None, reals(14), dtlz2_5));
    let mut ga5 = NSGAIII::new(Arc::clone(&dtlz5), 0, 6, ExecutionMode::MultiThreaded).with_seed(1);
    ga5.run(40_000);
    let parallel = ggplot(ga5.get_archive())
        .aes(aes().color(Obj(0)))
        .geom_parallel([Obj(0), Obj(1), Obj(2), Obj(3), Obj(4)])
        .title("DTLZ2, 5 objectives")
        .save(out.join("dtlz2_parallel.html"))?;

    for path in [&front, &convergence, &three, &parallel, &replay] {
        println!("wrote {}", path.display());
    }
    let _ = open_in_browser(&front.to_string_lossy());
    Ok(())
}
