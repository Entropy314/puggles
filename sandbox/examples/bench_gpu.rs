//! CPU vs GPU evaluation for NSGA-II, across objective cost.
//!
//! The GPU path (wgpu compute shader) has fixed per-generation overhead
//! (buffer upload + dispatch + synchronous readback). It only pays off when
//! per-evaluation compute is large enough to hide that overhead — this sweep
//! shows the crossover from "CPU wins" (cheap objective) to "GPU wins" (heavy).
//!
//! The objective (same on CPU and GPU): for a D-dim solution, with work factor W:
//!   f1 = Σ_i Σ_{k<W} cos(x_i · (k+1))      (heavy, scales with W)
//!   f2 = Σ_i (x_i − 0.5)²
//!
//! Run: cargo run --release --features gpu --example bench_gpu

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Build with the gpu feature:  cargo run --release --features gpu --example bench_gpu");
}

#[cfg(feature = "gpu")]
fn main() {
    use puggles::core::{EvalFn, Problem};
    use puggles::gatypes::{Real, SolutionDataTypes};
    use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
    use puggles::gpu_evaluator::GpuEvaluator;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    // Per-evaluation work factor, shared by the CPU objective (GPU bakes it into the shader).
    static WORK: AtomicUsize = AtomicUsize::new(1);

    fn heavy_obj(x: &Vec<f64>) -> Vec<f64> {
        let w = WORK.load(Ordering::Relaxed);
        let mut f1 = 0.0f64;
        let mut f2 = 0.0f64;
        for &xi in x {
            let mut acc = 0.0f64;
            for k in 0..w {
                acc += (xi * (k as f64 + 1.0)).cos();
            }
            f1 += acc;
            f2 += (xi - 0.5) * (xi - 0.5);
        }
        vec![f1, f2]
    }

    fn shader(work: usize) -> String {
        format!(
            r#"
@group(0) @binding(0) var<storage, read> solutions: array<f32>;
@group(0) @binding(1) var<storage, read_write> objectives: array<f32>;
struct Params {{ solution_length: u32, num_objectives: u32, pop_size: u32 }};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let s = id.x;
    if (s >= params.pop_size) {{ return; }}
    let base = s * params.solution_length;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    for (var i: u32 = 0u; i < params.solution_length; i = i + 1u) {{
        let xi = solutions[base + i];
        var acc: f32 = 0.0;
        for (var k: u32 = 0u; k < {work}u; k = k + 1u) {{
            acc = acc + cos(xi * f32(k + 1u));
        }}
        f1 = f1 + acc;
        f2 = f2 + (xi - 0.5) * (xi - 0.5);
    }}
    let o = s * params.num_objectives;
    objectives[o] = f1;
    objectives[o + 1u] = f2;
}}
"#
        )
    }

    fn build_problem(d: usize) -> Arc<Problem> {
        let types = (0..d)
            .map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))))
            .collect();
        Arc::new(Problem {
            solution_length: d,
            number_of_objectives: 2,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1]),
            solution_data_types: types,
            variable_constraints: None,
            eval_fn: EvalFn::Single(heavy_obj),
        })
    }

    fn time_run(problem: &Arc<Problem>, pop: usize, nfe: usize, mode: ExecutionMode,
                gpu: Option<GpuEvaluator>) -> f64 {
        let mut ga = NSGAII::new(Arc::clone(problem), pop, mode);
        #[allow(unused_mut)]
        if let Some(ev) = gpu {
            ga = ga.with_gpu_evaluator(ev);
        }
        let t0 = Instant::now();
        ga.run(nfe);
        t0.elapsed().as_secs_f64() * 1000.0
    }

    // (label, D, W, POP, NFE)
    let configs = [
        ("cheap    D=20  W=1",    20,   1,   256, 20_000),
        ("light    D=50  W=16",   50,  16,   256, 20_000),
        ("medium   D=100 W=128",  100, 128,  256, 20_000),
        ("heavy    D=200 W=512",  200, 512,  256, 20_000),
        ("v.heavy  D=400 W=2048", 400, 2048, 256, 20_000),
    ];

    println!("\nCPU vs GPU objective evaluation (NSGA-II, pop=256, NFE=20000)");
    println!("Apple M3 (Metal). Objective cost grows with D (dims) and W (work/dim).\n");
    println!("{:<22} {:>10} {:>10} {:>10}   {:>12}", "config", "seq(ms)", "par(ms)", "gpu(ms)", "gpu vs par");
    println!("{}", "-".repeat(70));

    for (label, d, w, pop, nfe) in configs {
        WORK.store(w, Ordering::Relaxed);
        let problem = build_problem(d);

        let seq = time_run(&problem, pop, nfe, ExecutionMode::Sequential, None);
        let par = time_run(&problem, pop, nfe, ExecutionMode::MultiThreaded, None);

        // Build the GPU evaluator (one-time device/pipeline init) BEFORE timing the run.
        let ev = GpuEvaluator::new_blocking(&shader(w), d, 2);
        let gpu = time_run(&problem, pop, nfe, ExecutionMode::GPU, Some(ev));

        let ratio = par / gpu;
        let verdict = if ratio > 1.0 { format!("{ratio:.1}x faster") } else { format!("{:.1}x slower", 1.0 / ratio) };
        println!("{label:<22} {seq:>10.1} {par:>10.1} {gpu:>10.1}   {verdict:>12}");
    }

    println!("\n# 'gpu vs par' > 1x means GPU beat the multithreaded CPU path.");
    println!("# GPU carries fixed per-generation upload+dispatch+readback latency;");
    println!("# it wins once per-evaluation compute (D×W) is large enough to amortize it.");
}
