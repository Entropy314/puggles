//! Native CPU and GPU Puggles benchmark for:
//! (x[0] - 3)^2 + (x[1] - 6)^2

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Run with: cargo run --release --features gpu --example native_cpu_gpu_quadratic");
}

#[cfg(feature = "gpu")]
fn main() {
    use puggles::core::Problem;
    use puggles::gatypes::{Real, SolutionDataTypes};
    use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
    use puggles::gpu_evaluator::GpuEvaluator;
    use std::sync::Arc;
    use std::time::Instant;

    const POPULATION: usize = 100;
    const NFE: usize = 10_000;
    const REPEATS: usize = 5;

    fn quadratic(x: &Vec<f64>) -> Vec<f64> {
        vec![(x[0] - 3.0).powi(2) + (x[1] - 6.0).powi(2)]
    }

    fn problem() -> Arc<Problem> {
        Arc::new(Problem::new(
            2,
            1,
            None,
            None,
            Some(vec![-1]),
            vec![
                SolutionDataTypes::Real(Real::new(Some(-100.0), Some(100.0))),
                SolutionDataTypes::Real(Real::new(Some(-100.0), Some(100.0))),
            ],
            quadratic,
        ))
    }

    fn run_cpu(mode: ExecutionMode) -> (f64, f64) {
        let mut ga = NSGAII::new(problem(), POPULATION, mode).with_seed(42);
        let started = Instant::now();
        ga.run(NFE);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let best = ga
            .population
            .iter()
            .map(|solution| solution.objective_fitness_values[0])
            .fold(f64::INFINITY, f64::min);
        (elapsed_ms, best)
    }

    fn shader() -> &'static str {
        r#"
@group(0) @binding(0) var<storage, read> solutions: array<f32>;
@group(0) @binding(1) var<storage, read_write> objectives: array<f32>;
struct Params { solution_length: u32, num_objectives: u32, pop_size: u32 };
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let candidate = id.x;
    if (candidate >= params.pop_size) { return; }
    let offset = candidate * params.solution_length;
    let x0 = solutions[offset];
    let x1 = solutions[offset + 1u];
    objectives[candidate] = (x0 - 3.0) * (x0 - 3.0) + (x1 - 6.0) * (x1 - 6.0);
}
"#
    }

    fn run_gpu(evaluator: GpuEvaluator) -> (f64, f64) {
        let mut ga = NSGAII::new(problem(), POPULATION, ExecutionMode::GPU)
            .with_seed(42)
            .with_gpu_evaluator(evaluator);
        let started = Instant::now();
        ga.run(NFE);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let best = ga
            .population
            .iter()
            .map(|solution| solution.objective_fitness_values[0])
            .fold(f64::INFINITY, f64::min);
        (elapsed_ms, best)
    }

    fn median(mut measurements: Vec<f64>) -> f64 {
        measurements.sort_by(f64::total_cmp);
        measurements[measurements.len() / 2]
    }

    fn cpu_median(mode: ExecutionMode) -> (f64, f64) {
        let results: Vec<(f64, f64)> = (0..REPEATS).map(|_| run_cpu(mode)).collect();
        (
            median(results.iter().map(|(elapsed, _)| *elapsed).collect()),
            results
                .iter()
                .map(|(_, best)| *best)
                .fold(f64::INFINITY, f64::min),
        )
    }

    println!(
        "Native Puggles quadratic benchmark (population={POPULATION}, NFE={NFE}, median of {REPEATS} runs)"
    );
    println!("Available CPU threads: {}", std::thread::available_parallelism().map_or(1, |n| n.get()));

    let (sequential_ms, sequential_best) = cpu_median(ExecutionMode::Sequential);
    println!("Rust sequential CPU:    {sequential_ms:.3} ms | best: {sequential_best:.3e}");

    let (threaded_ms, threaded_best) = cpu_median(ExecutionMode::MultiThreaded);
    println!(
        "Rust multithreaded CPU: {threaded_ms:.3} ms | best: {threaded_best:.3e} | speedup: {:.2}x",
        sequential_ms / threaded_ms
    );

    // Device/pipeline initialization is excluded from the timed GPU run.
    match std::panic::catch_unwind(|| GpuEvaluator::new_blocking(shader(), 2, 1)) {
        Ok(first_evaluator) => {
            let mut gpu_results = vec![run_gpu(first_evaluator)];
            for _ in 1..REPEATS {
                gpu_results.push(run_gpu(GpuEvaluator::new_blocking(shader(), 2, 1)));
            }
            let gpu_ms = median(gpu_results.iter().map(|(elapsed, _)| *elapsed).collect());
            let gpu_best = gpu_results
                .iter()
                .map(|(_, best)| *best)
                .fold(f64::INFINITY, f64::min);
            println!(
                "Rust GPU:               {gpu_ms:.3} ms | best: {gpu_best:.3e} | speedup vs threaded CPU: {:.2}x",
                threaded_ms / gpu_ms
            );
        }
        Err(_) => println!("Rust GPU: unavailable on this system"),
    }
}
