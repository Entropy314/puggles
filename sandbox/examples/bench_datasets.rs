//! Dataset-scaling worker: ONE (library, dataset, mode) run per process, so the driver
//! (`bench_datasets.py`) can measure each run's peak RSS and CPU time in isolation.
//!
//! Every evaluation scans EVERY row of the returns CSV (T rows × 20 assets), so the cost
//! of an evaluation grows with the dataset. w = x / Σx, r_t = Σ_i w_i·R_ti:
//!   multi  → minimize realized variance mean(r²) − mean(r)², maximize mean(r)
//!   single → maximize Sharpe mean(r) / std(r)
//!
//! Usage: bench_datasets <csv> <multi|single> <puggles-seq|puggles-par|puggles-gpu|genevo|ga> <nfe> <seed>
//! Output: RESULT\t<load_ms>\t<opt_ms>\t<evals>  then  FRONT\t<var>\t<mean>  lines (multi)
//!         or  BEST\t<sharpe>  (single). Final quality is always recomputed in f64 on the CPU.

use polars::prelude::*;
use puggles::core::{EvalFn, Problem};
use puggles::gatypes::{Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

const N: usize = 20; // assets
const POP: usize = 100;

/// Row-major returns matrix: DATA[t * N + i].
static DATA: OnceLock<Vec<f64>> = OnceLock::new();
/// Counts every real objective evaluation (the GPU path bypasses it; reported as NFE).
static EVALS: AtomicUsize = AtomicUsize::new(0);
/// Rows scanned per evaluation. Only lowered while calibrating genetic_algorithm (see `run_ga`).
static ROW_LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);

fn rows() -> usize {
    (DATA.get().unwrap().len() / N).min(ROW_LIMIT.load(Ordering::Relaxed))
}

/// (mean, variance) of the normalized portfolio's per-row return, one pass over the data.
fn port_stats(x: &[f64]) -> (f64, f64) {
    EVALS.fetch_add(1, Ordering::Relaxed);
    let total: f64 = x.iter().sum();
    let w: Vec<f64> = if total > 1e-12 { x.iter().map(|v| v / total).collect() } else { vec![1.0 / N as f64; N] };
    let t = rows();
    let (mut s, mut s2) = (0.0, 0.0);
    for row in DATA.get().unwrap()[..t * N].chunks_exact(N) {
        let r: f64 = row.iter().zip(&w).map(|(a, b)| a * b).sum();
        s += r;
        s2 += r * r;
    }
    let mean = s / t as f64;
    (mean, (s2 / t as f64 - mean * mean).max(0.0))
}

fn sharpe(x: &[f64]) -> f64 {
    let (m, v) = port_stats(x);
    if v > 0.0 { m / v.sqrt() } else { 0.0 }
}

fn multi_obj(x: &Vec<f64>) -> Vec<f64> {
    let (m, v) = port_stats(x);
    vec![v, -m]
}

fn single_obj(x: &Vec<f64>) -> Vec<f64> {
    vec![-sharpe(x)]
}

fn load(path: &str) -> f64 {
    let t0 = Instant::now();
    let df = LazyCsvReader::new(path).with_has_header(true).finish().unwrap().collect().unwrap();
    let cols: Vec<Vec<f64>> = df
        .get_column_names()
        .iter()
        .filter(|c| **c != "date" && **c != "scenario") // real_returns_*.csv also has `scenario`
        .map(|c| df.column(c).unwrap().f64().unwrap().into_no_null_iter().collect())
        .collect();
    drop(df);
    assert_eq!(cols.len(), N);
    let t = cols[0].len();
    let mut data = Vec::with_capacity(t * N);
    for r in 0..t {
        data.extend(cols.iter().map(|c| c[r]));
    }
    DATA.set(data).unwrap();
    t0.elapsed().as_secs_f64() * 1000.0
}

/// Map-reduce kernel for `GpuEvaluator::with_partials`: workgroup `b` owns a block of rows and
/// thread `p` owns a solution. Each tile of TILE rows is loaded once into workgroup memory and
/// shared by every solution, so a generation reads the dataset once, not once per solution.
/// Writes Σr and Σr² per (block, solution); the CPU sums blocks in f64 (`finish_*`). Sums go
/// tile → thread total, because one flat f32 running sum over many rows loses precision.
fn shader(t: usize) -> String {
    assert_eq!(N % 4, 0, "the kernel packs each row into N/4 vec4s");
    let q = N / 4;
    // Weights live in q named vec4 registers and the dot product is unrolled: a loop-indexed
    // per-thread array spills out of registers and ran ~5× slower.
    let load_w: String = (0..q)
        .map(|k| format!("var w{k} = vec4<f32>(solutions[b+{a}u], solutions[b+{b1}u], solutions[b+{c}u], solutions[b+{d}u]);\n",
                         a = 4 * k, b1 = 4 * k + 1, c = 4 * k + 2, d = 4 * k + 3))
        .collect();
    let total: String = (0..q).map(|k| format!("dot(w{k}, vec4<f32>(1.0))")).collect::<Vec<_>>().join(" + ");
    let norm: String = (0..q).map(|k| format!("w{k} = select(vec4<f32>(1.0 / {N}.0), w{k} / total, total > 1e-12);\n")).collect();
    let dot: String = (0..q).map(|k| format!("dot(w{k}, tile[r * {q}u + {k}u])")).collect::<Vec<_>>().join(" + ");
    format!(
        r#"
@group(0) @binding(0) var<storage, read> solutions: array<f32>;
@group(0) @binding(1) var<storage, read_write> objectives: array<f32>;
struct Params {{ solution_length: u32, num_objectives: u32, pop_size: u32 }};
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> data: array<vec4<f32>>;

const T: u32 = {t}u;
const WG: u32 = 128u;
const TILE: u32 = 64u;
var<workgroup> tile: array<vec4<f32>, {tile_len}>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {{
    let pop = params.pop_size;
    let per = (T + nwg.x - 1u) / nwg.x;
    let lo = min(wid.x * per, T);
    let hi = min(lo + per, T);
    // Loops below depend only on uniform values (pop, lo, hi), so every barrier is uniform.
    for (var pg: u32 = 0u; pg < pop; pg = pg + WG) {{
        let p = pg + lid;
        let live = p < pop;
        let b = min(p, pop - 1u) * {N}u;
        {load_w}
        let total = {total};
        {norm}
        var s: f32 = 0.0;
        var s2: f32 = 0.0;
        for (var t0: u32 = lo; t0 < hi; t0 = t0 + TILE) {{
            let rows = min(TILE, hi - t0);
            for (var j: u32 = lid; j < rows * {q}u; j = j + WG) {{ tile[j] = data[t0 * {q}u + j]; }}
            workgroupBarrier();
            var ts: f32 = 0.0;
            var ts2: f32 = 0.0;
            for (var r: u32 = 0u; r < rows; r = r + 1u) {{
                let x = {dot};
                ts = ts + x;
                ts2 = ts2 + x * x;
            }}
            s = s + ts;
            s2 = s2 + ts2;
            workgroupBarrier();
        }}
        if (live) {{
            let o = (wid.x * pop + p) * 2u;
            objectives[o] = s;
            objectives[o + 1u] = s2;
        }}
    }}
}}
"#,
        tile_len = 64 * q
    )
}

/// Turn summed (Σr, Σr²) into the objectives, in f64.
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

fn problem(multi: bool) -> Arc<Problem> {
    let n_obj = if multi { 2 } else { 1 };
    Arc::new(Problem {
        solution_length: N,
        number_of_objectives: n_obj,
        objective_constraint: None,
        objective_constraint_operands: None,
        direction: Some(vec![-1; n_obj]),
        solution_data_types: (0..N).map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))).collect(),
        variable_constraints: None,
        encoding: puggles::core::Encoding::PerGene,
        eval_fn: EvalFn::Single(if multi { multi_obj } else { single_obj }),
    })
}

/// Returns the final genomes worth reporting (archive for multi, best for single).
fn run_puggles(mode: &str, multi: bool, nfe: usize, seed: u64) -> Vec<Vec<f64>> {
    let exec = match mode {
        "puggles-par" => ExecutionMode::MultiThreaded,
        "puggles-gpu" => ExecutionMode::GPU,
        _ => ExecutionMode::Sequential,
    };
    let mut ga = NSGAII::new(problem(multi), POP, exec).with_seed(seed);
    #[cfg(feature = "gpu")]
    if mode == "puggles-gpu" {
        let data: Vec<f32> = DATA.get().unwrap().iter().map(|&v| v as f32).collect();
        // ~1k rows per block, capped so huge files still give a few hundred workgroups.
        let blocks = rows().div_ceil(1024).clamp(1, 256) as u32;
        let ev = puggles::gpu_evaluator::GpuEvaluator::new_blocking_with_data(
            &shader(rows()), N, if multi { 2 } else { 1 }, &data)
            .with_partials(blocks, 2, if multi { finish_multi } else { finish_single });
        ga = ga.with_gpu_evaluator(ev);
    }
    #[cfg(not(feature = "gpu"))]
    if mode == "puggles-gpu" {
        let _ = (shader, finish_multi, finish_single);
        panic!("build with --features gpu for puggles-gpu");
    }
    ga.run(nfe);
    let archive = ga.get_archive();
    if multi {
        archive.iter().map(|s| s.solution.clone()).collect()
    } else {
        let best = archive.iter().chain(ga.population.iter())
            .min_by(|a, b| a.objective_fitness_values[0].total_cmp(&b.objective_fitness_values[0]))
            .unwrap();
        vec![best.solution.clone()]
    }
}

/// genevo maximizes an i64 fitness → Sharpe scaled by 1e6. Stepped until NFE evaluations.
fn run_genevo(nfe: usize) -> Vec<Vec<f64>> {
    use genevo::operator::prelude::*;
    use genevo::population::{build_population, ValueEncodedGenomeBuilder};
    use genevo::prelude::*;

    #[derive(Clone, Debug)]
    struct SharpeFitness;
    impl FitnessFunction<Vec<f64>, i64> for SharpeFitness {
        fn fitness_of(&self, g: &Vec<f64>) -> i64 { (sharpe(g) * 1e6) as i64 }
        fn average(&self, v: &[i64]) -> i64 { v.iter().sum::<i64>() / v.len() as i64 }
        fn highest_possible_fitness(&self) -> i64 { 1_000_000_000 }
        fn lowest_possible_fitness(&self) -> i64 { -1_000_000_000 }
    }

    let initial: Population<Vec<f64>> = build_population()
        .with_genome_builder(ValueEncodedGenomeBuilder::new(N, 0.0, 1.0))
        .of_size(POP)
        .uniform_at_random();
    let mut sim = simulate(
        genetic_algorithm()
            .with_evaluation(SharpeFitness)
            .with_selection(MaximizeSelector::new(0.7, 2))
            .with_crossover(MultiPointCrossBreeder::new(3))
            .with_mutation(RandomValueMutator::new(0.1, 0.0, 1.0))
            .with_reinsertion(ElitistReinserter::new(SharpeFitness, true, 0.7))
            .with_initial_population(initial)
            .build(),
    )
    .until(GenerationLimit::new(u64::MAX)) // the real stop is the NFE check below
    .build();

    let (mut best_fit, mut best) = (i64::MIN, vec![]);
    loop {
        match sim.step() {
            Ok(SimResult::Intermediate(step)) => {
                let b = &step.result.best_solution.solution;
                if b.fitness > best_fit { best_fit = b.fitness; best = b.genome.clone(); }
                if EVALS.load(Ordering::Relaxed) >= nfe { break; }
            }
            Ok(SimResult::Final(step, _, _, _)) => {
                let b = &step.result.best_solution.solution;
                if b.fitness > best_fit { best = b.genome.clone(); }
                break;
            }
            Err(_) => break,
        }
    }
    vec![best]
}

/// genetic_algorithm caches fitness, so its generation count is calibrated to spend ≈ NFE
/// evaluations. Calibration scans only 64 rows: evaluations per generation depend on the
/// operators, not on the dataset.
fn run_ga(nfe: usize, seed: u64) -> Vec<Vec<f64>> {
    use genetic_algorithm::genotype::MutationType;
    use genetic_algorithm::strategy::evolve::prelude::*;

    #[derive(Clone, Debug)]
    struct SharpeFitness;
    impl Fitness for SharpeFitness {
        type Genotype = RangeGenotype<f64>;
        fn calculate_for_chromosome(
            &mut self,
            c: &FitnessChromosome<Self>,
            _g: &FitnessGenotype<Self>,
        ) -> Option<FitnessValue> {
            Some((sharpe(&c.genes) * 1e6) as FitnessValue)
        }
    }

    let run_gens = |gens: usize| -> Vec<f64> {
        let genotype = RangeGenotype::builder()
            .with_genes_size(N)
            .with_allele_range(0.0..=1.0)
            .with_mutation_type(MutationType::Range(0.1))
            .build()
            .unwrap();
        let evolve = Evolve::builder()
            .with_genotype(genotype)
            .with_fitness(SharpeFitness)
            .with_fitness_ordering(FitnessOrdering::Maximize)
            .with_target_population_size(POP)
            .with_max_generations(gens)
            .with_select(SelectTournament::new(0.5, 0.02, 4))
            .with_crossover(CrossoverUniform::new(0.7, 0.8))
            .with_mutate(MutateSingleGene::new(0.2))
            .with_rng_seed_from_u64(seed)
            .call()
            .unwrap();
        evolve.best_genes_and_fitness_score().unwrap().0
    };

    const PROBE_GENS: usize = 50;
    ROW_LIMIT.store(64, Ordering::Relaxed);
    run_gens(PROBE_GENS);
    let per_gen = (EVALS.swap(0, Ordering::Relaxed) as f64 / PROBE_GENS as f64).max(1.0);
    ROW_LIMIT.store(usize::MAX, Ordering::Relaxed);
    vec![run_gens(((nfe as f64 / per_gen).round() as usize).max(1))]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, multi, lib) = (&a[1], a[2] == "multi", a[3].as_str());
    let (nfe, seed): (usize, u64) = (a[4].parse().unwrap(), a[5].parse().unwrap());

    let load_ms = load(path);
    let t0 = Instant::now();
    let genomes = match lib {
        "genevo" => run_genevo(nfe),
        "ga" => run_ga(nfe, seed),
        _ => run_puggles(lib, multi, nfe, seed),
    };
    let opt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let evals = if lib == "puggles-gpu" { nfe } else { EVALS.load(Ordering::Relaxed) };

    println!("RESULT\t{load_ms:.1}\t{opt_ms:.1}\t{evals}");
    for g in &genomes {
        if multi {
            let (m, v) = port_stats(g);
            println!("FRONT\t{v:e}\t{m:e}");
        } else {
            println!("BEST\t{:.6}", sharpe(g));
        }
    }
}
