//! CubeCL feasibility check: the dataset benchmark's map-reduce GPU kernel written as a Rust
//! `#[cube]` function (instead of hand-written WGSL), run through CubeCL's wgpu backend and
//! plugged into NSGA-II as a `BatchEvaluator`.
//!
//! Same CLI and output as `bench_datasets` so the driver can add it as a row:
//!   bench_cubecl <csv> <multi|single> puggles-cubecl <nfe> <seed>
//! Run: cargo run --release --features cubecl --example bench_cubecl -- data/real_returns_1mb.csv multi puggles-cubecl 2000 1

#[cfg(not(feature = "cubecl"))]
fn main() {
    eprintln!("build with --features cubecl");
}

#[cfg(feature = "cubecl")]
fn main() {
    imp::main()
}

#[cfg(feature = "cubecl")]
mod imp {
    use cubecl::prelude::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use polars::prelude::*;
    use puggles::core::{reduce_partials, BatchEvaluator, EvalFn, Problem};
    use puggles::gatypes::{Real, SolutionDataTypes};
    use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, OnceLock};
    use std::time::Instant;

    const N: usize = 20;
    const POP: usize = 100;
    const WG: usize = 128;
    const TILE: usize = 64;

    static DATA: OnceLock<Vec<f64>> = OnceLock::new();

    fn rows() -> usize {
        DATA.get().unwrap().len() / N
    }

    /// Workgroup ("cube") = block of rows, unit = solution; each tile of rows is shared through
    /// cube memory so the dataset is read once per generation. Writes Σr, Σr² per (block, solution).
    // ponytail: assumes Σx > 0 (the GA's x ∈ [0,1]^20 never sums to 0 in practice; the f64 re-check would catch it).
    #[cube(launch)]
    fn partials_kernel(sol: &Array<f32>, data: &Array<f32>, out: &mut Array<f32>, rows: usize, pop: usize) {
        let blocks = CUBE_COUNT as usize;
        let b = CUBE_POS as usize;
        let per = (rows + blocks - 1) / blocks;
        let start = b * per;
        let lo = if start < rows { start } else { rows };
        let hi = if lo + per < rows { lo + per } else { rows };
        let mut tile = SharedMemory::<f32>::new(TILE * N);
        let p = UNIT_POS as usize;
        let live = p < pop;
        let pc = if live { p } else { pop - 1 };
        // Weights in a local array with unrolled loops, so they can stay in registers.
        let mut w = Array::<f32>::new(N);
        let mut total = 0.0f32;
        #[unroll]
        for i in 0..N {
            w[i] = sol[pc * N + i];
            total += w[i];
        }
        #[unroll]
        for i in 0..N {
            w[i] /= total;
        }
        let mut s = 0.0f32;
        let mut s2 = 0.0f32;
        let mut t0 = lo;
        while t0 < hi {
            let mut j = p;
            while j < TILE * N {
                if t0 * N + j < hi * N {
                    tile[j] = data[t0 * N + j];
                }
                j += WG;
            }
            sync_cube();
            let mut ts = 0.0f32;
            let mut ts2 = 0.0f32;
            for r in 0..TILE {
                if t0 + r < hi {
                    let mut x = 0.0f32;
                    #[unroll]
                    for i in 0..N {
                        x += w[i] * tile[r * N + i];
                    }
                    ts += x;
                    ts2 += x * x;
                }
            }
            s += ts;
            s2 += ts2;
            sync_cube();
            t0 += TILE;
        }
        if live {
            out[(b * pop + p) * 2] = s;
            out[(b * pop + p) * 2 + 1] = s2;
        }
    }

    struct CubeEval {
        client: ComputeClient<WgpuRuntime>,
        data: cubecl::server::Handle,
        blocks: u32,
        multi: bool,
        count: AtomicUsize,
    }

    impl BatchEvaluator for CubeEval {
        fn evaluate_batch(&self, solutions: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let pop = solutions.len();
            self.count.fetch_add(pop, Ordering::Relaxed);
            let flat: Vec<f32> = solutions.iter().flat_map(|s| s.iter().map(|&v| v as f32)).collect();
            let sol = self.client.create_from_slice(f32::as_bytes(&flat));
            let len = self.blocks as usize * pop * 2;
            let out = self.client.empty(len * core::mem::size_of::<f32>());
            partials_kernel::launch(
                &self.client,
                CubeCount::Static(self.blocks, 1, 1),
                CubeDim::new_1d(WG as u32),
                unsafe { ArrayArg::from_raw_parts(sol, flat.len()) },
                unsafe { ArrayArg::from_raw_parts(self.data.clone(), rows() * N) },
                unsafe { ArrayArg::from_raw_parts(out.clone(), len) },
                rows(),
                pop,
            );
            let raw = self.client.read_one_unchecked(out);
            let finish = if self.multi { finish_multi } else { finish_single };
            reduce_partials(f32::from_bytes(&raw), self.blocks as usize, pop, 2, finish)
        }
        fn device_description(&self) -> String {
            "CubeCL wgpu (default device)".into()
        }
        fn evaluated_count(&self) -> usize {
            self.count.load(Ordering::Relaxed)
        }
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

    /// f64 stats on the CPU, for the final report (same definition as bench_datasets).
    fn port_stats(x: &[f64]) -> (f64, f64) {
        let total: f64 = x.iter().sum();
        let w: Vec<f64> = x.iter().map(|v| v / total).collect();
        let (mut s, mut s2) = (0.0, 0.0);
        for row in DATA.get().unwrap().chunks_exact(N) {
            let r: f64 = row.iter().zip(&w).map(|(a, b)| a * b).sum();
            s += r;
            s2 += r * r;
        }
        let m = s / rows() as f64;
        (m, (s2 / rows() as f64 - m * m).max(0.0))
    }

    fn unused_objective(_: &Vec<f64>) -> Vec<f64> {
        unreachable!("evaluation runs on the CubeCL evaluator")
    }

    fn load(path: &str) -> f64 {
        let t0 = Instant::now();
        let df = LazyCsvReader::new(path).with_has_header(true).finish().unwrap().collect().unwrap();
        let cols: Vec<Vec<f64>> = df
            .get_column_names()
            .iter()
            .filter(|c| **c != "date" && **c != "scenario")
            .map(|c| df.column(c).unwrap().f64().unwrap().into_no_null_iter().collect())
            .collect();
        drop(df);
        let t = cols[0].len();
        let mut data = Vec::with_capacity(t * N);
        for r in 0..t {
            data.extend(cols.iter().map(|c| c[r]));
        }
        DATA.set(data).unwrap();
        t0.elapsed().as_secs_f64() * 1000.0
    }

    pub fn main() {
        let a: Vec<String> = std::env::args().collect();
        let (path, multi) = (&a[1], a[2] == "multi");
        let (nfe, seed): (usize, u64) = (a[4].parse().unwrap(), a[5].parse().unwrap());
        let load_ms = load(path);

        let t0 = Instant::now();
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let data_f32: Vec<f32> = DATA.get().unwrap().iter().map(|&v| v as f32).collect();
        let eval = CubeEval {
            data: client.create_from_slice(f32::as_bytes(&data_f32)),
            client,
            blocks: rows().div_ceil(1024).clamp(1, 256) as u32,
            multi,
            count: AtomicUsize::new(0),
        };
        drop(data_f32);
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
}
