# puggles: Rust-CUDA, cloud machines and faster evaluation, November 2026

One page on what changed in puggles, what the benchmarks on three machines show, what went wrong along the way, and what to do next. Full tables are in [`benchmarks_datasets.md`](benchmarks_datasets.md); the per-change detail is in [`performance_changes.md`](performance_changes.md). Raw data: `results_datasets_m3.json`, `results_datasets_gcp_c3.json`, `results_datasets_gcp_g2.json`.

## Summary

- **puggles is faster than pymoo on every machine at every size: 3.6–23×.** Against platypus and DEAP it is 6–79× faster. The gap is smallest on a CPU-only machine with a big dataset: at 1 GB on the CPU-only cloud machine, puggles' best path (Python + JAX) is only 1.2× faster than pymoo with a Numba objective.
- **Rust-CUDA works.** A GPU kernel written in Rust and compiled with Rust-CUDA runs on an NVIDIA L4, computes in f64, matches the CPU results exactly, and is 4.7–12.5× faster than pymoo (2.6 s vs 32.7 s at 1 GB). It is **not** the fastest GPU path: PyTorch on the same GPU (f32) takes 1.5 s at 1 GB.
- **The big win is reading the data once per generation, not once per solution.** The new GPU map-reduce kernel cut the 1 GB run on the Apple M3 from 6.6 s to 1.5 s.
- **New CPU features:** a row-scan evaluator (1.1–1.9× faster than 8 threads at 1 GB) and `ExecutionMode::Auto`, which times the first generation and picks the path.
- **Solution quality is about the same everywhere.** At 1 GB, hypervolume is within about 5% across libraries and Sharpe within 1% (genevo about 2% lower), with no consistent winner.
- **Several results were wrong at first and were caught:** a CUDA crash at exit, GPU rows that were really CPU emulation, and memory under-reported on macOS. See "Problems found".

## What changed in puggles

| change | why it is faster | measured |
| --- | --- | --- |
| **GPU map-reduce mode** (`GpuEvaluator::with_partials`) | Workgroups own blocks of rows and share each row across all solutions, so a generation reads the dataset once (about 0.9 GB) instead of once per solution (about 90 GB) | M3, 1 GB: 6.6 s → 1.5 s |
| **`BatchEvaluator` trait** + `with_batch_evaluator` | Any device backend (CUDA, CubeCL) plugs in without touching the GA | enabled the CubeCL and CUDA rows |
| **Adapter limits** requested | wgpu's default 128 MiB buffer limit blocked large datasets | enabled 1 GB |
| **`batch_numpy=True`** | One array copy per generation instead of building P·n Python floats | about 1.3× less overhead on small data |
| **`objective_cfunc`** (Numba `@cfunc`) | Called from Rust without the GIL, so all cores run a Python-written objective | M3, 1 GB: 19.6 s vs 99 s for a plain callback; only 1.5× on the Xeons |
| **`RowScanEvaluator`** (CPU map-reduce) | Same idea as the GPU kernel, with Rayon over row blocks | 1 GB: 14.4 vs 27.8 s (M3), 17.6 vs 26.3 s (c3), 34.8 vs 38.1 s (L4), each against 8 threads |
| **`ExecutionMode::Auto`** | Times the first generation and picks sequential, threads or the device | 1 GB: it ran like the GPU path on the M3 (2.7 s vs 1.7 s) and like plain 8 threads on both VMs (within 5%) |
| **`puggles-cuda`** (separate crate) | f64 map-reduce kernel written in Rust (Rust-CUDA) | L4, 1 GB: 2.6 s |
| **Software-renderer detection** | puggles now warns when its "GPU" is a CPU emulator, instead of timing it silently | correctness fix |

## How it was measured

- **Problem:** portfolio optimization on real daily returns of 20 US stocks (1986–2026), block-bootstrapped up to 1 GB (5.6M rows). Every evaluation scans every row. Multi-objective: minimize variance, maximize mean return. Single-objective: maximize Sharpe.
- **Fairness:** same population (100) and same evaluation budget for every library (10,000 × 3 runs up to 10 MB; 2,000 × 1 run at 100 MB and 1 GB), one process per run. The datasets were byte-identical on all machines (checked by checksum).
- **Machines:** Apple M3 (8 cores, 16 GB, Metal GPU); GCP `c3-standard-8` (Sapphire Rapids, 8 vCPU, no GPU); GCP `g2-standard-8` Spot (Intel Xeon 2.2 GHz, 8 vCPU, NVIDIA L4 23 GB, driver 580.178.04). Both VMs ran Ubuntu 24.04 and were deleted after the runs.

## Results

### Rust-CUDA on the L4 (multi-objective opt time, ↓ lower is better)

| size | Rust-CUDA (f64) | PyTorch CUDA (f32) | puggles 8 threads | pymoo | pymoo ÷ CUDA |
| --- | --- | --- | --- | --- | --- |
| 1 KB | 216 ms | 571 ms | **61 ms** | 1,209 ms | 5.6× |
| 1 MB | 310 ms | 445 ms | **225 ms** | 1,468 ms | 4.7× |
| 10 MB | **396 ms** | 427 ms | 1,657 ms | 3,116 ms | 7.9× |
| 100 MB | 407 ms | **423 ms** | 3,748 ms | 3,663 ms | 9.0× |
| 1 GB | 2.6 s | **1.5 s** | 38.1 s | 32.7 s | 12.4× |

Single-objective is nearly identical (2.6 s at 1 GB, 12.5× faster than pymoo). The CUDA path has about 200 ms of fixed start-up (context, shader load, data upload), so it loses to 8 CPU threads below about 1 MB. The best single-objective Sharpe at 1 GB is 0.0680 for CUDA, for the 8-thread CPU path, and for pymoo (0.06796). The L4 is weak at f64, which likely explains why f32 PyTorch wins at 1 GB; that is an untested guess.

### puggles' fastest path against the other libraries, per machine (opt time, multi-objective)

| machine | 1 KB: vs pymoo / platypus / DEAP | 1 GB: fastest puggles path | 1 GB: vs pymoo / pymoo + Numba / platypus / DEAP |
| --- | --- | --- | --- |
| Apple M3 | 19× / 54× / 45× | GPU, 1.7 s | 7.8× / 4.0× / 55× / 56× |
| c3-standard-8 (CPU) | 19× / 50× / 39× | Python + JAX, 6.9 s | 3.6× / 1.2× / 8.3× / 8.3× |
| g2 + L4 | 23× / 52× / 44× | Python + PyTorch CUDA, 1.5 s | 21× / 7.1× / 46× / 46× |

The same on one line for the Rust libraries (single-objective, 1 GB): genevo and genetic_algorithm take 41–224 s depending on the machine, about the same as sequential puggles (42–219 s), and 20–149× slower than puggles' best path on the same machine. genevo's solution quality is slightly lower (Sharpe 0.0662–0.0667 vs 0.0680); genetic_algorithm's matches (0.0677).

### The same path on each machine (1 GB, multi-objective, opt time)

| path | Apple M3 | c3 CPU | g2 + L4 |
| --- | --- | --- | --- |
| puggles 8 threads | 27.8 s | 26.3 s | 38.1 s |
| puggles CPU row scan | 14.4 s | 17.6 s | 34.8 s |
| puggles Python + Numba batch | 6.8 s | 8.1 s | 10.0 s |
| puggles Python + JAX | 4.7 s | 6.9 s | 9.1 s |
| puggles Python + PyTorch | 5.1 s (Apple GPU) | 24.1 s (CPU) | 1.5 s (CUDA) |
| puggles native GPU (wgpu) | 1.7 s | no GPU | no GPU (see below) |
| puggles Rust-CUDA | not built | no GPU | 2.6 s |
| pymoo | 13.5 s | 25.0 s | 32.7 s |
| platypus / DEAP | 96 s / 97 s | 57 s / 57 s | 71 s / 71 s |

Peak memory: native puggles uses 5–12 MB at 1 KB, against 57–153 MB for pymoo and DEAP. At 1 GB every library needs 2.6–3.3 GB because the dataset dominates; Rust-CUDA peaks at 1.95 GB.

## Problems found and fixed

| problem | cause | fix |
| --- | --- | --- |
| `puggles-cuda` would not build | Rust-CUDA's compiler backend feature was not enabled, and a dependency (`glam`) resolved to a version `cuda_std` cannot compile with | enabled the feature; pinned `glam` 0.30.9 in a committed lock file |
| CUDA runs "failed" even though results were right | The CUDA context was dropped before the buffers living in it, so every process crashed (SIGSEGV) at exit | declare the context field last |
| wgpu GPU, CubeCL and Auto rows on the L4 | wgpu picked `llvmpipe`, a CPU software renderer. The VM's NVIDIA driver loads but will not create a Vulkan instance, even with the matching Vulkan package installed | removed those rows; puggles now warns and the benchmark records "no GPU" |
| macOS memory numbers too low | Max RSS leaves out compressed and swapped pages, and the Mac was swapping | switched to peak memory footprint |
| Datasets differed between machines | Prices had been saved at 4 decimals | re-downloaded, saved at full precision; checksums now match on all machines |
| Spot L4 preempted twice | Normal for Spot | the driver resumes from its saved results |

## Caveats

- **The 100 MB and 1 GB rows are single runs.** Treat differences under about 1.5× as noise. The M3 in particular varies by up to 50% at 1 GB (sequential: 64 s multi vs 42 s single).
- **The machines differ in CPU, not just GPU.** The L4 VM's 2.2 GHz Xeon is slower than the c3's Sapphire Rapids, so its CPU rows are slower.
- **Hypervolume is normalized per result file.** Compare it within a machine, not across machines.
- **The CUDA worker uses a simple one-thread CSV parser,** so its CPU-seconds are mostly loading. Compare opt time.
- **Auto only knows sequential, threads and the wgpu device.** It cannot choose CUDA or row scan unless they are attached.
- **wgpu on NVIDIA Vulkan hardware is unverified.** wgpu is verified on the M3 only.
- **The row-scan evaluator is slower than hoped** (I expected 3–7 s at 1 GB; it takes 14–35 s). Its per-row callback prevents the loop layout that makes the Numba and JAX kernels fast.

## Next steps

1. **Make the CUDA kernel faster:** test an f32 mode to confirm the f64 explanation, then tune tiling. The target is PyTorch's 1.5 s.
2. **Get a GPU through Vulkan on NVIDIA,** or use CubeCL's CUDA backend, so the wgpu and CubeCL paths can be measured on the L4.
3. **Speed up the CPU row scan** with a dedicated dot-product kernel, aiming at Numba and JAX speeds.
4. **Let Auto consider more:** the row-scan and CUDA evaluators, and a re-probe if the first generation was unrepresentative.
5. **More seeds at 100 MB and 1 GB** to settle the quality comparison.
6. **Open the pull request.** The branch `fast-eval-paths` is committed locally and has not been pushed.

## Files

- Code: [`src/gpu_evaluator.rs`](../src/gpu_evaluator.rs), [`src/row_scan.rs`](../src/row_scan.rs), [`src/core.rs`](../src/core.rs), [`python/src/py_problem.rs`](../python/src/py_problem.rs), [`puggles-cuda/`](../puggles-cuda/).
- Benchmark drivers: [`bench_datasets.py`](bench_datasets.py), [`examples/bench_datasets.rs`](examples/bench_datasets.rs), [`examples/bench_cubecl.rs`](examples/bench_cubecl.rs); GCP setup scripts: [`scripts/gcp/`](scripts/gcp/).
- GCP cost: roughly $1–2 in total for both VMs (an estimate; check the billing console).
