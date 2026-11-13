# puggles Dataset-Scaling Benchmark: Real Stock Data, 1 KB → 1 GB

Single- and multi-objective portfolio optimization on real daily stock returns at five dataset sizes. puggles runs in native Rust (sequential, 8 threads, wgpu GPU, CubeCL GPU) and from Python (plain, NumPy batch, Numba batch, Numba cfunc, PyTorch MPS, JAX). It is compared with **pymoo, platypus and DEAP** (Python) and **genevo and genetic_algorithm** (Rust).

**Run date:** 2026-11-06, faster evaluation paths added 2026-11-13 · **Raw results:** [`results_datasets.json`](results_datasets.json) · **Drivers:** [`bench_datasets.py`](bench_datasets.py), [`examples/bench_datasets.rs`](examples/bench_datasets.rs), [`scripts/build_real_datasets.py`](scripts/build_real_datasets.py) · An earlier run on synthetic data: [`benchmarks_datasets_synthetic.md`](benchmarks_datasets_synthetic.md) · What changed and why: [`performance_changes.md`](performance_changes.md) · Three-machine summary: [`report_cuda_and_cloud.md`](report_cuda_and_cloud.md)

## Hardware and software

| | |
| --- | --- |
| Machine | Apple M3: 8 CPU cores (4 performance + 4 efficiency), 10-core integrated GPU (Metal), 16 GB unified memory |
| OS | macOS 27.2 (26B123) |
| Rust | rustc 1.94.0, release profile with `lto = true`, `codegen-units = 1` · wgpu 22.1 · CubeCL 0.10 |
| Python | 3.14.7 · numpy 2.5.4 · polars 2.1.0 · numba 0.68.1 · torch 2.15.0 · jax 0.12.0.1 |
| Libraries | puggles 0.3.0 · pymoo 0.6.2 · platypus-opt 1.4.1 · DEAP 1.4.4 · genevo 0.7.1 · genetic_algorithm 0.27.2 |

## How to read the metrics

| metric | better | meaning |
| --- | --- | --- |
| load ms | ↓ lower | Read the CSV into memory (polars in both Rust and Python) |
| opt ms | ↓ lower | Optimization wall time, the headline number (± = std over 3 runs) |
| µs/eval | ↓ lower | opt time ÷ objective evaluations |
| evals/s | ↑ higher | Evaluation throughput |
| CPU s | ↓ lower | user + sys CPU time of the whole process (load + any JIT compile + optimize). Above wall time = several cores in use |
| peak memory MB | ↓ lower | Peak memory footprint of the whole process, dataset included. Counts compressed and swapped pages, so memory pressure doesn't hide usage |
| HV (multi) | ↑ higher | Hypervolume of the final Pareto front, 0–1. Normalized across all libraries **at the same size**, so compare within a size, not across sizes |
| front size (multi) | ↑ higher | Points on the final front (informational; capped at the population of 100) |
| best Sharpe (single) | ↑ higher | Best Sharpe ratio found (daily, not annualized), recomputed in f64 |
| vs pymoo | ↑ higher | pymoo opt time ÷ this library's opt time. >1× = faster than pymoo |

## The data

**Source.** Real split- and dividend-adjusted daily closes from Yahoo Finance's public chart endpoint, for 20 US large caps with long histories: AAPL, MSFT, IBM, KO, PG, JNJ, XOM, CVX, WMT, PFE, MRK, INTC, ORCL, JPM, BAC, GE, MCD, DIS, HD and PEP. The series are inner-joined on date, giving **10,242 common trading days, 1986-03-14 → 2026-11-06**. That table is saved as `data/real_prices.csv`, and polars `pct_change` turns it into daily returns.

**Growing it to 1 GB.** 40 years of real data for 20 stocks is about 2 MB. Larger files start with the full real history (`scenario` 0), then add **block-bootstrap scenarios**. Each scenario replays the real calendar using randomly drawn **20-day blocks of whole real trading days**. That keeps the real correlations between stocks, the fat tails and short-term volatility clustering. Schema: `date,scenario,AAPL,…,PEP`.

| size | file bytes | rows (T) | content |
| --- | --- | --- | --- |
| 1kb | 1,121 | 5 | last 5 real trading days |
| 1mb | 1,114,082 | 5,500 | last 5,500 real trading days (2004 → 2026) |
| 10mb | 11,135,105 | 55,000 | real history + 5 bootstrap scenarios |
| 100mb | 111,590,346 | 549,000 | real history + 53 scenarios |
| 1gb | 1,143,757,780 | 5,600,000 | real history + 546 scenarios |

**How realistic it is.** Compared with the earlier synthetic file, the real data has much fatter tails and stronger correlation between stocks. The bootstrap preserves both:

| statistic (10mb file) | real history | bootstrap part | old synthetic |
| --- | --- | --- | --- |
| mean annualized volatility | 30.3% | 30.1% | 32.1% |
| mean pairwise correlation | 0.345 | 0.347 | 0.198 |
| kurtosis (normal = 3) | 17.4 | 14.3 | 3.0 |

The returns include real extremes such as Black Monday (1987) and AAPL's −52% day in September 2000.

## Problem and method

**Objective.** Every evaluation scans every row, so the cost of one evaluation grows linearly with the dataset. Weights are `w = x / Σx` with `x ∈ [0,1]^20`, and the portfolio return for row t is `r_t = Σ_i w_i·R_ti`.
- **Multi-objective:** minimize the realized variance `mean(r²) − mean(r)²` and maximize the mean return `mean(r)`.
- **Single-objective:** maximize Sharpe = `mean(r) / std(r)`.

**Fairness.**
- Every library gets the same problem, population (100) and number of objective evaluations (NFE), counted by a shared counter inside each objective. All libraries land within ±3% of the budget. genevo overshoots by 1% because it stops only at generation boundaries. genetic_algorithm is calibrated to about 2.5% under, because it caches fitness and its generation count has to be estimated.
- **Budget:** 10,000 evaluations × 3 seeded runs for 1kb–10mb. 2,000 evaluations × 1 run for 100mb and 1gb. The large sizes are single runs, so treat differences under about 10% as noise.
- **Isolation:** every run is its own process under `/usr/bin/time -l`. Library imports and JIT compiles (Numba about 2 s, JAX about 0.3 s) happen before the clock starts.
- **genevo and genetic_algorithm support only one objective**, so they appear only in the single-objective results.

**How each library evaluates:**

| library | evaluation style |
| --- | --- |
| puggles native, sequential and 8 threads | Rust objective, one solution at a time (8 threads via Rayon) |
| puggles native, GPU | WGSL compute shader, map-reduce mode (`GpuEvaluator::with_partials`): workgroup = block of rows, thread = solution, rows staged in workgroup memory. The dataset is read once per generation. f32 on the GPU; partial sums are combined in f64 on the CPU |
| puggles native, CubeCL GPU | The same kernel written in Rust with CubeCL 0.10 (`#[cube]`), run on its wgpu/Metal backend and plugged in through `with_batch_evaluator` |
| puggles native, CUDA (Rust-CUDA) | The same kernel in Rust via Rust-CUDA, in f64 (`puggles-cuda/`). **Not built here**: it needs Linux/Windows and an NVIDIA GPU |
| puggles (py) | Python objective called once per solution (`R @ w` in NumPy) |
| puggles (py, batch) | `batch_objective_function` with `batch_numpy=True`: one NumPy call per generation, with the population passed and returned as arrays |
| puggles (py, numba batch) | The same batch interface, with the objective compiled by Numba. Threads split the rows, and each row updates all 100 solutions while it's in cache |
| puggles (py, numba cfunc, 8 threads) | `objective_cfunc`: a Numba `@cfunc` per-solution objective that puggles calls from Rust with no GIL, on all cores |
| puggles (py, torch MPS batch) | Batch objective with the dataset as an f32 PyTorch tensor on the Apple GPU (MPS) |
| puggles (py, jax jit+vmap) | A single-solution JAX function turned into a batch objective with `jax.jit(jax.vmap(f))`, run on the CPU (rows scanned in blocks) |
| pymoo | Vectorized `_evaluate`: one NumPy call per generation (the same code as puggles batch) |
| pymoo (numba) | pymoo with the same Numba objective as puggles (numba batch) |
| platypus, DEAP | Python objective called once per solution (`R @ w` in NumPy) |
| genevo, genetic_algorithm | Rust objective, one solution at a time |

## puggles vs every other library

This is the fastest puggles mode at each size against every other library, using the same problem, the same evaluation budget and the same machine. For what changed in puggles to get here, and why it's faster, see [`performance_changes.md`](performance_changes.md).

#### Multi-objective: speed (opt time; × = how much slower than the fastest puggles mode)

| size | fastest puggles mode | pymoo | pymoo (numba) | platypus | DEAP |
| --- | --- | --- | --- | --- | --- |
| 1kb | puggles (native, sequential): **19 ms** | 344 ms (17.9×) | 305 ms (15.9×) | 1,086 ms (56.6×) | 855 ms (44.5×) |
| 1mb | puggles (py, jax jit+vmap): **48 ms** | 384 ms (7.9×) | 349 ms (7.2×) | 1,492 ms (30.8×) | 1,225 ms (25.3×) |
| 10mb | puggles (native, GPU): **202 ms** | 1,035 ms (5.1×) | 680 ms (3.4×) | 4,728 ms (23.4×) | 4,342 ms (21.5×) |
| 100mb | puggles (native, CubeCL GPU): **146 ms** | 1,532 ms (10.5×) | 836 ms (5.7×) | 9,537 ms (65.5×) | 9,396 ms (64.5×) |
| 1gb | puggles (native, GPU): **1,524 ms** | 13.5 s (8.8×) | 7,794 ms (5.1×) | 94.7 s (62.1×) | 94.6 s (62.1×) |

#### Multi-objective: quality (HV ↑; puggles = best of its modes)

| size | puggles | pymoo | pymoo (numba) | platypus | DEAP |
| --- | --- | --- | --- | --- | --- |
| 1kb | 0.749 | 0.758 | 0.758 | 0.757 | 0.716 |
| 1mb | 0.644 | 0.623 | 0.623 | 0.683 | 0.617 |
| 10mb | 0.751 | 0.748 | 0.748 | 0.715 | 0.737 |
| 100mb | 0.726 | 0.708 | 0.708 | 0.656 | 0.694 |
| 1gb | 0.668 | 0.702 | 0.702 | 0.702 | 0.693 |

#### Single-objective: speed (opt time; × = how much slower than the fastest puggles mode)

| size | fastest puggles mode | pymoo | pymoo (numba) | platypus | DEAP | genevo | genetic_algorithm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | puggles (native, sequential): **9 ms** | 360 ms (41.6×) | 304 ms (35.1×) | 471 ms (54.3×) | 141 ms (16.2×) | 5 ms (0.6×) | 2 ms (0.3×) |
| 1mb | puggles (py, jax jit+vmap): **38 ms** | 385 ms (10.2×) | 353 ms (9.3×) | 858 ms (22.7×) | 526 ms (13.9×) | 229 ms (6.1×) | 207 ms (5.5×) |
| 10mb | puggles (native, GPU): **187 ms** | 1,044 ms (5.6×) | 684 ms (3.7×) | 3,852 ms (20.6×) | 3,555 ms (19.0×) | 1,442 ms (7.7×) | 1,974 ms (10.6×) |
| 100mb | puggles (native, GPU): **128 ms** | 1,454 ms (11.3×) | 861 ms (6.7×) | 9,335 ms (72.6×) | 9,144 ms (71.2×) | 3,169 ms (24.7×) | 4,099 ms (31.9×) |
| 1gb | puggles (native, GPU): **1,587 ms** | 13.5 s (8.5×) | 7,602 ms (4.8×) | 94.0 s (59.2×) | 93.9 s (59.2×) | 35.0 s (22.0×) | 44.1 s (27.8×) |

#### Single-objective: quality (best Sharpe ↑; puggles = best of its modes)

| size | puggles | pymoo | pymoo (numba) | platypus | DEAP | genevo | genetic_algorithm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | 0.5582 | 0.5572 | 0.5572 | 0.5573 | 0.5573 | 0.4752 | 0.5571 |
| 1mb | 0.0707 | 0.0707 | 0.0707 | 0.0707 | 0.0707 | 0.0680 | 0.0707 |
| 10mb | 0.0681 | 0.0681 | 0.0681 | 0.0681 | 0.0681 | 0.0673 | 0.0681 |
| 100mb | 0.0664 | 0.0665 | 0.0665 | 0.0649 | 0.0661 | 0.0648 | 0.0662 |
| 1gb | 0.0678 | 0.0678 | 0.0678 | 0.0672 | 0.0675 | 0.0668 | 0.0676 |

#### Memory: peak footprint (MB ↓), single-objective

| size | puggles (native, sequential) | puggles (native, GPU) | puggles (py, batch) | pymoo | platypus | DEAP | genevo | genetic_algorithm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | 5 | 50 | 53 | 78 | 53 | 56 | 5 | 5 |
| 1mb | 10 | 56 | 61 | 87 | 58 | 61 | 10 | 10 |
| 10mb | 35 | 98 | 116 | 141 | 81 | 81 | 36 | 35 |
| 100mb | 312 | 525 | 372 | 397 | 337 | 336 | 312 | 312 |
| 1gb | 2,732 | 4,488 | 2,939 | 2,914 | 2,918 | 2,918 | 2,732 | 2,732 |

**In short:**
- **Speed (multi-objective):** puggles is fastest at every size, by 3.4–17.9× over pymoo (with or without Numba) and 21–66× over platypus and DEAP.
- **Speed (single-objective):** the same holds, with one exception. At 1 KB, genevo (5 ms) and genetic_algorithm (2 ms) beat puggles (9 ms), because they're lighter single-objective loops and evaluation on 5 rows is nearly free. From 1 MB up, puggles is 5.5–32× faster than both.
- **Quality:** comparable across libraries, and no library wins every size. Multi-objective HV: pymoo is best at 1 KB, platypus at 1 MB, puggles at 10 MB and 100 MB, and pymoo and platypus at 1 GB (0.702 vs puggles' 0.668, a single run). Single-objective Sharpe agrees within about 1%, except genevo, which is lowest.
- **Memory:** native puggles, genevo and genetic_algorithm need the least (5–36 MB up to 10 MB). The Python libraries carry a 53–141 MB interpreter and NumPy floor. At 1 GB everything is dominated by the dataset (2.7–2.9 GB), and the GPU path costs about 1.8 GB more for its f32 device copy.

## Headline results

### 1 GB, multi-objective (2,000 evaluations)

| library | opt time ↓ | CPU s ↓ | peak memory ↓ | HV ↑ | vs pymoo ↑ |
| --- | --- | --- | --- | --- | --- |
| **puggles (native, GPU)** | **1.5 s** | 6.2 | 4,488 MB | 0.668 | **8.83×** |
| puggles (native, CubeCL GPU) | 1.7 s | 6.2 | 5,956 MB | 0.668 | 8.04× |
| puggles (py, torch MPS batch) | 4.8 s | 5.2 | 3,704 MB | 0.668 | 2.78× |
| puggles (py, jax jit+vmap) | 6.0 s | 25.3 | 3,025 MB | 0.668 | 2.26× |
| puggles (py, numba batch) | 7.3 s | 52.0 | 2,920 MB | 0.668 | 1.83× |
| pymoo (numba) | 7.8 s | 54.5 | 2,931 MB | 0.702 | 1.73× |
| pymoo | 13.5 s | 16.6 | 2,929 MB | 0.702 | 1.00× |
| puggles (py, batch) | 15.7 s | 17.8 | 2,923 MB | 0.668 | 0.86× |
| puggles (native, 8 threads) | 18.6 s | 130.4 | 2,733 MB | 0.668 | 0.72× |
| puggles (py, numba cfunc, 8 threads) | 21.3 s | 141.0 | 2,931 MB | 0.668 | 0.63× |
| puggles (native, sequential) | 41.4 s | 46.9 | 2,732 MB | 0.668 | 0.33× |
| puggles (py) | 93.2 s | 96.2 | 2,936 MB | 0.668 | 0.14× |
| DEAP | 94.6 s | 97.2 | 2,929 MB | 0.693 | 0.14× |
| platypus | 94.7 s | 97.2 | 2,937 MB | 0.702 | 0.14× |

### 1 GB, single-objective (2,000 evaluations)

| library | opt time ↓ | CPU s ↓ | peak memory ↓ | best Sharpe ↑ | vs pymoo ↑ |
| --- | --- | --- | --- | --- | --- |
| **puggles (native, GPU)** | **1.6 s** | 4.3 | 4,488 MB | 0.0678 | **8.52×** |
| puggles (native, CubeCL GPU) | 1.9 s | 4.3 | 5,615 MB | 0.0678 | 7.10× |
| puggles (py, jax jit+vmap) | 4.3 s | 21.4 | 3,028 MB | 0.0678 | 3.17× |
| puggles (py, torch MPS batch) | 4.7 s | 4.8 | 3,732 MB | 0.0678 | 2.89× |
| puggles (py, numba batch) | 7.1 s | 50.8 | 2,923 MB | 0.0678 | 1.91× |
| pymoo (numba) | 7.6 s | 54.4 | 2,921 MB | 0.0678 | 1.78× |
| pymoo | 13.5 s | 16.8 | 2,914 MB | 0.0678 | 1.00× |
| puggles (py, batch) | 14.0 s | 17.1 | 2,939 MB | 0.0678 | 0.97× |
| puggles (py, numba cfunc, 8 threads) | 18.8 s | 134.9 | 2,937 MB | 0.0678 | 0.72× |
| puggles (native, 8 threads) | 18.9 s | 127.5 | 2,732 MB | 0.0678 | 0.72× |
| genevo | 35.0 s | 61.9 | 2,732 MB | 0.0668 | 0.39× |
| puggles (native, sequential) | 42.0 s | 45.1 | 2,732 MB | 0.0678 | 0.32× |
| genetic_algorithm | 44.1 s | 47.0 | 2,732 MB | 0.0676 | 0.31× |
| DEAP | 93.9 s | 96.7 | 2,918 MB | 0.0675 | 0.14× |
| platypus | 94.0 s | 96.8 | 2,918 MB | 0.0672 | 0.14× |
| puggles (py) | 97.9 s | 98.5 | 2,929 MB | 0.0678 | 0.14× |

### Fastest optimizer at each size (opt time, multi-objective)

| size | 1st | 2nd | 3rd | pymoo | pymoo (numba) | platypus | DEAP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | puggles (native, sequential) **19 ms** | puggles (py, batch) 22 ms | puggles (py, jax jit+vmap) 24 ms | 344 ms (17.9× slower) | 305 ms (15.9× slower) | 1,086 ms (56.6× slower) | 855 ms (44.5× slower) |
| 1mb | puggles (py, jax jit+vmap) **48 ms** | puggles (py, batch) 54 ms | puggles (native, 8 threads) 69 ms | 384 ms (7.9× slower) | 349 ms (7.2× slower) | 1,492 ms (30.8× slower) | 1,225 ms (25.3× slower) |
| 10mb | puggles (native, GPU) **202 ms** | puggles (native, CubeCL GPU) 211 ms | puggles (py, jax jit+vmap) 228 ms | 1,035 ms (5.1× slower) | 680 ms (3.4× slower) | 4,728 ms (23.4× slower) | 4,342 ms (21.5× slower) |
| 100mb | puggles (native, CubeCL GPU) **146 ms** | puggles (native, GPU) 247 ms | puggles (py, jax jit+vmap) 406 ms | 1,532 ms (10.5× slower) | 836 ms (5.7× slower) | 9,537 ms (65.5× slower) | 9,396 ms (64.5× slower) |
| 1gb | puggles (native, GPU) **1,524 ms** | puggles (native, CubeCL GPU) 1,674 ms | puggles (py, torch MPS batch) 4,844 ms | 13.5 s (8.8× slower) | 7,794 ms (5.1× slower) | 94.7 s (62.1× slower) | 94.6 s (62.1× slower) |

## What the faster evaluation paths changed

| change | 1 KB | 1 MB | 10 MB | 100 MB | 1 GB |
| --- | --- | --- | --- | --- | --- |
| **GPU kernel**: per-solution → map-reduce (`with_partials`) | 177 → 205 ms | 167 → 201 ms | 452 → 202 ms | 612 → 247 ms | **6.6 → 1.5 s (4.3×)** |
| **Python batch**: lists → arrays (`batch_numpy=True`) | 27.4 → 21.6 ms (1.27×) | 70.0 → 53.9 ms (1.30×) | 681 → 659 ms | 1.29 → 1.25 s | 13.3 → 15.7 s (single run; noise) |
| **Per-solution Python → `objective_cfunc`** (8 threads, no GIL) | 56 → 24 ms (2.3×) | 430 → 74 ms (5.8×) | 3.4 → 0.51 s (6.7×) | 10.0 → 1.28 s (7.8×) | 93 → 21 s (4.4×) |

Times are multi-objective opt times. The cfunc row compares against `puggles (py)`, the same per-solution objective written in plain Python.

## Findings

1. **The map-reduce GPU kernel is now the fastest option from 10 MB up: 8.8× faster than pymoo at 1 GB** (1.5 s vs 13.5 s), using 6 CPU-seconds.
   - The old kernel gave each solution its own workgroup, so every solution streamed the whole dataset. The new one stages each block of rows in workgroup memory and reuses it for all 100 solutions, so a generation reads the data once.
   - The other half of the speed-up came from holding each solution's weights in `vec4` registers with an unrolled dot product. A loop-indexed per-thread array was about 5× slower.
   - Results match the CPU exactly, because the final quality is recomputed in f64.
   - Below about 1 MB the GPU still loses to the CPU. Device and shader setup (about 150 ms) dominates.

2. **CubeCL is viable.** The same kernel written in Rust (`#[cube]`) runs within about 1.1–1.2× of the hand-written WGSL at 1 GB (1.7 s vs 1.5 s), with identical results. It needed the same register/unroll treatment (6.7 s → 1.9 s).
   - It would let puggles users write GPU objectives in Rust instead of WGSL, and target CUDA and ROCm from the same source.
   - It requires CubeCL 0.10 here: 0.11 needs rustc 1.95.
   - It adds a large dependency tree, so it fits better as an optional feature or companion crate than as a core dependency.

3. **Framework batch objectives work well through puggles' batch interface.**
   - **PyTorch on the Apple GPU (MPS):** 4.7–4.8 s at 1 GB, 2.8–2.9× faster than pymoo, and only about 5 CPU-seconds.
   - **JAX on the CPU:** 4.3–6.0 s at 1 GB. It's also the fastest Python option at 1 MB (48 ms), because XLA fuses the block scan.
   - Neither needed any puggles changes beyond `batch_numpy=True`.
   - MPS has high fixed per-call costs, so Torch is slow on small data (150–290 ms at 1 KB).

4. **The Numba cfunc path makes a Python-written per-solution objective run like native Rust.** It's within 15% of native 8 threads at 1 GB, and 2.3–7.8× faster than the same objective as a plain Python callable.
   - It does **not** beat batch evaluation at large sizes, because each call still reads the whole dataset for one solution. That makes it the right tool for per-solution objectives that can't be batched, such as simulations, not for data scans.

5. **The NumPy array hand-off cuts Python↔Rust overhead by about 1.3× on small data.** It doesn't matter at large sizes, where the objective dominates.
   - At 1 KB, puggles' Python batch path (22 ms) is now close to native single-threaded puggles (19 ms), and 16× faster than pymoo.

6. **Why native 8-thread puggles still loses to batch evaluation on big data.**
   - Native puggles evaluates one solution at a time, so each evaluation streams the whole 0.9 GB matrix. That's about 1.8 TB per run, at the M3's roughly 100 GB/s memory bandwidth.
   - Batch paths read each row once per generation for all solutions. A native Rust `EvalFn::Batch` objective written that way would close the gap; it isn't benchmarked here.

7. **Memory.**
   - **Small data:** native puggles uses **5–36 MB** up to 10 MB. Python libraries use 52–141 MB. Numba, JAX and Torch add their runtimes (110–250 MB at 1 KB).
   - **At 1 GB:** CPU runs land at 2.7–3.0 GB, dominated by the dataset. The loaders briefly hold about 3 copies of the 0.9 GB matrix.
   - The GPU rows are higher (4.5–6.0 GB) because of the f32 device copy. On unified memory, GPU buffers count toward the process.

8. **Solution quality is comparable. No library wins everywhere.**
   - **Multi-objective HV:** puggles is best at 100 MB, while pymoo and platypus are best at 1 GB (0.702 vs 0.668). Those are single 20-generation runs, so seed noise dominates.
   - **Single-objective Sharpe:** every library is within about 1%, except genevo, which is consistently lowest.
   - Every new evaluation path (GPU, CubeCL, cfunc, Torch, JAX) reproduces the same best Sharpe and fronts as the CPU path for the same seed.

9. **Real data against synthetic** ([`benchmarks_datasets_synthetic.md`](benchmarks_datasets_synthetic.md)). Timings and rankings barely moved, because the evaluation cost depends on the number of rows, not on the values.

### Caveats
- The 100 MB and 1 GB rows are single runs at 2,000 evaluations, so differences under about 10% are noise. For example, `puggles (py, batch)` at 1 GB measured 13.3 s and 15.7 s on two runs of identical code.
- JIT compile time is excluded from opt time but included in CPU s: Numba about 2 s, JAX about 0.3 s. GPU, CubeCL and Torch opt times *include* device setup and data upload.
- NumPy, Numba, JAX and XLA use several threads, so the Python rows aren't strictly single-threaded. The CPU s column shows it.
- **The CUDA backend (`puggles-cuda/`, kernels written with Rust-CUDA) was written but never compiled or run.** Rust-CUDA needs Linux/Windows, a pinned nightly, LLVM 7 or 21, CUDA 12+ and an NVIDIA GPU, none of which this Mac has. Its benchmark row shows `not built`.
- Bootstrapped scenarios replay real days in random 20-day blocks. They are not real history beyond the first 10,242 days.
- **Toolchain note:** on this macOS 27.2 / Xcode setup, the default `ld` produced a puggles Python extension that wouldn't load ("mis-aligned LINKEDIT string pool"). The bindings were built with the Command Line Tools linker instead: `DEVELOPER_DIR=/Library/Developer/CommandLineTools maturin develop --release`.

## Three machines (2026-11-13): Apple M3, GCP CPU, GCP NVIDIA L4 with Rust-CUDA

The whole suite was rerun on identical datasets (checksums match) on three machines. The short version, with problems found and next steps, is in [`report_cuda_and_cloud.md`](report_cuda_and_cloud.md). Raw data: `results_datasets_m3.json`, `results_datasets_gcp_c3.json`, `results_datasets_gcp_g2.json`. The M3 numbers in this section come from that rerun, so they differ slightly from the M3 tables above, which used an earlier download of the same prices.

**Not available on the L4 VM:** the wgpu GPU path and CubeCL, because the VM's NVIDIA Vulkan driver would not initialize and wgpu only found a CPU software renderer. Those rows are marked `no GPU` (wgpu) or omitted (CubeCL). CUDA and PyTorch-CUDA are unaffected. **Not built on the Mac:** Rust-CUDA, which needs Linux and an NVIDIA GPU.

### Machines

| | Apple M3 (local) | GCP c3-standard-8 | GCP g2-standard-8 (Spot) |
| --- | --- | --- | --- |
| CPU | Apple M3 (4P + 4E cores) | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | Intel(R) Xeon(R) CPU @ 2.20GHz |
| Threads | 8 | 8 vCPU | 8 vCPU |
| Memory | 16 GB unified | 31 GB | 31 GB |
| GPU | Apple M3, 10 cores (Metal) | none | NVIDIA L4, 23034 MiB, 580.178.04 |
| OS | macOS 27.2.1 (26B131) | Ubuntu 24.04.5 LTS (7.0.0-1017-gcp) | Ubuntu 24.04.5 LTS (7.0.0-1017-gcp) |
| Python / torch / jax | 3.14.7 / 2.15.0 / 0.12.0.1 | 3.14.10 / 2.15.0+cpu / 0.12.0.1 | 3.14.10 / 2.15.0 / 0.12.0.1 |

### 1 GB, multi-objective: the same path on each machine (opt time, ↓ lower is better)

| path | Apple M3 | GCP c3-standard-8 | GCP g2 + L4 |
| --- | --- | --- | --- |
| puggles (native, sequential) | 64.3 s | 184.0 s | 219.2 s |
| puggles (native, 8 threads) | 27.8 s | 26.3 s | 38.1 s |
| puggles (native, CPU row scan) | 14.4 s | 17.6 s | 34.8 s |
| puggles (native, Auto) | 2.7 s | 27.2 s | 39.8 s |
| puggles (native, GPU) | 1.7 s | n/a | no GPU |
| puggles (native, CubeCL GPU) | 2.0 s | n/a | n/a |
| puggles (native, CUDA, Rust-CUDA) | not built | n/a | 2.6 s |
| puggles (py, batch) | 13.4 s | 25.1 s | 33.0 s |
| puggles (py, numba batch) | 6.8 s | 8.1 s | 10.0 s |
| puggles (py, numba cfunc, 8 threads) | 19.6 s | 37.4 s | 46.1 s |
| puggles (py, jax jit+vmap) | 4.7 s | 6.9 s | 9.1 s |
| puggles (py, torch batch) | 5.1 s | 24.1 s | 1.5 s |
| pymoo | 13.5 s | 25.0 s | 32.7 s |
| pymoo (numba) | 7.0 s | 8.3 s | 10.9 s |
| platypus | 95.8 s | 57.2 s | 70.7 s |
| DEAP | 97.4 s | 57.2 s | 70.6 s |
| puggles (py) | 99.0 s | 57.9 s | 68.8 s |

### 1 GB, single-objective: the same path on each machine (opt time, ↓ lower is better)

| path | Apple M3 | GCP c3-standard-8 | GCP g2 + L4 |
| --- | --- | --- | --- |
| puggles (native, sequential) | 42.3 s | 183.5 s | 218.9 s |
| puggles (native, 8 threads) | 22.6 s | 26.4 s | 38.1 s |
| puggles (native, CPU row scan) | 13.0 s | 17.6 s | 34.8 s |
| puggles (native, Auto) | 2.0 s | 27.2 s | 40.3 s |
| puggles (native, GPU) | 1.9 s | n/a | no GPU |
| puggles (native, CubeCL GPU) | 1.8 s | n/a | n/a |
| puggles (native, CUDA, Rust-CUDA) | not built | n/a | 2.6 s |
| puggles (py, batch) | 14.1 s | 25.0 s | 34.1 s |
| puggles (py, numba batch) | 7.6 s | 8.1 s | 11.4 s |
| puggles (py, numba cfunc, 8 threads) | 21.6 s | 37.1 s | 46.2 s |
| puggles (py, jax jit+vmap) | 5.0 s | 6.9 s | 9.0 s |
| puggles (py, torch batch) | 5.3 s | 24.3 s | 1.5 s |
| pymoo | 14.4 s | 25.1 s | 32.8 s |
| pymoo (numba) | 7.6 s | 8.4 s | 10.9 s |
| platypus | 103.1 s | 56.7 s | 69.7 s |
| DEAP | 102.7 s | 56.7 s | 70.2 s |
| genevo | 41.1 s | 138.0 s | 162.9 s |
| genetic_algorithm | 55.2 s | 186.9 s | 224.1 s |
| puggles (py) | 113.3 s | 57.0 s | 70.9 s |

### Rust-CUDA on the NVIDIA L4 against the other paths on that machine

Opt time, multi-objective, ↓ lower is better. The ratio column is pymoo's time on the same machine divided by the CUDA time.

| size | puggles (native, CUDA, Rust-CUDA) | puggles (py, torch batch) | puggles (py, jax jit+vmap) | puggles (py, numba batch) | puggles (native, 8 threads) | pymoo | pymoo ÷ CUDA ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | 216 ms | 571 ms | 97 ms | 85 ms | 61 ms | 1,209 ms | 5.6× |
| 1mb | 310 ms | 445 ms | 166 ms | 154 ms | 225 ms | 1,468 ms | 4.7× |
| 10mb | 396 ms | 427 ms | 567 ms | 593 ms | 1,657 ms | 3,116 ms | 7.9× |
| 100mb | 407 ms | 423 ms | 896 ms | 978 ms | 3,748 ms | 3,663 ms | 9.0× |
| 1gb | 2,646 ms | 1,543 ms | 9,069 ms | 9,969 ms | 38.1 s | 32.7 s | 12.4× |

Same comparison, single-objective:

| size | puggles (native, CUDA, Rust-CUDA) | puggles (py, torch batch) | puggles (py, jax jit+vmap) | puggles (py, numba batch) | puggles (native, 8 threads) | pymoo | pymoo ÷ CUDA ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | 179 ms | 400 ms | 52 ms | 50 ms | 31 ms | 977 ms | 5.5× |
| 1mb | 281 ms | 412 ms | 129 ms | 102 ms | 197 ms | 1,483 ms | 5.3× |
| 10mb | 314 ms | 410 ms | 518 ms | 565 ms | 1,625 ms | 3,075 ms | 9.8× |
| 100mb | 406 ms | 619 ms | 900 ms | 1,174 ms | 3,721 ms | 3,750 ms | 9.2× |
| 1gb | 2,614 ms | 1,511 ms | 9,029 ms | 11.4 s | 38.1 s | 32.8 s | 12.5× |

At 1 GB on the L4 (multi-objective): CPU time and peak memory.

| path | opt time ↓ | CPU s ↓ | peak memory ↓ | HV ↑ |
| --- | --- | --- | --- | --- |
| puggles (native, CUDA, Rust-CUDA) | 2.6 s | 29.1 | 1,951 MB | 0.716 |
| puggles (py, torch batch) | 1.5 s | 18.0 | 2,992 MB | 0.716 |
| puggles (py, jax jit+vmap) | 9.1 s | 63.5 | 2,879 MB | 0.716 |
| puggles (py, numba batch) | 10.0 s | 100.1 | 2,810 MB | 0.716 |
| puggles (native, 8 threads) | 38.1 s | 315.4 | 2,578 MB | 0.716 |
| pymoo | 32.7 s | 271.1 | 2,814 MB | 0.720 |
| puggles (native, sequential) | 219.2 s | 241.4 | 2,578 MB | 0.716 |
| puggles (native, CPU row scan) | 34.8 s | 295.7 | 2,578 MB | 0.717 |

## Reproduce

```bash
cd sandbox
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python pymoo deap platypus-opt numpy polars psutil maturin numba torch jax
( cd ../python && DEVELOPER_DIR=/Library/Developer/CommandLineTools \
    VIRTUAL_ENV=$PWD/../sandbox/.venv ../sandbox/.venv/bin/maturin develop --release --uv )
.venv/bin/python -I scripts/build_real_datasets.py data/raw      # download (cached) + build all sizes
cargo build --release --features gpu,cubecl --example bench_datasets --example bench_cubecl
# optional, Linux + NVIDIA only: (cd ../puggles-cuda && cargo build --release --example bench_cuda)
.venv/bin/python bench_datasets.py        # all sizes and libraries, about 40 min on the M3
# subset: .venv/bin/python bench_datasets.py --sizes 1kb,1mb --nfe 500 --runs 1 --libs pymoo,puggles-gpu
# synthetic data: --prefix returns_
```

## Full results

Every cell is the mean over 3 runs (1kb–10mb) or a single run (100mb, 1gb). opt ms shows ± std where there were several runs.

### Multi-objective

| size | library | load ms ↓ | opt ms ↓ | µs/eval ↓ | evals/s ↑ | CPU s ↓ | peak memory MB ↓ | HV ↑ | front size | vs pymoo (opt time) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | puggles (native, sequential) | 0.7 | 19.2 ±1 | 1.92 | 520,833 | 0.02 | 5 | 0.7494 | 100 | 17.93× |
| 1kb | puggles (native, 8 threads) | 0.7 | 25.8 ±2 | 2.58 | 388,098 | 0.08 | 6 | 0.7494 | 100 | 13.36× |
| 1kb | puggles (native, GPU) | 0.9 | 204.7 ±8 | 20.47 | 48,844 | 0.06 | 51 | 0.7494 | 100 | 1.68× |
| 1kb | puggles (native, CubeCL GPU) | 1.0 | 212.6 ±2 | 21.26 | 47,029 | 0.07 | 68 | 0.7494 | 100 | 1.62× |
| 1kb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | | |
| 1kb | puggles (py) | 2.3 | 56.0 ±0 | 5.60 | 178,571 | 0.20 | 52 | 0.7494 | 100 | 6.15× |
| 1kb | puggles (py, batch) | 20.6 | 21.6 ±0 | 2.16 | 462,250 | 0.16 | 53 | 0.7494 | 100 | 15.91× |
| 1kb | puggles (py, numba batch) | 2.1 | 29.2 ±1 | 2.92 | 342,075 | 2.32 | 172 | 0.7494 | 100 | 11.77× |
| 1kb | puggles (py, numba cfunc, 8 threads) | 2.1 | 23.9 ±0 | 2.39 | 418,994 | 0.58 | 111 | 0.7494 | 100 | 14.42× |
| 1kb | puggles (py, torch MPS batch) | 2.1 | 292.9 ±231 | 29.29 | 34,141 | 0.83 | 247 | 0.7494 | 100 | 1.18× |
| 1kb | puggles (py, jax jit+vmap) | 2.1 | 23.5 ±0 | 2.35 | 424,929 | 0.44 | 122 | 0.7494 | 100 | 14.62× |
| 1kb | pymoo | 2.2 | 344.2 ±3 | 34.42 | 29,056 | 0.49 | 77 | 0.7576 | 100 | 1.00× |
| 1kb | pymoo (numba) | 7.6 | 304.9 ±22 | 30.49 | 32,794 | 2.65 | 180 | 0.7576 | 100 | 1.13× |
| 1kb | platypus | 2.5 | 1,086.2 ±20 | 108.62 | 9,207 | 1.22 | 53 | 0.7574 | 100 | 0.32× |
| 1kb | DEAP | 2.1 | 855.1 ±10 | 85.51 | 11,695 | 1.02 | 56 | 0.7162 | 100 | 0.40× |
| 1mb | puggles (native, sequential) | 3.7 | 251.6 ±57 | 25.16 | 39,751 | 0.23 | 10 | 0.6293 | 100 | 1.53× |
| 1mb | puggles (native, 8 threads) | 2.6 | 69.0 ±1 | 6.90 | 144,858 | 0.40 | 11 | 0.6293 | 100 | 5.56× |
| 1mb | puggles (native, GPU) | 3.8 | 201.3 ±4 | 20.13 | 49,685 | 0.07 | 56 | 0.6368 | 100 | 1.91× |
| 1mb | puggles (native, CubeCL GPU) | 2.9 | 205.5 ±2 | 20.55 | 48,662 | 0.07 | 75 | 0.6387 | 100 | 1.87× |
| 1mb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | | |
| 1mb | puggles (py) | 4.4 | 430.3 ±3 | 43.03 | 23,238 | 0.57 | 57 | 0.6293 | 100 | 0.89× |
| 1mb | puggles (py, batch) | 5.5 | 53.9 ±0 | 5.39 | 185,644 | 0.19 | 61 | 0.6293 | 100 | 7.13× |
| 1mb | puggles (py, numba batch) | 4.4 | 69.1 ±1 | 6.91 | 144,788 | 2.54 | 177 | 0.6293 | 100 | 5.56× |
| 1mb | puggles (py, numba cfunc, 8 threads) | 4.2 | 73.8 ±1 | 7.38 | 135,563 | 0.94 | 112 | 0.6293 | 100 | 5.21× |
| 1mb | puggles (py, torch MPS batch) | 4.1 | 209.7 ±60 | 20.97 | 47,680 | 0.80 | 282 | 0.6441 | 100 | 1.83× |
| 1mb | puggles (py, jax jit+vmap) | 4.1 | 48.4 ±0 | 4.84 | 206,469 | 0.49 | 126 | 0.6293 | 100 | 7.93× |
| 1mb | pymoo | 4.2 | 384.0 ±16 | 38.40 | 26,042 | 0.52 | 87 | 0.6235 | 100 | 1.00× |
| 1mb | pymoo (numba) | 4.6 | 348.7 ±5 | 34.87 | 28,675 | 2.96 | 185 | 0.6235 | 100 | 1.10× |
| 1mb | platypus | 4.2 | 1,491.8 ±26 | 149.18 | 6,703 | 1.61 | 58 | 0.6834 | 100 | 0.26× |
| 1mb | DEAP | 4.2 | 1,224.9 ±24 | 122.49 | 8,164 | 1.37 | 61 | 0.6173 | 100 | 0.31× |
| 10mb | puggles (native, sequential) | 19.2 | 2,110.6 ±53 | 211.06 | 4,738 | 2.14 | 36 | 0.7365 | 100 | 0.49× |
| 10mb | puggles (native, 8 threads) | 15.1 | 473.5 ±13 | 47.35 | 21,118 | 3.14 | 36 | 0.7365 | 100 | 2.19× |
| 10mb | puggles (native, GPU) | 18.9 | 201.7 ±6 | 20.17 | 49,579 | 0.11 | 97 | 0.7513 | 100 | 5.13× |
| 10mb | puggles (native, CubeCL GPU) | 14.7 | 211.1 ±2 | 21.11 | 47,371 | 0.12 | 183 | 0.7472 | 100 | 4.90× |
| 10mb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | | |
| 10mb | puggles (py) | 10.5 | 3,423.6 ±84 | 342.36 | 2,921 | 3.55 | 81 | 0.7365 | 100 | 0.30× |
| 10mb | puggles (py, batch) | 10.0 | 658.8 ±5 | 65.88 | 15,180 | 0.82 | 116 | 0.7365 | 100 | 1.57× |
| 10mb | puggles (py, numba batch) | 10.5 | 364.3 ±4 | 36.43 | 27,452 | 4.37 | 193 | 0.7365 | 100 | 2.84× |
| 10mb | puggles (py, numba cfunc, 8 threads) | 10.4 | 511.2 ±9 | 51.12 | 19,563 | 4.23 | 131 | 0.7365 | 100 | 2.02× |
| 10mb | puggles (py, torch MPS batch) | 10.3 | 384.3 ±2 | 38.43 | 26,024 | 0.87 | 1,333 | 0.7486 | 100 | 2.69× |
| 10mb | puggles (py, jax jit+vmap) | 11.0 | 227.7 ±2 | 22.77 | 43,924 | 1.30 | 169 | 0.7365 | 100 | 4.55× |
| 10mb | pymoo | 10.9 | 1,035.0 ±28 | 103.50 | 9,662 | 1.20 | 141 | 0.7478 | 100 | 1.00× |
| 10mb | pymoo (numba) | 11.7 | 680.5 ±12 | 68.05 | 14,695 | 4.77 | 202 | 0.7478 | 100 | 1.52× |
| 10mb | platypus | 10.9 | 4,728.0 ±479 | 472.80 | 2,115 | 4.76 | 81 | 0.7153 | 100 | 0.22× |
| 10mb | DEAP | 11.0 | 4,342.0 ±41 | 434.20 | 2,303 | 4.46 | 81 | 0.7373 | 100 | 0.24× |
| 100mb | puggles (native, sequential) | 268.5 | 4,069.6 | 2,034.80 | 491 | 4.60 | 312 | 0.7260 | 100 | 0.38× |
| 100mb | puggles (native, 8 threads) | 135.5 | 1,416.7 | 708.35 | 1,412 | 9.67 | 312 | 0.7260 | 100 | 1.08× |
| 100mb | puggles (native, GPU) | 266.0 | 246.9 | 123.45 | 8,100 | 0.58 | 526 | 0.7184 | 100 | 6.20× |
| 100mb | puggles (native, CubeCL GPU) | 134.2 | 145.6 | 72.80 | 13,736 | 0.60 | 840 | 0.7184 | 100 | 10.52× |
| 100mb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | | |
| 100mb | puggles (py) | 70.1 | 9,980.7 | 4,990.35 | 200 | 10.29 | 336 | 0.7260 | 100 | 0.15× |
| 100mb | puggles (py, batch) | 64.3 | 1,253.0 | 626.50 | 1,596 | 1.69 | 373 | 0.7260 | 100 | 1.22× |
| 100mb | puggles (py, numba batch) | 73.9 | 661.3 | 330.65 | 3,024 | 6.75 | 373 | 0.7260 | 100 | 2.32× |
| 100mb | puggles (py, numba cfunc, 8 threads) | 68.2 | 1,278.4 | 639.20 | 1,564 | 10.09 | 337 | 0.7260 | 100 | 1.20× |
| 100mb | puggles (py, torch MPS batch) | 67.9 | 599.3 | 299.65 | 3,337 | 1.16 | 1,520 | 0.7260 | 100 | 2.56× |
| 100mb | puggles (py, jax jit+vmap) | 66.3 | 405.9 | 202.95 | 4,927 | 2.51 | 425 | 0.7260 | 100 | 3.77× |
| 100mb | pymoo | 67.8 | 1,531.5 | 765.75 | 1,306 | 1.98 | 398 | 0.7078 | 100 | 1.00× |
| 100mb | pymoo (numba) | 69.1 | 836.5 | 418.25 | 2,391 | 7.02 | 382 | 0.7078 | 100 | 1.83× |
| 100mb | platypus | 71.2 | 9,536.7 | 4,768.35 | 210 | 9.90 | 336 | 0.6559 | 85 | 0.16× |
| 100mb | DEAP | 70.6 | 9,396.5 | 4,698.25 | 213 | 9.80 | 336 | 0.6939 | 100 | 0.16× |
| 1gb | puggles (native, sequential) | 1,860.2 | 41,400.9 | 20,700.45 | 48 | 46.85 | 2,732 | 0.6679 | 100 | 0.33× |
| 1gb | puggles (native, 8 threads) | 2,307.7 | 18,620.4 | 9,310.20 | 107 | 130.45 | 2,733 | 0.6679 | 100 | 0.72× |
| 1gb | puggles (native, GPU) | 2,821.1 | 1,524.0 | 762.00 | 1,312 | 6.22 | 4,488 | 0.6679 | 100 | 8.83× |
| 1gb | puggles (native, CubeCL GPU) | 1,355.4 | 1,674.1 | 837.05 | 1,195 | 6.23 | 5,956 | 0.6679 | 100 | 8.04× |
| 1gb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | | |
| 1gb | puggles (py) | 751.5 | 93,218.5 | 46,609.25 | 21 | 96.17 | 2,936 | 0.6679 | 100 | 0.14× |
| 1gb | puggles (py, batch) | 660.9 | 15,711.4 | 7,855.70 | 127 | 17.76 | 2,923 | 0.6679 | 100 | 0.86× |
| 1gb | puggles (py, numba batch) | 992.7 | 7,349.5 | 3,674.75 | 272 | 52.03 | 2,920 | 0.6679 | 100 | 1.83× |
| 1gb | puggles (py, numba cfunc, 8 threads) | 722.3 | 21,331.2 | 10,665.60 | 94 | 140.99 | 2,931 | 0.6679 | 100 | 0.63× |
| 1gb | puggles (py, torch MPS batch) | 1,249.4 | 4,843.6 | 2,421.80 | 413 | 5.20 | 3,704 | 0.6679 | 100 | 2.78× |
| 1gb | puggles (py, jax jit+vmap) | 1,069.1 | 5,953.1 | 2,976.55 | 336 | 25.32 | 3,025 | 0.6679 | 100 | 2.26× |
| 1gb | pymoo | 707.7 | 13,463.0 | 6,731.50 | 149 | 16.56 | 2,929 | 0.7016 | 100 | 1.00× |
| 1gb | pymoo (numba) | 704.9 | 7,794.0 | 3,897.00 | 257 | 54.48 | 2,931 | 0.7016 | 100 | 1.73× |
| 1gb | platypus | 735.8 | 94,707.4 | 47,353.70 | 21 | 97.22 | 2,937 | 0.7017 | 100 | 0.14× |
| 1gb | DEAP | 876.2 | 94,566.0 | 47,283.00 | 21 | 97.22 | 2,929 | 0.6928 | 100 | 0.14× |

### Single-objective

| size | library | load ms ↓ | opt ms ↓ | µs/eval ↓ | evals/s ↑ | CPU s ↓ | peak memory MB ↓ | best Sharpe ↑ | vs pymoo (opt time) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | puggles (native, sequential) | 0.7 | 8.7 ±0 | 0.87 | 1,153,846 | 0.01 | 5 | 0.5582 | 41.58× |
| 1kb | puggles (native, 8 threads) | 0.7 | 14.0 ±0 | 1.40 | 712,589 | 0.07 | 5 | 0.5582 | 25.68× |
| 1kb | puggles (native, GPU) | 1.0 | 191.8 ±8 | 19.18 | 52,138 | 0.05 | 50 | 0.5582 | 1.88× |
| 1kb | puggles (native, CubeCL GPU) | 1.0 | 189.1 ±4 | 18.91 | 52,873 | 0.05 | 68 | 0.5582 | 1.91× |
| 1kb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | |
| 1kb | genevo | 0.9 | 5.1 ±0 | 0.51 | 1,963,636 | 0.02 | 5 | 0.4752 | 70.20× |
| 1kb | genetic_algorithm | 0.8 | 2.5 ±0 | 0.26 | 3,898,533 | 0.00 | 5 | 0.5571 | 144.15× |
| 1kb | puggles (py) | 2.2 | 55.0 ±16 | 5.50 | 181,818 | 0.18 | 52 | 0.5582 | 6.55× |
| 1kb | puggles (py, batch) | 2.2 | 10.7 ±0 | 1.07 | 937,500 | 0.14 | 53 | 0.5582 | 33.78× |
| 1kb | puggles (py, numba batch) | 2.1 | 19.9 ±1 | 1.99 | 502,513 | 2.30 | 172 | 0.5582 | 18.11× |
| 1kb | puggles (py, numba cfunc, 8 threads) | 2.1 | 13.2 ±0 | 1.32 | 755,668 | 0.56 | 111 | 0.5582 | 27.23× |
| 1kb | puggles (py, torch MPS batch) | 2.1 | 149.4 ±1 | 14.94 | 66,949 | 0.78 | 245 | 0.5582 | 2.41× |
| 1kb | puggles (py, jax jit+vmap) | 2.2 | 13.0 ±0 | 1.30 | 769,231 | 0.42 | 122 | 0.5582 | 27.72× |
| 1kb | pymoo | 2.2 | 360.4 ±19 | 36.04 | 27,750 | 0.50 | 78 | 0.5572 | 1.00× |
| 1kb | pymoo (numba) | 2.2 | 304.1 ±9 | 30.41 | 32,884 | 2.71 | 180 | 0.5572 | 1.19× |
| 1kb | platypus | 2.1 | 470.8 ±2 | 47.08 | 21,240 | 0.60 | 53 | 0.5573 | 0.77× |
| 1kb | DEAP | 2.2 | 140.7 ±2 | 14.07 | 71,090 | 0.29 | 56 | 0.5573 | 2.56× |
| 1mb | puggles (native, sequential) | 2.6 | 213.6 ±8 | 21.36 | 46,824 | 0.21 | 10 | 0.0707 | 1.80× |
| 1mb | puggles (native, 8 threads) | 2.7 | 58.2 ±2 | 5.82 | 171,821 | 0.38 | 10 | 0.0707 | 6.62× |
| 1mb | puggles (native, GPU) | 3.0 | 184.9 ±2 | 18.49 | 54,083 | 0.05 | 56 | 0.0707 | 2.08× |
| 1mb | puggles (native, CubeCL GPU) | 2.9 | 187.4 ±2 | 18.74 | 53,352 | 0.05 | 75 | 0.0707 | 2.05× |
| 1mb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | |
| 1mb | genevo | 2.9 | 229.2 ±92 | 22.74 | 43,979 | 0.27 | 10 | 0.0680 | 1.68× |
| 1mb | genetic_algorithm | 2.8 | 206.7 ±15 | 21.33 | 46,880 | 0.20 | 10 | 0.0707 | 1.86× |
| 1mb | puggles (py) | 4.4 | 428.6 ±9 | 42.86 | 23,332 | 0.57 | 56 | 0.0707 | 0.90× |
| 1mb | puggles (py, batch) | 4.1 | 43.7 ±0 | 4.37 | 228,659 | 0.17 | 61 | 0.0707 | 8.80× |
| 1mb | puggles (py, numba batch) | 4.2 | 57.8 ±1 | 5.78 | 173,010 | 2.53 | 177 | 0.0707 | 6.66× |
| 1mb | puggles (py, numba cfunc, 8 threads) | 4.2 | 63.1 ±1 | 6.31 | 158,479 | 0.92 | 113 | 0.0707 | 6.10× |
| 1mb | puggles (py, torch MPS batch) | 4.1 | 167.7 ±1 | 16.77 | 59,618 | 0.79 | 282 | 0.0707 | 2.30× |
| 1mb | puggles (py, jax jit+vmap) | 4.5 | 37.7 ±0 | 3.77 | 265,018 | 0.49 | 126 | 0.0707 | 10.20× |
| 1mb | pymoo | 4.2 | 385.1 ±3 | 38.51 | 25,970 | 0.54 | 87 | 0.0707 | 1.00× |
| 1mb | pymoo (numba) | 4.2 | 352.8 ±1 | 35.28 | 28,347 | 2.96 | 185 | 0.0707 | 1.09× |
| 1mb | platypus | 4.2 | 857.9 ±4 | 85.79 | 11,656 | 1.00 | 58 | 0.0707 | 0.45× |
| 1mb | DEAP | 4.2 | 526.0 ±9 | 52.60 | 19,011 | 0.69 | 61 | 0.0707 | 0.73× |
| 10mb | puggles (native, sequential) | 15.3 | 2,057.7 ±37 | 205.77 | 4,860 | 2.08 | 35 | 0.0681 | 0.51× |
| 10mb | puggles (native, 8 threads) | 14.8 | 473.2 ±11 | 47.32 | 21,134 | 3.14 | 36 | 0.0681 | 2.21× |
| 10mb | puggles (native, GPU) | 15.9 | 186.8 ±5 | 18.68 | 53,543 | 0.08 | 98 | 0.0681 | 5.59× |
| 10mb | puggles (native, CubeCL GPU) | 15.5 | 196.8 ±2 | 19.68 | 50,813 | 0.09 | 183 | 0.0681 | 5.30× |
| 10mb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | |
| 10mb | genevo | 15.3 | 1,441.6 ±34 | 143.02 | 6,992 | 2.20 | 36 | 0.0673 | 0.72× |
| 10mb | genetic_algorithm | 15.0 | 1,973.9 ±18 | 203.09 | 4,924 | 2.00 | 35 | 0.0681 | 0.53× |
| 10mb | puggles (py) | 10.2 | 3,351.3 ±55 | 335.13 | 2,984 | 3.49 | 81 | 0.0681 | 0.31× |
| 10mb | puggles (py, batch) | 9.8 | 644.5 ±7 | 64.45 | 15,517 | 0.80 | 116 | 0.0681 | 1.62× |
| 10mb | puggles (py, numba batch) | 10.9 | 351.2 ±7 | 35.12 | 28,474 | 4.33 | 194 | 0.0681 | 2.97× |
| 10mb | puggles (py, numba cfunc, 8 threads) | 10.7 | 508.0 ±6 | 50.80 | 19,684 | 4.21 | 131 | 0.0681 | 2.05× |
| 10mb | puggles (py, torch MPS batch) | 10.1 | 372.0 ±9 | 37.20 | 26,879 | 0.87 | 1,333 | 0.0681 | 2.81× |
| 10mb | puggles (py, jax jit+vmap) | 9.9 | 219.3 ±4 | 21.93 | 45,607 | 1.30 | 170 | 0.0681 | 4.76× |
| 10mb | pymoo | 10.6 | 1,043.7 ±38 | 104.37 | 9,581 | 1.22 | 141 | 0.0681 | 1.00× |
| 10mb | pymoo (numba) | 12.3 | 684.0 ±12 | 68.40 | 14,620 | 4.74 | 201 | 0.0681 | 1.53× |
| 10mb | platypus | 11.3 | 3,852.3 ±57 | 385.23 | 2,596 | 3.99 | 81 | 0.0681 | 0.27× |
| 10mb | DEAP | 11.0 | 3,555.4 ±80 | 355.54 | 2,813 | 3.69 | 81 | 0.0681 | 0.29× |
| 100mb | puggles (native, sequential) | 164.2 | 4,023.1 | 2,011.55 | 497 | 4.36 | 312 | 0.0664 | 0.36× |
| 100mb | puggles (native, 8 threads) | 133.2 | 1,329.7 | 664.85 | 1,504 | 8.85 | 312 | 0.0664 | 1.09× |
| 100mb | puggles (native, GPU) | 130.9 | 128.5 | 64.25 | 15,564 | 0.37 | 525 | 0.0664 | 11.32× |
| 100mb | puggles (native, CubeCL GPU) | 130.9 | 135.6 | 67.80 | 14,749 | 0.38 | 840 | 0.0664 | 10.73× |
| 100mb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | |
| 100mb | genevo | 134.7 | 3,168.9 | 1,467.08 | 682 | 5.27 | 312 | 0.0648 | 0.46× |
| 100mb | genetic_algorithm | 134.4 | 4,098.7 | 2,054.49 | 487 | 4.38 | 312 | 0.0662 | 0.35× |
| 100mb | puggles (py) | 69.9 | 9,336.3 | 4,668.15 | 214 | 9.70 | 336 | 0.0664 | 0.16× |
| 100mb | puggles (py, batch) | 64.4 | 1,254.1 | 627.05 | 1,595 | 1.69 | 372 | 0.0664 | 1.16× |
| 100mb | puggles (py, numba batch) | 73.1 | 648.1 | 324.05 | 3,086 | 6.75 | 373 | 0.0664 | 2.24× |
| 100mb | puggles (py, numba cfunc, 8 threads) | 65.9 | 1,366.4 | 683.20 | 1,464 | 10.27 | 337 | 0.0664 | 1.06× |
| 100mb | puggles (py, torch MPS batch) | 68.9 | 497.9 | 248.95 | 4,017 | 1.18 | 1,521 | 0.0664 | 2.92× |
| 100mb | puggles (py, jax jit+vmap) | 66.9 | 404.7 | 202.35 | 4,942 | 2.48 | 425 | 0.0664 | 3.59× |
| 100mb | pymoo | 71.6 | 1,454.5 | 727.25 | 1,375 | 1.91 | 397 | 0.0665 | 1.00× |
| 100mb | pymoo (numba) | 68.8 | 860.8 | 430.40 | 2,323 | 7.01 | 382 | 0.0665 | 1.69× |
| 100mb | platypus | 73.2 | 9,335.3 | 4,667.65 | 214 | 9.71 | 337 | 0.0649 | 0.16× |
| 100mb | DEAP | 72.1 | 9,144.2 | 4,572.10 | 219 | 9.58 | 336 | 0.0661 | 0.16× |
| 1gb | puggles (native, sequential) | 1,605.0 | 41,954.3 | 20,977.15 | 48 | 45.07 | 2,732 | 0.0678 | 0.32× |
| 1gb | puggles (native, 8 threads) | 1,870.8 | 18,851.3 | 9,425.65 | 106 | 127.52 | 2,732 | 0.0678 | 0.72× |
| 1gb | puggles (native, GPU) | 1,615.1 | 1,587.0 | 793.50 | 1,260 | 4.28 | 4,488 | 0.0678 | 8.52× |
| 1gb | puggles (native, CubeCL GPU) | 2,139.2 | 1,903.7 | 951.85 | 1,051 | 4.32 | 5,615 | 0.0678 | 7.10× |
| 1gb | puggles (native, CUDA, Rust-CUDA) | not built | | | | | | | |
| 1gb | genevo | 1,700.8 | 34,960.3 | 16,185.32 | 62 | 61.95 | 2,732 | 0.0668 | 0.39× |
| 1gb | genetic_algorithm | 1,350.5 | 44,079.8 | 21,202.41 | 47 | 47.02 | 2,732 | 0.0676 | 0.31× |
| 1gb | puggles (py) | 900.2 | 97,880.0 | 48,940.00 | 20 | 98.47 | 2,929 | 0.0678 | 0.14× |
| 1gb | puggles (py, batch) | 660.8 | 13,963.9 | 6,981.95 | 143 | 17.07 | 2,939 | 0.0678 | 0.97× |
| 1gb | puggles (py, numba batch) | 724.0 | 7,079.3 | 3,539.65 | 283 | 50.82 | 2,923 | 0.0678 | 1.91× |
| 1gb | puggles (py, numba cfunc, 8 threads) | 647.4 | 18,802.5 | 9,401.25 | 106 | 134.94 | 2,937 | 0.0678 | 0.72× |
| 1gb | puggles (py, torch MPS batch) | 686.4 | 4,678.4 | 2,339.20 | 427 | 4.79 | 3,732 | 0.0678 | 2.89× |
| 1gb | puggles (py, jax jit+vmap) | 627.5 | 4,266.6 | 2,133.30 | 469 | 21.38 | 3,028 | 0.0678 | 3.17× |
| 1gb | pymoo | 1,106.8 | 13,520.7 | 6,760.35 | 148 | 16.82 | 2,914 | 0.0678 | 1.00× |
| 1gb | pymoo (numba) | 727.7 | 7,602.1 | 3,801.05 | 263 | 54.43 | 2,921 | 0.0678 | 1.78× |
| 1gb | platypus | 1,064.1 | 93,973.7 | 46,986.85 | 21 | 96.82 | 2,918 | 0.0672 | 0.14× |
| 1gb | DEAP | 1,058.5 | 93,902.9 | 46,951.45 | 21 | 96.74 | 2,918 | 0.0675 | 0.14× |
