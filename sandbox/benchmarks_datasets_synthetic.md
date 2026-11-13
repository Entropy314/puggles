# puggles Dataset-Scaling Benchmark: 1 KB → 1 GB (synthetic data)

Single- and multi-objective portfolio optimization on five dataset sizes. puggles runs in native Rust (sequential, 8 threads, GPU) and from Python. It is compared with **pymoo, platypus and DEAP** (Python) and **genevo and genetic_algorithm** (Rust).

**Run date:** 2026-11-06 · **Raw results:** [`results_datasets_synthetic.json`](results_datasets_synthetic.json) · **Drivers:** [`bench_datasets.py`](bench_datasets.py), [`examples/bench_datasets.rs`](examples/bench_datasets.rs)

> **Correction (memory at 100 MB / 1 GB):** this run recorded *maximum resident set size*. The
> machine was under memory pressure at the time (about 6.8 GB of swap in use), and macOS leaves
> compressed and swapped-out pages out of RSS. As a result the 1 GB peak-memory figures here are
> understated by up to about 2×. Re-measured afterwards, pymoo at 1 GB peaks at 3.05 GB, not 1.56 GB.
> Timings are unaffected. The real-data report ([`benchmarks_datasets.md`](benchmarks_datasets.md))
> records *peak memory footprint* instead, which does count those pages.

## Hardware and software

| | |
| --- | --- |
| Machine | Apple M3: 8 CPU cores (4 performance + 4 efficiency), 10-core integrated GPU (Metal), 16 GB unified memory |
| OS | macOS 27.2 (26B123) |
| Rust | rustc 1.94.0, release profile with `lto = true`, `codegen-units = 1` · wgpu 22.1 |
| Python | 3.14.7 · numpy 2.5.4 · polars 2.1.0 |
| Libraries | puggles 0.3.0 · pymoo 0.6.2 · platypus-opt 1.4.1 · DEAP 1.4.4 · genevo 0.7.1 · genetic_algorithm 0.27.2 |

## How to read the metrics

| metric | better | meaning |
| --- | --- | --- |
| load ms | ↓ lower | Read the CSV into memory (polars in both Rust and Python) |
| opt ms | ↓ lower | Optimization wall time, the headline number (± = std over 3 runs) |
| µs/eval | ↓ lower | opt time ÷ objective evaluations |
| evals/s | ↑ higher | Evaluation throughput |
| CPU s | ↓ lower | user + sys CPU time of the whole process (load + optimize). Above wall time = several cores in use |
| peak RSS MB | ↓ lower | Peak resident memory of the whole process, dataset included |
| HV (multi) | ↑ higher | Hypervolume of the final Pareto front, 0–1. Normalized across all libraries **at the same size**, so compare within a size, not across sizes |
| front size (multi) | ↑ higher | Points on the final front (informational; capped at the population of 100) |
| best Sharpe (single) | ↑ higher | Best Sharpe ratio found (daily, not annualized), recomputed in f64 |
| vs pymoo | ↑ higher | pymoo opt time ÷ this library's opt time. >1× = faster than pymoo |

## Problem and method

**Data.** Synthetic daily returns for 20 assets from a single-factor model (`scripts/gen_1gb.py`), with the same `date,A00..A19` schema at every size:

| size | file bytes | rows (T) |
| --- | --- | --- |
| 1kb | 1,086 | 5 |
| 1mb | 1,104,364 | 5,500 |
| 10mb | 11,047,344 | 55,000 |
| 100mb | 110,256,833 | 549,000 |
| 1gb | 1,128,254,396 | 5,600,000 |

**Objective.** Every evaluation scans every row, so the cost of one evaluation grows linearly with the dataset. (The older §3 of [`benchmarks.md`](benchmarks.md) optimized on a 20×20 covariance matrix, which made the optimizer's speed independent of dataset size.) Weights are `w = x / Σx` with `x ∈ [0,1]^20`, and the portfolio return for row t is `r_t = Σ_i w_i·R_ti`.
- **Multi-objective:** minimize the realized variance `mean(r²) − mean(r)²` and maximize the mean return `mean(r)`.
- **Single-objective:** maximize Sharpe = `mean(r) / std(r)`.

**Fairness.**
- Every library gets the same problem, population (100) and number of objective evaluations (NFE), counted by a shared counter inside each objective. All libraries land within ±3% of the budget. genevo overshoots by 1% because it stops only at generation boundaries. genetic_algorithm is calibrated to about 2.5% under, because it caches fitness and its generation count has to be estimated.
- **Budget:** 10,000 evaluations × 3 seeded runs for 1kb–10mb. 2,000 evaluations × 1 run for 100mb and 1gb. The large sizes are single runs, so treat differences under about 10% as noise.
- **Isolation:** every run is its own process under `/usr/bin/time -l`, so peak RSS and CPU time aren't contaminated by other runs. Library imports happen before the clock starts.
- **genevo and genetic_algorithm support only one objective**, so they appear only in the single-objective results.

**How each library evaluates:**

| library | evaluation style |
| --- | --- |
| puggles native, sequential and 8 threads | Rust objective, one solution at a time (8 threads via Rayon) |
| puggles native, GPU | WGSL compute shader with the dataset in GPU memory (f32), one 256-thread workgroup per solution. Final results are recomputed in f64 on the CPU |
| puggles (py) | Python objective called once per solution (`R @ w` in NumPy) |
| puggles (py, batch) | `batch_objective_function`: one NumPy call per generation for the whole population |
| pymoo | Vectorized `_evaluate`: one NumPy call per generation (same block code as puggles batch) |
| platypus, DEAP | Python objective called once per solution (`R @ w` in NumPy) |
| genevo, genetic_algorithm | Rust objective, one solution at a time |

## Headline results

### 1 GB, multi-objective (2,000 evaluations)

| library | opt time ↓ | CPU s ↓ | peak RSS ↓ | HV ↑ | vs pymoo ↑ |
| --- | --- | --- | --- | --- | --- |
| **puggles (native, GPU)** | **6.8 s** | **6.1** | 1,664 MB | 0.736 | **2.16×** |
| puggles (py, batch) | 13.8 s | 17.7 | **1,260 MB** | 0.736 | 1.07× |
| pymoo | 14.8 s | 18.1 | 1,560 MB | 0.668 | 1.00× |
| puggles (native, 8 threads) | 20.4 s | 129.0 | 2,014 MB | 0.736 | 0.72× |
| puggles (native, sequential) | 44.6 s | 48.4 | 1,475 MB | 0.736 | 0.33× |
| puggles (py) | 101.2 s | 103.0 | 2,560 MB | 0.736 | 0.15× |
| DEAP | 106.0 s | 106.6 | 1,487 MB | **0.773** | 0.14× |
| platypus | 119.7 s | 113.5 | 1,486 MB | 0.692 | 0.12× |

### 1 GB, single-objective (2,000 evaluations)

| library | opt time ↓ | CPU s ↓ | peak RSS ↓ | best Sharpe ↑ | vs pymoo ↑ |
| --- | --- | --- | --- | --- | --- |
| **puggles (native, GPU)** | **8.8 s** | **5.2** | 1,590 MB | 0.0526 | **1.93×** |
| puggles (py, batch) | 14.8 s | 18.4 | 1,462 MB | 0.0526 | 1.14× |
| pymoo | 16.9 s | 19.3 | 1,369 MB | 0.0525 | 1.00× |
| puggles (native, 8 threads) | 22.0 s | 128.7 | 1,949 MB | 0.0526 | 0.77× |
| genevo | 37.3 s | 66.7 | 2,290 MB | 0.0489 | 0.45× |
| genetic_algorithm | 43.2 s | 46.3 | 2,051 MB | 0.0523 | 0.39× |
| puggles (native, sequential) | 44.9 s | 47.4 | 1,641 MB | 0.0526 | 0.38× |
| platypus | 103.8 s | 105.2 | 1,379 MB | 0.0528 | 0.16× |
| DEAP | 105.7 s | 104.7 | **1,332 MB** | **0.0530** | 0.16× |
| puggles (py) | 106.0 s | 105.9 | 1,313 MB | 0.0526 | 0.16× |

### Fastest optimizer at each size (opt time, multi-objective)

| size | 1st | 2nd | pymoo | platypus | DEAP |
| --- | --- | --- | --- | --- | --- |
| 1kb | puggles seq **20.8 ms** | puggles 8 thr 23.3 ms | 414 ms (20× slower) | 1,117 ms (54×) | 882 ms (42×) |
| 1mb | puggles py-batch **61.5 ms** | puggles 8 thr 74.8 ms | 414 ms (6.7×) | 1,484 ms (24×) | 1,280 ms (21×) |
| 10mb | puggles GPU **509 ms** | puggles 8 thr 557 ms | 1,097 ms (2.2×) | 4,550 ms (8.9×) | 4,361 ms (8.6×) |
| 100mb | puggles GPU **776 ms** | puggles py-batch 1,423 ms | 1,634 ms (2.1×) | 10,200 ms (13×) | 9,985 ms (13×) |
| 1gb | puggles GPU **6.8 s** | puggles py-batch 13.8 s | 14.8 s (2.2×) | 119.7 s (18×) | 106.0 s (16×) |

## Findings

1. **A puggles mode is fastest at every size, but which mode depends on the size.**
   - **1 KB:** the plain sequential CPU run wins. Threads and the GPU only add overhead on 5 rows.
   - **1 MB:** the batch Python objective and 8 threads lead.
   - **10 MB and up:** the GPU wins, about 2.2× faster than pymoo, the fastest other library. It also beats puggles' own 8-thread mode by 2.3–3× at 100 MB–1 GB.

2. **The GPU is the most resource-efficient option at scale.** At 1 GB it used **6 CPU-seconds**, against **129** for the 8-thread run and 18 for pymoo. The CPU stays almost idle while the GPU works.
   - Below about 10 MB the GPU loses. Its opt time includes about 150 ms of one-time device and shader setup, plus a fixed upload and readback every generation.

3. **Batch evaluation beats more threads when evaluation is limited by memory bandwidth.** pymoo and `puggles (py, batch)` beat puggles' native 8-thread run from 100 MB up.
   - The per-solution objective streams the whole matrix (about 900 MB at 1 GB) once for **every solution**. The batch path does one matrix multiply per generation, which reads each row once for all 100 solutions.
   - That's also why 8 threads give only 2.2× over sequential at 1 GB, for 2.7× the CPU time: the run is limited by memory bandwidth, not compute.
   - puggles' core already supports native batch objectives (`EvalFn::Batch`). A native-Rust batch run wasn't benchmarked here and is the obvious next row to add.

4. **One Python call per solution costs the most at scale.** puggles (py), platypus and DEAP all take about 100–120 s at 1 GB, about 7× slower than the batch versions of the same NumPy maths. puggles' own sorting and selection don't help when 2,000 separate Python calls dominate. For data-heavy objectives, pass `batch_objective_function`.

5. **Small datasets: puggles uses 5–10× less memory.** Native puggles peaks at 11–17 MB up to 1 MB of data. The Python libraries peak at 81–123 MB, because the interpreter, NumPy and polars set a floor. pymoo is the heaviest at every small size (113–176 MB).

6. **Large datasets: memory is dominated by the data, and native puggles has no advantage there.**
   - Every library peaks at 1.3–2.6 GB at 1 GB.
   - The native worker briefly holds the polars DataFrame, the column vectors and a row-major copy at the same time, so its peak exceeds the 0.9 GB matrix. Building the row-major copy straight from the CSV would remove most of that. That's a property of this benchmark's loader, not of puggles.
   - The 8-thread run adds per-thread buffers (2.0 GB). The GPU run adds an f32 copy of the data (1.6 GB).

7. **Solution quality is comparable. No library wins everywhere.**
   - **Multi-objective HV:**
     - pymoo is slightly ahead at 1 MB and 10 MB (0.732 / 0.772 against puggles' 0.724 / 0.754).
     - puggles leads at 100 MB (0.806 against 0.687–0.767 for the rest).
     - DEAP leads at 1 GB (0.773 against puggles' 0.736). At 100 MB and 1 GB that's a single run of only 20 generations, so the spread is mostly seed noise.
     - Every CPU puggles mode gives an identical front for a given seed. The GPU's front differs slightly at small sizes because it ranks solutions in f32.
   - **Single-objective Sharpe:** from 1 MB up, every library is within about 1–2%, except genevo, which is consistently lowest (0.0489 against about 0.053 at 1 GB).

8. **Load time:** loading the CSV takes about 3.6–4.0 s in the native Rust worker at 1 GB, against 1.3–2.2 s in Python polars. The Rust load also includes building the row-major matrix, and the reader configuration differs (see §3a of [`benchmarks.md`](benchmarks.md)). It doesn't affect the opt times.

### Caveats
- **1 KB has 5 rows.** With 5 returns an optimizer can push variance close to zero, so the 1 KB **Sharpe values (hundreds to thousands) are meaningless** as quality. Use that size only for overhead and timing. Multi-objective HV at 1 KB is fine.
- The 100 MB and 1 GB rows are single runs at 2,000 evaluations.
- NumPy's matrix multiply can use several threads, so the Python libraries aren't strictly single-threaded. The CPU s column shows how much extra CPU each run used.
- **Toolchain note:** on this macOS 27.2 / Xcode setup, the default `ld` produced a puggles Python extension that wouldn't load ("mis-aligned LINKEDIT string pool"). The bindings were built with the Command Line Tools linker instead: `DEVELOPER_DIR=/Library/Developer/CommandLineTools maturin develop --release`. Rust executables were unaffected.
- The GPU dispatch mode used here (`GpuEvaluator::with_solutions_per_workgroup(1)`) was added for this benchmark. With the default dispatch (one thread per solution), the 1 GB GPU run was about 8× slower than 8 CPU threads.

## Reproduce

```bash
cd sandbox
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python pymoo deap platypus-opt numpy polars psutil maturin
( cd ../python && DEVELOPER_DIR=/Library/Developer/CommandLineTools \
    VIRTUAL_ENV=$PWD/../sandbox/.venv ../sandbox/.venv/bin/maturin develop --release --uv )
head -n 6 data/returns_1mb.csv > data/returns_1kb.csv          # 1 KB (5 rows)
.venv/bin/python scripts/gen_1gb.py 100mb 549000                # 100 MB
.venv/bin/python scripts/gen_1gb.py 1gb 5600000                 # 1 GB
# 1 MB / 10 MB: scripts/gen_datasets.py
cargo build --release --features gpu --example bench_datasets
.venv/bin/python bench_datasets.py --prefix returns_ --out results_datasets_synthetic.json  # all sizes, about 25 min on the M3
# subset: .venv/bin/python bench_datasets.py --sizes 1kb,1mb --nfe 500 --runs 1 --libs pymoo,puggles-gpu
```

## Full results

Every cell is the mean over 3 runs (1kb–10mb) or a single run (100mb, 1gb). opt ms shows ± std where there were several runs.

### Multi-objective

| size | library | load ms ↓ | opt ms ↓ | µs/eval ↓ | evals/s ↑ | CPU s ↓ | peak RSS MB ↓ | HV ↑ | front size | vs pymoo (opt time) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | puggles (native, sequential) | 1.4 | 20.8 ±4 | 2.08 | 481,541 | 0.02 | 12 | 0.9320 | 100 | 19.93× |
| 1kb | puggles (native, 8 threads) | 1.0 | 23.3 ±1 | 2.33 | 429,799 | 0.09 | 12 | 0.9320 | 100 | 17.79× |
| 1kb | puggles (native, GPU) | 0.8 | 190.2 ±24 | 19.02 | 52,585 | 0.05 | 20 | 0.9321 | 100 | 2.18× |
| 1kb | puggles (py) | 12.2 | 54.4 ±2 | 5.44 | 183,711 | 0.21 | 82 | 0.9320 | 100 | 7.61× |
| 1kb | puggles (py, batch) | 2.2 | 28.0 ±2 | 2.80 | 356,718 | 0.19 | 82 | 0.9320 | 100 | 14.77× |
| 1kb | pymoo | 2.4 | 414.0 ±59 | 41.40 | 24,157 | 0.54 | 113 | 0.9309 | 99 | 1.00× |
| 1kb | platypus | 2.3 | 1,116.9 ±42 | 111.69 | 8,954 | 1.23 | 82 | 0.8816 | 97 | 0.37× |
| 1kb | DEAP | 2.2 | 881.9 ±13 | 88.19 | 11,340 | 1.05 | 86 | 0.8708 | 84 | 0.47× |
| 1mb | puggles (native, sequential) | 3.6 | 228.5 ±6 | 22.85 | 43,770 | 0.23 | 17 | 0.7241 | 100 | 1.81× |
| 1mb | puggles (native, 8 threads) | 2.7 | 74.8 ±2 | 7.48 | 133,630 | 0.41 | 17 | 0.7241 | 100 | 5.54× |
| 1mb | puggles (native, GPU) | 2.9 | 211.5 ±16 | 21.15 | 47,289 | 0.06 | 25 | 0.7253 | 100 | 1.96× |
| 1mb | puggles (py) | 5.2 | 436.7 ±3 | 43.67 | 22,901 | 0.59 | 87 | 0.7241 | 100 | 0.95× |
| 1mb | puggles (py, batch) | 3.9 | 61.5 ±1 | 6.15 | 162,514 | 0.20 | 91 | 0.7241 | 100 | 6.73× |
| 1mb | pymoo | 5.5 | 414.2 ±1 | 41.42 | 24,141 | 0.57 | 123 | 0.7320 | 100 | 1.00× |
| 1mb | platypus | 5.0 | 1,484.4 ±18 | 148.44 | 6,737 | 1.63 | 88 | 0.7241 | 100 | 0.28× |
| 1mb | DEAP | 4.4 | 1,279.9 ±21 | 127.99 | 7,813 | 1.45 | 91 | 0.7133 | 100 | 0.32× |
| 10mb | puggles (native, sequential) | 19.7 | 2,205.5 ±130 | 220.55 | 4,534 | 2.21 | 42 | 0.7539 | 100 | 0.50× |
| 10mb | puggles (native, 8 threads) | 15.2 | 556.7 ±88 | 55.67 | 17,963 | 3.31 | 42 | 0.7539 | 100 | 1.97× |
| 10mb | puggles (native, GPU) | 16.7 | 508.7 ±55 | 50.87 | 19,659 | 0.14 | 62 | 0.7433 | 100 | 2.16× |
| 10mb | puggles (py) | 28.5 | 4,434.3 ±704 | 443.43 | 2,255 | 4.26 | 110 | 0.7539 | 100 | 0.25× |
| 10mb | puggles (py, batch) | 13.3 | 787.4 ±44 | 78.74 | 12,699 | 0.93 | 145 | 0.7539 | 100 | 1.39× |
| 10mb | pymoo | 11.1 | 1,097.1 ±67 | 109.71 | 9,115 | 1.26 | 176 | 0.7718 | 100 | 1.00× |
| 10mb | platypus | 12.8 | 4,550.4 ±41 | 455.04 | 2,198 | 4.66 | 110 | 0.7683 | 100 | 0.24× |
| 10mb | DEAP | 11.6 | 4,361.2 ±37 | 436.12 | 2,293 | 4.50 | 110 | 0.7617 | 100 | 0.25× |
| 100mb | puggles (native, sequential) | 324.0 | 4,064.2 | 2,032.10 | 492 | 4.55 | 312 | 0.8064 | 44 | 0.40× |
| 100mb | puggles (native, 8 threads) | 132.9 | 1,812.6 | 906.30 | 1,103 | 10.02 | 313 | 0.8064 | 44 | 0.90× |
| 100mb | puggles (native, GPU) | 139.6 | 775.8 | 387.90 | 2,578 | 0.51 | 446 | 0.8064 | 44 | 2.11× |
| 100mb | puggles (py) | 73.0 | 10,597.5 | 5,298.75 | 189 | 10.69 | 360 | 0.8064 | 44 | 0.15× |
| 100mb | puggles (py, batch) | 101.1 | 1,423.0 | 711.50 | 1,405 | 1.87 | 396 | 0.8064 | 44 | 1.15× |
| 100mb | pymoo | 71.1 | 1,633.6 | 816.80 | 1,224 | 2.02 | 428 | 0.6866 | 37 | 1.00× |
| 100mb | platypus | 71.3 | 10,199.7 | 5,099.85 | 196 | 10.56 | 360 | 0.7326 | 27 | 0.16× |
| 100mb | DEAP | 127.2 | 9,985.3 | 4,992.65 | 200 | 10.36 | 361 | 0.7673 | 59 | 0.16× |
| 1gb | puggles (native, sequential) | 3,993.6 | 44,587.0 | 22,293.50 | 45 | 48.35 | 1,475 | 0.7358 | 38 | 0.33× |
| 1gb | puggles (native, 8 threads) | 3,856.5 | 20,386.0 | 10,193.00 | 98 | 128.99 | 2,014 | 0.7358 | 38 | 0.72× |
| 1gb | puggles (native, GPU) | 4,005.9 | 6,823.1 | 3,411.55 | 293 | 6.05 | 1,664 | 0.7358 | 38 | 2.16× |
| 1gb | puggles (py) | 1,286.7 | 101,228.6 | 50,614.30 | 20 | 103.03 | 2,560 | 0.7358 | 38 | 0.15× |
| 1gb | puggles (py, batch) | 1,795.3 | 13,797.4 | 6,898.70 | 145 | 17.67 | 1,260 | 0.7358 | 38 | 1.07× |
| 1gb | pymoo | 1,751.8 | 14,771.1 | 7,385.55 | 135 | 18.09 | 1,560 | 0.6681 | 30 | 1.00× |
| 1gb | platypus | 2,026.5 | 119,714.5 | 59,857.25 | 17 | 113.53 | 1,486 | 0.6922 | 33 | 0.12× |
| 1gb | DEAP | 2,047.4 | 105,989.1 | 52,994.55 | 19 | 106.64 | 1,487 | 0.7727 | 28 | 0.14× |

### Single-objective

| size | library | load ms ↓ | opt ms ↓ | µs/eval ↓ | evals/s ↑ | CPU s ↓ | peak RSS MB ↓ | best Sharpe ↑ | vs pymoo (opt time) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1kb | puggles (native, sequential) | 0.8 | 9.5 ±1 | 0.95 | 1,056,338 | 0.01 | 11 | 3,375.4314 | 41.49× |
| 1kb | puggles (native, 8 threads) | 0.8 | 14.6 ±1 | 1.46 | 683,371 | 0.06 | 12 | 3,375.4314 | 26.84× |
| 1kb | puggles (native, GPU) | 0.9 | 168.6 ±12 | 16.86 | 59,312 | 0.04 | 20 | 3,507.3657 | 2.33× |
| 1kb | genevo | 1.2 | 4.8 ±0 | 0.48 | 2,085,517 | 0.02 | 11 | 11.6995 | 81.26× |
| 1kb | genetic_algorithm | 0.9 | 2.5 ±0 | 0.26 | 3,894,800 | 0.00 | 11 | 248.9195 | 157.09× |
| 1kb | puggles (py) | 2.7 | 48.3 ±2 | 4.83 | 206,897 | 0.21 | 81 | 3,375.4314 | 8.13× |
| 1kb | puggles (py, batch) | 2.5 | 17.7 ±0 | 1.77 | 563,910 | 0.17 | 81 | 3,375.4314 | 22.15× |
| 1kb | pymoo | 2.6 | 392.7 ±19 | 39.27 | 25,463 | 0.55 | 113 | 9,738.6520 | 1.00× |
| 1kb | platypus | 2.3 | 512.4 ±22 | 51.24 | 19,515 | 0.65 | 82 | 318.2844 | 0.77× |
| 1kb | DEAP | 2.4 | 149.1 ±4 | 14.91 | 67,084 | 0.31 | 85 | 2,748.5724 | 2.63× |
| 1mb | puggles (native, sequential) | 6.2 | 216.9 ±8 | 21.69 | 46,111 | 0.21 | 16 | 0.0818 | 1.91× |
| 1mb | puggles (native, 8 threads) | 2.8 | 61.2 ±1 | 6.12 | 163,310 | 0.38 | 17 | 0.0818 | 6.77× |
| 1mb | puggles (native, GPU) | 2.9 | 178.8 ±8 | 17.88 | 55,918 | 0.05 | 25 | 0.0818 | 2.32× |
| 1mb | genevo | 3.6 | 155.8 ±7 | 15.46 | 64,698 | 0.26 | 17 | 0.0791 | 2.66× |
| 1mb | genetic_algorithm | 2.8 | 215.8 ±22 | 22.07 | 45,319 | 0.21 | 16 | 0.0818 | 1.92× |
| 1mb | puggles (py) | 4.0 | 433.9 ±20 | 43.39 | 23,049 | 0.57 | 86 | 0.0818 | 0.96× |
| 1mb | puggles (py, batch) | 4.3 | 66.8 ±24 | 6.68 | 149,701 | 0.20 | 91 | 0.0818 | 6.20× |
| 1mb | pymoo | 4.0 | 414.4 ±38 | 41.44 | 24,131 | 0.55 | 123 | 0.0818 | 1.00× |
| 1mb | platypus | 4.2 | 886.7 ±30 | 88.67 | 11,278 | 1.02 | 87 | 0.0818 | 0.47× |
| 1mb | DEAP | 4.4 | 542.0 ±8 | 54.20 | 18,451 | 0.69 | 91 | 0.0818 | 0.76× |
| 10mb | puggles (native, sequential) | 18.6 | 2,095.0 ±60 | 209.50 | 4,773 | 2.10 | 41 | 0.0552 | 0.53× |
| 10mb | puggles (native, 8 threads) | 15.7 | 542.2 ±59 | 54.22 | 18,442 | 3.28 | 42 | 0.0552 | 2.06× |
| 10mb | puggles (native, GPU) | 17.3 | 512.8 ±77 | 51.28 | 19,501 | 0.12 | 62 | 0.0552 | 2.18× |
| 10mb | genevo | 16.7 | 1,513.6 ±78 | 150.16 | 6,660 | 2.28 | 42 | 0.0532 | 0.74× |
| 10mb | genetic_algorithm | 16.1 | 2,123.4 ±78 | 217.54 | 4,597 | 2.12 | 41 | 0.0552 | 0.53× |
| 10mb | puggles (py) | 24.1 | 3,847.3 ±502 | 384.73 | 2,599 | 3.84 | 110 | 0.0552 | 0.29× |
| 10mb | puggles (py, batch) | 11.9 | 725.7 ±39 | 72.57 | 13,779 | 0.88 | 145 | 0.0552 | 1.54× |
| 10mb | pymoo | 11.2 | 1,115.6 ±71 | 111.56 | 8,964 | 1.26 | 176 | 0.0552 | 1.00× |
| 10mb | platypus | 12.1 | 4,692.7 ±279 | 469.27 | 2,131 | 4.70 | 110 | 0.0552 | 0.24× |
| 10mb | DEAP | 11.8 | 3,698.7 ±45 | 369.87 | 2,704 | 3.82 | 110 | 0.0552 | 0.30× |
| 100mb | puggles (native, sequential) | 311.7 | 4,183.8 | 2,091.90 | 478 | 4.52 | 312 | 0.0529 | 0.41× |
| 100mb | puggles (native, 8 threads) | 134.8 | 1,410.1 | 705.05 | 1,418 | 9.52 | 313 | 0.0529 | 1.21× |
| 100mb | puggles (native, GPU) | 171.0 | 729.2 | 364.60 | 2,743 | 0.50 | 446 | 0.0529 | 2.33× |
| 100mb | genevo | 152.2 | 3,811.4 | 1,764.54 | 567 | 5.84 | 312 | 0.0500 | 0.45× |
| 100mb | genetic_algorithm | 166.1 | 4,193.6 | 2,103.11 | 475 | 4.53 | 312 | 0.0526 | 0.41× |
| 100mb | puggles (py) | 145.2 | 9,849.8 | 4,924.90 | 203 | 10.23 | 360 | 0.0529 | 0.17× |
| 100mb | puggles (py, batch) | 101.8 | 1,349.5 | 674.75 | 1,482 | 1.81 | 396 | 0.0529 | 1.26× |
| 100mb | pymoo | 135.3 | 1,700.4 | 850.20 | 1,176 | 2.12 | 426 | 0.0523 | 1.00× |
| 100mb | platypus | 74.9 | 10,659.0 | 5,329.50 | 188 | 10.74 | 360 | 0.0511 | 0.16× |
| 100mb | DEAP | 157.1 | 10,208.2 | 5,104.10 | 196 | 10.60 | 360 | 0.0527 | 0.17× |
| 1gb | puggles (native, sequential) | 3,636.6 | 44,907.4 | 22,453.70 | 45 | 47.40 | 1,641 | 0.0526 | 0.38× |
| 1gb | puggles (native, 8 threads) | 3,789.5 | 21,994.1 | 10,997.05 | 91 | 128.72 | 1,949 | 0.0526 | 0.77× |
| 1gb | puggles (native, GPU) | 3,776.2 | 8,758.6 | 4,379.30 | 228 | 5.20 | 1,590 | 0.0526 | 1.93× |
| 1gb | genevo | 2,070.5 | 37,275.7 | 17,257.27 | 58 | 66.66 | 2,290 | 0.0489 | 0.45× |
| 1gb | genetic_algorithm | 3,526.4 | 43,150.5 | 21,804.19 | 46 | 46.30 | 2,051 | 0.0523 | 0.39× |
| 1gb | puggles (py) | 2,129.9 | 106,048.8 | 53,024.40 | 19 | 105.90 | 1,313 | 0.0526 | 0.16× |
| 1gb | puggles (py, batch) | 1,709.7 | 14,769.1 | 7,384.55 | 135 | 18.42 | 1,462 | 0.0526 | 1.14× |
| 1gb | pymoo | 2,168.4 | 16,872.2 | 8,436.10 | 119 | 19.26 | 1,369 | 0.0525 | 1.00× |
| 1gb | platypus | 2,120.9 | 103,813.0 | 51,906.50 | 19 | 105.23 | 1,379 | 0.0528 | 0.16× |
| 1gb | DEAP | 1,878.3 | 105,700.3 | 52,850.15 | 19 | 104.66 | 1,332 | 0.0530 | 0.16× |

