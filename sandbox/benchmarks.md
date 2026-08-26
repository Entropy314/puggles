# rustypus Benchmark Analysis

**Machine:** Apple M3 — 8 logical CPU cores, 10-core GPU (Metal)
**Rust:** release profile with `lto = true`, `codegen-units = 1`, rustc 1.94
**Python (.venv):** polars-py 1.40.1 · numpy 2.4.6 · pymoo 0.6.1.6 · DEAP 1.4 · platypus-opt 1.4.1
**Rust GA crates (§5):** genevo 0.7.1 · genetic_algorithm 0.27.2 (both single-objective)
**Date:** 2026-07-15

> **Optimized build.** These numbers reflect several performance changes: (1) `CrossoverManager`
> now calls each operator **once per pair** instead of once per gene (was O(D²) work + ~2D
> full-solution clones per pair); (2) parent selection reuses the ranks from environmental
> selection, so there is **one non-dominated sort per generation instead of two**; (3)
> survivors are **moved** out of the combined population and parents are indexed in place —
> no per-generation `Solution` clones for selection; (4) the RNG is `SmallRng` (xoshiro) rather
> than `StdRng` (ChaCha); and (5) `lto = true` + `codegen-units = 1`. Combined, the optimizer is
> **~3.8× faster** on the portfolio problem (176 ms → 46 ms) and **~2.3×** on ZDT1 (62 → 27 ms),
> with identical solution quality (IGD unchanged) and memory.
>
> **rustypus is measured two ways.** *native* rows run the pure-Rust examples; *py* rows run the
> restored pyo3/maturin Python bindings (`python/` crate) with a **Python-callable objective** —
> apples-to-apples with how pymoo/platypus/DEAP are used. Every eval crosses the FFI boundary and
> re-acquires the GIL, so `py` runs always execute Sequential; it is still far faster than the
> other Python libraries because the GA machinery (sort/select/variation) stays in Rust. Install
> the bindings with `cd python && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release`.

**Run:**

```bash
cd sandbox
# Build + install the Python bindings once (for every "rustypus (py)" row):
( cd ../python && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 ../sandbox/.venv/bin/maturin develop --release )

cargo run --release --example bench_zdt1                       # §1 + §2 (CPU, native)
.venv/bin/python bench_python.py                               # §1 (5-way: +rustypus-py/pymoo/platypus/DEAP)
cargo run --release --features gpu --example bench_gpu         # §2 (GPU crossover)
cargo run --release --example bench_large_data                 # §3a pipeline timing + memory
cargo run --release --example bench_mem                        # §3b optimizer RSS (rustypus native, isolated)
for l in rustypus pymoo platypus deap; do .venv/bin/python bench_opt_mem.py $l; done  # §3b optimizer RSS (isolated)
.venv/bin/python bench_dtlz.py                                 # §4 (DTLZ2 3-obj 5-way: time + HV + IGD)
cargo run --release --example bench_singleobj                  # §5 (single-obj: rustypus vs genevo vs genetic_algorithm)
```

---

## Contents

1. [ZDT1 NSGA-II — Algorithm Comparison](#1-zdt1-nsga-ii--algorithm-comparison)
2. [Objective Cost — When Parallelism and the GPU Win](#2-objective-cost--when-parallelism-and-the-gpu-win)
3. [Large Dataset Pipeline — 10 KB → 1 GB](#3-large-dataset-pipeline)
   - [3a. Preprocessing: CSV load + covariance](#3a-preprocessing-csv-load--covariance)
   - [3b. Optimizer comparison (size-invariant)](#3b-optimizer-comparison-size-invariant)
4. [Many-Objective: DTLZ2 (3 objectives)](#4-many-objective-dtlz2)
5. [Single-Objective: rustypus vs genevo vs genetic_algorithm](#5-single-objective)
6. [Bugs Fixed During Benchmarking](#6-bugs-fixed)
7. [Summary and Guidance](#7-summary-and-guidance)

---

## 1. ZDT1 NSGA-II — Algorithm Comparison

**Problem:** ZDT1 — canonical two-objective benchmark, N=10 continuous variables in [0,1].
`f1 = x1`, `g = 1 + 9/(N−1)·Σ(x2…xN)`, `f2 = g·(1 − √(f1/g))`. True front: `x2…xN = 0`, `f2 = 1 − √f1`.

**Settings:** pop=100 · NFE=10,000 · 5 runs (Rust) / 3 runs (Python) · 1 warm-up
**Metric:** IGD — mean distance from 100 reference points on the true front to the nearest archive point. Lower is better.

| library | ms / run | ±std | IGD | ±std |
| --- | --- | --- | --- | --- |
| **rustypus (native, sequential)** | **27.1** | 2.1 | 0.0050 | 0.0002 |
| rustypus (native, parallel) | 35.5 | 0.2 | 0.0051 | 0.0001 |
| **rustypus (py)** | **31.8** | 3.1 | 0.0051 | 0.0003 |
| pymoo | 215.4 | 12.8 | 0.0045 | 0.0004 |
| DEAP | 843.4 | 4.2 | 0.0048 | 0.0003 |
| platypus | 1062.1 | 3.9 | 0.0108 | 0.0108 |

**Convergence:** all libraries reach IGD ≈ 0.005 — identical solution quality (platypus shows one poorly-converged run in this batch, inflating its mean/std; its median is ≈ 0.005). ZDT1 converges for any correct NSGA-II within 10,000 NFE.

**Why rustypus is fastest (27 ms vs pymoo 215 ms — ~8×):** ZDT1 costs ~30–70 ns per evaluation, so the ~0.7 ms of actual objective math is <1% of runtime. Everything else is **algorithm overhead** — non-dominated sort, crowding distance, selection, variation, allocation — which compiled Rust does with far less per-item cost than pymoo's NumPy batch dispatch, and ~13–16× less than the pure-Python loops in platypus/DEAP.

**rustypus (py) — 31.8 ms, ~7× faster than pymoo — despite a Python objective.** The bindings call a Python `lambda` per evaluation (FFI + GIL re-acquire each time), yet the whole GA loop stays in Rust, so it lands within ~1.2× of native rustypus and still crushes the other Python libraries. This is the fair "use rustypus from Python" number; a native Rust objective (no per-eval GIL) matches the native row.

**Why parallel ≈ sequential here:** Rayon's task spawn/join (~1–10 µs) dwarfs a 30 ns objective call, so threading only adds overhead. This is the **algorithm-dominated regime** — see §2 for the opposite.

---

## 2. Objective Cost — When Parallelism and the GPU Win

### 2a. Expensive objective on the CPU

An O(N²) coupling simulation (~300 µs/eval) puts the run in the **evaluation-dominated regime**, where Rayon pays off (same GA settings as §1, timing only):

| mode | ms / run | ±std |
| --- | --- | --- |
| rustypus (sequential) | 130.1 | 0.1 |
| **rustypus (parallel)** | **58.7** | 0.3 |

**2.2× on 8 cores.** Sub-linear because the serial algorithm overhead (§1) doesn't parallelize; net evaluation time alone scales ~3×. **Python cannot do this** — the GIL blocks threads and multiprocessing serialization dominates at this granularity (estimated ≥ 30× slower).

### 2b. CPU vs GPU across objective cost

Same objective on CPU (Rust) and GPU (generated WGSL compute shader), sweeping per-evaluation cost via dimension D and inner work W (pop=256, NFE=20,000). Objective: `f1 = Σᵢ Σ_{k<W} cos(xᵢ·(k+1))`, `f2 = Σᵢ (xᵢ−0.5)²`. Device auto-selected: **Apple M3 (Metal)**.

| config | dims · work | seq (ms) | par (ms) | gpu (ms) | par vs seq | gpu vs par |
| --- | --- | --- | --- | --- | --- | --- |
| cheap | D=20 · W=1 | 156 | 136 | 254 | 1.15× | 1.9× slower |
| light | D=50 · W=16 | 216 | 173 | 268 | 1.25× | 1.6× slower |
| medium | D=100 · W=128 | 1,104 | 386 | 332 | 2.86× | 1.2× faster |
| heavy | D=200 · W=512 | 7,038 | 1,665 | 697 | 4.23× | 2.4× faster |
| v.heavy | D=400 · W=2048 | 52,001 | 11,572 | 2,980 | 4.49× | 3.9× faster |

Two crossovers as the objective gets heavier: CPU **parallel** pulls further ahead of sequential (1.15× → 4.5× on 8 cores), and the **GPU** goes from losing (overhead-bound) to winning decisively — **up to 3.9× over the 8-core CPU** (≈17× over single-thread) at v.heavy. The crossover point is around D=100.

(These are much lower than the pre-optimization report because the O(D²)→O(D) crossover fix runs on the CPU in *every* path — seq, par, and the GPU run's host-side variation — so all three columns dropped, most at high D. The GPU column dropped the most, which is why GPU now wins clearly: with host-side variation no longer dominating, the run is bounded by evaluation, exactly what the GPU accelerates. This table is from the crossover-fix build; the later round (SmallRng, one sort/gen, fewer clones) lowers the CPU columns a further ~10–15%, nudging the crossover point slightly later — the qualitative story is unchanged.)

**GPU + data-driven objectives:** `GpuEvaluator::new_blocking_with_data(...)` now uploads read-only data bound at `@group(0) @binding(3)`, so a data-driven objective (e.g. the §3 portfolio's 20×20 covariance) *can* run on the GPU — though for that tiny objective the CPU still wins (per §2b, GPU only pays off once per-evaluation compute is large).

**Threshold rule:** use `ExecutionMode::MultiThreaded` when evaluation exceeds ~100 µs; below that, Rayon overhead erases the benefit. Reach for GPU only when per-evaluation compute is large *and* readback is amortized.

---

## 3. Large Dataset Pipeline

**Problem:** portfolio optimization — minimize variance and maximize return over 20 assets, weights normalized to sum to 1 inside the objective. **Pipeline:** CSV load → 20×20 covariance → NSGA-II (pop=200, NFE=10,000).

Datasets are synthetic single-factor daily returns:

| label | rows | file size |
| --- | --- | --- |
| 10kb | 53 | 11 KB |
| 1mb | 5,500 | 1.1 MB |
| 10mb | 55,000 | 11 MB |
| 100mb | 549,000 | 105 MB |
| 1gb | 5,600,000 | 1.07 GB |

Generate: `head -n 54 data/returns_1mb.csv > data/returns_10kb.csv` (10kb), `scripts/gen_datasets.py` (1–100 MB), `scripts/gen_1gb.py` (1 GB). Memory figures are peak RSS sampled every 10 ms (sysinfo in Rust, psutil in Python).

### 3a. Preprocessing: CSV load + covariance

This is the only stage that scales with dataset size. Rust uses polars(Rust) + a serial covariance loop; Python uses polars(Python) + numpy BLAS `cov`.

| size | rows | Python load (ms) | Python cov (ms) | Rust load (ms) | Rust cov (ms) |
| --- | --- | --- | --- | --- | --- |
| 10kb | 53 | 16.7 | 49.4 | 4.9 | 0.1 |
| 1mb | 5,500 | 12.0 | 12.6 | 4.1 | 1.1 |
| 10mb | 55,000 | 16.9 | 12.7 | 25.9 | 10.4 |
| 100mb | 549,000 | 82.0 | 49.1 | 237.6 | 101.3 |
| 1gb | 5,600,000 | 622.9 | 1091.4 | 3546.8 | 1186.2 |

- Small-dataset Python numbers (10kb/1mb) are interpreter/numpy warm-up, not real work.
- At 1 GB, Python-polars `read_csv` (eager, multithreaded) loads faster than the Rust example's `LazyCsvReader().collect()` (which also reads the date column) — a reader-config artifact on the same polars core, not a fundamental gap. Covariance is comparable (Rust 1.14 s vs numpy 1.09 s). Parallelizing the Rust cov loop and skipping the date column would close both gaps.

**Pipeline memory scales with the dataset, not the optimizer.** Peak RSS (whole Rust process, sampled every 5 ms) at each stage:

| size | load (MB) | cov (MB) | opt stage (MB) |
| --- | --- | --- | --- |
| 10kb | 8 | 11 | 13 |
| 1mb | 13 | 17 | 18 |
| 10mb | 39 | 29 | 36 |
| 100mb | 259 | 243 | 243 |
| 1gb | 2105 | 1841 | 1789 |

The "opt stage" RSS retains the loaded returns matrix (the process doesn't free it), so it tracks
dataset size — but that memory is the *DataFrame*, not the optimizer: run in isolation the
optimizer needs only ~8 MB (see §3b). At 1 GB the process peaks around ~2 GB (the 1.07 GB CSV
expands in memory during parsing).

### 3b. Optimizer comparison (size-invariant)

The optimizer never touches raw returns — it only sees the **20×20 covariance and 20 means**, so optimization time is **independent of dataset size**. Native Rust confirms it directly: sequential opt is 45.3 / 44.4 / 45.0 / 44.7 / 43.9 ms at 10kb / 1mb / 10mb / 100mb / 1gb (flat within noise); parallel likewise ~50 ms.

Head-to-head on the identical problem (pop 200, NFE 10,000). **Each optimizer runs in its own
process on a synthetic 20×20 covariance (no dataset load)**, so peak RSS is a clean
"runtime + optimizer working set" figure — not contaminated by a shared process that loaded a
big DataFrame. Drivers: [`bench_mem.rs`](examples/bench_mem.rs) (Rust) and
[`bench_opt_mem.py`](bench_opt_mem.py) (one Python process per library).

| optimizer | opt time (ms) | peak RSS (MB) | vs rustypus (time) |
| --- | --- | --- | --- |
| **rustypus (native, sequential)** | **46.9** | **8** | 1.0× |
| **rustypus (native, parallel)** | **49.5** | **9** | 1.1× |
| rustypus (py) | 73.9 | 36 | 1.6× slower |
| pymoo (NSGA2, vectorized numpy) | 217.4 | 72 | 4.6× slower |
| DEAP (NSGA-II) | 1681.2 | 42 | 36× slower |
| platypus (NSGA-II) | 1700.2 | 38 | 36× slower |

**rustypus (native) is ~4.6× faster than pymoo and ~36× faster than the pure-Python libraries — at
~5–9× less memory** (8 MB vs 38–72 MB). The Python figures are dominated by the interpreter +
numpy baseline; pymoo's vectorized batch evaluation allocates the largest arrays (72 MB), while
the pure-Python loops in platypus/DEAP stay smaller (38–42 MB) but pay ~36× in time. As in §1,
sequential ≈ parallel — a 20-variable quadratic is far too cheap for threading to help.

**rustypus (py) — 74 ms / 36 MB.** Still ~3× faster than pymoo from Python. Its 36 MB is the
numpy-backed objective's baseline (the portfolio objective calls numpy, so numpy is imported into
the process), *not* the optimizer: the optimizer working set is the same ~8 MB as native — the
extra memory is the shared numpy/interpreter cost every Python library here also pays.

---

## 4. Many-Objective: DTLZ2

**Problem:** DTLZ2 — a canonical *three-objective* benchmark. n=12 variables in [0,1]; the first
M−1 are angles, the remaining k=10 drive the distance term `g = Σ(xᵢ−0.5)²`. The true Pareto
front is the unit-sphere first octant (`Σ fᵢ² = 1`). Standard formulation used identically across
all libraries — native Rust ([`bench_dtlz.rs`](examples/bench_dtlz.rs)) and Python
([`bench_dtlz.py`](bench_dtlz.py)). (The library's built-in `dtlz2` couples its objective count to
n and is *not* used here; the benchmark uses the standard M=3 form so fronts are comparable.)

**Settings:** pop=100 · NFE=20,000 · 3 runs · 1 warm-up
**Metrics — computed uniformly for every library** with pymoo's indicators against the true front:
**HV** (hypervolume, ref point `[1.1,1.1,1.1]`, higher = better) and **IGD** (lower = better).

| library | ms / run | ±std | HV ↑ | IGD ↓ |
| --- | --- | --- | --- | --- |
| **rustypus (native)** | **71.2** | 2.1 | 0.6857 | 0.0739 |
| **rustypus (py)** | **104.1** | 1.2 | 0.6797 | 0.0772 |
| pymoo | 460.0 | 21.1 | 0.7029 | 0.0716 |
| platypus | 2549.4 | 4.4 | 0.7042 | 0.0733 |
| DEAP | 1811.1 | 39.6 | 0.7043 | 0.0751 |

**Speed:** rustypus stays fastest — native **~6.5× faster than pymoo** and ~25–36× faster than
platypus/DEAP; the Python-callable binding is still ~4.4× faster than pymoo.

**Quality — the honest trade-off:** on three objectives rustypus's HV/IGD is slightly *behind*
(HV 0.686 vs ~0.704). NSGA-II ranks a many-objective population by crowding distance, which
diversifies less effectively as objectives grow; pymoo/platypus/DEAP reach a marginally better
spread. rustypus buys a large speed win for a small quality cost here. For genuinely
many-objective problems the library ships **NSGA-III** (`src/nsga3.rs`, reference-point based),
which is designed to close exactly this gap — not yet exposed through the Python bindings.

---

## 5. Single-Objective — rustypus vs genevo vs genetic_algorithm

Everything above is **multi-objective** (NSGA-II, Pareto fronts). Two popular Rust GA crates,
[**genevo**](https://docs.rs/genevo) 0.7 and
[**genetic_algorithm**](https://docs.rs/genetic_algorithm) 0.27, are **single-objective only** —
scalar fitness, no Pareto/NSGA machinery (genetic_algorithm's own docs say *"for multiple
objectives, combine them into a weighted sum"*). They therefore **cannot** run ZDT1/DTLZ2/the
portfolio front; comparing them there would be meaningless. So this section drops to a level playing
field: minimize the **Rastrigin** function (continuous, N=10, global minimum 0 at the origin), which
all three can do — rustypus runs its NSGA-II with a single objective.

**Apples-to-apples budget.** The fair budget for a GA comparison is a fixed number of *actual
objective-function evaluations* (NFE) — **not** generations, because `genetic_algorithm` caches
fitness (unchanged chromosomes aren't re-evaluated), so equal generations would let it do far fewer
real evaluations than the others. A shared atomic counter inside the objective enforces the budget:
rustypus's `run(NFE)` counts its own evaluations, genevo is stepped manually until the counter hits
NFE, and `genetic_algorithm`'s generation count is calibrated to spend ≈ NFE. All three land within
±1% of the same 20,000 evaluations.

**Settings:** N=10 · pop=100 · **budget = 20,000 evaluations** · 5 runs · 1 warm-up.
Driver: [`examples/bench_singleobj.rs`](examples/bench_singleobj.rs).

| library | ms / run | ±std | best f (→ 0) | µs / eval |
| --- | --- | --- | --- | --- |
| genetic_algorithm | **3.1** | 0.0 | 5.98 | **0.16** |
| genevo | 8.6 | 0.3 | 2.19 | 0.43 |
| rustypus | 55.0 | 0.2 | **1.05** | 2.75 |

At an equal evaluation budget the result splits cleanly into **two independent axes**:

- **Quality per evaluation — rustypus wins** (best f ≈ 1.05 vs 2.19 / 5.98). Its real-coded SBX
  crossover + polynomial mutation and rank-based selection extract the most progress from each of the
  20,000 evaluations. `genetic_algorithm`'s default uniform crossover + single-gene mutation explore
  the Rastrigin landscape weakly, so it converts the same budget into the poorest solution.
- **Wall-time per evaluation — rustypus loses badly** (2.75 µs vs 0.43 / 0.16 — ~6–17× more). This is
  the Pareto tax: NSGA-II runs an O(N²) non-dominated sort + crowding-distance pass every generation,
  dead weight when there is only one objective, which a single-objective GA skips entirely.

So the earlier "genevo is faster *and* better" impression was an artifact of an unequal budget
(genevo had been given ~1.8× more evaluations). Equalized, it's a genuine trade-off: **rustypus turns
each evaluation into the best solution but spends the most time per evaluation.** That matters when
evaluations are *expensive* (a real simulation), where wall-time is dominated by the objective, not
the GA bookkeeping — there rustypus's superior quality-per-evaluation can outweigh its overhead. When
evaluations are *cheap* (like Rastrigin), the per-generation overhead dominates and a single-objective
crate is far faster in wall-clock.

**Takeaway:** rustypus is built for *multi-objective* work, where it dominates (§1–§4). For
single-objective problems, a dedicated crate wins on cheap-evaluation throughput — **genetic_algorithm**
fastest per evaluation, **genevo** a balance — while rustypus still finds the best solution per
evaluation. (Not an algorithm-controlled comparison: the *budget, population, and problem* are
identical, but each crate uses its own operator set — no two expose the same one, and
`genetic_algorithm` would likely close the quality gap with stronger real-coded operators.)

---

## 6. Bugs Fixed During Benchmarking

Two correctness bugs surfaced while building these benchmarks (both fixed; the fixes are in `NSGAII`).

### Bug 1 — Wrong default mutation operator

**Symptom:** IGD = **0.84** vs 0.005 for the other libraries on ZDT1 — 180× worse convergence.

**Root cause:** `MutationManager::new()` defaulted Real variables to `UniformMutation` with `probability = 1.0` — every gene replaced by a uniform-random value each generation, making the search completely non-local.

**Fix:** default to `PolynomialMutation(η=20, probability=1/N)`, the standard NSGA-II operator (Deb et al., 2002):

```rust
// src/genetic_algorithms_v2.rs — NSGAII::new()
let n = problem.solution_length;
let mut mutation_manager = MutationManager::new();
mutation_manager.set_default_real_mutation(Arc::new(PolynomialMutation::new(
    Some(1.0 / n as f64),  // per-variable mutation rate = 1/N
    Some(20.0),            // distribution index
)));
```

**Result:** IGD 0.84 → 0.005; runtime also improved (the algorithm now converges to a compact front).

### Bug 2 — Unbounded archive

**Symptom:** after fixing mutation, runtime jumped to **~1,764 ms** on ZDT1.

**Root cause:** `update_archive()` kept *all* non-dominated front-0 solutions. On a well-converged front every pair is mutually non-dominated, so the archive grew ~100/generation → ~9,900 solutions, and `fast_non_dominated_sort` is O(n²) — ~100 M comparisons/generation.

**Fix:** cap the archive at `population_size`, evicting the most crowded solutions (crowding distance) to preserve front diversity. Runtime returned to ~62 ms with IGD held at 0.005.

---

## 7. Summary and Guidance

### Optimizer ranking (portfolio, N=20, pop=200, NFE=10,000; isolated-process memory)

| rank | library | opt time | peak RSS | time relative |
| --- | --- | --- | --- | --- |
| 1 | rustypus (native seq/par) | ~47 ms | **8 MB** | 1.0× |
| 2 | rustypus (py) | ~74 ms | 36 MB | 1.6× |
| 3 | pymoo | ~217 ms | 72 MB | 4.6× |
| 4 | DEAP | ~1,681 ms | 42 MB | 36× |
| 5 | platypus | ~1,700 ms | 38 MB | 36× |

rustypus is both the fastest and by far the leanest — native is **~4.6× faster than pymoo, ~36×
faster than the pure-Python libraries, at ~5–9× less memory**; even called from Python with a
Python objective it stays ~3× ahead of pymoo. Python footprints are dominated by the interpreter +
numpy baseline. On two objectives (ZDT1, portfolio) solution quality is identical across libraries;
on three-plus objectives (DTLZ2, §4) rustypus trades a small quality gap for its speed, which
NSGA-III is meant to close. (Full-pipeline RSS scales with dataset size, but that is the loaded
DataFrame, not the optimizer — see §3a/§3b.)

**Scope note — this is a multi-objective story.** All of the above compares rustypus against other
*multi-objective* optimizers, its design point. For **single-objective** problems (§5), at an equal
20,000-evaluation budget the axes split: rustypus reaches the **best** solution per evaluation
(Rastrigin best f ≈ 1.05 vs genevo 2.19, genetic_algorithm 5.98) but is the **slowest** in wall-time
(~6–17× more µs/evaluation), because NSGA-II's per-generation Pareto sort is dead weight on one
objective. Pick the tool to the problem: rustypus for Pareto/multi-objective (or single-objective
where each evaluation is expensive), a dedicated single-objective crate when evaluations are cheap and
wall-clock throughput matters.

### Execution-mode guidance

| scenario | mode | reason |
| --- | --- | --- |
| eval < ~100 µs (e.g. ZDT1, 20-asset portfolio) | `Sequential` | Rayon/GPU overhead exceeds the work |
| eval > ~100 µs (expensive simulation) | `MultiThreaded` | ~2.2× on 8 cores here (up to ~4.5× for heavier evals); Python's GIL can't compete |
| eval very heavy *and* self-contained | `GPU` (`--features gpu`) | wins once compute amortizes per-generation readback |

### Dataset size vs. work

Scaling the dataset 100× changes **preprocessing** time but leaves **optimization** time flat (±3%) — the optimizer only ever sees a 20×20 covariance. Investing in a faster optimizer (rustypus over platypus: ~38×) pays off equally at every dataset size.

### Standard NSGA-II operator defaults (applied by `NSGAII::new()`)

| operator | parameter | value |
| --- | --- | --- |
| `PolynomialMutation` | probability / η | `1/N` / 20 |
| `SimulatedBinaryCrossover` | probability / η | 1.0 / 20 |
| Archive | max size | `population_size` |
