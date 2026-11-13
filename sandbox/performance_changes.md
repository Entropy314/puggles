# puggles Performance Changes, November 2026


This report covers what changed in puggles between 2026-11-06 and 2026-11-13, where it changed, and why each change makes puggles faster. All numbers come from the dataset benchmark ([`benchmarks_datasets.md`](benchmarks_datasets.md), raw data in [`results_datasets.json`](results_datasets.json)): a portfolio problem on real stock returns from 1 KB to 1 GB, on an Apple M3 (8 CPU cores, 10-core GPU, 16 GB), with the same evaluation budget for every library.

## Summary


- **The GPU path is 4.3× faster at 1 GB (6.6 s → 1.5 s).** It is now the fastest option from 10 MB up, and **8.8× faster than pymoo** and about 60× faster than platypus and DEAP at 1 GB. The fix was reading the dataset once per generation instead of once per solution.
- **Any GPU or accelerator backend can now plug into puggles** through a new `BatchEvaluator` trait, without changing the GA. CubeCL and CUDA backends use it.
- **Python↔Rust overhead is down about 1.3× on small data.** Batch objectives can exchange NumPy arrays instead of nested lists (`batch_numpy=True`).
- **Python-written per-solution objectives can now use all CPU cores.** A Numba `@cfunc` runs from Rust without the GIL (`objective_cfunc`). That's 2.3–7.8× faster than the same objective as a plain Python function, and within about 15% of native Rust threads.
- **PyTorch and JAX objectives work through the batch interface with no puggles changes.** At 1 GB, Torch on the Apple GPU takes 4.7–4.8 s (2.8–2.9× faster than pymoo) and JAX on the CPU takes 4.3–6.0 s (2.3–3.2× faster).
- **GPU kernels can be written in Rust** (CubeCL), within about 1.1–1.2× of hand-written WGSL. A CUDA backend with Rust kernels (Rust-CUDA) now builds and runs on an NVIDIA L4: 2.6 s at 1 GB, 12× faster than pymoo, with results matching the CPU (see [`report_cuda_and_cloud.md`](report_cuda_and_cloud.md)).

## 1. Changes in the core library


### 1.1 `BatchEvaluator`: one extension point for device backends

**Where:**

- the trait, [src/core.rs:52](../src/core.rs#L52),
- `with_batch_evaluator`, [src/genetic_algorithms_v2.rs:247](../src/genetic_algorithms_v2.rs#L247) and [src/nsga3.rs:134](../src/nsga3.rs#L134),
- routing through a private `device_evaluator()` accessor, [src/genetic_algorithms_v2.rs:253](../src/genetic_algorithms_v2.rs#L253).

**What:** NSGA-II and NSGA-III used to accept exactly one device type, the built-in wgpu `GpuEvaluator`. They now accept anything that implements `evaluate_batch`, `device_description` and `evaluated_count`. The existing `with_gpu_evaluator` API is unchanged.

**Why it matters for performance:**

- Evaluation is where the time goes on large problems. This makes faster evaluators, such as CubeCL, CUDA, or future XLA/PJRT backends, a drop-in change instead of a fork of the GA.
- It also lets those backends live in separate crates. That matters for Rust-CUDA, whose git-only dependencies can't be published to crates.io.

**Check:** a CPU mock evaluator test, [src/genetic_algorithms_v2.rs:1007](../src/genetic_algorithms_v2.rs#L1007), confirms every evaluation is routed through the attached evaluator.

### 1.2 GPU: map-reduce ("partials") mode

**Where:** `GpuEvaluator::with_partials`, [src/gpu_evaluator.rs:228](../src/gpu_evaluator.rs#L228), documented at [src/gpu_evaluator.rs:37](../src/gpu_evaluator.rs#L37). The shared CPU reduction is `reduce_partials`, [src/core.rs:64](../src/core.rs#L64).

**What:**

- The old mode gave each solution its own workgroup, so each solution streamed the whole dataset.
- In the new mode, workgroup *b* owns a block of rows and thread *p* owns solution *p*. Each tile of rows is loaded once into workgroup memory and reused by every solution. The GPU writes per-block partial sums (Σr, Σr²).
- The CPU then adds the blocks in f64 and a `finish` function turns the sums into objectives.

**Why it matters:**

- An objective that scans data is limited by memory traffic, not arithmetic. At 1 GB, per-solution evaluation reads about 90 GB per generation, because each of 100 solutions reads the full 0.9 GB. The map-reduce mode reads 0.9 GB per generation.
- Summing the blocks in f64 on the CPU keeps accuracy even though the GPU computes in f32. Results match the CPU path exactly for the same seed.

**A second, equally important detail:** each solution's 20 weights must live in `vec4` registers, with the dot product unrolled ([sandbox/examples/bench_datasets.rs:90](examples/bench_datasets.rs#L90)). The first version of the tiled kernel used a loop-indexed per-thread array and took **8.0 s** at 1 GB, slower than the old kernel. The `vec4` version took **1.5 s**. Indexing a per-thread array in a loop pushes it out of registers into slow memory.

**Measured** (opt time, same seed, before → after):

| mode | 1 KB | 1 MB | 10 MB | 100 MB | 1 GB |
| --- | --- | --- | --- | --- | --- |
| multi-objective | 177 → 205 ms | 167 → 201 ms | 452 → 202 ms | 612 → 247 ms | **6,572 → 1,524 ms (4.3×)** |
| single-objective | 158 → 192 ms | 164 → 185 ms | 410 → 187 ms | 578 → 128 ms | **6,701 → 1,587 ms (4.2×)** |

Below 1 MB the GPU is dominated by about 150 ms of device and shader setup, so either kernel loses to the CPU there. The small differences at 1 KB and 1 MB are setup noise.

**Check:** an ignored-by-default GPU test, [src/gpu_evaluator.rs:381](../src/gpu_evaluator.rs#L381), checks both modes against a CPU sum.

### 1.3 GPU: request the adapter's real limits

**Where:** [src/gpu_evaluator.rs:129](../src/gpu_evaluator.rs#L129).

**What:** the device now requests `adapter.limits()` instead of wgpu's defaults.

**Why it matters:** wgpu's default caps a storage buffer at 128 MiB. The 1 GB dataset is about 450 MB as f32, so without this change the GPU path couldn't run on large data at all.

## 2. Changes in the Python bindings


### 2.1 `batch_numpy=True`: arrays instead of nested lists

**Where:**

- the batch trampoline, [python/src/py_problem.rs:82](../python/src/py_problem.rs#L82),
- the argument, [python/src/py_problem.rs:282](../python/src/py_problem.rs#L282),
- the new dependency `numpy = "0.29"` (rust-numpy).

**What:**

- **Input (opt-in):** the population is passed as one `(P, n)` float64 array, a single contiguous copy.
- **Output (always accepted):** a 2-D float64 array is read back directly.
- Lists still work, so existing code is unaffected.

**Why it matters:**

- Previously every generation built P·n Python float objects (2,000 per generation here) plus 100 lists. The user's function then called `np.asarray`, and `.tolist()` converted the results back into Python objects.
- That overhead is fixed per generation, so it dominates when the objective is cheap.

**Measured** (opt time):

| row | 1 KB | 1 MB | 10 MB | 100 MB | 1 GB |
| --- | --- | --- | --- | --- | --- |
| NumPy batch, multi | 27.4 → 21.6 ms (1.27×) | 70.0 → 53.9 ms (1.30×) | 681 → 659 ms | 1,287 → 1,253 ms | 13.3 → 15.7 s (noise, single run) |
| NumPy batch, single | 17.1 → 10.7 ms (1.60×) | 50.6 → 43.7 ms (1.16×) | 694 → 645 ms | 1,301 → 1,254 ms | 13.6 → 14.0 s |
| Numba batch, multi | 37.2 → 29.2 ms (1.27×) | 79.4 → 69.1 ms (1.15×) | 392 → 364 ms | 773 → 661 ms | 7.3 → 7.3 s |

The gain is in the 1 KB–1 MB range, where the per-generation hand-off is a large share of the run. At 1 KB, puggles' Python batch path (22 ms) is now close to native single-threaded puggles (19 ms). At 1 GB the objective is almost the entire run, so the hand-off no longer matters.

### 2.2 `objective_cfunc`: GIL-free compiled per-solution objectives

**Where:**

- `CFunc`, which validates the signature, [python/src/py_problem.rs:148](../python/src/py_problem.rs#L148) and [:159](../python/src/py_problem.rs#L159),
- the trampoline, [python/src/py_problem.rs:204](../python/src/py_problem.rs#L204),
- activation at the single entry point every run path goes through, [python/src/py_nsgaii.rs:42](../python/src/py_nsgaii.rs#L42).

**What:**

- You pass a Numba `@cfunc("void(CPointer(float64), int64, CPointer(float64), int64, voidptr)")`, plus an optional C-contiguous NumPy array as `cfunc_data`.
- puggles calls the compiled function pointer directly from Rust, with no GIL and no Python objects.
- Because it doesn't need the GIL, `execution_mode="multithreaded"` runs it on all cores. A plain Python callable is always forced to one thread.
- The ctypes signature is checked when the `Problem` is created and raises `TypeError` on a mismatch. A wrong pointer would otherwise crash the process.

**Why it matters:** before this change, a per-solution objective written in Python paid for a GIL acquisition, a list conversion and an interpreter call on every evaluation, all serialized on one core.

**Measured** (multi-objective opt time):

| row | 1 KB | 1 MB | 10 MB | 100 MB | 1 GB |
| --- | --- | --- | --- | --- | --- |
| plain Python callable | 56.0 ms | 430 ms | 3,424 ms | 9,981 ms | 93.2 s |
| **Numba cfunc, 8 threads** | **23.9 ms (2.3×)** | **73.8 ms (5.8×)** | **511 ms (6.7×)** | **1,278 ms (7.8×)** | **21.3 s (4.4×)** |
| native Rust, 8 threads (reference) | 25.8 ms | 69.0 ms | 474 ms | 1,417 ms | 18.6 s |

The limitation: like native per-solution evaluation, each call still reads the whole dataset, so it doesn't beat batch evaluation on data scans. It's the right tool for per-solution objectives that can't be batched, such as simulations.

## 3. Backends outside the core library


### 3.1 CubeCL: GPU kernels written in Rust (feasibility check)

**Where:** [sandbox/examples/bench_cubecl.rs:46](examples/bench_cubecl.rs#L46) holds the `#[cube]` kernel. Its `BatchEvaluator` is at [:113](examples/bench_cubecl.rs#L113). It sits behind the sandbox `cubecl` feature, so it adds no core dependency.

**Why it matters:** WGSL shaders are the main usability barrier to the GPU path. CubeCL lets the same kernel be written in Rust and compiled for wgpu/Metal, CUDA or ROCm.

**Measured** (CubeCL vs hand-written WGSL; identical results):

| mode | 10 MB | 100 MB | 1 GB |
| --- | --- | --- | --- |
| multi | 211 vs 202 ms | 146 vs 247 ms | 1,674 vs 1,524 ms (1.10×) |
| single | 197 vs 187 ms | 136 vs 129 ms | 1,904 vs 1,587 ms (1.20×) |

CubeCL needed the same register/unroll treatment, going from 6.7 s to 1.9 s at 1 GB. Two constraints:

- **Version:** CubeCL 0.11 needs rustc 1.95, so this uses 0.10.
- **Size:** it pulls in a large dependency tree.

Recommendation: adopt it as an optional feature or companion crate, not a core dependency.

### 3.2 CUDA with Rust kernels (Rust-CUDA): written, not built

**Where:** [puggles-cuda/](../puggles-cuda/). It's a standalone, unpublished crate:

- an f64 map-reduce kernel at [puggles-cuda/kernels/src/lib.rs:13](../puggles-cuda/kernels/src/lib.rs#L13),
- a `CudaEvaluator` implementing `BatchEvaluator` at [puggles-cuda/src/lib.rs:62](../puggles-cuda/src/lib.rs#L62),
- dependencies pinned to Rust-CUDA commit `3a82c35`, with toolchain `nightly-2026-04-02`.

**Why it would matter:** native f64 on NVIDIA GPUs, which WGSL lacks, plus CUDA's mature compiler.

**Status:** built and benchmarked on a GCP g2-standard-8 (NVIDIA L4) in Rust-CUDA's CUDA 12 container. It needed three fixes (the codegen-backend feature, a `glam` version pin, and dropping the CUDA context last to avoid a crash at exit). Results: 2.6 s at 1 GB multi-objective vs pymoo's 32.7 s and PyTorch-CUDA's 1.5 s (f32). It's a separate crate because crates.io refuses git dependencies, and so that puggles keeps building on stable Rust. The Mac benchmark row reports `not built`, as it needs Linux and an NVIDIA GPU.

## 4. Changes to how performance is measured


These changes don't make puggles faster. They make the numbers trustworthy.

| change | where | why it matters |
| --- | --- | --- |
| Every evaluation scans every row | [examples/bench_datasets.rs](examples/bench_datasets.rs), [bench_datasets.py](bench_datasets.py) | The older portfolio benchmark optimized on a 20×20 covariance, so dataset size never affected the optimizer and the GPU could never win. Now evaluation cost scales with the data, as in real data-driven objectives. |
| Real stock data | [scripts/build_real_datasets.py](scripts/build_real_datasets.py) | 20 US large caps, 1986–2026, from Yahoo Finance. Fat tails (kurtosis 17 vs 3 synthetic) and real correlations. Block bootstrap of real days extends it to 1 GB. |
| Peak memory footprint instead of max RSS | [bench_datasets.py:441](bench_datasets.py#L441) | Under memory pressure macOS leaves compressed and swapped pages out of RSS. The earlier synthetic run under-reported 1 GB memory by up to 2× (pymoo: 1.56 GB reported vs 3.05 GB re-measured). |
| One process per run, equal evaluation budget | [bench_datasets.py](bench_datasets.py) | Clean memory and CPU numbers. Libraries are compared at the same number of objective evaluations, not generations, because genetic_algorithm caches fitness. |
| Imports and JIT compiles happen before the clock | [bench_datasets.py](bench_datasets.py) | Numba (about 2 s) and JAX (about 0.3 s) compile time isn't counted as optimizer time; it shows up in CPU seconds. GPU, CubeCL and Torch opt times *include* device setup and data upload. |

**Toolchain workaround:** on macOS 27.2, Xcode's linker produced a Python extension that failed to load ("mis-aligned LINKEDIT string pool"). Build the bindings with `DEVELOPER_DIR=/Library/Developer/CommandLineTools maturin develop --release`.

## 5. What didn't help, and the limits


- **Native 8 threads is limited by memory bandwidth on big data.** It takes 18.6 s at 1 GB, 2.2× over sequential for 2.8× the CPU time, because each per-solution evaluation streams the whole matrix. Batch evaluation (pymoo, the NumPy batch path) still beats it there.
- **A Numba batch kernel needs the right loop order.** The first version vectorized poorly: 10.2 s at 1 GB, only 1.3× faster than pymoo. Making the solutions the inner loop gave 6.7 s.
- **Torch on MPS is slow on small data** (150–290 ms at 1 KB), because each call has fixed GPU overhead.
- **The GPU loses below about 1 MB** to setup cost.
- **The 100 MB and 1 GB results are single runs**, so differences under about 10% are noise.
- **Solution quality didn't change.** Every new path reproduces the same fronts and best Sharpe as the CPU path for the same seed. These changes are purely about speed.

## 6. Next steps


1. **A native Rust batch objective** (`EvalFn::Batch`) written in the map-reduce style. It would remove native puggles' 1 GB memory-bandwidth gap on the CPU without needing a GPU.
2. **Speed up `puggles-cuda`:** test an f32 mode (the L4 is weak at f64) and tune tiling, aiming at PyTorch's 1.5 s at 1 GB.
3. **Optional CubeCL integration**, so users can write GPU objectives in Rust.
4. **A loader that holds the dataset once.** At 1 GB every worker briefly keeps about 3 copies of the 0.9 GB matrix. Building the row-major array straight from the CSV would cut peak memory to about 1 GB.
