# puggles

A multi-objective evolutionary optimization library written in Rust. Implements **NSGA-II** and the reference-point **NSGA-III** (for 3+ objectives) for single- and multi-objective problems over continuous, integer, and binary decision variables, with constraint handling, parallel (Rayon) evaluation, seeded reproducible runs, and optional GPU acceleration and browser plots.

> **Python:** `pip install puggles` ([PyPI](https://pypi.org/project/puggles/), source in `python/`). The live dashboard is an optional add-on: `pip install "puggles[dashboard]"` (source in `python-dashboard/`).
>
> **Full documentation:** see [GUIDE.md](GUIDE.md) for walkthroughs, the complete API, and configuration reference. Release notes: [CHANGELOG.md](CHANGELOG.md).


---

## Add to your project

```toml
[dependencies]
puggles = "0.3"
# Optional features, all off by default:
# puggles = { version = "0.3", features = ["plot"] }  # browser plots and a live dashboard (adds serde_json)
# puggles = { version = "0.3", features = ["gpu"] }   # GPU evaluation via wgpu
```

| Feature | Adds | Dependencies |
| --- | --- | --- |
| *(none)* | NSGA-II, NSGA-III, all encodings, constraints, Rayon parallelism, checkpoints, metrics | `rand`, `rayon`, `serde`, `smallvec` |
| `plot` | `puggles::plot`: ggplot-style figures, run recording, a live browser dashboard | + `serde_json` |
| `gpu` | `puggles::gpu_evaluator`: objectives in a WGSL compute shader | + `wgpu`, `pollster` |

---

## Quick start

```rust
use std::sync::Arc;
use puggles::core::Problem;
use puggles::gatypes::{Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};

// Objective: minimize two conflicting functions. Returns one value per objective.
fn objectives(x: &Vec<f64>) -> Vec<f64> {
    let f1 = x[0] * x[0] + x[1] * x[1];
    let f2 = (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2);
    vec![f1, f2]
}

fn main() {
    let problem = Arc::new(Problem::new(
        2,                                  // solution_length (decision variables)
        2,                                  // number_of_objectives
        None,                               // objective constraints
        None,                               // constraint operators
        Some(vec![-1, -1]),                 // direction: -1 = minimize, 1 = maximize
        vec![
            SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))),
            SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))),
        ],
        objectives,
    ));

    let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded);
    ga.run(10_000); // budget in objective-function evaluations (NFE); auto-initializes

    for sol in ga.get_archive() {
        println!("x = {:?}  f = {:?}", sol.solution, sol.objective_fitness_values);
    }
}
```

`get_archive()` returns the non-dominated (Pareto-optimal) set. On a `Solution`, decision variables are `sol.solution` and objective values are `sol.objective_fitness_values`.

---

## Variable types

```rust
use puggles::gatypes::{Real, Integer, BitBinary, SolutionDataTypes};

SolutionDataTypes::Real(Real::new(Some(-10.0), Some(10.0)))  // continuous float in [-10, 10)
SolutionDataTypes::Integer(Integer::new(Some(-100), Some(100))) // integer in [-100, 100)
SolutionDataTypes::BitBinary(BitBinary::new())               // 0 or 1
```

Mix them freely within one problem — crossover and mutation adapt per variable type. Defaults: `Real` → SBX crossover + polynomial mutation; `Integer` → uniform crossover + polynomial mutation (rounded); `BitBinary` → uniform crossover + bit-flip mutation. Every mutation runs at the conventional per-gene rate `1/n`.

---

## Constraints

Constraints are bounds on the objective values after evaluation: `objective[i] <op> bound`, where `<op>` is one of `<`, `>`, `<=`, `>=`, `==`, `!=`. Pass one entry per objective (`None` = unconstrained).

```rust
let problem = Arc::new(Problem::new(
    2,
    1,
    Some(vec![Some(1.0)]),               // bound per objective
    Some(vec![Some("<".to_string())]),   // objective[0] must be < 1.0
    Some(vec![-1]),
    vec![
        SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))),
        SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))),
    ],
    sphere,
));
```

Constraints on the decision variables, `g(x) <= 0`, attach with a builder:

```rust
let problem = Problem::new(/* ... */)
    .with_variable_constraints(vec![|x: &Vec<f64>| 1.0 - x.iter().sum::<f64>()]); // sum(x) >= 1
```

Both optimizers use constraint-based dominance: a feasible solution dominates any infeasible one, and among infeasible solutions the one with fewer violations wins. `sol.constraint_violation` counts how many constraints a solution breaks; `sol.feasible` is `true` when none are broken.

---

## Many objectives: NSGA-III

For 3 or more objectives, crowding distance stops spreading the front well; NSGA-III keeps diversity with a grid of reference points instead:

```rust
use puggles::nsga3::NSGAIII;

// 12 divisions on 3 objectives -> 91 reference points; population 0 = derived from them (92).
let mut ga = NSGAIII::new(Arc::clone(&problem), 0, 12, ExecutionMode::MultiThreaded).with_seed(42);
ga.run(40_000);
let front = ga.get_archive();
```

`focus_on(&aspiration, spread)` concentrates the search near target objective values mid-run (from `run_with`), and `clear_focus()` returns to the whole front.

---

## Reproducible runs

`with_seed(seed)` (on both optimizers) seeds the initial population, crossover, mutation and selection, so a run repeats exactly — pair it with `ExecutionMode::Sequential`, since multithreaded evaluation interleaves non-deterministically. `save_state()` / `load_state()` checkpoint a run (see [`checkpoint`](src/checkpoint.rs)), and [`metrics`](src/metrics.rs) has `igd`, `hypervolume_2d` and `spacing` for scoring fronts.

---

## Execution modes

Pass an [`ExecutionMode`](src/genetic_algorithms_v2.rs) to `NSGAII::new`:

| Mode | What runs |
| --- | --- |
| `Sequential` | Single-threaded CPU |
| `MultiThreaded` | All CPU cores via Rayon (**default**) |
| `GPU` | GPU via a wgpu compute shader — see below |

Because `GPU` can silently fall back to CPU (feature off, or no evaluator attached) and a batch objective bypasses per-solution parallelism, ask the optimizer what it will *actually* do:

```rust
let ga = NSGAII::new(problem, 100, ExecutionMode::GPU);
assert_eq!(ga.effective_mode(), ExecutionMode::MultiThreaded); // no GPU evaluator attached → CPU
```

---

## GPU acceleration (optional)

Build with `--features gpu` and attach a `GpuEvaluator` backed by a WGSL compute shader (see the shader interface documented at the top of [src/gpu_evaluator.rs](src/gpu_evaluator.rs)):

```rust
# // requires: features = ["gpu"]
use puggles::gpu_evaluator::GpuEvaluator;

let evaluator = GpuEvaluator::new_blocking(shader_wgsl, solution_length, num_objectives);
let mut ga = NSGAII::new(problem, 200, ExecutionMode::GPU)
    .with_gpu_evaluator(evaluator);
ga.run(50_000);
```

On construction the evaluator prints the selected adapter to stderr (`puggles: GPU = ...`), so you can confirm a real device is in use. GPU applies only to single-objective-function problems (`EvalFn::Single`); batch problems always run their batch closure on the CPU.

---

## Built-in benchmark objectives

[`benchmark_objective_functions`](src/benchmark_objective_functions.rs) ships standard test functions usable directly as objectives: `parabloid_5`, `parabloid_5_loc`, `parabloid_hyper_5`, `simple_objective`, `xyz_objective`, and `dtlz1`–`dtlz7` (note `dtlz4` takes an extra `alpha` argument, so wrap it in a closure to use as an objective).

```rust
use puggles::benchmark_objective_functions::dtlz2;
let problem = Arc::new(Problem::new(12, 3, None, None, Some(vec![-1; 3]),
    types, dtlz2));
```

---

## Plots and a live dashboard (optional)

Enable the `plot` feature to watch a run in the browser, or to make ggplot-style interactive figures. Nothing visual is compiled in without it:

```rust
use puggles::plot::{aes, ggplot, Dashboard, RunMeta, Field::*};

let dash = Dashboard::start(RunMeta::from_problem(&problem), "127.0.0.1", 0)?;
dash.open_browser();
ga.run_with(10_000, |ga| dash.record(ga)); // live Pareto front, variables, convergence

ggplot(ga.get_archive()).aes(aes().x(Obj(0)).y(Obj(1)).color(Var(0))).geom_point().show()?;
```

The dashboard header shows whether evaluations run on the GPU, and on which device, or on CPU threads. Try it:

```bash
cargo run --release --features plot --example plot_pareto -- --live                              # ZDT1
cargo run --release --features plot --example plot_pareto -- --live --problem knapsack           # 250-item 0/1 knapsack
cargo run --release --features plot,gpu --example plot_pareto -- --live --problem knapsack --gpu # ...on the GPU
```

See [GUIDE.md](GUIDE.md#visualization--dashboard) for details.

---

## Runnable examples

```bash
cargo run --release --features plot --example plot_pareto        # figures; add `-- --live` for the dashboard
cargo run --release --features gpu --example native_cpu_gpu_quadratic  # CPU vs GPU on one problem
```

In a checkout of the repository, the `sandbox/` workspace (not shipped with the crate) has more: `cd sandbox && cargo run --release --example generic_opt` (also `portfolio_opt`, `supply_chain`, `bench_zdt1`).

---

## Testing

```bash
cargo test                              # core library (CPU paths)
cargo test --features gpu               # also compiles/exercises the GPU-gated code
cargo test --features plot,recorder-abi # plots, run recording, the dashboard server, and the
                                        # native interface the Python dashboard add-on uses
python scripts/check_versions.py        # the crate and both Python packages share one version
```

See [GUIDE.md](GUIDE.md) for the full API reference, custom operators, dominance/sorting internals, and tuning tips.
