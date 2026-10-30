# puggles — User Guide

puggles is a Rust implementation of NSGA-II (Non-dominated Sorting Genetic Algorithm II) and NSGA-III (its reference-point successor for many objectives). It handles single- and multi-objective optimization over continuous, integer, and binary decision variables, with constraint handling, parallel evaluation (Rayon), seeded reproducible runs, and optional GPU acceleration and browser visualization.

> This guide covers the Rust library. For Python, `pip install puggles` (source in `python/`); the live dashboard is a separate, optional wheel, `pip install "puggles[dashboard]"` (source in `python-dashboard/`). To build them from a checkout: `maturin develop --release -m python/Cargo.toml` and, for the dashboard, `maturin develop --release -m python-dashboard/Cargo.toml`.

---

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Quickstart](#quickstart)
4. [Walkthroughs](#walkthroughs)
   - [Single-objective minimization](#1-single-objective-minimization)
   - [Multi-objective Pareto optimization](#2-multi-objective-pareto-optimization)
   - [Constraints](#3-constraints)
   - [Mixed variable types](#4-mixed-variable-types)
   - [Batch evaluation](#5-batch-evaluation)
   - [Built-in benchmark objectives](#6-built-in-benchmark-objectives)
   - [Custom operators](#7-custom-operators)
   - [Per-generation inspection & early stop](#8-per-generation-inspection--early-stop)
   - [Many objectives with NSGA-III](#9-many-objectives-with-nsga-iii)
   - [Permutation encoding](#10-permutation-encoding)
   - [Island model](#11-island-model)
   - [Checkpoint & resume](#12-checkpoint--resume)
   - [Reproducibility & metrics](#13-reproducibility--metrics)
5. [Execution Modes](#execution-modes)
6. [GPU Acceleration](#gpu-acceleration)
7. [Visualization & Dashboard](#visualization--dashboard)
8. [Dominance & Sorting](#dominance--sorting)
9. [Reference](#reference)
10. [Tips](#tips)

---

## Installation

```toml
[dependencies]
puggles = "0.3"
```

Everything beyond the optimizers is behind optional features, off by default:

| Feature | Adds | Extra dependencies |
|---|---|---|
| `plot` | [`puggles::plot`](#visualization--dashboard): ggplot-style figures, run recording, a live browser dashboard | `serde_json` |
| `gpu` | [`puggles::gpu_evaluator`](#gpu-acceleration): objectives in a WGSL compute shader | `wgpu`, `pollster` |

```toml
puggles = { version = "0.3", features = ["plot", "gpu"] }
```

(`recorder-abi` is internal: the native interface between the Python wheel and its dashboard add-on.)

Requires a recent stable Rust toolchain (the `gpu` feature needs a wgpu-compatible driver at runtime).

---

## Core Concepts

| Concept | Description |
|---|---|
| `Problem` | The search space: variable types & bounds, number of objectives, optional constraints, direction, and the objective function. Shared as `Arc<Problem>`. |
| `Solution` | One candidate. Holds `solution` (decision variables), `objective_fitness_values`, `constraint_values`, `constraint_violation`, `feasible`, `evaluated`. |
| `NSGAII` | The optimizer. Takes an `Arc<Problem>`, a population size, and an `ExecutionMode`. |
| `ExecutionMode` | `Sequential`, `MultiThreaded` (default), or `GPU`. |

**Directions:** `-1` = minimize, `1` = maximize, one per objective. `Problem::new` defaults `None` to minimize all.

**Variable types** (`puggles::gatypes`):
- `Real::new(lower, upper)` — continuous float in `[lower, upper)`
- `Integer::new(lower, upper)` — integer in `[lower, upper)`
- `BitBinary::new()` — 0 or 1

Bounds are `Option`: `None` means unbounded (`f64::MIN`/`MAX` or `i64::MIN`/`MAX`). Each is wrapped in a `SolutionDataTypes` variant.

---

## Quickstart

```rust
use std::sync::Arc;
use puggles::core::Problem;
use puggles::gatypes::{Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};

fn sphere(x: &Vec<f64>) -> Vec<f64> {
    vec![x.iter().map(|xi| xi * xi).sum()] // one objective: minimize sum of squares
}

fn main() {
    let problem = Arc::new(Problem::new(
        5,                                   // solution_length
        1,                                   // number_of_objectives
        None,                                // objective_constraint
        None,                                // objective_constraint_operands
        Some(vec![-1]),                      // direction (minimize)
        vec![SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))); 5],
        sphere,
    ));

    let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded);
    ga.run(10_000); // NFE budget; auto-initializes the population

    let best = ga.get_archive().iter()
        .min_by(|a, b| a.objective_fitness_values[0]
            .partial_cmp(&b.objective_fitness_values[0]).unwrap())
        .unwrap();

    println!("Best objective: {:.6}", best.objective_fitness_values[0]);
    println!("Variables:      {:?}", best.solution);
    println!("NFE used:       {}", ga.get_nfe());
}
```

An objective function has the signature `fn(&Vec<f64>) -> Vec<f64>`: it receives the decision variables and returns one value per objective.

---

## Walkthroughs

### 1. Single-Objective Minimization

Minimize the 5-variable sphere `f(x) = Σ xᵢ²` over `[-5, 5]⁵` — see the [Quickstart](#quickstart). The archive holds the non-dominated set; for a single objective that is the single best solution.

### 2. Multi-Objective Pareto Optimization

Two conflicting objectives. NSGA-II returns a Pareto front — solutions where you cannot improve one objective without worsening another.

```rust
fn two_obj(x: &Vec<f64>) -> Vec<f64> {
    let f1 = x[0] * x[0] + x[1] * x[1];        // distance from origin
    let f2 = (x[0] - 2.0).powi(2) + x[1] * x[1]; // distance from (2, 0)
    vec![f1, f2]
}

let problem = Arc::new(Problem::new(
    2,
    2,
    None,
    None,
    Some(vec![-1, -1]),                        // minimize both
    vec![
        SolutionDataTypes::Real(Real::new(Some(-3.0), Some(3.0))),
        SolutionDataTypes::Real(Real::new(Some(-3.0), Some(3.0))),
    ],
    two_obj,
));

let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded);
ga.run(20_000);

let mut front = ga.get_archive().to_vec();
front.sort_by(|a, b| a.objective_fitness_values[0]
    .partial_cmp(&b.objective_fitness_values[0]).unwrap());
println!("Pareto front size: {}", front.len());
for s in &front {
    println!("  f1={:.3}  f2={:.3}", s.objective_fitness_values[0], s.objective_fitness_values[1]);
}
```

### 3. Constraints

Constraints are bounds on the objective values after evaluation: `objective[i] <op> bound`, one per objective (`None` = unconstrained). Operators: `<`, `>`, `<=`, `>=`, `==`, `!=`.

```rust
let problem = Arc::new(Problem::new(
    2,
    1,
    Some(vec![Some(1.0)]),                     // bound per objective
    Some(vec![Some("<".to_string())]),         // objective[0] must be < 1.0
    Some(vec![-1]),
    vec![
        SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))),
        SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))),
    ],
    sphere,
));

let mut ga = NSGAII::new(Arc::clone(&problem), 50, ExecutionMode::Sequential);
ga.run(5_000);

for s in ga.get_archive().iter().filter(|s| s.feasible) {
    println!("f={:.4}  x={:?}", s.objective_fitness_values[0], s.solution);
}
```

Constraint-based dominance: a feasible solution dominates any infeasible one; among infeasible solutions, fewer violations wins. `sol.constraint_violation` counts violated constraints; `sol.constraint_values` holds `1.0` (satisfied) / `0.0` (violated) per constraint.

### 4. Mixed Variable Types

Mix `Real`, `Integer`, and `BitBinary` in one problem — each operator adapts to the variable type.

```rust
use puggles::gatypes::{BitBinary, Integer, Real, SolutionDataTypes};

fn mixed(x: &Vec<f64>) -> Vec<f64> {
    let binary_penalty = 5.0 * (1.0 - x[0]); // prefer x[0] = 1
    let int_cost = (x[1] - 7.0).abs();        // prefer x[1] = 7
    let real_cost: f64 = x[2..].iter().map(|v| v * v).sum();
    vec![binary_penalty + int_cost + real_cost]
}

let problem = Arc::new(Problem::new(
    5, 1, None, None, Some(vec![-1]),
    vec![
        SolutionDataTypes::BitBinary(BitBinary::new()),
        SolutionDataTypes::Integer(Integer::new(Some(0), Some(20))),
        SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))),
        SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))),
        SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))),
    ],
    mixed,
));
```

Default operators by type (every mutation at the per-gene rate `1/n`):
- `Real` → SBX crossover + polynomial mutation
- `Integer` → uniform crossover + polynomial mutation (rounded to the integer grid)
- `BitBinary` → uniform crossover + bit-flip mutation

### 5. Batch Evaluation

When the objective is expensive, evaluate the whole unevaluated set in one call. Use the `Problem` struct literal with `EvalFn::Batch` (the positional `Problem::new` only builds `EvalFn::Single`):

```rust
use puggles::core::{EvalFn, Problem};

fn batch_sphere(inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    inputs.iter()
        .map(|x| vec![x.iter().map(|xi| xi * xi).sum()])
        .collect()
}

let problem = Arc::new(Problem {
    solution_length: 5,
    number_of_objectives: 1,
    objective_constraint: None,
    objective_constraint_operands: None,
    direction: Some(vec![-1]),
    solution_data_types: vec![SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))); 5],
    eval_fn: EvalFn::Batch(batch_sphere),
});
```

The batch closure is called once per generation with all unevaluated solutions and runs on the calling thread — parallelize inside it if you want (e.g. Rayon). A batch problem bypasses the per-solution `ExecutionMode` parallelism (`effective_mode()` reports `Sequential`).

### 6. Built-in Benchmark Objectives

[`benchmark_objective_functions`](src/benchmark_objective_functions.rs) provides ready objectives with the `fn(&Vec<f64>) -> Vec<f64>` signature: `parabloid_5`, `parabloid_5_loc`, `parabloid_hyper_5`, `simple_objective`, `xyz_objective`, and `dtlz1`–`dtlz7`.

```rust
use puggles::benchmark_objective_functions::dtlz2;

let problem = Arc::new(Problem::new(
    12, 3, None, None, Some(vec![-1; 3]),
    vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))); 12],
    dtlz2,
));
let mut ga = NSGAII::new(Arc::clone(&problem), 200, ExecutionMode::MultiThreaded);
ga.run(50_000);
```

> `dtlz4` takes an extra `alpha` parameter, so it isn't a bare `fn(&Vec<f64>) -> Vec<f64>`. Wrap it: `|x| dtlz4(x, 100.0)` — note a non-capturing closure coerces to a function pointer.

### 7. Custom Operators

`NSGAII` exposes its `crossover_manager` and `mutation_manager` as public fields. Override the default operator per variable type after construction:

```rust
use std::sync::Arc;
use puggles::genetic_operators::crossover::DifferentialEvolutionCrossover;
use puggles::genetic_operators::mutation::PolynomialMutation;

let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded);

ga.crossover_manager.set_default_real_crossover(
    Box::new(DifferentialEvolutionCrossover::new(Some(0.9), Some(0.8))),
);
ga.mutation_manager.set_default_real_mutation(
    Arc::new(PolynomialMutation::new(Some(1.0), Some(20.0))),
);

ga.run(30_000);
```

`set_default_{real,integer,binary}_crossover` take a `Box<dyn Crossover + Send>`; `set_default_{real,integer,binary}_mutation` take an `Arc<dyn Mutation>`.

**Crossover operators** (`genetic_operators::crossover`):

| Type | Applies to | Constructor |
|---|---|---|
| `SimulatedBinaryCrossover` | Real | `::new(probability, distribution_index)` — default for Real |
| `DifferentialEvolutionCrossover` | Real | `::new(probability, scaling_factor)` |
| `BlendCrossover` | Real | `{ probability, alpha }` |
| `UnimodalDistributionCrossover` | Real | `{ probability, distribution_index, nparents, zeta, eta }` |
| `ParentCentricCrossover` | Real, Integer | `{ nparents, noffspring, eta, zeta }` |
| `UniformCrossover` | Integer, BitBinary | `{ probability }` — per-gene **swap** probability; default for Integer/BitBinary at 0.5. Setting it to 1.0 swaps every gene, which merely exchanges the parents. |
| `ArithmeticCrossover` | Real, Integer | `{ probability }` — per-gene blend `a*p1+(1-a)*p2` with random `a`; the two children are complements |

**Mutation operators** (`genetic_operators::mutation`):

| Type | Applies to | Constructor |
|---|---|---|
| `UniformMutation` | Real, Integer | `::default()` or `{ probability }` — resamples a gene uniformly; use a rate near `1/n`, never 1.0 |
| `PolynomialMutation` | Real, Integer | `::new(probability, distribution_index)` — **default for Real/Integer** at `1/n`, eta=20 |
| `GaussianMutation` | Real, Integer | `::new(probability, standard_deviation)` — symmetric normal perturbation (Box-Muller) |
| `BitFlipMutation` | BitBinary | `::default()` or `{ probability }` — **default for BitBinary** at `1/n` |

`::new` constructors take `Option<f64>` arguments (`None` uses the documented default).

`MutationManager::new(solution_length)` installs all three defaults at the conventional `1/n`
per-gene rate. A rate of 1.0 replaces the whole genome every generation — that is random search,
not evolution — so override the rate only if you know why.

### 8. Per-Generation Inspection & Early Stop

`run(max_nfe)` is **resumable**: it continues from the current NFE, skips already-evaluated solutions, and refreshes the archive each generation. So to inspect progress or stop early, call it in increasing chunks and read `get_archive()` / `get_nfe()` between calls:

```rust
let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded);

let mut budget = 0;
while budget < 20_000 {
    budget += 2_000;
    ga.run(budget); // resumes; runs until total NFE reaches `budget`, archive updated

    let best = ga.get_archive().iter()
        .map(|s| s.objective_fitness_values[0])
        .fold(f64::INFINITY, f64::min);
    println!("nfe={:6}  archive={:3}  best_f1={best:.4}",
        ga.get_nfe(), ga.get_archive().len());

    // inspect `best` and `break` here to stop early
}
```

For a wall-clock budget instead, use `run_timed(max_nfe, Duration)` which stops at whichever of the NFE budget or the time limit comes first.

To observe every generation without chunking, use `run_with` (on both `NSGAII` and `NSGAIII`). The closure sees the optimizer after each generation; return `false` to stop:

```rust
ga.run_with(20_000, |ga| {
    println!("nfe={:6}  archive={:3}", ga.get_nfe(), ga.get_archive().len());
    ga.get_archive().len() < 80 // stop once the front has 80 solutions
});
```

Observing a run doesn't change it: with the same seed, `run_with` produces exactly the archive `run` does.

To stop when the front stops moving rather than at a fixed budget, `run_until_converged(max_nfe, patience, epsilon)` ends once no objective's best value has improved by more than `epsilon` for `patience` generations, and returns the number of generations run.

### 9. Many Objectives with NSGA-III

Crowding distance spreads a front well for 2 objectives but poorly from about 3 on. [`NSGAIII`](src/nsga3.rs) replaces it with a grid of structured reference points (Das–Dennis): survivors are spread across the grid, which keeps the front diverse with 3, 5 or 10 objectives.

```rust
use puggles::nsga3::{das_dennis, NSGAIII};

assert_eq!(das_dennis(3, 12).len(), 91); // reference points for 3 objectives, 12 divisions

// population_size 0 = the reference-point count rounded up to a multiple of 4 (here 92).
let mut ga = NSGAIII::new(Arc::clone(&problem), 0, 12, ExecutionMode::MultiThreaded).with_seed(7);
ga.run(40_000);
println!("{} solutions on the front", ga.get_archive().len());
```

The API mirrors `NSGAII`: `run`, `run_with`, `get_archive`, `get_nfe`, `with_seed`, `execution_info`, `effective_mode`, `last_generation_times`, and (feature `gpu`) `with_gpu_evaluator` / `gpu_evaluations`. Batch objectives work too. Operators are the crate defaults (SBX + polynomial mutation).

**Steering.** Between generations, `focus_on(&aspiration, spread)` shrinks the reference grid around target objective values (one per objective, `f64::NAN` = no preference) so the search concentrates there; `spread` in (0, 1] sets how tightly (1 = the whole front). `clear_focus()` returns to the uniform grid:

```rust
ga.run_with(60_000, |ga| {
    if ga.get_nfe() > 20_000 {
        ga.focus_on(&[0.2, f64::NAN, 0.5], 0.3); // near f1 = 0.2, f3 = 0.5
    }
    true
});
```

---

### 10. Permutation Encoding

Ordering problems (TSP, scheduling, sequencing) need whole-vector variation: blending or
resampling a single gene would duplicate one value and drop another. Switch the problem to
permutation encoding and the GA uses order crossover (OX1) and swap mutation instead.

```rust
use puggles::core::Problem;
use puggles::gatypes::{Integer, SolutionDataTypes};

const N: usize = 8;
const POS: [f64; N] = [0.0, 9.0, 3.0, 7.0, 1.0, 5.0, 8.0, 2.0];

// Total travel distance for the visiting order in `x`.
fn tour_length(x: &Vec<f64>) -> Vec<f64> {
    vec![x.windows(2).map(|w| (POS[w[0] as usize] - POS[w[1] as usize]).abs()).sum()]
}

let problem = Arc::new(
    Problem::new(
        N, 1, None, None, Some(vec![-1]),
        (0..N).map(|_| SolutionDataTypes::Integer(Integer::new(Some(0), Some(N as i64 - 1)))).collect(),
        tour_length,
    )
    .with_permutation_encoding(),
);

let mut ga = NSGAII::new(problem, 40, ExecutionMode::Sequential).with_seed(5);
ga.run(6_000);
// Every solution is a valid permutation of 0..N.
```

Under this encoding `solution_data_types` is ignored for generation — the value space is fixed
by the permutation itself. Supply entries anyway so `solution_length` matches.

Customise the operator with `crossover_manager.set_permutation_crossover(...)` and the swap rate
with `mutation_manager.set_permutation_swap_probability(...)`.

---

### 11. Island Model

One large population can collapse onto a single region of the front. Several smaller
populations, explored independently and pooled at the end, trade per-island depth for coverage.
Islands are embarrassingly parallel, so they run under Rayon with no coordination.

```rust
use puggles::islands::{run_islands, IslandConfig};

let front = run_islands(
    Arc::clone(&problem),
    IslandConfig {
        islands: 8,
        population_size: 50,
        max_nfe_per_island: 10_000,
        seed: Some(0),
        ..Default::default()
    },
);
```

Island `i` is seeded with `seed + i`: the islands differ from each other, and the whole run still
reproduces exactly for a given base seed. `execution_mode` applies *within* each island and
defaults to `Sequential`, since the islands already supply the parallelism.

The return value is the non-dominated set of the pooled archives. `merge_fronts` does that
pooling on its own if you run the GAs yourself.

---

### 12. Checkpoint & Resume

`GaState` holds only evolved data — decision variables and evaluation results — because a
`Problem` carries a function pointer that cannot be serialized. Rebuild the same `Problem` on
resume and the state rehydrates against it.

```rust
let json = serde_json::to_string(&ga.save_state())?;
std::fs::write("run.json", &json)?;

// ... later, against an identically-built Problem ...
let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded);
ga.load_state(serde_json::from_str(&std::fs::read_to_string("run.json")?)?);
ga.run(200_000); // cumulative budget — already-spent evaluations are not repeated
```

From Python the same thing is a JSON string:

```python
state = ga.save_state()
ga2 = puggles.NSGAII(problem, population_size=100, seed=1)
ga2.load_state(state)
ga2.run(200_000)
```

---

### 13. Reproducibility & Metrics

**Seeds.** `with_seed(seed)` on either optimizer seeds the initial population, crossover, mutation and selection; the same seed and problem give a bit-identical run, and observing it with `run_with` doesn't change it. Use `ExecutionMode::Sequential` with it: multithreaded evaluation interleaves non-deterministically. To pause and resume, see [§12](#12-checkpoint--resume).

**Metrics.** [`puggles::metrics`](src/metrics.rs) scores minimization fronts given as objective vectors (`metrics::objective_vectors(ga.get_archive())`):

| Function | Measures |
|---|---|
| `igd(front, reference_set)` | Inverted generational distance to a known front; lower is better |
| `hypervolume_2d(front, [r1, r2])` | Area dominated by a 2-objective front up to a reference point; higher is better |
| `spacing(front)` | How evenly the front is spread; lower is more even |

`NSGAII::archive_hypervolume(reference)` reports the current archive's hypervolume, and `run_until_hv_converged(max_nfe, patience, epsilon, reference)` stops once it plateaus — diversity-aware, unlike `run_until_converged`'s per-objective bests.

---

## Execution Modes

`ExecutionMode` controls how population evaluation is parallelized:

| Mode | What runs |
|---|---|
| `Sequential` | Single CPU thread |
| `MultiThreaded` | All CPU cores via Rayon (**default**, `ExecutionMode::default()`) |
| `GPU` | wgpu compute shader — requires `--features gpu` and an attached `GpuEvaluator` |

The requested mode isn't always the effective one: `GPU` falls back to `MultiThreaded` when the `gpu` feature is off or no evaluator is attached, and a batch objective bypasses per-solution parallelism entirely. Ask what will actually run:

```rust
let ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::GPU);
assert_eq!(ga.effective_mode(), ExecutionMode::MultiThreaded); // no evaluator → CPU fallback
```

To pin the Rayon thread count, configure the global pool once at startup:

```rust
rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();
```

---

## GPU Acceleration

Build with `--features gpu`. Supply a WGSL compute shader (bindings documented at the top of [src/gpu_evaluator.rs](src/gpu_evaluator.rs)), construct a `GpuEvaluator`, and attach it:

```rust
use puggles::gpu_evaluator::GpuEvaluator;

let evaluator = GpuEvaluator::new_blocking(shader_wgsl, solution_length, num_objectives);
let mut ga = NSGAII::new(Arc::clone(&problem), 200, ExecutionMode::GPU)
    .with_gpu_evaluator(evaluator);
ga.run(50_000);
```

- `new_blocking` panics if no GPU adapter is available, and prints the selected adapter to stderr (`puggles: GPU = <name> (<type>, <backend>)`) — a reliable confirmation a real device is in use.
- GPU applies only to `EvalFn::Single` problems; batch problems always run their closure on the CPU.
- Objective values are computed in `f32` on the GPU and returned as `f64`.

---

## Visualization & Dashboard

Enable the `plot` feature (adds only `serde_json`; nothing visual is compiled without it). From Python, the same dashboard comes as the optional `puggles-dashboard` wheel (`pip install "puggles[dashboard]"`). Everything renders in the browser with Plotly.js, which the page loads from a pinned CDN URL, so viewing needs an internet connection. There's no GUI code in Rust: the crate writes data and serves a small web page compiled into it.

### Live dashboard

```rust
use puggles::plot::{Dashboard, RunMeta};

let meta = RunMeta::from_problem(&problem).title("ZDT1").objective_names(["cost", "risk"]);
let dash = Dashboard::start(meta, "127.0.0.1", 0)?; // port 0 = any free port
dash.open_browser();                                  // prints the URL, opens a tab
ga.run_with(50_000, |ga| dash.record(ga));            // false once Stop is pressed
dash.finish();
dash.save_html("run.html")?;                          // self-contained replay page
dash.wait();                                          // keep serving until Ctrl+C
```

The page shows:

- **Objective space**: a 2-D scatter, 3-D scatter, parallel coordinates, or scatter matrix, picked by default from the objective count. Axes and colour-by are selectable. Optional layers:
  - **True front:** drawn faintly when the run's metadata includes one.
  - **Trails:** earlier fronts, fading with age.
  - **Constraint violations:** colours the population; feasible members stay grey.
- **Decision variables**: parallel coordinates against each variable's bounds, or a heatmap for many variables.
- **Convergence**, plotted against evaluations. The options are:
  - IGD to the true front.
  - Normalized hypervolume: exact for 2–3 objectives, a Monte Carlo estimate for 4 or more.
  - Best value per objective.
  - Front size.
  - Feasible share of the population.
  - Evaluations per second.
  - **Time per generation**: a stacked breakdown of evaluation, sorting and selection, variation, archive, dashboard overhead, and unexplained time. This tells you, for example, whether the GPU is actually your bottleneck.
- **Solutions table**: sortable.
  - Click a row or point to inspect a solution in every view.
  - **Pin** up to four solutions and open the *Pinned* tab to compare them side by side. The best value per objective is in bold, and variables that differ from P1 are shaded. Export the pinned solutions as CSV or JSON.
  - **★ Balanced** selects the most even trade-off: the solution whose worst objective, normalized to the front's range, is smallest.
  - **Filter** by lassoing points (2-D or matrix) or by dragging along any parallel-coordinates axis. The filters combine, survive live updates, and limit both the table and the CSV export.
- **Goals** (side panel): set a target for any objective.
  - Solutions meeting every target are marked "meets".
  - The rest are ranked by their *goal gap*: how far their worst objective misses, as a share of the front's range. It's a table column and a colour-by option, and you can list only the solutions that meet the goals.
  - With NSGA-III, **Steer** concentrates the run near the goals (R-NSGA-III-style focused reference points), and the change is marked on the convergence chart.
- **Drivers** (decision-variables card): rank correlation between each variable and each objective or custom column across the front, strongest first. Click a cell to colour the front by that variable.
- **Results** (side panel, with two or more runs): each group's final metric (median, IQR, mean ± sd) and a two-sided Mann–Whitney U test against the best group. The panel warns when the group sizes are too small for any p-value to reach 0.05. The convergence chart can also show *Median by group* with IQR bands instead of one line per run.
- **Plateau**: a marker where the run came within 1% of its final value ("the last 60% of evaluations added <1%").
- **Notifications** (bell): a browser notification when a run finishes or plateaus.
- **Report**: a one-page HTML report with the summary, results, goals, chosen and pinned solutions, and the charts.
- **Accessibility**: a shapes mode that tells runs apart by marker shape and line style, not only colour (on automatically in forced-colours mode). Keyboard shortcuts (press `?`), and screen-reader announcements for state and selection changes.
- **Pause / Resume / Stop**: hold a live run after the current generation, inspect it, then continue.
- **Compare runs**: overlay saved runs (*Compare runs…* accepts `.pgrun.json` and `.pgrun.json.gz`), each in its own colour. Their fronts are shown at the same evaluation budget as the generation you're viewing, and their convergence curves share one axis.
- **Scrubber**: replay generation by generation. The view, axes, metric and generation are kept in the URL, so a link reopens the same view. Every chart's camera button saves a PNG; the disk button saves an SVG.

The header shows **where evaluations actually run**: a green `GPU · <device>` badge with a live count of solutions evaluated on the GPU, `CPU · N threads`, or an orange `GPU not used` badge when GPU was requested but isn't happening. Hover the badge for the reason, e.g. no `GpuEvaluator` attached, built without the `gpu` feature, or NSGA-III, which has no GPU path. The header also shows live evaluations per second, and the same information is available in code as `ga.execution_info()` and `ga.gpu_evaluations()`.

The server binds to 127.0.0.1 and pushes at most 10 updates a second, so it never slows a fast run. Recording costs one copy of the archive and population objectives per generation. History is capped at 500 frames by default and thinned evenly beyond that; use `Dashboard::start_with(RunHistory::new(meta).with_max_frames(n), ..)` to change the cap.

### Comparing runs, true fronts, pausing

```rust
use puggles::plot::{reference, Dashboard, RunHistory, RunMeta};

let meta = RunMeta::from_problem(&problem).reference_front(reference::dtlz2(3, 12)); // enables IGD
let dash = Dashboard::start(meta, "127.0.0.1", 0)?;
dash.keep_runs(true); // each finished run stays overlaid when the next one starts
for seed in 0..3 {
    let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded).with_seed(seed);
    ga.run_with(20_000, |ga| dash.record(ga));
}
dash.compare_with("baseline", RunHistory::load_json("baseline.pgrun.json")?);
dash.pause(); // what the page's Pause button does; `resume()` continues
```

Steering, your own columns, groups, events and remote access:

```rust
// Steer: goals set in the page focus an NSGA-III run (call apply_focus before record).
ga.run_with(50_000, |ga| { dash.apply_focus(ga); dash.record(ga) });
// Or focus directly: ga.focus_on(&[0.2, 0.2, f64::NAN], 0.25); ga.clear_focus();

dash.set_columns(vec!["weight".into()], |x| vec![x.iter().sum()]); // computed per front solution
dash.set_group("NSGA-II, pop 100");      // applies to the run that starts next
dash.add_event(ga.get_nfe(), "raised mutation rate");
dash.set_max_points(2_000);              // population sample per frame (default 2,000)

// Remote: bound beyond loopback, a token is required by default and included in dash.url().
let remote = Dashboard::start(meta, "0.0.0.0", 8765)?; // or dash.require_token(None) behind a tunnel
```

To view a dashboard running on a remote machine without exposing it, tunnel the port instead: `ssh -L 8765:127.0.0.1:8765 user@host`, then open `http://127.0.0.1:8765/` locally.

`reference` has `zdt1`, `zdt2`, `dtlz1`, `dtlz2` (also covering DTLZ3 and DTLZ4), and `by_name`. Phase timings are available without the dashboard too, as `ga.last_generation_times()`. If you embed the dashboard and need to release a lock while a run is paused (as the Python binding does with the GIL), use `record_without_hold` followed by `hold_for(timeout)` in a loop.

### Recording without a server

```rust
use puggles::plot::{RunHistory, RunMeta};

let mut history = RunHistory::new(RunMeta::from_problem(&problem));
ga.run_with(50_000, |ga| { history.record(ga); true });
history.save_html("run.html")?;        // full dashboard, offline
history.save_json("run.pgrun.json")?;  // `python -m puggles.dashboard run.pgrun.json`
history.to_html_compared(&[("other seed".into(), other_history)]); // replay page with overlays
```

### ggplot-style figures

```rust
use puggles::plot::{aes, ggplot, Field::*};

ggplot(ga.get_archive())                            // one row per solution
    .aes(aes().x(Obj(0)).y(Obj(1)).color(Var(2)))
    .geom_point()
    .title("Pareto front")
    .save("front.html")?;                           // or .show() to open it

ggplot(&history).aes(aes().x(Nfe).y(Obj(0))).geom_line().show()?;           // convergence
ggplot(&population).aes(aes().x(Obj(0)).y(Obj(1)).z(Obj(2)).color(Rank))
    .geom_point().scale_color_categorical().show()?;                       // 3-D
ggplot(ga.get_archive()).geom_parallel([Obj(0), Obj(1), Obj(2), Obj(3)]).show()?;
```

| Piece | Options |
|---|---|
| Data | `&[Solution]` (columns `f1..`, `x1..`, `feasible`, `rank`), `&RunHistory` (`gen`, `nfe`, `time`, `front_size`, best `f1..`), or `PlotData::new().column(name, values)` |
| Fields | `Obj(i)`, `Var(j)`, `Feasible`, `Rank`, `Gen`, `Nfe`, `Time`, `FrontSize`, `"any column"` |
| Geoms | `geom_point` (3-D when `z` is mapped), `geom_line`, `geom_histogram().bins(n)`, `geom_parallel(dims)`, `geom_splom(dims)` |
| Per layer | `layer_aes(..)`, `size`, `alpha`, `width`, `bins` (apply to the latest geom) |
| Labels | `title`, `xlab`, `ylab`, `zlab`, `color_lab` |
| Scales | `scale_x_log10`, `scale_y_log10`, `scale_color_sequential` (default for numbers), `scale_color_diverging`, `scale_color_categorical` |
| Theme | `theme_auto` (follows the viewer), `theme_light`, `theme_dark` |

Mistakes such as an unknown field (`Obj(5)` on a 2-objective problem) or `geom_point` without `y` come back as `Err(PlotError)` from `save`/`show`/`to_json`; they don't panic. Set `PUGGLES_NO_BROWSER=1` to stop `show()` and `open_browser()` from launching a browser, e.g. in CI. A runnable tour: `cargo run --release --features plot --example plot_pareto`. For a live run, add `-- --live [--problem zdt1|knapsack|knapsack3|dtlz2-5] [--gpu] [--pace-ms N]`. `knapsack` is a 250-item, 2-knapsack 0/1 problem with capacity constraints; `knapsack3` uses 3 knapsacks with NSGA-III. `--gpu` evaluates ZDT1 or the knapsack in a WGSL shader and needs `--features plot,gpu`.

---

## Dominance & Sorting

The default and only built-in comparator is **Pareto dominance**. The sorting primitives are public in [`puggles::dominance`](src/dominance.rs):

```rust
use puggles::dominance::{ParetoDominance, fast_non_dominated_sort, crowding_distance};

// Fronts as indices into `population`; front 0 is the Pareto-optimal set.
let fronts: Vec<Vec<usize>> = fast_non_dominated_sort(&population, &ParetoDominance);

// Crowding distance within one front (higher = more diverse).
let cd: Vec<f64> = crowding_distance(&population, &fronts[0]);
```

`compare_solutions` returns `-1` if the first solution dominates, `1` if the second does, `0` if non-dominated. Constraint violations are compared first (feasible beats infeasible; fewer violations beats more). `NSGAII` uses `ParetoDominance` for environmental selection and a `CrowdingTournamentSelector` (rank, then crowding distance) for parent selection.

---

## Reference

### `Problem`

Construct with `Problem::new(...)` (panics on invalid config), `Problem::try_new(...)`
(returns `Result<Problem, ConfigError>`), or a struct literal (needed for `EvalFn::Batch`).

| Field / param | Type | Description |
|---|---|---|
| `solution_length` | `usize` | Number of decision variables |
| `number_of_objectives` | `usize` | Values returned by the objective function |
| `objective_constraint` | `Option<Vec<Option<f64>>>` | Bound per objective, or `None` |
| `objective_constraint_operands` | `Option<Vec<Option<String>>>` | Operator per objective: `<`, `>`, `<=`, `>=`, `==`, `!=` |
| `direction` | `Option<Vec<i8>>` | `-1` = minimize, `1` = maximize (default: all `-1`) |
| `solution_data_types` | `Vec<SolutionDataTypes>` | One `Real`/`Integer`/`BitBinary` per variable |
| `eval_fn` | `EvalFn` | `Single(fn(&Vec<f64>) -> Vec<f64>)` or `Batch(fn(&Vec<Vec<f64>>) -> Vec<Vec<f64>>)` |
| `variable_constraints` | `Option<Vec<fn(&Vec<f64>) -> f64>>` | Decision-variable constraints `g(x) <= 0` |
| `encoding` | `Encoding` | `PerGene` (default) or `Permutation` |

### `Solution`

| Field | Type | Description |
|---|---|---|
| `solution` | `Vec<f64>` | Decision variable values |
| `objective_fitness_values` | `ObjVec` | Objective values (a `SmallVec<[f64; 4]>`: indexes and iterates like a `Vec`; `.to_vec()` for one) |
| `constraint_values` | `ObjVec` | `1.0` = satisfied, `0.0` = violated, per constraint |
| `feasible` | `bool` | `true` if all constraints satisfied |
| `constraint_violation` | `usize` | Count of violated constraints |
| `constraint_violation_magnitude` | `f64` | Total violation distance; breaks ties between equally-infeasible solutions |
| `evaluated` | `bool` | `true` once the objective has been computed |
| `problem` | `Arc<Problem>` | The problem it belongs to |

### `NSGAII`

`NSGAII::new(problem: Arc<Problem>, population_size: usize, execution_mode: ExecutionMode)`.

| Method | Returns | Description |
|---|---|---|
| `initialize()` | `()` | Generate the initial population |
| `evaluate_population(&mut Vec<Solution>)` | `()` | Evaluate a population slice (respects mode / batch) |
| `iterate()` | `()` | One full generation |
| `iterate_n(n)` | `()` | One generation producing at most `n` offspring |
| `with_seed(seed)` | `Self` | Seed all randomness for a reproducible run; builder-style |
| `run(max_nfe)` | `()` | Run to the NFE budget (auto-initializes) |
| `run_with(max_nfe, \|ga\| -> bool)` | `()` | `run`, calling the closure after each generation; `false` stops |
| `run_timed(max_nfe, Duration)` | `()` | Stop at the NFE budget or time limit, whichever first |
| `run_until_converged(max_nfe, patience, epsilon)` | `usize` | Stop once the per-objective bests plateau; returns generations run |
| `run_until_hv_converged(max_nfe, patience, epsilon, ref)` | `usize` | Stop on a hypervolume plateau (2 objectives) |
| `archive_hypervolume([f64; 2])` | `f64` | Hypervolume of the current archive |
| `update_archive()` | `()` | Fold the current non-dominated feasible set into the archive |
| `get_archive()` | `&[Solution]` | Pareto archive |
| `take_archive()` | `Vec<Solution>` | Move the archive out (leaves it empty) |
| `get_nfe()` | `usize` | Total evaluations so far |
| `effective_mode()` | `ExecutionMode` | The path evaluation will actually take |
| `execution_info()` | `ExecutionInfo` | Requested vs effective mode, CPU threads, GPU device, and why they differ |
| `gpu_evaluations()` | `Option<usize>` | Solutions evaluated on the GPU so far, if an evaluator is attached |
| `last_generation_times()` | `GenerationTimes` | Seconds the last generation spent in variation, evaluation, selection and archive update |
| `save_state()` / `load_state(GaState)` | `GaState` / `()` | Checkpoint and resume ([§12](#12-checkpoint--resume)) |
| `with_gpu_evaluator(GpuEvaluator)` | `Self` | (feature `gpu`) attach a GPU evaluator; builder-style |

Public fields worth knowing: `population: Vec<Solution>`, `archive: Vec<Solution>`, `crossover_manager`, `mutation_manager`, `execution_mode`.

### `NSGAIII`

`NSGAIII::new(problem: Arc<Problem>, population_size: usize /* 0 = derived */, divisions: usize, execution_mode: ExecutionMode)`. Methods: `with_seed`, `run`, `run_with`, `get_archive`, `take_archive`, `get_nfe`, `reference_points`, `focus_on`, `clear_focus`, `execution_info`, `effective_mode`, `gpu_evaluations`, `with_gpu_evaluator` (feature `gpu`), `last_generation_times` — see [§9](#9-many-objectives-with-nsga-iii).

---

## Tips

- **NFE budget:** a common starting point is `population_size × 100` (≈100 generations). Hard multi-objective problems may need 1000+ generations.
- **Population size:** 50–200 is typical. Larger populations cover the Pareto front better but cost more per generation.
- **Execution mode:** use `MultiThreaded` (the default) for cheap-to-evaluate objectives to get free CPU parallelism; `Sequential` for tiny problems or when debugging; `GPU` only with the feature enabled and an evaluator attached.
- **Constraints** are compared by violation count first, then by total violation magnitude. Magnitude only breaks ties between solutions with the same number of broken constraints, so constraints on wildly different scales are still worth normalizing if you rely on that tiebreak.
- **Bounds are closed:** `[lower, upper]` everywhere — generation, crossover, and mutation.
- **Ordering problems** need `with_permutation_encoding()`; per-gene operators cannot keep a permutation valid.
- **Archive vs. population:** the archive is the running Pareto front accumulated across iterations; the population is the current generation. For the final answer, use `get_archive()`.
