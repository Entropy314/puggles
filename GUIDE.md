# rustypus — User Guide

rustypus is a Rust implementation of NSGA-II (Non-dominated Sorting Genetic Algorithm II). It handles single- and multi-objective optimization over continuous, integer, and binary decision variables, with optional constraint handling, parallel evaluation (Rayon), and optional GPU acceleration.

> **Python bindings** are temporarily removed and will be reimplemented in a later PR. This guide covers the Rust library.

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
5. [Execution Modes](#execution-modes)
6. [GPU Acceleration](#gpu-acceleration)
7. [Dominance & Sorting](#dominance--sorting)
8. [Reference](#reference)
9. [Tips](#tips)

---

## Installation

Not published to crates.io. Depend on it by path or git in your `Cargo.toml`:

```toml
[dependencies]
rustypus = { path = "path/to/rustypus" }
```

GPU evaluation is behind an optional feature (adds `wgpu` + `pollster`):

```toml
rustypus = { path = "path/to/rustypus", features = ["gpu"] }
```

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

**Variable types** (`rustypus::gatypes`):
- `Real::new(lower, upper)` — continuous float in `[lower, upper)`
- `Integer::new(lower, upper)` — integer in `[lower, upper)`
- `BitBinary::new()` — 0 or 1

Bounds are `Option`: `None` means unbounded (`f64::MIN`/`MAX` or `i64::MIN`/`MAX`). Each is wrapped in a `SolutionDataTypes` variant.

---

## Quickstart

```rust
use std::sync::Arc;
use rustypus::core::Problem;
use rustypus::gatypes::{Real, SolutionDataTypes};
use rustypus::genetic_algorithms_v2::{ExecutionMode, NSGAII};

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
use rustypus::gatypes::{BitBinary, Integer, Real, SolutionDataTypes};

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

Default operators by type:
- `Real` → SBX crossover + uniform mutation
- `Integer` → uniform crossover + uniform mutation
- `BitBinary` → uniform crossover + bit-flip mutation

### 5. Batch Evaluation

When the objective is expensive, evaluate the whole unevaluated set in one call. Use the `Problem` struct literal with `EvalFn::Batch` (the positional `Problem::new` only builds `EvalFn::Single`):

```rust
use rustypus::core::{EvalFn, Problem};

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
use rustypus::benchmark_objective_functions::dtlz2;

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
use rustypus::genetic_operators::crossover::DifferentialEvolutionCrossover;
use rustypus::genetic_operators::mutation::PolynomialMutation;

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
| `UniformCrossover` | Integer, BitBinary | `{ probability }` — default for Integer/BitBinary |
| `ArithmeticCrossover` | Integer | `{ probability }` |

**Mutation operators** (`genetic_operators::mutation`):

| Type | Applies to | Constructor |
|---|---|---|
| `UniformMutation` | Real, Integer | `::default()` or `{ probability }` — default for Real/Integer |
| `PolynomialMutation` | Real, Integer | `::new(probability, distribution_index)` |
| `GaussianMutation` | Real | `::new(probability, standard_deviation)` |
| `BitFlipMutation` | BitBinary | `::default()` or `{ probability }` — default for BitBinary |

`::new` constructors take `Option<f64>` arguments (`None` uses the documented default).

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
use rustypus::gpu_evaluator::GpuEvaluator;

let evaluator = GpuEvaluator::new_blocking(shader_wgsl, solution_length, num_objectives);
let mut ga = NSGAII::new(Arc::clone(&problem), 200, ExecutionMode::GPU)
    .with_gpu_evaluator(evaluator);
ga.run(50_000);
```

- `new_blocking` panics if no GPU adapter is available, and prints the selected adapter to stderr (`rustypus: GPU = <name> (<type>, <backend>)`) — a reliable confirmation a real device is in use.
- GPU applies only to `EvalFn::Single` problems; batch problems always run their closure on the CPU.
- Objective values are computed in `f32` on the GPU and returned as `f64`.

---

## Dominance & Sorting

The default and only built-in comparator is **Pareto dominance**. The sorting primitives are public in [`rustypus::dominance`](src/dominance.rs):

```rust
use rustypus::dominance::{ParetoDominance, fast_non_dominated_sort, crowding_distance};

// Fronts as indices into `population`; front 0 is the Pareto-optimal set.
let fronts: Vec<Vec<usize>> = fast_non_dominated_sort(&population, &ParetoDominance);

// Crowding distance within one front (higher = more diverse).
let cd: Vec<f64> = crowding_distance(&population, &fronts[0]);
```

`compare_solutions` returns `-1` if the first solution dominates, `1` if the second does, `0` if non-dominated. Constraint violations are compared first (feasible beats infeasible; fewer violations beats more). `NSGAII` uses `ParetoDominance` for environmental selection and a `CrowdingTournamentSelector` (rank, then crowding distance) for parent selection.

---

## Reference

### `Problem`

Construct with `Problem::new(...)` (single objective fn) or a struct literal (needed for `EvalFn::Batch`).

| Field / param | Type | Description |
|---|---|---|
| `solution_length` | `usize` | Number of decision variables |
| `number_of_objectives` | `usize` | Values returned by the objective function |
| `objective_constraint` | `Option<Vec<Option<f64>>>` | Bound per objective, or `None` |
| `objective_constraint_operands` | `Option<Vec<Option<String>>>` | Operator per objective: `<`, `>`, `<=`, `>=`, `==`, `!=` |
| `direction` | `Option<Vec<i8>>` | `-1` = minimize, `1` = maximize (default: all `-1`) |
| `solution_data_types` | `Vec<SolutionDataTypes>` | One `Real`/`Integer`/`BitBinary` per variable |
| `eval_fn` | `EvalFn` | `Single(fn(&Vec<f64>) -> Vec<f64>)` or `Batch(fn(&Vec<Vec<f64>>) -> Vec<Vec<f64>>)` |

### `Solution`

| Field | Type | Description |
|---|---|---|
| `solution` | `Vec<f64>` | Decision variable values |
| `objective_fitness_values` | `Vec<f64>` | Objective values |
| `constraint_values` | `Vec<f64>` | `1.0` = satisfied, `0.0` = violated, per constraint |
| `feasible` | `bool` | `true` if all constraints satisfied |
| `constraint_violation` | `usize` | Count of violated constraints |
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
| `run(max_nfe)` | `()` | Run to the NFE budget (auto-initializes) |
| `run_timed(max_nfe, Duration)` | `()` | Stop at the NFE budget or time limit, whichever first |
| `update_archive()` | `()` | Fold the current non-dominated feasible set into the archive |
| `get_archive()` | `&[Solution]` | Pareto archive |
| `get_nfe()` | `usize` | Total evaluations so far |
| `effective_mode()` | `ExecutionMode` | The path evaluation will actually take |
| `with_gpu_evaluator(GpuEvaluator)` | `Self` | (feature `gpu`) attach a GPU evaluator; builder-style |

Public fields worth knowing: `population: Vec<Solution>`, `archive: Vec<Solution>`, `crossover_manager`, `mutation_manager`, `execution_mode`.

---

## Tips

- **NFE budget:** a common starting point is `population_size × 100` (≈100 generations). Hard multi-objective problems may need 1000+ generations.
- **Population size:** 50–200 is typical. Larger populations cover the Pareto front better but cost more per generation.
- **Execution mode:** use `MultiThreaded` (the default) for cheap-to-evaluate objectives to get free CPU parallelism; `Sequential` for tiny problems or when debugging; `GPU` only with the feature enabled and an evaluator attached.
- **Constraints** need no normalization — constraint-based dominance only compares violation counts.
- **Archive vs. population:** the archive is the running Pareto front accumulated across iterations; the population is the current generation. For the final answer, use `get_archive()`.
