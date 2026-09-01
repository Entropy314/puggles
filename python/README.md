# puggles

A multi-objective evolutionary optimization library written in Rust. Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for solving single- and multi-objective optimization problems over continuous, integer, and binary decision variables, with optional constraint handling, parallel (Rayon) evaluation, and optional GPU acceleration.

> **Python bindings:** in `python/`. Build with `cd python && maturin develop --release`.
>
> **Full documentation:** see [GUIDE.md](GUIDE.md) for walkthroughs, the complete API, and configuration reference.

---

## Add to your project

Not published to crates.io yet — depend on it by path or git:

```toml
[dependencies]
puggles = { path = "path/to/puggles" }
# GPU evaluation is optional and off by default:
# puggles = { path = "path/to/puggles", features = ["gpu"] }
```

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

Mix them freely within one problem — crossover and mutation adapt per variable type. Defaults: `Real` → SBX crossover + uniform mutation; `Integer` → uniform crossover + uniform mutation; `BitBinary` → uniform crossover + bit-flip mutation.

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

NSGA-II uses constraint-based dominance: a feasible solution dominates any infeasible one, and among infeasible solutions the one with fewer violations wins. `sol.constraint_violation` counts how many constraints a solution breaks; `sol.feasible` is `true` when none are broken.

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

## Runnable examples

The [`sandbox/`](sandbox/) workspace has complete, runnable examples:

```bash
cd sandbox
cargo run --release --example generic_opt     # ZDT1 bi-objective
cargo run --release --example portfolio_opt
cargo run --release --example supply_chain
cargo run --release --example bench_zdt1
```

---

## Testing

```bash
cargo test                 # core library (CPU paths)
cargo test --features gpu  # also compiles/exercises the GPU-gated code
```

See [GUIDE.md](GUIDE.md) for the full API reference, custom operators, dominance/sorting internals, and tuning tips.
