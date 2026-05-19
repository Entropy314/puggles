# rustypus — User Guide

rustypus is a Rust implementation of NSGA-II (Non-dominated Sorting Genetic Algorithm II) with Python bindings. It handles single- and multi-objective optimization over continuous, integer, and binary decision variables, with optional constraint handling, parallel evaluation, and GPU acceleration.

---

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Python Quickstart](#python-quickstart)
4. [Walkthroughs](#walkthroughs)
   - [Single-objective minimization](#1-single-objective-minimization)
   - [Multi-objective Pareto optimization](#2-multi-objective-pareto-optimization)
   - [Constraints](#3-constraints)
   - [Mixed variable types](#4-mixed-variable-types)
   - [Parallel batch evaluation](#5-parallel-batch-evaluation)
   - [Built-in benchmark problems](#6-built-in-benchmark-problems)
   - [Custom operators](#7-custom-operators)
   - [Per-iteration callbacks](#8-per-iteration-callbacks)
5. [Rust API](#rust-api)
6. [Reference](#reference)

---

## Installation

### Python (via maturin)

```bash
# Install maturin if needed
pip install maturin

# Build and install into the current environment
cd python
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

> **Note:** If your Python is newer than 3.13, the `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` env var is required until pyo3 is updated.

### Rust (as a library)

Add to `Cargo.toml`:

```toml
[dependencies]
rustypus = { path = "." }
```

---

## Core Concepts

| Concept | Description |
|---|---|
| `Problem` | Defines the search space: variable types, bounds, number of objectives, optional constraints, and the objective function. |
| `Solution` | One candidate in the population. Holds `variables`, `objectives`, `constraints`, `feasible`, `evaluated`. |
| `NSGAII` | The optimizer. Takes a `Problem`, a population size, and an execution mode, then runs the genetic algorithm. |
| `CrossoverConfig` | Optional: selects and tunes the crossover operator for each variable type. |
| `MutationConfig` | Optional: selects and tunes the mutation operator for each variable type. |

**Directions:** `-1` = minimize, `1` = maximize. Default is minimize all objectives.

**Variable types:**
- `Real(lower, upper)` — continuous float in `[lower, upper)`
- `Integer(lower, upper)` — integer in `[lower, upper)`
- `BitBinary()` — 0 or 1

---

## Python Quickstart

```python
import rustypus as rp

# Define the objective function
def sphere(x):
    return [sum(xi**2 for xi in x)]   # one objective: minimize sum of squares

# Define the problem
problem = rp.Problem(
    solution_length=5,
    number_of_objectives=1,
    solution_data_types=[rp.Real(-5.0, 5.0)] * 5,
    objective_function=sphere,
)

# Create and run the optimizer
ga = rp.NSGAII(problem, population_size=100)
ga.run(max_nfe=10_000)

# Inspect results
best = min(ga.get_archive(), key=lambda s: s.objectives[0])
print(f"Best objective: {best.objectives[0]:.6f}")
print(f"Variables:      {[round(v, 4) for v in best.variables]}")
print(f"NFE used:       {ga.nfe}")
```

---

## Walkthroughs

### 1. Single-Objective Minimization

Minimize the 5-variable sphere function `f(x) = Σ xᵢ²` over `[-5, 5]⁵`.

```python
import rustypus as rp

def sphere(x):
    return [sum(xi**2 for xi in x)]

problem = rp.Problem(
    solution_length=5,
    number_of_objectives=1,
    solution_data_types=[rp.Real(-5.0, 5.0)] * 5,
    objective_function=sphere,
    direction=[-1],        # -1 = minimize (default)
)

ga = rp.NSGAII(problem, population_size=50)
ga.run(10_000)

archive = ga.get_archive()
best = min(archive, key=lambda s: s.objectives[0])
print(f"f = {best.objectives[0]:.4f} at x = {[round(v,3) for v in best.variables]}")
```

The archive contains all non-dominated (Pareto-optimal) solutions found. For single-objective problems this is the single best solution.

---

### 2. Multi-Objective Pareto Optimization

Optimize two conflicting objectives simultaneously. NSGA-II returns a Pareto front — the set of solutions where you cannot improve one objective without worsening another.

```python
import rustypus as rp

# Classic ZDT1-style trade-off: minimize f1 = x[0], minimize f2 = 1 - sqrt(x[0])
# Using a simpler two-objective toy problem here
def two_obj(x):
    f1 = x[0] ** 2 + x[1] ** 2          # distance from origin
    f2 = (x[0] - 2) ** 2 + x[1] ** 2   # distance from (2, 0)
    return [f1, f2]

problem = rp.Problem(
    solution_length=2,
    number_of_objectives=2,
    solution_data_types=[rp.Real(-3.0, 3.0), rp.Real(-3.0, 3.0)],
    objective_function=two_obj,
    direction=[-1, -1],    # minimize both
)

ga = rp.NSGAII(problem, population_size=100)
ga.run(20_000)

print(f"Pareto front size: {len(ga.get_archive())}")
for sol in sorted(ga.get_archive(), key=lambda s: s.objectives[0]):
    print(f"  f1={sol.objectives[0]:.3f}  f2={sol.objectives[1]:.3f}")
```

**Visualizing the Pareto front** (requires matplotlib):

```python
import matplotlib.pyplot as plt

archive = ga.get_archive()
f1 = [s.objectives[0] for s in archive]
f2 = [s.objectives[1] for s in archive]

plt.scatter(f1, f2)
plt.xlabel("f1"); plt.ylabel("f2")
plt.title("Pareto Front")
plt.show()
```

---

### 3. Constraints

Constraints are defined as bounds on the objective values after evaluation. Each constraint is a comparison: `objective[i] op bound` where `op` is one of `<`, `>`, `<=`, `>=`, `==`, `!=`.

```python
import rustypus as rp

# Minimize x² + y², but require the result to be < 1.0
def sphere_2d(x):
    return [x[0]**2 + x[1]**2]

problem = rp.Problem(
    solution_length=2,
    number_of_objectives=1,
    solution_data_types=[rp.Real(-5.0, 5.0), rp.Real(-5.0, 5.0)],
    objective_function=sphere_2d,
    objective_constraints=[1.0],   # bound for each objective
    constraint_operands=["<"],     # objective[0] must be < 1.0
)

ga = rp.NSGAII(problem, population_size=50)
ga.run(5_000)

feasible = [s for s in ga.get_archive() if s.feasible]
print(f"Feasible solutions in archive: {len(feasible)}")
for s in feasible:
    print(f"  f={s.objectives[0]:.4f}  x={[round(v,3) for v in s.variables]}")
```

Solutions that violate constraints are ranked worse in the selection tournament and tend to be pushed out of the archive. The `constraint_violation` field on a `Solution` counts how many constraints are violated.

---

### 4. Mixed Variable Types

Problems can mix `Real`, `Integer`, and `BitBinary` variables. Each crossover and mutation operator automatically adapts to the variable type.

```python
import rustypus as rp

def mixed_objective(x):
    # x[0] is binary (0/1), x[1] is integer, x[2]-x[4] are real
    binary_penalty = 5.0 * (1 - x[0])    # prefer x[0] = 1
    int_cost       = abs(x[1] - 7)        # prefer x[1] = 7
    real_cost      = sum(xi**2 for xi in x[2:])
    return [binary_penalty + int_cost + real_cost]

problem = rp.Problem(
    solution_length=5,
    number_of_objectives=1,
    solution_data_types=[
        rp.BitBinary(),
        rp.Integer(0, 20),
        rp.Real(-5.0, 5.0),
        rp.Real(-5.0, 5.0),
        rp.Real(-5.0, 5.0),
    ],
    objective_function=mixed_objective,
)

ga = rp.NSGAII(problem, population_size=80)
ga.run(10_000)

best = min(ga.get_archive(), key=lambda s: s.objectives[0])
print(f"binary={int(best.variables[0])}  integer={int(best.variables[1])}  "
      f"reals={[round(v,3) for v in best.variables[2:]]}")
print(f"objective = {best.objectives[0]:.4f}")
```

Default operators by type:
- `Real` → SBX crossover + uniform mutation
- `Integer` → uniform crossover + uniform mutation
- `BitBinary` → uniform crossover + bit-flip mutation

---

### 5. Parallel Batch Evaluation

When your objective function is expensive, use `batch_objective_function` to evaluate the whole population at once. This lets you use Python's `ProcessPoolExecutor` or any other parallel framework inside the batch call.

```python
import rustypus as rp
from concurrent.futures import ProcessPoolExecutor

def evaluate_one(x):
    import time, math
    time.sleep(0.001)               # simulate expensive evaluation
    return [sum(xi**2 for xi in x)]

def batch_evaluate(population):
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(evaluate_one, population))
    return results

problem = rp.Problem(
    solution_length=5,
    number_of_objectives=1,
    solution_data_types=[rp.Real(-5.0, 5.0)] * 5,
    batch_objective_function=batch_evaluate,   # batch mode
)

ga = rp.NSGAII(problem, population_size=50)
ga.run(2_000)

best = min(ga.get_archive(), key=lambda s: s.objectives[0])
print(f"Best f = {best.objectives[0]:.4f}")
```

> **When to use batch vs. single:** Use `batch_objective_function` when your objective is slow (external simulation, neural network inference, etc.). For fast pure-Python functions, single-evaluation with `execution_mode="sequential"` is simpler.

---

### 6. Built-in Benchmark Problems

`create_benchmark_problem` wraps standard test functions in a `Problem` with zero Python overhead — the function runs entirely in Rust, enabling true Rayon parallelism.

```python
import rustypus as rp

# DTLZ2: 3-objective, 12-variable benchmark, minimize all objectives
problem = rp.create_benchmark_problem(
    name="dtlz2",
    solution_length=12,
    number_of_objectives=3,
    bounds=[(0.0, 1.0)] * 12,
)

ga = rp.NSGAII(
    problem,
    population_size=200,
    execution_mode="multithreaded",   # true parallelism — no Python GIL
)
ga.run(50_000)

print(f"Pareto front size: {len(ga.get_archive())}")
print(f"NFE used:          {ga.nfe}")
```

Available benchmarks: `"dtlz1"` through `"dtlz7"`, `"paraboloid_5"`, `"paraboloid_5_loc"`, `"paraboloid_hyper_5"`, `"simple_objective"`, `"xyz_objective"`.

**Execution modes:**
| Mode | When to use |
|---|---|
| `"sequential"` | Python callables (GIL prevents true parallelism) |
| `"multithreaded"` | Rust / benchmark objectives — true CPU parallelism via Rayon |
| `"gpu"` | GPU-compiled WGSL shader objectives |

When a Python `objective_function` is detected, the mode automatically falls back to `"sequential"` to avoid GIL deadlocks. Benchmark problems and batch callables are not affected.

---

### 7. Custom Operators

`CrossoverConfig` and `MutationConfig` let you swap operators without touching the core algorithm.

```python
import rustypus as rp

problem = rp.create_benchmark_problem(
    "dtlz1", solution_length=7, number_of_objectives=3,
    bounds=[(0.0, 1.0)] * 7,
)

crossover = rp.CrossoverConfig(
    real_crossover="de",         # Differential Evolution crossover
    de_probability=0.9,
    de_scaling_factor=0.8,
)

mutation = rp.MutationConfig(
    real_mutation="polynomial",
    probability=0.05,            # per-variable mutation probability
    polynomial_distribution_index=20.0,
)

ga = rp.NSGAII(
    problem,
    population_size=100,
    execution_mode="multithreaded",
    crossover_config=crossover,
    mutation_config=mutation,
    num_threads=8,               # pin Rayon thread count
)
ga.run(30_000)
```

**Available crossover operators:**
| Key | Applies to | Description |
|---|---|---|
| `"sbx"` | Real | Simulated Binary Crossover (default) |
| `"de"` | Real | Differential Evolution |
| `"blend"` | Real | BLX-α crossover |
| `"pcx"` | Real, Integer | Parent-Centric Crossover |
| `"undx"` | Real | Unimodal Distribution Crossover |
| `"uniform"` | Integer, BitBinary | Uniform bit/integer crossover (default) |
| `"arithmetic"` | Integer | Arithmetic average crossover |

**PCX parameters** (exposed on `CrossoverConfig`):
```python
crossover = rp.CrossoverConfig(
    real_crossover="pcx",
    pcx_nparents=3,
    pcx_noffspring=2,
    pcx_eta=0.1,
    pcx_zeta=0.1,
)
```

**Available mutation operators:**
| Key | Applies to | Description |
|---|---|---|
| `"uniform"` | Real, Integer | Uniform random replacement (default) |
| `"polynomial"` | Real, Integer | Polynomial mutation |
| `"gaussian"` | Real | Gaussian perturbation |
| `"bitflip"` | BitBinary | Bit-flip (default) |

---

### 8. Per-Iteration Callbacks

Pass a `callback` to `run()` to inspect the population after every generation and optionally stop early.

```python
import rustypus as rp

problem = rp.create_benchmark_problem(
    "dtlz2", solution_length=12, number_of_objectives=3,
    bounds=[(0.0, 1.0)] * 12,
)

history = []

def on_iteration(archive, population, nfe):
    best_obj = min(s.objectives[0] for s in archive) if archive else float("inf")
    history.append((nfe, best_obj))
    print(f"  nfe={nfe:6d}  archive={len(archive):3d}  best_f1={best_obj:.4f}")

    # Return False to stop early
    if nfe >= 20_000:
        return False

ga = rp.NSGAII(problem, population_size=100, execution_mode="multithreaded")
ga.run(100_000, callback=on_iteration)

print(f"\nStopped at NFE={ga.nfe}")
```

Callback signature: `callback(archive: list[Solution], population: list[Solution], nfe: int) -> bool | None`

Returning `False` stops the run immediately. Any other return value (including `None`) continues.

---

## Rust API

If you are using rustypus as a Rust library directly:

```rust
use std::sync::Arc;
use rustypus::core::{EvalFn, Problem};
use rustypus::gatypes::{SolutionDataTypes, Real};
use rustypus::genetic_algorithms_v2::{NSGAII, ExecutionMode, GeneticAlgorithm};

fn sphere(x: &Vec<f64>) -> Vec<f64> {
    vec![x.iter().map(|xi| xi * xi).sum()]
}

fn main() {
    let problem = Arc::new(Problem::new(
        5,                                           // solution_length
        1,                                           // number_of_objectives
        None,                                        // objective_constraints
        None,                                        // constraint_operands
        Some(vec![-1]),                              // direction: minimize
        vec![SolutionDataTypes::Real(Real::new(Some(-5.0), Some(5.0))); 5],
        sphere,                                      // objective function
    ));

    let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded);
    ga.run(10_000);

    let best = ga.archive.iter()
        .min_by(|a, b| a.objective_fitness_values[0]
            .partial_cmp(&b.objective_fitness_values[0]).unwrap())
        .unwrap();

    println!("Best f = {:.6}", best.objective_fitness_values[0]);
    println!("NFE    = {}", ga.get_nfe());
}
```

### Key types

```rust
// EvalFn enum — set during Problem construction
pub enum EvalFn {
    Single(fn(&Vec<f64>) -> Vec<f64>),
    Batch(fn(&Vec<Vec<f64>>) -> Vec<Vec<f64>>),
}

// NSGAII methods
ga.initialize();                          // generate initial population
ga.evaluate_population(&mut pop);        // evaluate a population slice
ga.iterate();                             // one full generation
ga.iterate_n(n);                         // one generation with at most n offspring
ga.run(max_nfe);                         // run until NFE budget exhausted (auto-inits)
ga.run_timed(max_nfe, Duration::from_secs(10)); // stop at NFE or time limit
ga.update_archive();                     // rebuild Pareto archive from current population
ga.get_archive()                         // &[Solution]
ga.get_nfe()                             // usize
```

### Batch evaluation in Rust

```rust
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

### Dominance strategies

```rust
use rustypus::dominance::{ParetoDominance, EpsilonDominance, AttributeDominance};

// Standard Pareto dominance (default in NSGAII)
let d = ParetoDominance;

// Epsilon-box dominance — groups nearby solutions, reduces Pareto front size
let d = EpsilonDominance { epsilon: vec![0.01, 0.01] };

// Attribute-based — compare by pre-computed rank and crowding distance
let cmp = AttributeDominance::compare_by_rank_and_crowd(rank_a, crowd_a, rank_b, crowd_b);
```

---

## Reference

### `Problem`

| Parameter | Type | Description |
|---|---|---|
| `solution_length` | `int` | Number of decision variables |
| `number_of_objectives` | `int` | Number of objectives returned by the objective function |
| `solution_data_types` | `list` | One `Real`, `Integer`, or `BitBinary` per variable |
| `objective_function` | `callable` | `f(x: list[float]) -> list[float]` — use when not batching |
| `batch_objective_function` | `callable` | `f(pop: list[list[float]]) -> list[list[float]]` — whole-population evaluation |
| `direction` | `list[int]` | `-1` = minimize, `1` = maximize per objective (default: all `-1`) |
| `objective_constraints` | `list[float\|None]` | Bound value per objective, or `None` for no constraint |
| `constraint_operands` | `list[str\|None]` | Operator string per objective: `"<"`, `">"`, `"<="`, `">="`, `"=="`, `"!="` |

### `Solution` (read-only after `run()`)

| Field | Type | Description |
|---|---|---|
| `variables` | `list[float]` | Decision variable values |
| `objectives` | `list[float]` | Objective function values |
| `constraints` | `list[float]` | Constraint satisfaction: `1.0` = satisfied, `0.0` = violated |
| `feasible` | `bool` | `True` if all constraints satisfied |
| `constraint_violation` | `int` | Count of violated constraints |
| `evaluated` | `bool` | `True` if objective function has been called |

### `NSGAII`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `problem` | `Problem` | — | The problem to optimize |
| `population_size` | `int` | `100` | Number of individuals per generation |
| `execution_mode` | `str` | `"sequential"` | `"sequential"`, `"multithreaded"`, or `"gpu"` |
| `crossover_config` | `CrossoverConfig` | `None` | Operator selection and parameters |
| `mutation_config` | `MutationConfig` | `None` | Operator selection and parameters |
| `num_threads` | `int` | `None` | Rayon thread count (multithreaded mode only) |

| Method | Returns | Description |
|---|---|---|
| `run(max_nfe, callback=None)` | `None` | Run until `max_nfe` evaluations (auto-initializes) |
| `get_archive()` | `list[Solution]` | Pareto-optimal solutions from last run |
| `get_population()` | `list[Solution]` | Final population from last run |
| `nfe` | `int` | Total evaluations performed in last run |

### `CrossoverConfig` — all parameters

| Parameter | Default | Description |
|---|---|---|
| `real_crossover` | `"sbx"` | Crossover for `Real` variables |
| `integer_crossover` | `"uniform"` | Crossover for `Integer` variables |
| `binary_crossover` | `"uniform"` | Crossover for `BitBinary` variables |
| `sbx_probability` | `1.0` | SBX: per-variable crossover probability |
| `sbx_distribution_index` | `20.0` | SBX: distribution index η (higher = closer to parents) |
| `de_probability` | `0.9` | DE: crossover probability |
| `de_scaling_factor` | `0.8` | DE: scaling factor F |
| `blend_alpha` | `0.5` | BLX: exploration range α |
| `uniform_probability` | `1.0` | Uniform: per-variable swap probability |
| `pcx_nparents` | `2` | PCX: number of parents |
| `pcx_noffspring` | `2` | PCX: number of offspring |
| `pcx_eta` | `0.1` | PCX: standard deviation along mean direction |
| `pcx_zeta` | `0.1` | PCX: standard deviation orthogonal to mean |

### `MutationConfig` — all parameters

| Parameter | Default | Description |
|---|---|---|
| `real_mutation` | `"uniform"` | Mutation for `Real` variables |
| `integer_mutation` | `"uniform"` | Mutation for `Integer` variables |
| `binary_mutation` | `"bitflip"` | Mutation for `BitBinary` variables |
| `probability` | `1.0` | Per-variable mutation probability |
| `polynomial_distribution_index` | `20.0` | Polynomial: distribution index η |
| `gaussian_std_dev` | `0.1` | Gaussian: standard deviation |

---

## Tips

**NFE budget sizing:** A common starting point is `population_size × 100` evaluations (100 generations). For complex multi-objective problems with large populations, 1000+ generations may be needed.

**Population size:** 50–200 is typical. Larger populations explore more of the Pareto front but cost more per generation.

**Execution mode selection:**
- Python callable → always `"sequential"` (GIL prevents parallel Python calls)
- Batch callable with ProcessPoolExecutor → `"sequential"` for the GA loop, parallelism happens inside your batch function
- Rust benchmark → `"multithreaded"` for free CPU parallelism

**Constraint handling:** NSGA-II uses constraint-based dominance: a feasible solution always dominates an infeasible one, and among infeasible solutions, fewer violations wins. Constraints do not need to be normalized.

**Archive vs. population:** The archive is the running Pareto front accumulated across all iterations. The population is the current generation. For the final answer, use `get_archive()`.
