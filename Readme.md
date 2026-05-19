# rustypus

A multi-objective evolutionary optimization library written in Rust with Python bindings. Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for solving single- and multi-objective optimization problems over continuous, integer, and binary decision variables.

> **Full documentation:** See [GUIDE.md](GUIDE.md) for a complete walkthrough with examples.

---

## Installation

```bash
pip install maturin

cd python
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

> If your Python is 3.14+, the `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` env var is required.

---

## Quick Start

```python
import rustypus as rp

def my_objective(x):
    f1 = x[0]**2 + x[1]**2
    f2 = (x[0] - 1)**2 + (x[1] - 1)**2
    return [f1, f2]

problem = rp.Problem(
    solution_length=2,
    number_of_objectives=2,
    solution_data_types=[rp.Real(-5.0, 5.0), rp.Real(-5.0, 5.0)],
    objective_function=my_objective,
    direction=[-1, -1],   # -1 = minimize, 1 = maximize
)

ga = rp.NSGAII(problem, population_size=100)
ga.run(10_000)

for sol in ga.get_archive():
    print(f"x={[round(v,3) for v in sol.variables]}  f={[round(f,3) for f in sol.objectives]}")
```

---

## Variable Types

```python
rp.Real(-10.0, 10.0)   # continuous float in [-10, 10)
rp.Integer(-100, 100)  # discrete integer in [-100, 100)
rp.BitBinary()         # 0 or 1
```

Mix them freely within a single problem — operators adapt automatically per type.

---

## Constraints

```python
problem = rp.Problem(
    solution_length=2,
    number_of_objectives=1,
    solution_data_types=[rp.Real(0.0, 10.0), rp.Real(0.0, 10.0)],
    objective_function=my_objective,
    objective_constraints=[50.0, None],      # bound per objective (None = unconstrained)
    constraint_operands=["<", None],         # operator: "<", ">", "<=", ">=", "==", "!="
)
```

Feasible solutions dominate infeasible ones; among infeasible solutions, fewer violations wins.

---

## Execution Modes

| Mode | Use when |
| --- | --- |
| `"sequential"` | Python callable objective (default; GIL prevents true parallelism) |
| `"multithreaded"` | Rust/benchmark objective — real CPU parallelism via Rayon |
| `"gpu"` | GPU-compiled WGSL shader objective |

When a Python `objective_function` is detected, the mode automatically falls back to `"sequential"` to prevent GIL deadlocks.

```python
# True parallelism with a built-in Rust benchmark
problem = rp.create_benchmark_problem(
    name="dtlz2",
    solution_length=10,
    number_of_objectives=3,
    bounds=[(0.0, 1.0)] * 10,
)
ga = rp.NSGAII(problem, population_size=200, execution_mode="multithreaded", num_threads=8)
ga.run(50_000)
```

---

## Batch Evaluation (Python parallelism)

Use `batch_objective_function` to evaluate the whole generation at once. Plug in any Python parallel framework inside the batch call:

```python
from concurrent.futures import ProcessPoolExecutor

def evaluate_one(x):
    return [sum(xi**2 for xi in x)]

def batch_evaluate(population):
    with ProcessPoolExecutor() as pool:
        return list(pool.map(evaluate_one, population))

problem = rp.Problem(
    solution_length=5,
    number_of_objectives=1,
    solution_data_types=[rp.Real(-5.0, 5.0)] * 5,
    batch_objective_function=batch_evaluate,
)
ga = rp.NSGAII(problem, population_size=50)
ga.run(5_000)
```

---

## Custom Operators

```python
crossover = rp.CrossoverConfig(
    real_crossover="de",      # "sbx" | "de" | "blend" | "pcx" | "undx"
    de_probability=0.9,
    de_scaling_factor=0.8,
)

mutation = rp.MutationConfig(
    real_mutation="polynomial",   # "uniform" | "polynomial" | "gaussian"
    probability=0.05,
    polynomial_distribution_index=20.0,
)

ga = rp.NSGAII(problem, crossover_config=crossover, mutation_config=mutation)
```

---

## Callbacks

Inspect progress after every generation or stop early:

```python
def on_iteration(archive, population, nfe):
    print(f"nfe={nfe}  archive_size={len(archive)}")
    if nfe >= 20_000:
        return False   # stop early

ga.run(100_000, callback=on_iteration)
```

---

## Results

```python
ga.run(10_000)

archive    = ga.get_archive()     # Pareto-optimal solutions
population = ga.get_population()  # final generation
nfe        = ga.nfe               # total evaluations used

sol = archive[0]
sol.variables           # list[float] — decision variables
sol.objectives          # list[float] — objective values
sol.feasible            # bool
sol.constraint_violation  # int — number of violated constraints
```

---

## Built-in Benchmarks

Available: `"dtlz1"` – `"dtlz7"`, `"paraboloid_5"`, `"paraboloid_5_loc"`, `"paraboloid_hyper_5"`, `"simple_objective"`, `"xyz_objective"`.

```python
problem = rp.create_benchmark_problem(
    name="dtlz2",
    solution_length=10,
    number_of_objectives=3,
    bounds=[(0.0, 1.0)] * 10,
)
```

Call them directly too:

```python
rp.dtlz2([0.5] * 10)   # -> list[float]
```

---

## Full Example: Beam Design

```python
import rustypus as rp

def beam_design(x):
    width, height = x[0], x[1]
    weight     = width * height
    deflection = 1000.0 / (width * height**3)
    return [weight, deflection]

problem = rp.Problem(
    solution_length=2,
    number_of_objectives=2,
    solution_data_types=[rp.Real(1.0, 50.0), rp.Real(1.0, 100.0)],
    objective_function=beam_design,
    direction=[-1, -1],
)

ga = rp.NSGAII(problem, population_size=100)
ga.run(10_000)

print("Pareto front (weight vs deflection):")
for sol in sorted(ga.get_archive(), key=lambda s: s.objectives[0]):
    w, d = sol.objectives
    print(f"  w={sol.variables[0]:.1f}  h={sol.variables[1]:.1f}  "
          f"-> weight={w:.2f}  deflection={d:.6f}")
```

---

See [GUIDE.md](GUIDE.md) for the complete API reference, Rust usage, dominance strategies, and advanced configuration.
