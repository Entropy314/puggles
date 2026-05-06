# Rustypus

A multi-objective evolutionary optimization library written in Rust with Python bindings. Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for solving multi-objective optimization problems.

## Python Installation

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop
```

## Quick Start

```python
from rustypus import Problem, NSGAII, Real, Integer, BitBinary

# Define your objective function
# Takes a list of floats, returns a list of objective values
def my_objective(x):
    f1 = x[0]**2 + x[1]**2
    f2 = (x[0] - 1)**2 + (x[1] - 1)**2
    return [f1, f2]

# Define the problem
problem = Problem(
    solution_length=2,
    number_of_objectives=2,
    solution_data_types=[Real(-5.0, 5.0), Real(-5.0, 5.0)],
    objective_function=my_objective,
    direction=[-1, -1],  # -1 = minimize, 1 = maximize
)

# Run the optimizer
algo = NSGAII(problem, population_size=100, execution_mode="sequential")
algo.run(10000)

# Get the Pareto-optimal solutions
for sol in algo.get_archive():
    print(f"x = {sol.variables}, objectives = {sol.objectives}")
```

## Variable Types

Rustypus supports three variable types that can be mixed within a single problem:

```python
from rustypus import Real, Integer, BitBinary

solution_data_types = [
    Real(-10.0, 10.0),      # Continuous float in [-10, 10]
    Integer(-100, 100),      # Discrete integer in [-100, 100]
    BitBinary(),             # Binary 0 or 1
]
```

## Constraints

You can add objective constraints to filter feasible solutions:

```python
problem = Problem(
    solution_length=3,
    number_of_objectives=2,
    solution_data_types=[Real(0.0, 10.0)] * 3,
    objective_function=my_objective,
    direction=[-1, -1],
    objective_constraints=[Some(50.0), None],      # Constrain first objective
    constraint_operands=[Some("<"), None],           # First objective must be < 50
)
```

## Execution Modes

Rustypus supports three execution modes that control how population evaluation and initialization are parallelized:

```python
# Sequential — single-threaded, best for debugging or cheap objective functions
algo = NSGAII(problem, population_size=100, execution_mode="sequential")

# Multithreaded — uses all available CPU cores via Rayon (Rust's parallel iterator library)
algo = NSGAII(problem, population_size=100, execution_mode="multithreaded")

# GPU — reserved for GPU-accelerated evaluation (currently falls back to multithreaded)
algo = NSGAII(problem, population_size=100, execution_mode="gpu")
```

### Python objective functions

When using a Python callable as your objective function, evaluation is always run sequentially regardless of the `execution_mode` you pass. This is because Python's GIL (Global Interpreter Lock) prevents true parallel execution of Python code — running objective evaluations across threads would cause a deadlock. You can still pass any mode, but it will be overridden to `"sequential"` internally.

### Built-in Rust benchmarks (true parallelism)

To get full multi-threaded parallelism — including the objective function — use `create_benchmark_problem()`. These run entirely in Rust with no GIL involvement, so `"multithreaded"` gives real speedups across all CPU cores:

```python
from rustypus import NSGAII, create_benchmark_problem

problem = create_benchmark_problem(
    name="dtlz2",
    solution_length=10,
    number_of_objectives=3,
    bounds=[(0.0, 1.0)] * 10,
)

# True multi-core parallelism — objective runs in pure Rust
algo = NSGAII(problem, population_size=200, execution_mode="multithreaded")
algo.run(50000)
```

### When to use each mode

| Mode | Python objective | Rust benchmark (`create_benchmark_problem`) |
|------|-----------------|----------------------------------------------|
| `"sequential"` | Default (always used) | Works, but single-core |
| `"multithreaded"` | Falls back to sequential | Full speedup across all cores |
| `"gpu"` | Falls back to sequential | Same as multithreaded (GPU support planned) |

## Built-in Benchmarks

Use built-in benchmark functions for testing. These run entirely in Rust with no Python overhead:

```python
from rustypus import NSGAII, create_benchmark_problem, dtlz2

# Call a benchmark function directly
result = dtlz2([0.5] * 8)

# Create a problem using a built-in benchmark
problem = create_benchmark_problem(
    name="dtlz2",
    solution_length=10,
    number_of_objectives=3,
    bounds=[(0.0, 1.0)] * 10,
    direction=[-1, -1, -1],
)

algo = NSGAII(problem, population_size=200)
algo.run(50000)
```

Available benchmarks: `dtlz1` through `dtlz7`, `paraboloid_5`, `paraboloid_5_loc`, `paraboloid_hyper_5`, `simple_objective`, `xyz_objective`.

## Accessing Results

```python
algo.run(10000)

# Pareto-optimal archive
archive = algo.get_archive()

# Final population
population = algo.get_population()

# Number of function evaluations
print(algo.nfe)

# Each solution has:
sol = archive[0]
sol.variables            # Decision variable values
sol.objectives           # Objective function values
sol.constraints          # Constraint values
sol.feasible             # Whether the solution is feasible
sol.evaluated            # Whether the solution has been evaluated
sol.constraint_violation # Number of constraint violations
```

## Full Example: Engineering Design

```python
from rustypus import Problem, NSGAII, Real

def beam_design(x):
    width, height = x[0], x[1]
    # Minimize weight
    weight = width * height
    # Minimize deflection (simplified)
    deflection = 1000.0 / (width * height**3)
    return [weight, deflection]

problem = Problem(
    solution_length=2,
    number_of_objectives=2,
    solution_data_types=[
        Real(1.0, 50.0),   # width
        Real(1.0, 100.0),  # height
    ],
    objective_function=beam_design,
    direction=[-1, -1],
)

algo = NSGAII(problem, population_size=100)
algo.run(10000)

print("Pareto front (weight vs deflection):")
for sol in sorted(algo.get_archive(), key=lambda s: s.objectives[0]):
    w, d = sol.objectives
    print(f"  width={sol.variables[0]:.2f}, height={sol.variables[1]:.2f} -> weight={w:.2f}, deflection={d:.6f}")
```
