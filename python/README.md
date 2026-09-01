# puggles

Fast multi-objective evolutionary optimization for Python — **NSGA-II** and **NSGA-III**, with a
Rust core. Roughly **7–8× faster than pymoo** and **20–30× faster than DEAP and platypus** at
equal solution quality.

```bash
pip install puggles
```

Prebuilt wheels for Linux, macOS, and Windows; Python 3.9+. No Rust toolchain needed.

## Quick start

```python
import puggles as pg

# ZDT1: two conflicting objectives over 10 variables in [0, 1].
def zdt1(x):
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    return [x[0], g * (1 - (x[0] / g) ** 0.5)]

problem = pg.Problem(
    solution_length=10,
    number_of_objectives=2,
    solution_data_types=[pg.Real(0.0, 1.0)] * 10,
    objective_function=zdt1,     # must return a LIST, one entry per objective
    direction=[-1, -1],          # -1 = minimize, 1 = maximize
)

ga = pg.NSGAII(problem, population_size=100)
ga.run(10_000)                   # budget in objective-function evaluations

for s in ga.get_archive():       # the Pareto front
    print(s.variables, s.objectives)
```

## Benchmarks

ZDT1, population 100, 10,000 evaluations, Apple M3. IGD lower is better — all libraries converge
to the same quality, so the difference is pure algorithm overhead.

| library | ms / run | IGD |
| --- | --- | --- |
| **puggles** | **31.8** | 0.0051 |
| pymoo | 215.4 | 0.0045 |
| DEAP | 843.4 | 0.0048 |
| platypus | 1062.1 | 0.0108 |

On DTLZ2 (3 objectives), `NSGAIII` reaches **IGD 0.011** — the best of any library measured,
against pymoo's 0.076.

## Features

- **NSGA-II** and **NSGA-III** (reference-point based; prefer it for 3+ objectives)
- **Real, Integer, and Binary** decision variables, mixable in one problem
- **Constraints** — objective bounds and decision-variable `g(x) <= 0`
- **Batch objectives** — `batch_objective_function=f` evaluates a whole population per call,
  amortizing the GIL. ~1.8× on an expensive objective; a wash on a cheap one.
- **GPU evaluation** via `GpuProblem` and a WGSL compute shader (experimental)
- Built-in DTLZ1–7 benchmark problems

## Many objectives

```python
ga = pg.NSGAIII(problem, divisions=12)   # population derived from the reference points
ga.run(20_000)
```

## Tuning operators

```python
ga = pg.NSGAII(
    problem,
    population_size=100,
    crossover_config=pg.CrossoverConfig(real_crossover="sbx", sbx_distribution_index=20.0),
    mutation_config=pg.MutationConfig(real_mutation="polynomial"),  # probability=None -> 1/n
)
```

Leave `probability` at `None` unless you have a reason: it defaults to the conventional `1/n`
per-gene rate. A rate of `1.0` replaces the whole genome every generation, which is random
search, not evolution.

## Status

The NSGA-II core across all three encodings is well tested and benchmarked. The GPU evaluator is
experimental and untested in CI. Runs are not yet seeded from Python, so results vary between
invocations; and one Python-callable problem may be optimized at a time per process (concurrent
GAs over different Python objectives in threads are not supported).

## Links

- Source, Rust API, and full guide: https://github.com/Entropy314/puggles
- Issues: https://github.com/Entropy314/puggles/issues

## License

MIT OR Apache-2.0
