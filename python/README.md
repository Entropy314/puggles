# puggles

Fast multi-objective evolutionary optimization for Python — **NSGA-II** and **NSGA-III**, with a
Rust core. Roughly **7–8× faster than pymoo** and **20–30× faster than DEAP and platypus** at
equal solution quality.

```bash
pip install puggles                  # the optimizers
pip install "puggles[dashboard]"     # ...plus the optional live browser dashboard
```

Prebuilt wheels for Linux, macOS, and Windows; Python 3.9+. No Rust toolchain needed. The core
wheel has no visual code; the dashboard is a separate wheel,
[`puggles-dashboard`](https://pypi.org/project/puggles-dashboard/), that the `[dashboard]` extra
installs at the matching version.

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

ga = pg.NSGAII(problem, population_size=100, seed=42)
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
- **Constraints** — bounds on objective values (`objective_constraints` + `constraint_operands`,
  e.g. `[1.0]` with `["<"]`); feasible solutions always beat infeasible ones
- **Batch objectives** — `batch_objective_function=f` evaluates a whole population per call,
  amortizing the GIL. ~1.8× on an expensive objective; a wash on a cheap one.
- **Reproducible runs** — `seed=42` fixes the initial population, crossover, mutation, and
  selection, so a run repeats exactly. Omit it for fresh randomness each time.
- **GPU evaluation** via `GpuProblem` and a WGSL compute shader (experimental; build from source
  with `--features gpu` — the PyPI wheels are CPU-only)
- Built-in DTLZ1–7 benchmark problems
- **Live dashboard** (optional add-on) — `callback=pg.Dashboard(ga)` shows the evolving front,
  variables, and convergence in the browser

## Reproducibility

A genetic algorithm is randomised: without a seed, two runs of identical code give different
answers (on ZDT1, roughly 10% apart in IGD). Pass a seed to make a run exactly repeatable —
essential for publishing a result, writing a regression test, or comparing two parameter
settings without measuring noise.

```python
a = pg.NSGAII(problem, seed=42); a.run(10_000)
b = pg.NSGAII(problem, seed=42); b.run(10_000)
# a and b produce identical fronts, down to the last bit
```

Seeding applies to `NSGAIII` too. Use it with `execution_mode="sequential"`: multithreaded runs
interleave evaluations non-deterministically, so they are not reproducible even when seeded.

## Many objectives

```python
ga = pg.NSGAIII(problem, divisions=12, seed=42)   # population derived from the reference points
ga.run(20_000)
```

## Dashboard (optional add-on)

`pip install "puggles[dashboard]"` adds a live browser view of a run. It shows the Pareto front
in 2-D, 3-D, parallel coordinates or a scatter matrix, plus the decision variables, convergence
and a sortable table of solutions. To use it, pass the dashboard as the callback:

```python
dash = pg.Dashboard(ga, objective_names=["cost", "risk"])   # opens a tab
ga.run(20_000, callback=dash)                              # live; the page's Stop button ends the run
dash.save("run.pgrun.json.gz")                              # replay: python -m puggles.dashboard run.pgrun.json.gz
dash.wait()                                                 # in a script: keep the page up until Ctrl+C
```

The add-on can also pause and steer a run, compare runs and seeds, show a known true front,
export standalone HTML, and work remotely or in Jupyter. Its
[README](https://github.com/Entropy314/puggles/tree/master/python-dashboard#readme) covers all of
this. It records natively, straight from the optimizer's Rust solutions, so with a built-in
objective the run keeps its full multithreaded speed while you watch it.

Without the add-on, `import puggles` never loads it. `pg.Dashboard`, `pg.show`, `pg.load` and
`pg.reference_front` then raise an `ImportError` that gives the install command. Any other
callable still works as `callback=`.

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

The NSGA-II and NSGA-III cores across all three encodings are well tested and benchmarked. The GPU evaluator is
experimental and untested in CI. One Python-callable problem may be optimized at a time per
process — concurrent GAs over different Python objectives in threads are not supported.

## Links

- Source, Rust API, and full guide: https://github.com/Entropy314/puggles
- Issues: https://github.com/Entropy314/puggles/issues

## License

MIT OR Apache-2.0
