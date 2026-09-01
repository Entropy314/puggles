#!/usr/bin/env python3
"""
bench_python.py  —  ZDT1 NSGA-II comparison: Platypus / pymoo / DEAP / puggles-python

Install Python libraries:
    pip install platypus-opt pymoo deap

For puggles Python bindings, cd to python/ and run:
    maturin develop --release

Run:
    python3 bench_python.py
    # or via run_benchmarks.sh for a combined Rust+Python table
"""

import math
import random
import statistics
import time

N = 10
POP = 100
NFE = 10_000
RUNS = 3
N_REF = 100  # reference points on true ZDT1 front


# ── Shared ZDT1 objective ─────────────────────────────────────────────────────

def zdt1(x):
    f1 = x[0]
    g = 1 + 9 / (N - 1) * sum(x[1:])
    f2 = g * (1 - (f1 / g) ** 0.5)
    return f1, f2


# ── IGD vs ZDT1 true Pareto front (f2 = 1 − √f1) ────────────────────────────

_ZDT1_REF = [(t / N_REF, 1 - (t / N_REF) ** 0.5) for t in range(1, N_REF + 1)]


def igd(points):
    """
    Inverted Generational Distance vs the true ZDT1 Pareto front.
    For each of N_REF reference points, finds the nearest point in `points`,
    then returns the mean of those distances.  Lower = better; 0 = optimal.
    """
    if not points:
        return float("inf")
    return sum(
        min(math.hypot(af1 - rf1, af2 - rf2) for af1, af2 in points)
        for rf1, rf2 in _ZDT1_REF
    ) / N_REF


# ── Benchmark runners ─────────────────────────────────────────────────────────

def bench_puggles_py(mode="sequential"):
    """
    puggles Python bindings.
    Note: every objective call crosses the FFI boundary and re-acquires the GIL,
    so this measures FFI + GIL overhead, not raw Rust speed.
    Install:  cd python/ && maturin develop --release
    """
    import puggles as rp

    times, igds = [], []
    for _ in range(RUNS):
        problem = rp.Problem(
            solution_length=N,
            number_of_objectives=2,
            solution_data_types=[rp.Real(0.0, 1.0)] * N,
            objective_function=lambda x: list(zdt1(x)),
            direction=[-1, -1],
        )
        ga = rp.NSGAII(problem, population_size=POP, execution_mode=mode)
        t0 = time.perf_counter()
        ga.run(NFE)
        times.append((time.perf_counter() - t0) * 1000)
        igds.append(igd([(s.objectives[0], s.objectives[1]) for s in ga.get_archive()]))
    return times, igds


def bench_puggles_py_batch():
    """
    puggles with a BATCH objective: the GA calls one vectorized function per generation
    (whole population at once) instead of one Python call per solution — so the GIL is
    crossed ~POP× less often. The batch fn evaluates all solutions with NumPy.
    """
    import numpy as np
    import puggles as rp

    def zdt1_batch(pop):
        X = np.asarray(pop, dtype=float)  # (P, N)
        f1 = X[:, 0]
        g = 1 + 9 / (N - 1) * X[:, 1:].sum(1)
        f2 = g * (1 - (f1 / g) ** 0.5)
        return np.column_stack([f1, f2]).tolist()

    times, igds = [], []
    for _ in range(RUNS):
        problem = rp.Problem(
            solution_length=N, number_of_objectives=2,
            solution_data_types=[rp.Real(0.0, 1.0)] * N,
            batch_objective_function=zdt1_batch,
            direction=[-1, -1],
        )
        ga = rp.NSGAII(problem, population_size=POP, execution_mode="sequential")
        t0 = time.perf_counter()
        ga.run(NFE)
        times.append((time.perf_counter() - t0) * 1000)
        igds.append(igd([(s.objectives[0], s.objectives[1]) for s in ga.get_archive()]))
    return times, igds


def bench_pymoo():
    import numpy as np
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as _Prob
    from pymoo.optimize import minimize as _min

    class ZDT1Prob(_Prob):
        def __init__(self):
            super().__init__(n_var=N, n_obj=2, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            f1 = x[:, 0]
            g = 1 + 9 / (N - 1) * x[:, 1:].sum(axis=1)
            f2 = g * (1 - (f1 / g) ** 0.5)
            out["F"] = np.column_stack([f1, f2])

    times, igds = [], []
    for _ in range(RUNS):
        alg = NSGA2(pop_size=POP)
        t0 = time.perf_counter()
        res = _min(ZDT1Prob(), alg, ("n_eval", NFE), verbose=False, seed=None)
        times.append((time.perf_counter() - t0) * 1000)
        pts = [(r[0], r[1]) for r in res.F] if res.F is not None else []
        igds.append(igd(pts))
    return times, igds


def bench_platypus():
    from platypus import NSGAII as _NSGA, Problem as _Prob, Real as _Real

    def obj(v):
        f1, f2 = zdt1(v)
        return [f1, f2]

    times, igds = [], []
    for _ in range(RUNS):
        prob = _Prob(N, 2)
        prob.types[:] = _Real(0, 1)
        prob.function = obj
        alg = _NSGA(prob, population_size=POP)
        t0 = time.perf_counter()
        alg.run(NFE)
        times.append((time.perf_counter() - t0) * 1000)
        igds.append(igd([(s.objectives[0], s.objectives[1]) for s in alg.result]))
    return times, igds


def bench_deap():
    """
    NSGA-II hand-rolled with DEAP primitives (SBX crossover, polynomial mutation).
    NFE budget: POP initial + (NFE−POP)//POP generations × POP offspring ≈ NFE.
    Parent selection: binary tournament on rank + crowding distance (selTournamentDCD).
    Environmental selection: fast non-dominated sort (selNSGA2).
    """
    from deap import base, creator, tools

    # Use unique class names to allow repeated imports safely.
    if not hasattr(creator, "_ZDT1Fit"):
        creator.create("_ZDT1Fit", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "_ZDT1Ind"):
        creator.create("_ZDT1Ind", list, fitness=creator._ZDT1Fit)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator._ZDT1Ind, toolbox.attr_float, n=N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: zdt1(ind))
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=0.0, up=1.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=0.0, up=1.0, indpb=1.0 / N)
    toolbox.register("select", tools.selNSGA2)

    GENS = (NFE - POP) // POP  # 99 → total evals = 100 + 99*100 = 10,000

    times, igds = [], []
    for _ in range(RUNS):
        t0 = time.perf_counter()

        pop = toolbox.population(n=POP)
        for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
            ind.fitness.values = fit
        # Initial selNSGA2 assigns ranks and crowding_dist for selTournamentDCD.
        pop = toolbox.select(pop, len(pop))

        for _ in range(GENS):
            # Tournament selection on current (sorted) population.
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # SBX crossover (prob 0.9) + polynomial mutation (always).
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.9:
                    toolbox.mate(ind1, ind2)
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values
                del ind2.fitness.values

            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

            # Environmental selection: non-dominated sort of union.
            pop = toolbox.select(pop + offspring, POP)

        times.append((time.perf_counter() - t0) * 1000)
        first_front = tools.sortNondominated(pop, POP, first_front_only=True)[0]
        igds.append(igd([(ind.fitness.values[0], ind.fitness.values[1]) for ind in first_front]))

    return times, igds


# ── Output formatting ─────────────────────────────────────────────────────────

# Note: a Python-callable objective always runs Sequential (the GIL serialises
# every eval; "multithreaded" would only deadlock), so a single puggles (py) row.
BENCHMARKS = [
    ("puggles (py)",             lambda: bench_puggles_py("sequential")),
    ("puggles (py, batch)",      bench_puggles_py_batch),
    ("pymoo",                     bench_pymoo),
    ("platypus",                  bench_platypus),
    ("DEAP",                      bench_deap),
]

W = 74


def fmt_row(name, times, igds):
    mt = statistics.mean(times)
    st = statistics.stdev(times) if len(times) > 1 else 0.0
    mi = statistics.mean(igds)
    si = statistics.stdev(igds) if len(igds) > 1 else 0.0
    return f"{name:<32} {mt:>10.1f} {st:>10.1f} {mi:>9.4f} {si:>9.4f}"


if __name__ == "__main__":
    print(
        f"# ZDT1  N={N}  pop={POP}  NFE={NFE}  {RUNS} runs"
        f"  metric=IGD (lower=better, 0=optimal)"
    )
    print(f"{'library':<32} {'ms/run':>10} {'±std':>10} {'IGD':>9} {'±std':>9}")
    print("─" * W)

    for name, fn in BENCHMARKS:
        try:
            t_list, igd_list = fn()
            print(fmt_row(name, t_list, igd_list))
        except ImportError as exc:
            pkg = str(exc).split("'")[-2] if "'" in str(exc) else str(exc)
            print(f"{name:<32} {'not installed: ' + pkg:>{W - 33}}")
        except Exception as exc:
            print(f"{name:<32} {'ERROR: ' + str(exc)[:35]:>{W - 33}}")

    print("─" * W)
    print("# IGD ref: 100 pts on ZDT1 true front (f2=1−√f1).  IGD→0 as front converges.")
