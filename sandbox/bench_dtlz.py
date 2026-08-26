#!/usr/bin/env python3
"""
bench_dtlz.py — DTLZ2 (3-objective) NSGA-II 5-way comparison.

Libraries: rustypus (py) / pymoo / platypus / DEAP / rustypus (native Rust).
Metrics (computed uniformly with pymoo indicators against the true DTLZ2 front):
    ms/run   — wall-clock time (lower better)
    HV       — hypervolume, ref point [1.1,1.1,1.1] (higher better)
    IGD      — inverted generational distance (lower better)

Uses the STANDARD DTLZ2 (first M-1 vars are angles; front is the unit-sphere
octant), identical to sandbox/examples/bench_dtlz.rs, so all fronts are comparable.

Run:  ./.venv/bin/python3 bench_dtlz.py
"""
import math
import os
import random
import statistics
import subprocess
import time

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

N, M, POP, NFE, RUNS = 12, 3, 100, 20_000, 3
HERE = os.path.dirname(os.path.abspath(__file__))


# ── Standard DTLZ2 (matches bench_dtlz.rs) ────────────────────────────────────
def dtlz2(x):
    g = sum((x[i] - 0.5) ** 2 for i in range(M - 1, len(x)))
    f = []
    for i in range(M):
        v = 1.0 + g
        for j in range(M - 1 - i):
            v *= math.cos(x[j] * math.pi / 2)
        if i > 0:
            v *= math.sin(x[M - 1 - i] * math.pi / 2)
        f.append(v)
    return f


# ── Uniform quality metrics (shared across every library) ─────────────────────
_REF_DIRS = get_reference_directions("das-dennis", M, n_partitions=12)
_PF = get_problem("dtlz2", n_obj=M).pareto_front(_REF_DIRS)
_HV = HV(ref_point=np.array([1.1] * M))
_IGD = IGD(_PF)


def score(front):
    F = np.asarray(front, dtype=float)
    if F.size == 0:
        return 0.0, float("inf")
    return float(_HV(F)), float(_IGD(F))


def summarize(times, hvs, igds):
    mt = statistics.mean(times)
    st = statistics.stdev(times) if len(times) > 1 else 0.0
    return mt, st, statistics.mean(hvs), statistics.mean(igds)


# ── Runners ───────────────────────────────────────────────────────────────────
def bench_rustypus_py():
    import rustypus as rp

    times, hvs, igds = [], [], []
    for _ in range(RUNS):
        p = rp.Problem(
            solution_length=N, number_of_objectives=M,
            solution_data_types=[rp.Real(0.0, 1.0)] * N,
            objective_function=lambda x: dtlz2(x), direction=[-1] * M,
        )
        ga = rp.NSGAII(p, population_size=POP, execution_mode="sequential")
        t0 = time.perf_counter(); ga.run(NFE); times.append((time.perf_counter() - t0) * 1000)
        front = [[s.objectives[k] for k in range(M)] for s in ga.get_archive()]
        hv, igd = score(front); hvs.append(hv); igds.append(igd)
    return summarize(times, hvs, igds)


def bench_pymoo():
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as _P
    from pymoo.optimize import minimize

    class DTLZ2P(_P):
        def __init__(s):
            super().__init__(n_var=N, n_obj=M, xl=0.0, xu=1.0)

        def _evaluate(s, X, out, *a, **k):
            g = ((X[:, M - 1:] - 0.5) ** 2).sum(axis=1)
            cols = []
            for i in range(M):
                v = 1.0 + g
                for j in range(M - 1 - i):
                    v = v * np.cos(X[:, j] * np.pi / 2)
                if i > 0:
                    v = v * np.sin(X[:, M - 1 - i] * np.pi / 2)
                cols.append(v)
            out["F"] = np.column_stack(cols)

    times, hvs, igds = [], [], []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        res = minimize(DTLZ2P(), NSGA2(pop_size=POP), ("n_eval", NFE), verbose=False, seed=None)
        times.append((time.perf_counter() - t0) * 1000)
        F = res.F if res.F is not None else np.empty((0, M))
        hv, igd = score(F); hvs.append(hv); igds.append(igd)
    return summarize(times, hvs, igds)


def bench_platypus():
    from platypus import NSGAII as _NS, Problem as _P, Real as _R

    times, hvs, igds = [], [], []
    for _ in range(RUNS):
        prob = _P(N, M); prob.types[:] = _R(0, 1); prob.function = lambda v: dtlz2(v)
        alg = _NS(prob, population_size=POP)
        t0 = time.perf_counter(); alg.run(NFE); times.append((time.perf_counter() - t0) * 1000)
        front = [[s.objectives[k] for k in range(M)] for s in alg.result]
        hv, igd = score(front); hvs.append(hv); igds.append(igd)
    return summarize(times, hvs, igds)


def bench_deap():
    from deap import base, creator, tools

    if not hasattr(creator, "_D2F"):
        creator.create("_D2F", base.Fitness, weights=(-1.0,) * M)
    if not hasattr(creator, "_D2I"):
        creator.create("_D2I", list, fitness=creator._D2F)
    tb = base.Toolbox()
    tb.register("attr", random.random)
    tb.register("ind", tools.initRepeat, creator._D2I, tb.attr, n=N)
    tb.register("pop", tools.initRepeat, list, tb.ind)
    tb.register("evaluate", lambda i: dtlz2(i))
    tb.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=0.0, up=1.0)
    tb.register("mutate", tools.mutPolynomialBounded, eta=20, low=0.0, up=1.0, indpb=1.0 / N)
    tb.register("select", tools.selNSGA2)
    GENS = (NFE - POP) // POP

    times, hvs, igds = [], [], []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        pop = tb.pop(n=POP)
        for i, f in zip(pop, map(tb.evaluate, pop)):
            i.fitness.values = f
        pop = tb.select(pop, len(pop))
        for _ in range(GENS):
            off = [tb.clone(i) for i in tools.selTournamentDCD(pop, len(pop))]
            for a, b in zip(off[::2], off[1::2]):
                if random.random() < 0.9:
                    tb.mate(a, b)
                tb.mutate(a); tb.mutate(b); del a.fitness.values, b.fitness.values
            for i in off:
                if not i.fitness.valid:
                    i.fitness.values = tb.evaluate(i)
            pop = tb.select(pop + off, POP)
        times.append((time.perf_counter() - t0) * 1000)
        ff = tools.sortNondominated(pop, POP, first_front_only=True)[0]
        front = [list(i.fitness.values) for i in ff]
        hv, igd = score(front); hvs.append(hv); igds.append(igd)
    return summarize(times, hvs, igds)


def bench_native_rust():
    subprocess.run(["cargo", "build", "--release", "--example", "bench_dtlz"],
                   cwd=HERE, check=True, capture_output=True)
    out = subprocess.run(["cargo", "run", "--release", "--example", "bench_dtlz"],
                         cwd=HERE, check=True, capture_output=True, text=True).stdout
    seq_ms = seq_std = None
    front = []
    for line in out.splitlines():
        p = line.split("\t")
        if p[0] == "RESULT" and p[1] == "seq":
            seq_ms, seq_std = float(p[2]), float(p[3])
        elif p[0] == "PT":
            front.append([float(p[1]), float(p[2]), float(p[3])])
    hv, igd = score(front)
    return seq_ms, seq_std, hv, igd


# ── Output ────────────────────────────────────────────────────────────────────
BENCHMARKS = [
    ("rustypus (native)", bench_native_rust),
    ("rustypus (py)", bench_rustypus_py),
    ("pymoo", bench_pymoo),
    ("platypus", bench_platypus),
    ("DEAP", bench_deap),
]
W = 74

if __name__ == "__main__":
    print(f"# DTLZ2  n={N}  m={M}  pop={POP}  NFE={NFE}  {RUNS} runs")
    print(f"{'library':<20} {'ms/run':>10} {'±std':>10} {'HV↑':>9} {'IGD↓':>9}")
    print("─" * W)
    for name, fn in BENCHMARKS:
        try:
            mt, st, hv, igd = fn()
            print(f"{name:<20} {mt:>10.1f} {st:>10.1f} {hv:>9.4f} {igd:>9.4f}")
        except ImportError as exc:
            pkg = str(exc).split("'")[-2] if "'" in str(exc) else str(exc)
            print(f"{name:<20} {'not installed: ' + pkg:>{W - 21}}")
        except Exception as exc:
            print(f"{name:<20} {'ERROR: ' + str(exc)[:40]:>{W - 21}}")
    print("─" * W)
    print("# HV ref=[1.1,1.1,1.1] (higher=better).  IGD vs das-dennis front (lower=better).")
