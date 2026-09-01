#!/usr/bin/env python3
"""Run ONE optimizer in this (isolated) process on a synthetic 20-asset portfolio,
print time + peak RSS. One process per library → clean, uncontaminated memory.

Usage:  python bench_opt_mem.py {pymoo|platypus|deap|puggles}
Output: MEM\t<lib>\t<ms>\t<peak_mb>
"""
import sys, time, resource
import numpy as np

N, POP, NFE = 20, 200, 10_000
rng = np.random.default_rng(0)
A = rng.standard_normal((64, N)) * 0.02
cov = A.T @ A / 64.0            # synthetic PSD covariance (values don't affect memory)
means = rng.standard_normal(N) * 0.01

def portfolio(w):
    w = np.asarray(w, float); t = w.sum()
    w = w / t if t > 1e-12 else np.full(N, 1.0 / N)
    return float(w @ cov @ w), -float(w @ means)

def run_pymoo():
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    class P(Problem):
        def __init__(s): super().__init__(n_var=N, n_obj=2, xl=0.0, xu=1.0)
        def _evaluate(s, x, out, *a, **k):
            w = x / x.sum(axis=1, keepdims=True).clip(1e-12)
            out["F"] = np.column_stack([np.einsum("pi,ij,pj->p", w, cov, w), -(w @ means)])
    minimize(P(), NSGA2(pop_size=POP), ("n_eval", NFE), seed=1, verbose=False)

def run_platypus():
    from platypus import NSGAII, Problem, Real
    p = Problem(N, 2); p.types[:] = Real(0.0, 1.0)
    p.function = lambda v: list(portfolio(v))
    NSGAII(p, population_size=POP).run(NFE)

def run_deap():
    import random
    from deap import base, creator, tools
    if not hasattr(creator, "_F"): creator.create("_F", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "_I"): creator.create("_I", list, fitness=creator._F)
    tb = base.Toolbox()
    tb.register("attr", random.random)
    tb.register("ind", tools.initRepeat, creator._I, tb.attr, n=N)
    tb.register("pop", tools.initRepeat, list, tb.ind)
    tb.register("evaluate", lambda i: portfolio(i))
    tb.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=0.0, up=1.0)
    tb.register("mutate", tools.mutPolynomialBounded, eta=20, low=0.0, up=1.0, indpb=1.0 / N)
    tb.register("select", tools.selNSGA2)
    GENS = (NFE - POP) // POP
    pop = tb.pop(n=POP)
    for i, f in zip(pop, map(tb.evaluate, pop)): i.fitness.values = f
    pop = tb.select(pop, len(pop))
    for _ in range(GENS):
        off = [tb.clone(i) for i in tools.selTournamentDCD(pop, len(pop))]
        for a, b in zip(off[::2], off[1::2]):
            if random.random() < 0.9: tb.mate(a, b)
            tb.mutate(a); tb.mutate(b); del a.fitness.values, b.fitness.values
        for i in off:
            if not i.fitness.valid: i.fitness.values = tb.evaluate(i)
        pop = tb.select(pop + off, POP)

def run_puggles():
    import puggles as rp
    p = rp.Problem(solution_length=N, number_of_objectives=2,
                   solution_data_types=[rp.Real(0.0, 1.0)] * N,
                   objective_function=lambda w: list(portfolio(w)),
                   direction=[-1, -1])
    rp.NSGAII(p, population_size=POP, execution_mode="sequential").run(NFE)

DISPATCH = {"pymoo": run_pymoo, "platypus": run_platypus, "deap": run_deap,
            "puggles": run_puggles}
lib = sys.argv[1]
DISPATCH[lib]()  # warm-up
t0 = time.perf_counter()
DISPATCH[lib]()
ms = (time.perf_counter() - t0) * 1000
# macOS ru_maxrss is in bytes; Linux in KiB.
raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = raw / 1e6 if sys.platform == "darwin" else raw / 1024.0
print(f"MEM\t{lib}\t{ms:.1f}\t{peak_mb:.0f}")
