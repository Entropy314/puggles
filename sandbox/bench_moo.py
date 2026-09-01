#!/usr/bin/env python3
"""
bench_moo.py — MOO benchmark: puggles (native Rust CPU) vs pymoo / platypus / DEAP,
on a data-driven portfolio problem (minimize variance, maximize return; 20 assets).

Two measurements, because they scale differently:
  A. Preprocessing (CSV load + covariance) — this is what scales with dataset size.
  B. Optimization — the optimizer only sees the 20x20 covariance, so it is
     size-invariant; measured once on a representative dataset.

puggles numbers come from the native Rust example `bench_large_data` (the Python
bindings were removed and return in a later PR).

Run:  .venv/bin/python bench_moo.py
"""
import os, statistics, subprocess, threading, time
import numpy as np, polars as pl, psutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))

N_ASSETS, POP, NFE, OPT_RUNS, LOAD_RUNS = 20, 200, 10_000, 3, 3
OPT_SIZE = "100mb"  # dataset whose covariance drives the (size-invariant) optimizer comparison

SIZES = [
    ("10kb",  "data/returns_10kb.csv"),
    ("1mb",   "data/returns_1mb.csv"),
    ("10mb",  "data/returns_10mb.csv"),
    ("100mb", "data/returns_100mb.csv"),
    ("1gb",   "data/returns_1gb.csv"),
]

_proc = psutil.Process(os.getpid())

def peak_mem_mb(fn):
    peak, done = [_proc.memory_info().rss], [False]
    def sampler():
        while not done[0]:
            try: peak[0] = max(peak[0], _proc.memory_info().rss)
            except psutil.NoSuchProcess: break
            time.sleep(0.01)
    t = threading.Thread(target=sampler, daemon=True); t.start()
    r = fn(); done[0] = True; t.join()
    return r, peak[0] / 1024 / 1024

# ── puggles native (subprocess) ──────────────────────────────────────────────

def run_rust_native():
    """Build+run bench_large_data, parse RESULT lines → {label: {...}}."""
    binary = "target/release/examples/bench_large_data"
    if not os.path.exists(binary):
        print("  [building Rust native benchmark …]", flush=True)
        subprocess.run(["cargo", "build", "--release", "--example", "bench_large_data", "-q"],
                       check=True)
    out = subprocess.run([f"./{binary}"], capture_output=True, text=True, timeout=1200).stdout
    res = {}
    for line in out.splitlines():
        if not line.startswith("RESULT\t"): continue
        _, label, rows, load, load_mb, cov, cov_mb, seq, seq_mb, par, par_mb = line.split("\t")
        res[label] = dict(rows=int(rows), load_ms=float(load), load_mb=float(load_mb),
                          cov_ms=float(cov), cov_mb=float(cov_mb), seq_ms=float(seq),
                          seq_mb=float(seq_mb), par_ms=float(par), par_mb=float(par_mb))
    return res

# ── preprocessing ─────────────────────────────────────────────────────────────

def load_polars(path):
    times, mems, df = [], [], None
    for _ in range(LOAD_RUNS):
        t0 = time.perf_counter()
        df, mem = peak_mem_mb(lambda: pl.read_csv(path))
        times.append((time.perf_counter() - t0) * 1000); mems.append(mem)
    return df, statistics.mean(times), statistics.mean(mems)

def compute_cov(df):
    def _run():
        mat = df.select(pl.all().exclude("date")).to_numpy()
        return mat.mean(axis=0), np.cov(mat, rowvar=False)
    t0 = time.perf_counter()
    (means, cov), mem = peak_mem_mb(_run)
    return means, cov, (time.perf_counter() - t0) * 1000, mem

# ── optimizers (all minimize [variance, -return]) ─────────────────────────────

def _portfolio(w_raw, means, cov):
    w = np.asarray(w_raw, float); t = w.sum()
    w = w / t if t > 1e-12 else np.full(len(w), 1.0 / len(w))
    return float(w @ cov @ w), -float(w @ means)

def _measure(one_run):
    one_run()  # warm-up
    times, mems = [], []
    for _ in range(OPT_RUNS):
        t0 = time.perf_counter()
        _, mem = peak_mem_mb(one_run)
        times.append((time.perf_counter() - t0) * 1000); mems.append(mem)
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0), max(mems)

def bench_pymoo(means, cov):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as P
    from pymoo.optimize import minimize
    class Port(P):
        def __init__(s): super().__init__(n_var=N_ASSETS, n_obj=2, xl=0.0, xu=1.0)
        def _evaluate(s, x, out, *a, **k):
            w = x / x.sum(axis=1, keepdims=True).clip(1e-12)
            out["F"] = np.column_stack([np.einsum("pi,ij,pj->p", w, cov, w), -(w @ means)])
    return _measure(lambda: minimize(Port(), NSGA2(pop_size=POP), ("n_eval", NFE), seed=1, verbose=False))

def bench_platypus(means, cov):
    from platypus import NSGAII, Problem, Real
    def one_run():
        p = Problem(N_ASSETS, 2); p.types[:] = Real(0.0, 1.0)
        p.function = lambda v: list(_portfolio(v, means, cov))
        NSGAII(p, population_size=POP).run(NFE)
    return _measure(one_run)

def bench_deap(means, cov):
    import random
    from deap import base, creator, tools
    if not hasattr(creator, "_F"): creator.create("_F", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "_I"): creator.create("_I", list, fitness=creator._F)
    tb = base.Toolbox()
    tb.register("attr", random.random)
    tb.register("ind", tools.initRepeat, creator._I, tb.attr, n=N_ASSETS)
    tb.register("pop", tools.initRepeat, list, tb.ind)
    tb.register("evaluate", lambda i: _portfolio(i, means, cov))
    tb.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=0.0, up=1.0)
    tb.register("mutate", tools.mutPolynomialBounded, eta=20, low=0.0, up=1.0, indpb=1.0 / N_ASSETS)
    tb.register("select", tools.selNSGA2)
    GENS = (NFE - POP) // POP
    def one_run():
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
    return _measure(one_run)

# ── main ──────────────────────────────────────────────────────────────────────

print(f"\nMOO benchmark  |  N_ASSETS={N_ASSETS}  pop={POP}  NFE={NFE}  {OPT_RUNS} runs+warmup")
print(f"cores={os.cpu_count()}  polars={pl.__version__}  numpy={np.__version__}\n")

print("Running native Rust benchmark (load + cov + seq/par optimize, all sizes) …", flush=True)
rust = run_rust_native()

# ── A. Preprocessing scaling ──────────────────────────────────────────────────
print("\n== A. Preprocessing: CSV load + covariance (scales with dataset size) ==")
print(f"{'size':>6} {'rows':>9} | {'py load ms':>11} {'py MB':>8} {'py cov ms':>10} "
      f"| {'rust load ms':>13} {'rust cov ms':>12}")
print("-" * 82)
cov_cache = {}
for label, path in SIZES:
    if not os.path.exists(path):
        print(f"{label:>6} {'(missing)':>9}"); continue
    df, l_ms, l_mb = load_polars(path)
    means, cov, c_ms, c_mb = compute_cov(df)
    cov_cache[label] = (means, cov)
    r = rust.get(label, {})
    print(f"{label:>6} {df.height:>9,} | {l_ms:>11.1f} {l_mb:>8.0f} {c_ms:>10.1f} "
          f"| {r.get('load_ms', float('nan')):>13.1f} {r.get('cov_ms', float('nan')):>12.1f}")
    del df

# ── B. Optimizer comparison (size-invariant: uses the OPT_SIZE covariance) ─────
means, cov = cov_cache.get(OPT_SIZE) or next(iter(cov_cache.values()))
print(f"\n== B. Optimization (size-invariant; covariance from {OPT_SIZE}) ==")
print(f"{'optimizer':<22} {'opt ms':>9} {'±std':>7} {'peak MB':>8}")
print("-" * 50)
r = rust.get(OPT_SIZE, {})
if r:
    print(f"{'puggles (Rust/seq)':<22} {r['seq_ms']:>9.1f} {'—':>7} {r['seq_mb']:>8.0f}")
    print(f"{'puggles (Rust/par)':<22} {r['par_ms']:>9.1f} {'—':>7} {r['par_mb']:>8.0f}")
for name, fn in [("pymoo", bench_pymoo), ("platypus", bench_platypus), ("DEAP", bench_deap)]:
    try:
        ms, std, mem = fn(means, cov)
        print(f"{name:<22} {ms:>9.1f} {std:>7.1f} {mem:>8.0f}")
    except Exception as e:
        print(f"{name:<22} ERROR: {str(e)[:40]}")

print("\n# opt is size-invariant because every optimizer works on the 20x20 covariance,")
print("# not the raw returns; only preprocessing (Table A) scales with dataset size.")
