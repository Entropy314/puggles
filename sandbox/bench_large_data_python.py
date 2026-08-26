#!/usr/bin/env python3
"""
bench_large_data_python.py — portfolio pipeline benchmark, all Python libraries.

For each dataset (1 MB, 10 MB, 100 MB):
  1. Load CSV once with polars-py (fastest Python loader)
  2. Compute covariance matrix once with numpy BLAS
  3. Time each optimizer independently on the identical 20-asset problem:
       rustypus (py/seq)  rustypus (py/par)  pymoo  platypus  DEAP

Run:  python3 bench_large_data_python.py
"""

import os
import random
import statistics
import subprocess
import threading
import time

import numpy as np
import polars as pl
import psutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))

N_ASSETS = 20
POP      = 200
NFE      = 10_000
OPT_RUNS = 3
LOAD_RUNS = 5

DATASETS = [
    ("  1 MB", "data/returns_1mb.csv"),
    (" 10 MB", "data/returns_10mb.csv"),
    ("100 MB", "data/returns_100mb.csv"),
]


# ── Peak-RSS sampler ──────────────────────────────────────────────────────────

_proc = psutil.Process(os.getpid())

def peak_mem_mb(fn):
    """Run fn(), return (result, peak_rss_mb) sampled every 10 ms."""
    peak = [_proc.memory_info().rss]
    done = [False]

    def sampler():
        while not done[0]:
            try:
                peak[0] = max(peak[0], _proc.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    result = fn()
    done[0] = True
    t.join()
    return result, peak[0] / 1024 / 1024  # bytes → MB


# ── Native Rust benchmark ─────────────────────────────────────────────────────

def run_rust_native():
    """
    Build (if needed) and run bench_large_data, parse its table.
    Returns {size_label: (load_ms, cov_ms, opt_seq_ms, opt_par_ms)} or {}.
    """
    binary = "target/release/examples/bench_large_data"
    if not os.path.exists(binary):
        print("  [building Rust native benchmark …]", flush=True)
        subprocess.run(
            ["cargo", "build", "--release", "--example", "bench_large_data", "-q"],
            check=True, capture_output=True,
        )
    try:
        out = subprocess.run([f"./{binary}"], capture_output=True, text=True,
                             timeout=300).stdout
    except Exception as exc:
        print(f"  [Rust native run failed: {exc}]")
        return {}

    results = {}
    for line in out.splitlines():
        parts = line.strip().split()
        # Data rows now have 11 tokens:
        # label(2) rows(1) load_ms mem load_cov mem seq_ms mem par_ms mem
        # e.g.: "1 MB  5500  9.8  42.1  3.1  44.0  174.1  80.2  173.8  80.5"
        if len(parts) >= 7 and parts[1] == "MB":
            label = parts[0] + " MB"
            try:
                if len(parts) == 11:
                    # parts: ['1','MB', rows, load_ms, load_mb, cov_ms, cov_mb, seq_ms, seq_mb, par_ms, par_mb]
                    results[label] = {
                        "load_ms":   float(parts[3]),
                        "load_mb":   float(parts[4]),
                        "cov_ms":    float(parts[5]),
                        "cov_mb":    float(parts[6]),
                        "seq_ms":    float(parts[7]),
                        "seq_mb":    float(parts[8]),
                        "par_ms":    float(parts[9]),
                        "par_mb":    float(parts[10]),
                    }
                else:
                    # Fallback for old format: parts: ['1','MB', rows, load_ms, cov_ms, seq_ms, par_ms]
                    results[label] = {
                        "load_ms": float(parts[3]),
                        "cov_ms":  float(parts[4]),
                        "seq_ms":  float(parts[5]),
                        "par_ms":  float(parts[6]),
                    }
            except (ValueError, IndexError):
                pass
    return results


# ── Preprocessing ─────────────────────────────────────────────────────────────

def bench_load_polars(path):
    times, mems = [], []
    df = None
    for _ in range(LOAD_RUNS):
        t0 = time.perf_counter()
        df, mem = peak_mem_mb(lambda: pl.read_csv(path))
        times.append((time.perf_counter() - t0) * 1000)
        mems.append(mem)
    return df, statistics.mean(times), statistics.mean(mems)


def bench_load_pandas(path):
    import pandas as pd
    times = []
    for _ in range(LOAD_RUNS):
        t0 = time.perf_counter()
        df = pd.read_csv(path, index_col=0)
        times.append((time.perf_counter() - t0) * 1000)
    return df, statistics.mean(times)


def compute_cov(df_polars):
    def _run():
        mat   = df_polars.select(pl.all().exclude("date")).to_numpy()
        means = mat.mean(axis=0)
        cov   = np.cov(mat, rowvar=False)
        return means, cov
    t0 = time.perf_counter()
    (means, cov), mem = peak_mem_mb(_run)
    ms = (time.perf_counter() - t0) * 1000
    return means, cov, ms, mem


# ── Shared objective helpers ──────────────────────────────────────────────────

def _portfolio(w_raw, means, cov):
    """Normalize weights, return (variance, -return)."""
    w = np.asarray(w_raw, dtype=float)
    t = w.sum()
    w = w / t if t > 1e-12 else np.full(len(w), 1.0 / len(w))
    return float(w @ cov @ w), -float(w @ means)


# ── Optimizers ────────────────────────────────────────────────────────────────

def bench_rustypus(means, cov, mode="sequential"):
    import rustypus as rp

    def obj(x):
        var, neg_ret = _portfolio(x, means, cov)
        return [var, neg_ret]

    def one_run():
        p  = rp.Problem(N_ASSETS, 2, [rp.Real(0.0, 1.0)] * N_ASSETS, obj, [-1, -1])
        ga = rp.NSGAII(p, POP, mode)
        ga.run(NFE)

    one_run()  # warm-up
    times, mems = [], []
    for _ in range(OPT_RUNS):
        t0 = time.perf_counter()
        _, mem = peak_mem_mb(one_run)
        times.append((time.perf_counter() - t0) * 1000)
        mems.append(mem)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, max(mems)


def bench_pymoo(means, cov):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as _Prob
    from pymoo.optimize import minimize as _min

    class _Port(_Prob):
        def __init__(self):
            super().__init__(n_var=N_ASSETS, n_obj=2, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            totals = x.sum(axis=1, keepdims=True).clip(1e-12)
            w = x / totals
            out["F"] = np.column_stack([
                np.einsum("pi,ij,pj->p", w, cov, w),
                -(w @ means),
            ])

    def one_run():
        _min(_Port(), NSGA2(pop_size=POP), ("n_eval", NFE), verbose=False, seed=None)

    one_run()  # warm-up
    times, mems = [], []
    for _ in range(OPT_RUNS):
        t0 = time.perf_counter()
        _, mem = peak_mem_mb(one_run)
        times.append((time.perf_counter() - t0) * 1000)
        mems.append(mem)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, max(mems)


def bench_platypus(means, cov):
    from platypus import NSGAII as _NSGA, Problem as _Prob, Real as _Real

    def obj(v):
        var, neg_ret = _portfolio(v, means, cov)
        return [var, neg_ret]

    def one_run():
        p = _Prob(N_ASSETS, 2)
        p.types[:] = _Real(0.0, 1.0)
        p.function = obj
        _NSGA(p, population_size=POP).run(NFE)

    one_run()  # warm-up
    times, mems = [], []
    for _ in range(OPT_RUNS):
        t0 = time.perf_counter()
        _, mem = peak_mem_mb(one_run)
        times.append((time.perf_counter() - t0) * 1000)
        mems.append(mem)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, max(mems)


def bench_deap(means, cov):
    from deap import base, creator, tools

    if not hasattr(creator, "_PortFit"):
        creator.create("_PortFit", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "_PortInd"):
        creator.create("_PortInd", list, fitness=creator._PortFit)

    tb = base.Toolbox()
    tb.register("attr", random.random)
    tb.register("individual", tools.initRepeat, creator._PortInd, tb.attr, n=N_ASSETS)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: _portfolio(ind, means, cov))
    tb.register("mate",   tools.cxSimulatedBinaryBounded, eta=20, low=0.0, up=1.0)
    tb.register("mutate", tools.mutPolynomialBounded, eta=20, low=0.0, up=1.0,
                indpb=1.0 / N_ASSETS)
    tb.register("select", tools.selNSGA2)
    GENS = (NFE - POP) // POP

    def one_run():
        pop = tb.population(n=POP)
        for ind, fit in zip(pop, map(tb.evaluate, pop)):
            ind.fitness.values = fit
        pop = tb.select(pop, len(pop))
        for _ in range(GENS):
            offspring = [tb.clone(ind) for ind in tools.selTournamentDCD(pop, len(pop))]
            for i1, i2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.9:
                    tb.mate(i1, i2)
                tb.mutate(i1); tb.mutate(i2)
                del i1.fitness.values; del i2.fitness.values
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = tb.evaluate(ind)
            pop = tb.select(pop + offspring, POP)

    one_run()  # warm-up
    times, mems = [], []
    for _ in range(OPT_RUNS):
        t0 = time.perf_counter()
        _, mem = peak_mem_mb(one_run)
        times.append((time.perf_counter() - t0) * 1000)
        mems.append(mem)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, max(mems)


# ── Runner ────────────────────────────────────────────────────────────────────

OPTIMIZERS = [
    ("rustypus (py/seq)", lambda m, c: bench_rustypus(m, c, "sequential")),
    ("rustypus (py/par)", lambda m, c: bench_rustypus(m, c, "multithreaded")),
    ("pymoo",             bench_pymoo),
    ("platypus",          bench_platypus),
    ("DEAP",              bench_deap),
]


def fmt_row(name, opt_ms, opt_std, preproc_ms):
    total = preproc_ms + opt_ms
    return f"  {name:<26} {opt_ms:>8.1f}  {opt_std:>6.1f}   {total:>9.1f}"


# ── Main ──────────────────────────────────────────────────────────────────────

print("\n╔═ Portfolio Optimization — All Libraries × All Dataset Sizes ════════════════╗")
print(f"║  N_ASSETS={N_ASSETS}  pop={POP}  NFE={NFE}  {OPT_RUNS} measured runs + 1 warm-up each           ║")
print( "║  † native Rust: Rust-polars load + serial Rust cov (own preprocessing)    ║")
print( "║  All others:    polars-py load + numpy BLAS cov (shared preprocessing)    ║")
print( "╚════════════════════════════════════════════════════════════════════════════╝")

# Fetch native Rust numbers once up front
rust_data = run_rust_native()

col_hr = "─" * 76

all_results = []

for label, path in DATASETS:
    df, t_load, m_load = bench_load_polars(path)
    _, t_load_pd        = bench_load_pandas(path)
    n_rows              = df.height
    means, cov, t_cov, m_cov = compute_cov(df)
    preproc_py = t_load + t_cov

    print(f"\n┌─ {label.strip()} · {n_rows:,} rows")
    print(f"│  polars-py load: {t_load:.1f} ms ({m_load:.0f} MB)"
          f"  │  pandas load: {t_load_pd:.1f} ms"
          f"  │  numpy cov: {t_cov:.1f} ms ({m_cov:.0f} MB)"
          f"  │  Python preprocessing: {preproc_py:.1f} ms")
    print(col_hr)
    print(f"  {'library':<28} {'opt(ms)':>8} {'±std':>6} {'peak MB':>8}  {'total(ms)':>10}")
    print(col_hr)

    row = {}

    # ── Native Rust rows (from subprocess) ────────────────────────────────────
    size_key = label.strip()
    if size_key in rust_data:
        rd = rust_data[size_key]
        r_preproc = rd["load_ms"] + rd["cov_ms"]
        for rname, ropt, rmem in [
            ("rustypus (native/seq) †", rd["seq_ms"], rd.get("seq_mb", float("nan"))),
            ("rustypus (native/par) †", rd["par_ms"], rd.get("par_mb", float("nan"))),
        ]:
            rtotal = r_preproc + ropt
            mem_s  = f"{rmem:.0f}" if rmem == rmem else "—"
            print(f"  {rname:<28} {ropt:>8.1f} {'—':>6} {mem_s:>8}  {rtotal:>10.1f}")
            row[rname] = (ropt, 0.0, rmem)
        print(f"  {'·'*72}")
        print(f"  {'† Rust load':>14} {rd['load_ms']:.1f} ms ({rd.get('load_mb', float('nan')):.0f} MB)"
              f"  +  cov {rd['cov_ms']:.1f} ms ({rd.get('cov_mb', float('nan')):.0f} MB)"
              f"  =  {r_preproc:.1f} ms preprocessing")
        print(f"  {'·'*72}")

    # ── Python library rows ───────────────────────────────────────────────────
    for name, fn in OPTIMIZERS:
        try:
            ms, std, mem = fn(means, cov)
            row[name] = (ms, std, mem)
            total = preproc_py + ms
            print(f"  {name:<28} {ms:>8.1f} {std:>6.1f} {mem:>8.0f}  {total:>10.1f}")
        except Exception as exc:
            row[name] = (float("nan"), 0.0, float("nan"))
            print(f"  {name:<28} {'ERROR: ' + str(exc)[:40]:>50}")

    print(col_hr)
    all_results.append((label.strip(), n_rows, t_load, t_load_pd, t_cov, row))

# ── Cross-size summary ────────────────────────────────────────────────────────
print("\n┌─ Optimization: time (ms) and peak RSS (MB) across dataset sizes")
header = f"  {'library':<30}"
for label, *_ in all_results:
    header += f"  {label+' ms':>10} {label+' MB':>10}"
print(header)
print("─" * (34 + 22 * len(all_results)))
first_row = all_results[0][-1]
for name in first_row:
    line = f"  {name:<30}"
    for *_, row in all_results:
        ms, _, mem = row.get(name, (float("nan"), 0.0, float("nan")))
        ms_s  = f"{ms:>9.1f}" if ms  == ms  else f"{'—':>9}"
        mem_s = f"{mem:>9.0f}" if mem == mem else f"{'—':>9}"
        line += f"  {ms_s} {mem_s}"
    print(line)

# ── Load comparison summary ────────────────────────────────────────────────────
print("\n┌─ CSV load time: polars-py vs pandas")
print(f"  {'size':<8} {'rows':>8}  {'polars-py':>10}  {'pandas':>8}  {'speedup':>8}")
print("─" * 52)
for label, n_rows, t_pl, t_pd, t_cov, _ in all_results:
    spd = t_pd / t_pl if t_pl > 0 else 0
    print(f"  {label:<8} {n_rows:>8}  {t_pl:>8.1f} ms  {t_pd:>6.1f} ms  {spd:>6.1f}×")

print(f"\n# polars-py {pl.__version__}  numpy {np.__version__}")
print("# total = polars-py load + numpy cov + opt")
print("# native Rust numbers (polars-Rust load + serial cov + rustypus) in bench_large_data.rs output")
