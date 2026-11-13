#!/usr/bin/env python3
"""Dataset-scaling benchmark: 1 KB → 1 GB returns CSVs, single- and multi-objective.

Every evaluation scans EVERY row (T rows × 20 assets), so the cost of an evaluation
grows with the dataset. w = x / Σx, r_t = Σ_i w_i·R_ti:
  multi  → minimize realized variance mean(r²) − mean(r)², maximize mean(r)
  single → maximize Sharpe mean(r) / std(r)
Same problem, same population (100) and same evaluation budget (NFE) for every library.

Two roles:
  python bench_datasets.py                     driver: every (size, mode, library, run) in its
                                               own process under /usr/bin/time -l → isolated peak memory
                                               footprint + CPU time; writes results_datasets.json and
                                               prints markdown tables.
  python bench_datasets.py worker <lib> <csv> <multi|single> <nfe> <seed>
                                               one Python-library run (prints RESULT/FRONT/BEST,
                                               the same format as examples/bench_datasets.rs).

Driver flags: --sizes 1kb,1mb  --nfe 500  --runs 1  --libs pymoo,puggles-seq  --out file.json  --resume 1
              --prefix real_returns_ (default; real-price data) | returns_ (synthetic)
Prereqs: cargo build --release --features gpu,cubecl --example bench_datasets --example bench_cubecl
         (cd ../python && DEVELOPER_DIR=/Library/Developer/CommandLineTools \
              ../sandbox/.venv/bin/maturin develop --release)
         optional, Linux + NVIDIA only: (cd ../puggles-cuda && cargo build --release --example bench_cuda)
"""
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N, POP = 20, 100
CHUNK = 65_536  # rows per block in the vectorized objective (bounds the T×POP temporary)

# (label, NFE, runs). The big files get a smaller budget and one run (user-approved).
SIZES = [("1kb", 10_000, 3), ("1mb", 10_000, 3), ("10mb", 10_000, 3),
         ("100mb", 2_000, 1), ("1gb", 2_000, 1)]
RUST_LIBS = ["puggles-seq", "puggles-par", "puggles-rowscan", "puggles-auto", "puggles-gpu", "puggles-cubecl", "puggles-cuda", "genevo", "ga"]
# Rust workers other than examples/bench_datasets (same CLI and output lines).
BINARIES = {
    "puggles-cubecl": os.path.join(HERE, "target/release/examples/bench_cubecl"),
    "puggles-cuda": os.path.join(HERE, "..", "puggles-cuda/target/release/examples/bench_cuda"),
}
PY_LIBS = ["puggles-py", "puggles-py-batch", "puggles-py-numba", "puggles-py-cfunc", "puggles-py-torch",
           "puggles-py-jax", "pymoo", "pymoo-numba", "platypus", "deap"]
SINGLE_ONLY = {"genevo", "ga"}
TIMEOUT_S = 900
NAMES = {
    "puggles-seq": "puggles (native, sequential)", "puggles-par": "puggles (native, 8 threads)",
    "puggles-gpu": "puggles (native, GPU)", "genevo": "genevo", "ga": "genetic_algorithm",
    "puggles-rowscan": "puggles (native, CPU row scan)", "puggles-auto": "puggles (native, Auto)",
    "puggles-cubecl": "puggles (native, CubeCL GPU)", "puggles-cuda": "puggles (native, CUDA, Rust-CUDA)",
    "puggles-py": "puggles (py)", "puggles-py-batch": "puggles (py, batch)",
    "puggles-py-numba": "puggles (py, numba batch)", "pymoo-numba": "pymoo (numba)",
    "puggles-py-cfunc": "puggles (py, numba cfunc, 8 threads)", "puggles-py-torch": "puggles (py, torch batch)",
    "puggles-py-jax": "puggles (py, jax jit+vmap)",
    "pymoo": "pymoo", "platypus": "platypus", "deap": "DEAP",
}


# ══ worker ═════════════════════════════════════════════════════════════════════

R = None   # (T, N) float64 returns
EVALS = 0  # real objective evaluations, counted the same way for every Python library


def load(path):
    import polars as pl
    global R
    t0 = time.perf_counter()
    R = np.ascontiguousarray(pl.read_csv(path).drop("date", "scenario", strict=False).to_numpy(), dtype=np.float64)
    return (time.perf_counter() - t0) * 1000


def stats(x):
    """(mean, var) of one portfolio's per-row return."""
    global EVALS
    EVALS += 1
    x = np.asarray(x, dtype=np.float64)
    t = x.sum()
    w = x / t if t > 1e-12 else np.full(N, 1.0 / N)
    r = R @ w
    m = r.mean()
    return m, max(r @ r / len(r) - m * m, 0.0)


def batch_stats(X):
    """(means, vars) for a whole population, accumulated over row blocks."""
    global EVALS
    X = np.asarray(X, dtype=np.float64)
    EVALS += len(X)
    t = X.sum(1, keepdims=True)
    W = np.where(t > 1e-12, X / np.where(t > 1e-12, t, 1.0), 1.0 / N)
    s, s2 = np.zeros(len(X)), np.zeros(len(X))
    for c in range(0, len(R), CHUNK):
        rc = R[c:c + CHUNK] @ W.T
        s += rc.sum(0)
        s2 += np.einsum("ij,ij->j", rc, rc)
    m = s / len(R)
    return m, np.maximum(s2 / len(R) - m * m, 0.0)


def numba_batch_stats():
    """`batch_stats` compiled with Numba. Threads split the ROWS into blocks; for each row every
    solution's running sums are updated while the row is in cache, so the data is read once per
    generation instead of once per solution. Compiled here, before the clock starts."""
    import numba as nb

    @nb.njit(parallel=True)
    def kernel(WT, R, nblk):
        # WT is (N, P): the innermost loops run over solutions p, which are contiguous and
        # independent, so they vectorize (SIMD) instead of being a 20-term dot-product chain.
        T, P = R.shape[0], WT.shape[1]
        S, S2 = np.zeros((nblk, P)), np.zeros((nblk, P))
        for b in nb.prange(nblk):
            s, s2, r = np.zeros(P), np.zeros(P), np.empty(P)
            for t in range(b * T // nblk, (b + 1) * T // nblk):
                r[:] = 0.0
                for i in range(WT.shape[0]):
                    x = R[t, i]
                    for p in range(P):
                        r[p] += WT[i, p] * x
                for p in range(P):
                    s[p] += r[p]
                    s2[p] += r[p] * r[p]
            S[b], S2[b] = s, s2
        return S.sum(0), S2.sum(0)

    t0 = time.perf_counter()
    kernel(np.full((N, 2), 1.0 / N), R[:8], 1)
    print(f"# numba compile {(time.perf_counter() - t0) * 1000:.0f} ms", file=sys.stderr)
    nblk = max(1, min(len(R) // 256, nb.get_num_threads() * 8))

    def stats(X):
        global EVALS
        X = np.asarray(X, dtype=np.float64)
        EVALS += len(X)
        t = X.sum(1, keepdims=True)
        W = np.where(t > 1e-12, X / np.where(t > 1e-12, t, 1.0), 1.0 / N)
        s, s2 = kernel(np.ascontiguousarray(W.T), R, nblk)
        m = s / len(R)
        return m, np.maximum(s2 / len(R) - m * m, 0.0)
    return stats


BATCH = batch_stats  # swapped for the Numba version by the *-numba libraries


def sharpe(m, v):
    return m / np.sqrt(v) if v > 0 else 0.0


def obj(x, multi):
    m, v = stats(x)
    return [v, -m] if multi else [-sharpe(m, v)]


def batch_obj(X, multi):
    m, v = BATCH(X)
    if multi:
        return np.column_stack([v, -m])
    return (-np.where(v > 0, m / np.sqrt(np.where(v > 0, v, 1.0)), 0.0))[:, None]


def run_puggles_py(multi, nfe, seed, batch, kw=None, mode="sequential"):
    """puggles from Python. Batch objectives get the population as one ndarray and return one."""
    global EVALS
    import puggles as rp
    if kw is None:
        kw = ({"batch_objective_function": lambda pop: batch_obj(pop, multi), "batch_numpy": True} if batch
              else {"objective_function": lambda x: obj(x, multi)})
    p = rp.Problem(solution_length=N, number_of_objectives=2 if multi else 1,
                   solution_data_types=[rp.Real(0.0, 1.0)] * N,
                   direction=[-1, -1] if multi else [-1], **kw)
    ga = rp.NSGAII(p, population_size=POP, execution_mode=mode, seed=seed)
    ga.run(nfe)
    if "objective_cfunc" in kw:
        EVALS = ga.nfe  # the compiled objective can't touch a Python counter
    return [list(s.objectives) for s in ga.get_archive()]


def finish(s, s2, multi):
    """(Σr, Σr²) per solution → objectives array, in f64."""
    m = s / len(R)
    v = np.maximum(s2 / len(R) - m * m, 0.0)
    if multi:
        return np.column_stack([v, -m])
    return (-np.where(v > 0, m / np.sqrt(np.where(v > 0, v, 1.0)), 0.0))[:, None]


def normalized(X):
    X = np.asarray(X, dtype=np.float64)
    t = X.sum(1, keepdims=True)
    return np.where(t > 1e-12, X / np.where(t > 1e-12, t, 1.0), 1.0 / N)


def cfunc_objective(multi):
    """Per-solution objective as a Numba @cfunc: puggles calls it from Rust with no GIL, so its
    8 threads run it in parallel. Like native puggles, each call scans the whole dataset."""
    import numba as nb
    T = len(R)

    @nb.cfunc("void(CPointer(float64), int64, CPointer(float64), int64, voidptr)")
    def f(x, n, out, m, data):
        Rv = nb.carray(data, (T, n), dtype=np.float64)
        tot = 0.0
        for i in range(n):
            tot += x[i]
        s = s2 = 0.0
        for t in range(T):
            r = 0.0
            for i in range(n):
                r += x[i] * Rv[t, i]
            r = r / tot if tot > 1e-12 else Rv[t].sum() / n
            s += r
            s2 += r * r
        mean = s / T
        var = max(s2 / T - mean * mean, 0.0)
        if m == 2:
            out[0] = var
            out[1] = -mean
        else:
            out[0] = -mean / np.sqrt(var) if var > 0 else 0.0
    return {"objective_cfunc": f, "cfunc_data": R}


def torch_batch(multi):
    """Batch objective on the Apple GPU via PyTorch MPS (f32: MPS has no f64). Row chunks keep
    the (chunk × P) intermediate small; the per-solution sums come back once per generation."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    Rt = torch.as_tensor(R, dtype=torch.float32).to(dev)  # upload counts as optimizer time, like the GPU row
    chunk = 262_144

    def f(X):
        global EVALS
        EVALS += len(X)
        W = torch.as_tensor(normalized(X).T, dtype=torch.float32).to(dev)
        s = torch.zeros(len(X), device=dev)
        s2 = torch.zeros(len(X), device=dev)
        for c in range(0, Rt.shape[0], chunk):
            rc = Rt[c:c + chunk] @ W
            s += rc.sum(0)
            s2 += (rc * rc).sum(0)
        return finish(s.cpu().double().numpy(), s2.cpu().double().numpy(), multi)
    return {"batch_objective_function": f, "batch_numpy": True}


def jax_batch(multi):
    """Single-solution objective vectorized by `jax.jit(jax.vmap(f))`, on the CPU here. Rows are
    scanned in blocks so no T × P intermediate is built; the data is passed as an argument
    (closing over it would embed 0.9 GB in the compiled program). Compiled before the clock."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    B = 8192
    nb_ = len(R) // B
    Rm = jnp.asarray(R[:nb_ * B].reshape(nb_, B, N))
    Rtail = jnp.asarray(R[nb_ * B:])

    def one(w, Rm, Rtail):
        def body(c, blk):
            r = blk @ w
            return (c[0] + r.sum(), c[1] + (r * r).sum()), None
        (s, s2), _ = jax.lax.scan(body, (0.0, 0.0), Rm)
        r = Rtail @ w
        return s + r.sum(), s2 + (r * r).sum()

    batched = jax.jit(jax.vmap(one, in_axes=(0, None, None)))
    jax.block_until_ready(batched(jnp.asarray(normalized(np.ones((POP, N)))), Rm, Rtail))  # compile

    def f(X):
        global EVALS
        EVALS += len(X)
        s, s2 = batched(jnp.asarray(normalized(X)), Rm, Rtail)
        return finish(np.asarray(s), np.asarray(s2), multi)
    return {"batch_objective_function": f, "batch_numpy": True}


def run_pymoo(multi, nfe, seed):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    class P(Problem):
        def __init__(s):
            super().__init__(n_var=N, n_obj=2 if multi else 1, xl=0.0, xu=1.0)

        def _evaluate(s, x, out, *a, **k):
            out["F"] = batch_obj(x, multi)

    alg = NSGA2(pop_size=POP) if multi else GA(pop_size=POP)
    res = minimize(P(), alg, ("n_eval", nfe), seed=seed, verbose=False)
    return np.atleast_2d(res.F).tolist()


def run_platypus(multi, nfe, seed):
    from platypus import NSGAII, GeneticAlgorithm, Problem, Real, nondominated
    random.seed(seed)
    p = Problem(N, 2 if multi else 1)
    p.types[:] = Real(0.0, 1.0)
    p.function = lambda v: obj(v, multi)
    alg = NSGAII(p, population_size=POP) if multi else GeneticAlgorithm(p, population_size=POP)
    alg.run(nfe)
    return [list(s.objectives) for s in (nondominated(alg.result) if multi else alg.result)]


def run_deap(multi, nfe, seed):
    from deap import base, creator, tools
    random.seed(seed)
    n_obj = 2 if multi else 1
    fit, ind = f"_F{n_obj}", f"_I{n_obj}"
    creator.create(fit, base.Fitness, weights=(-1.0,) * n_obj)
    creator.create(ind, list, fitness=getattr(creator, fit))
    tb = base.Toolbox()
    tb.register("ind", tools.initRepeat, getattr(creator, ind), random.random, n=N)
    tb.register("evaluate", lambda i: tuple(obj(i, multi)))
    tb.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=0.0, up=1.0)
    tb.register("mutate", tools.mutPolynomialBounded, eta=20, low=0.0, up=1.0, indpb=1.0 / N)
    pop = [tb.ind() for _ in range(POP)]
    for i in pop:
        i.fitness.values = tb.evaluate(i)
    if multi:
        pop = tools.selNSGA2(pop, POP)  # assigns crowding distance for selTournamentDCD
    for _ in range((nfe - POP) // POP):
        parents = tools.selTournamentDCD(pop, POP) if multi else tools.selTournament(pop, POP, 3)
        off = [tb.clone(i) for i in parents]
        for a, b in zip(off[::2], off[1::2]):
            if random.random() < 0.9:
                tb.mate(a, b)
            tb.mutate(a)
            tb.mutate(b)
            del a.fitness.values, b.fitness.values
        for i in off:
            if not i.fitness.valid:
                i.fitness.values = tb.evaluate(i)
        # NSGA-II survival for multi, (μ+λ) elitism for single
        pop = tools.selNSGA2(pop + off, POP) if multi else tools.selBest(pop + off, POP)
    best = tools.sortNondominated(pop, POP, first_front_only=True)[0] if multi else pop
    return [list(i.fitness.values) for i in best]


def worker(lib, path, mode, nfe, seed):
    global BATCH
    multi = mode == "multi"
    load_ms = load(path)
    if lib.endswith("-numba"):
        BATCH = numba_batch_stats()
    # Compiled objectives are built (and JIT-compiled) before the clock, like Numba above.
    kw = {"puggles-py-cfunc": lambda: cfunc_objective(multi),
          "puggles-py-jax": lambda: jax_batch(multi)}.get(lib, lambda: None)()
    runners = {
        "puggles-py": lambda: run_puggles_py(multi, nfe, seed, batch=False),
        "puggles-py-batch": lambda: run_puggles_py(multi, nfe, seed, batch=True),
        "puggles-py-numba": lambda: run_puggles_py(multi, nfe, seed, batch=True),
        "puggles-py-cfunc": lambda: run_puggles_py(multi, nfe, seed, True, kw, mode="multithreaded"),
        "puggles-py-torch": lambda: run_puggles_py(multi, nfe, seed, True, torch_batch(multi)),
        "puggles-py-jax": lambda: run_puggles_py(multi, nfe, seed, True, kw),
        "pymoo": lambda: run_pymoo(multi, nfe, seed),
        "pymoo-numba": lambda: run_pymoo(multi, nfe, seed),
        "platypus": lambda: run_platypus(multi, nfe, seed),
        "deap": lambda: run_deap(multi, nfe, seed),
    }
    # Import the library before the clock starts: a cold import isn't optimizer time.
    __import__({"pymoo": "pymoo.optimize", "pymoo-numba": "pymoo.optimize", "platypus": "platypus",
                "deap": "deap.tools", "puggles-py-torch": "torch"}.get(lib, "puggles"))
    __import__("puggles")
    t0 = time.perf_counter()
    F = runners[lib]()
    opt_ms = (time.perf_counter() - t0) * 1000
    print(f"RESULT\t{load_ms:.1f}\t{opt_ms:.1f}\t{EVALS}")
    if multi:
        for f in F:
            print(f"FRONT\t{f[0]:e}\t{-f[1]:e}")
    else:
        print(f"BEST\t{-min(f[0] for f in F):.6f}")


# ══ driver ═════════════════════════════════════════════════════════════════════

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def hardware():
    import importlib.metadata as md
    vers = {}
    for pkg in ["puggles", "pymoo", "platypus-opt", "deap", "numpy", "polars", "numba", "torch", "jax"]:
        try:
            vers[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            vers[pkg] = "?"
    common = {"rustc": sh("rustc --version"), "python": sys.version.split()[0], "versions": vers}
    if sys.platform == "linux":
        lscpu = sh("lscpu")
        field = lambda k: (re.search(rf"^{k}:\s*(.+)$", lscpu, re.M) or [None, "?"])[1].strip()
        return {
            "cpu": field("Model name"),
            "cores": sh("nproc"),
            "threads_per_core": field("Thread\\(s\\) per core"),
            "ram_gb": round(int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read())[1]) / 2**20),
            "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader") or "none",
            "cuda": sh("nvcc --version | tail -2 | head -1") or "?",
            "os": sh(". /etc/os-release && echo $PRETTY_NAME") + " (" + sh("uname -r") + ")",
            "instance": sh("curl -s -m 2 -H 'Metadata-Flavor: Google' "
                           "http://metadata.google.internal/computeMetadata/v1/instance/machine-type | sed 's|.*/||'") or "?",
            **common,
        }
    gpu = re.search(r"Total Number of Cores: (\d+)", sh("system_profiler SPDisplaysDataType"))
    return {
        "cpu": sh("sysctl -n machdep.cpu.brand_string"),
        "cores": sh("sysctl -n hw.ncpu"),
        "p_cores": sh("sysctl -n hw.perflevel0.physicalcpu"),
        "e_cores": sh("sysctl -n hw.perflevel1.physicalcpu"),
        "gpu_cores": gpu.group(1) if gpu else "?",
        "ram_gb": round(int(sh("sysctl -n hw.memsize") or 0) / 2**30),
        "os": f"macOS {sh('sw_vers -productVersion')} ({sh('sw_vers -buildVersion')})",
        **common,
    }


def run_one(lib, size, mode, nfe, seed, prefix):
    path = os.path.join(HERE, "data", f"{prefix}{size}.csv")
    if lib in PY_LIBS:
        cmd = [sys.executable, os.path.abspath(__file__), "worker", lib, path, mode, str(nfe), str(seed)]
    else:
        binary = BINARIES.get(lib, os.path.join(HERE, "target/release/examples/bench_datasets"))
        cmd = [binary, path, mode, lib, str(nfe), str(seed)]
    rec = {"lib": lib, "size": size, "mode": mode, "nfe": nfe, "seed": seed}
    if lib not in PY_LIBS and not os.path.exists(cmd[0]):  # e.g. puggles-cuda, which needs Linux + an NVIDIA GPU
        return {**rec, "status": "not built"}
    t0 = time.perf_counter()
    try:
        # BSD time on macOS (-l), GNU time on Linux (-v): different flags and report formats.
        tflag = "-v" if sys.platform == "linux" else "-l"
        p = subprocess.run(["/usr/bin/time", tflag, *cmd], capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {**rec, "status": "timeout", "wall_s": TIMEOUT_S}
    rec["wall_s"] = time.perf_counter() - t0
    res = [l.split("\t") for l in p.stdout.splitlines() if l.startswith("RESULT")]
    if p.returncode == 3:  # the worker found no real GPU adapter (see bench_datasets.rs)
        return {**rec, "status": "no GPU"}
    if p.returncode != 0 or not res:
        return {**rec, "status": "error", "err": re.split(r"\s+[\d.]+ real|\tCommand being timed", p.stderr)[0][-800:]}
    _, load_ms, opt_ms, evals = res[0]
    rec.update(status="ok", load_ms=float(load_ms), opt_ms=float(opt_ms), evals=int(evals))
    if sys.platform == "linux":
        num = lambda k: float(re.search(rf"{k}:\s*([\d.]+)", p.stderr)[1])
        # Linux max RSS is reliable here (GCE VMs run without swap), so it is the peak-memory figure.
        rss_mb = num(r"Maximum resident set size \(kbytes\)") / 1024
        rec.update(cpu_s=num(r"User time \(seconds\)") + num(r"System time \(seconds\)"),
                   peak_mb=rss_mb, rss_mb=rss_mb)
    else:
        t = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", p.stderr)
        rss = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
        # Peak footprint counts compressed/swapped pages too, so unlike max RSS it doesn't
        # under-report when macOS is under memory pressure.
        foot = re.search(r"(\d+)\s+peak memory footprint", p.stderr)
        rec.update(cpu_s=float(t.group(2)) + float(t.group(3)), peak_mb=int(foot.group(1)) / 2**20,
                   rss_mb=int(rss.group(1)) / 2**20)
    if mode == "multi":
        rec["front"] = [[float(a), -float(b)] for _, a, b in
                        (l.split("\t") for l in p.stdout.splitlines() if l.startswith("FRONT"))]
    else:
        rec["sharpe"] = max(float(l.split("\t")[1]) for l in p.stdout.splitlines() if l.startswith("BEST"))
    return rec


def add_hypervolume(records):
    """2-D HV in (variance, −mean) space, normalized by the union of every library's fronts
    at that dataset size, reference point (1.1, 1.1), divided by 1.21 so it lies in [0, 1].
    Higher is better; comparable within a size."""
    from pymoo.indicators.hv import HV
    for size in {r["size"] for r in records}:
        rs = [r for r in records if r["size"] == size and r.get("front")]
        if not rs:
            continue
        allp = np.array([p for r in rs for p in r["front"]])
        lo, hi = allp.min(0), allp.max(0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        hv = HV(ref_point=np.array([1.1, 1.1]))
        for r in rs:
            r["hv"] = float(hv((np.array(r["front"]) - lo) / span)) / 1.21
            r["front_size"] = len(r["front"])


def fmt(x, nd=1):
    return "—" if x is None else f"{x:,.{nd}f}"


def tables(records):
    out = []
    for mode in ["multi", "single"]:
        q = "HV ↑" if mode == "multi" else "best Sharpe ↑"
        out.append(f"\n### {'Multi' if mode == 'multi' else 'Single'}-objective\n")
        out.append(f"| size | library | load ms ↓ | opt ms ↓ | µs/eval ↓ | evals/s ↑ | CPU s ↓ | peak memory MB ↓ "
                   f"| {q} |{' front size |' if mode == 'multi' else ''} vs pymoo (opt time) ↑ |")
        ncol = 11 if mode == "multi" else 10
        out.append("|" + " --- |" * ncol)
        for size, _, _ in SIZES:
            rows = [r for r in records if r["size"] == size and r["mode"] == mode]
            if not rows:
                continue
            ok = lambda lib: [r for r in rows if r["lib"] == lib and r["status"] == "ok"]
            mean = lambda lib, k: statistics.mean(r[k] for r in ok(lib)) if ok(lib) else None
            pymoo_ms = mean("pymoo", "opt_ms")
            for lib in RUST_LIBS + PY_LIBS:
                rs = [r for r in rows if r["lib"] == lib]
                if not rs:
                    continue
                if not ok(lib):
                    out.append(f"| {size} | {NAMES[lib]} | {rs[0]['status']} |" + " |" * (ncol - 3))
                    continue
                opt = mean(lib, "opt_ms")
                evals = mean(lib, "evals")
                sd = statistics.stdev(r["opt_ms"] for r in ok(lib)) if len(ok(lib)) > 1 else None
                opt_s = fmt(opt) + (f" ±{sd:,.0f}" if sd is not None else "")
                qual = fmt(mean(lib, "hv"), 4) if mode == "multi" else fmt(mean(lib, "sharpe"), 4)
                front = f" {fmt(mean(lib, 'front_size'), 0)} |" if mode == "multi" else ""
                vs = f"{pymoo_ms / opt:.2f}×" if pymoo_ms else "—"
                out.append(f"| {size} | {NAMES[lib]} | {fmt(mean(lib, 'load_ms'))} | {opt_s} "
                           f"| {fmt(opt * 1000 / evals, 2)} | {fmt(evals / opt * 1000, 0)} "
                           f"| {fmt(mean(lib, 'cpu_s'), 2)} | {fmt(mean(lib, 'peak_mb'), 0)} "
                           f"| {qual} |{front} {vs} |")
    return "\n".join(out)


def driver(args):
    opt = dict(zip(args[::2], args[1::2]))
    sizes = opt.get("--sizes", ",".join(s for s, _, _ in SIZES)).split(",")
    libs = opt.get("--libs", ",".join(RUST_LIBS + PY_LIBS)).split(",")
    out = opt.get("--out", os.path.join(HERE, "results_datasets.json"))
    prefix = opt.get("--prefix", "real_returns_")  # "returns_" = the older synthetic files
    # --resume 1: keep finished runs from an earlier (e.g. preempted) invocation with the same --out.
    records = json.load(open(out))["records"] if opt.get("--resume") and os.path.exists(out) else []
    finished = {(r["lib"], r["size"], r["mode"], r["seed"]) for r in records if r["status"] in ("ok", "not built", "no GPU")}
    for size, nfe, runs in SIZES:
        if size not in sizes:
            continue
        nfe, runs = int(opt.get("--nfe", nfe)), int(opt.get("--runs", runs))
        for mode in ["multi", "single"]:
            for lib in libs:
                if mode == "multi" and lib in SINGLE_ONLY:
                    continue
                for run in range(runs):
                    if (lib, size, mode, run + 1) in finished:
                        continue
                    r = run_one(lib, size, mode, nfe, seed=run + 1, prefix=prefix)
                    records.append(r)
                    print(f"{size:>6} {mode:<6} {lib:<17} run{run} {r['status']:<7} "
                          f"opt={fmt(r.get('opt_ms'))}ms mem={fmt(r.get('peak_mb'), 0)}MB "
                          f"{r.get('err', '')[-200:]}", flush=True)
                    if r["status"] in ("timeout", "not built", "no GPU"):
                        break  # remaining runs would fail the same way
                    json.dump({"records": records}, open(out, "w"))  # checkpoint
    add_hypervolume(records)
    hw = hardware()
    json.dump({"hardware": hw, "records": records}, open(out, "w"), indent=1)
    print("\nHARDWARE", json.dumps(hw))
    print(tables(records))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        lib, path, mode, nfe, seed = sys.argv[2:7]
        worker(lib, path, mode, int(nfe), int(seed))
    else:
        driver(sys.argv[1:])
