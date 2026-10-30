"""Watch puggles optimize in the browser.

    python dashboard_demo.py            # ZDT1, 2 objectives, NSGA-II
    python dashboard_demo.py dtlz2-3    # 3 objectives, NSGA-III (3-D view)
    python dashboard_demo.py dtlz2-5    # 5 objectives, NSGA-III (parallel coordinates)
    python dashboard_demo.py dtlz2-5 --gpu   # objectives in a WGSL shader on the GPU (NSGA-II)

Needs the add-on: pip install "puggles[dashboard]"  (or build both: see python-dashboard/README.md).
--gpu needs the GPU build of the core:  maturin develop --release --features gpu -m python/Cargo.toml

Ctrl+C exits once you're done exploring. Add --save run.pgrun.json.gz to keep the run, then
replay it any time with `python -m puggles.dashboard run.pgrun.json.gz`.
"""

import argparse
import math
import time

import puggles as pg


def zdt1(x):
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    return [x[0], g * (1 - math.sqrt(x[0] / g))]


def dtlz2(m):
    def f(x):
        g = sum((v - 0.5) ** 2 for v in x[m - 1:])
        out = []
        for i in range(m):
            v = 1 + g
            for xj in x[: m - 1 - i]:
                v *= math.cos(xj * math.pi / 2)
            if i > 0:
                v *= math.sin(x[m - 1 - i] * math.pi / 2)
            out.append(v)
        return out
    return f


# The same objectives as WGSL compute shaders, one invocation per solution.
WGSL_HEADER = """
@group(0) @binding(0) var<storage, read> solutions: array<f32>;
@group(0) @binding(1) var<storage, read_write> objectives: array<f32>;
struct Params { solution_length: u32, num_objectives: u32, pop_size: u32 };
@group(0) @binding(2) var<uniform> params: Params;
"""

ZDT1_WGSL = WGSL_HEADER + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if (s >= params.pop_size) { return; }
    let n = params.solution_length;
    let off = s * n;
    var rest = 0.0;
    for (var i = 1u; i < n; i = i + 1u) { rest = rest + solutions[off + i]; }
    let x0 = solutions[off];
    let g = 1.0 + 9.0 * rest / f32(n - 1u);
    objectives[s * 2u] = x0;
    objectives[s * 2u + 1u] = g * (1.0 - sqrt(x0 / g));
}
"""

DTLZ2_WGSL = WGSL_HEADER + """
const HALF_PI: f32 = 1.5707963;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if (s >= params.pop_size) { return; }
    let n = params.solution_length;
    let m = params.num_objectives;
    let off = s * n;
    var g = 0.0;
    for (var i = m - 1u; i < n; i = i + 1u) { let d = solutions[off + i] - 0.5; g = g + d * d; }
    for (var k = 0u; k < m; k = k + 1u) {
        var f = 1.0 + g;
        for (var j = 0u; j < m - 1u - k; j = j + 1u) { f = f * cos(solutions[off + j] * HALF_PI); }
        if (k > 0u) { f = f * sin(solutions[off + m - 1u - k] * HALF_PI); }
        objectives[s * m + k] = f;
    }
}
"""


def paced(f, seconds):
    """Slow a cheap objective down so the evolution is visible."""
    def g(x):
        time.sleep(seconds)
        return f(x)
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", nargs="?", default="zdt1", choices=["zdt1", "dtlz2-3", "dtlz2-5"])
    ap.add_argument("--nfe", type=int, help="evaluation budget (default 30,000; 1,000,000 with --gpu)")
    ap.add_argument("--pace", type=float, default=0.00005, help="seconds of sleep per evaluation (CPU only)")
    ap.add_argument("--gpu", action="store_true", help="evaluate the objectives on the GPU")
    ap.add_argument("--save", help="save the run to this .pgrun.json[.gz] file")
    args = ap.parse_args()

    if args.problem == "zdt1":
        n, m, f = 10, 2, zdt1
        names = ["f1", "f2"]
    else:
        m = int(args.problem[-1])
        n, f = m + 9, dtlz2(m)
        names = [f"f{i + 1}" for i in range(m)]

    if args.gpu:
        if not hasattr(pg, "GpuProblem"):
            raise SystemExit("this puggles was built without GPU support; rebuild it with:\n"
                             "    maturin develop --release --features gpu")
        # The GPU evaluates whole generations at once, so it runs NSGA-II (NSGA-III evaluates
        # one solution at a time). There's no Python in the loop, so it runs at full speed.
        problem = pg.GpuProblem(
            solution_length=n,
            number_of_objectives=m,
            solution_data_types=[pg.Real(0.0, 1.0)] * n,
            shader_wgsl=ZDT1_WGSL if args.problem == "zdt1" else DTLZ2_WGSL,
            direction=[-1] * m,
        )
        ga = pg.NSGAII(problem, population_size=100, execution_mode="gpu", seed=42)
        nfe = args.nfe or 1_000_000
    else:
        problem = pg.Problem(
            solution_length=n,
            number_of_objectives=m,
            solution_data_types=[pg.Real(0.0, 1.0)] * n,
            objective_function=paced(f, args.pace),
            direction=[-1] * m,
        )
        ga = pg.NSGAII(problem, population_size=100, seed=42) if m == 2 else pg.NSGAIII(problem, divisions=12 if m == 3 else 4, seed=42)
        nfe = args.nfe or 30_000

    # The true fronts are known for these benchmarks: drawn faintly, and used for IGD.
    true_front = pg.reference_front("zdt1" if args.problem == "zdt1" else "dtlz2", m, 300)
    dash = pg.Dashboard(ga, objective_names=names, title=args.problem.upper() + (" · GPU" if args.gpu else ""),
                        reference_front=true_front)
    ga.run(nfe, callback=dash)
    print(f"done: {ga.nfe} evaluations, {len(ga.get_archive())} solutions on the front")
    if args.save:
        print("saved", dash.save(args.save))
    dash.wait()


if __name__ == "__main__":
    main()
