"""Compare optimizers on one dashboard: every finished run stays overlaid on the next.

    python compare_runs.py                # NSGA-II vs NSGA-III on DTLZ2 (3 objectives), 3 seeds each
    python compare_runs.py --objectives 5 --seeds 5

Each run's front appears in its own color (at the same evaluation budget as the run you're
scrubbing). Pick "Median by group" above the convergence chart for median + IQR bands, and open
the Results tab for each group's final IGD / hypervolume with a significance test. Ctrl+C exits.
"""

import argparse
import math

import puggles as pg


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objectives", type=int, default=3)
    ap.add_argument("--nfe", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    m = args.objectives
    n = m + 9

    problem = pg.Problem(
        solution_length=n,
        number_of_objectives=m,
        solution_data_types=[pg.Real(0.0, 1.0)] * n,
        objective_function=dtlz2(m),
        direction=[-1] * m,
    )
    # A column of your own: DTLZ2's objective vector has length 1 + g, so 1.0 means "on the true front".
    radius = lambda x: math.sqrt(sum(v * v for v in dtlz2(m)(x)))
    dash = pg.Dashboard(problem, title=f"DTLZ2 · {m} objectives", keep_runs=True,
                        reference_front=pg.reference_front("dtlz2", m, 300), columns={"radius": radius})
    for name, make in (("NSGA-II", lambda s: pg.NSGAII(problem, population_size=100, seed=s)),
                       ("NSGA-III", lambda s: pg.NSGAIII(problem, divisions=12 if m == 3 else 4, seed=s))):
        for seed in range(1, args.seeds + 1):
            dash.set_group(name)  # applies to the run that starts next
            ga = make(seed)
            ga.run(args.nfe, callback=dash)
            print(f"{name} seed {seed}: {len(ga.get_archive())} solutions on the front")
    print("runs on the dashboard:", dash.comparisons + ["(latest)"])
    dash.wait()


if __name__ == "__main__":
    main()
