"""Compare sequential and process-parallel independent Puggles runs.

This is process-level parallelism: each worker optimizes an independent restart.
It does not attempt to run a Python objective concurrently inside one optimizer,
because Python callbacks are serialized by the GIL.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
import sys
import time


POPULATION = 100
DEFAULT_NFE = 500_000


def objective_function(x: list[float]) -> float:
    return (x[0] - 3.0) ** 2 + (x[1] - 6.0) ** 2


def run_restart(seed: int, nfe: int) -> tuple[float, float]:
    """Run one independent Puggles optimization and return duration and best value."""
    import puggles as pg

    problem = pg.Problem(
        solution_length=2,
        number_of_objectives=1,
        solution_data_types=[pg.Real(-100.0, 100.0)] * 2,
        objective_function=lambda x: [objective_function(x)],
        direction=[-1],
    )
    ga = pg.NSGAII(
        problem,
        population_size=POPULATION,
        execution_mode="sequential",
        seed=seed,
    )
    started = time.perf_counter()
    ga.run(nfe)
    elapsed = time.perf_counter() - started
    best = min(solution.objectives[0] for solution in ga.get_population())
    return elapsed, best


def main() -> None:
    nfe = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NFE
    workers = min(4, os.cpu_count() or 1)
    seeds = [42 + index for index in range(workers)]

    print(
        f"Puggles process-parallel restarts: workers={workers}, "
        f"population={POPULATION}, NFE/restart={nfe:,}"
    )

    started = time.perf_counter()
    sequential_results = [run_restart(seed, nfe) for seed in seeds]
    sequential_wall = time.perf_counter() - started

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        parallel_results = list(executor.map(run_restart, seeds, [nfe] * workers))
    parallel_wall = time.perf_counter() - started

    sequential_best = min(best for _, best in sequential_results)
    parallel_best = min(best for _, best in parallel_results)
    speedup = sequential_wall / parallel_wall
    print(f"Sequential restarts: {sequential_wall:.3f} s | best: {sequential_best:.3e}")
    print(f"Process-parallel:    {parallel_wall:.3f} s | best: {parallel_best:.3e}")
    print(f"Process speedup:     {speedup:.2f}x")


if __name__ == "__main__":
    main()
