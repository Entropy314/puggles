# Changelog

One version number covers the Rust crate (`puggles` on crates.io) and the Python packages
(`puggles` and, from 0.3.0, `puggles-dashboard` on PyPI). `scripts/check_versions.py` keeps
them in step.

## 0.3.0 — unreleased

### Visualization (optional everywhere)

- **Rust:** a new `plot` feature (adds only `serde_json`) with three parts:
  - `ggplot`-style interactive figures.
  - `RunHistory` run recording, saved as `.pgrun.json`.
  - A live browser `Dashboard`. The page shows:
    - the front in 2-D, 3-D, parallel coordinates or a scatter matrix
    - decision variables and convergence (IGD, hypervolume, ...)
    - pause and stop controls, and goals with NSGA-III steering
    - run comparison and grouped results with significance tests
    - custom columns, remote access with a token, and a standalone HTML export

  Nothing visual is compiled without the feature.
- **Python:** the dashboard is a separate, optional wheel, `puggles-dashboard`, installed with
  `pip install "puggles[dashboard]"`. The core `puggles` wheel contains no visual code.
  - With the add-on installed, `puggles.Dashboard`, `show`, `load`, `reference_front` and
    `python -m puggles.dashboard` load on first use.
  - Without it, those names raise an `ImportError` that gives the install command.
- The add-on records natively through a small versioned C ABI (`puggles::recorder_abi`,
  internal feature `recorder-abi`). With a built-in objective, a watched run keeps the GIL
  released and its full multithreaded speed.
- If the core and add-on versions don't match, the core issues a `RuntimeWarning` and falls
  back to calling the dashboard as a plain Python callback.

### Optimizers

- `run_with(max_nfe, |ga| -> bool)` on `NSGAII` and `NSGAIII` observes or steers every
  generation. With a seed, an observed run is identical to `run`.
- NSGA-III steering: `focus_on(&aspiration, spread)` and `clear_focus()`.
- `execution_info()` reports requested vs. effective execution mode, CPU threads, the GPU device,
  and the reason for any fallback.
- `last_generation_times()` reports seconds spent in variation, evaluation, selection and the
  archive.
- `take_archive()` added.
- NSGA-III also gets `effective_mode()` and `gpu_evaluations()`. From Python it now runs a
  `GpuProblem`, and its `run()` accepts `callback=`.
- The archive no longer collects copies of identical genomes.

### Performance

- **NSGA-II:** about 9% faster on ZDT1 (30 real variables). A small mixed-integer constrained
  problem runs at the same speed. Seeded runs evolve exactly the same populations as 0.2.0; the
  archive differs only by the duplicate removal above.
  - Archive de-duplication hashes genomes in place.
  - Archive survivors and batch/GPU outputs are moved rather than cloned.
  - Mutation runs on the owned child, with no extra clone and no per-gene operator lookup.
  - Crossover returns a pair rather than a `Vec`.
- **NSGA-III:** about 24% faster on DTLZ2. It sorts once per generation instead of two or three times,
  because the next generation reuses the survival ranks. Crowding ties are now broken by
  population index, so seeded NSGA-III results differ from 0.2.0. Front quality is unchanged
  (DTLZ2, 60 seeds).
  - Survivors are moved, not cloned.
  - Niching is single-pass, and reference-point norms are computed once.
- **Dashboard:**
  - The population is sampled while it's being copied, not copied and then thinned.
  - Each new frame is serialized once for all open tabs.
  - Getters no longer clone the whole run history.
- **Python:**
  - Results move into `Solution` objects instead of being copied.
  - Objective trampolines take the callable registry lock only for the lookup.

### Fixed

- Python objectives were called with the callable registry locked. A callable that dropped a
  `Problem` (whose destructor takes the same lock) deadlocked the run.

### Packaging

- crates.io goes from 0.1.0 straight to 0.3.0. The 0.2.0 crate was never published, so its
  changes below reach crates.io for the first time here.
- The published crate now excludes the Python packages, CI files and scripts.

## 0.2.0 — 2026-09-11 (PyPI only)

- **Permutation encoding** for ordering problems (TSP, scheduling): `with_permutation_encoding()`
  in Rust, `permutation=True` in Python. It uses order crossover (OX1) and swap mutation.
- **Island model:** `islands::run_islands` and Python `run_island_model` run independent,
  seeded populations in parallel and merge their fronts.
- **Checkpoint & resume from Python:** `save_state()` returns JSON and `load_state(...)`
  restores it.
- **Front-quality metrics in Python:** `hypervolume_2d`, `igd`, `spacing`,
  `archive_hypervolume`, and `run_until_hv_converged` (stop when hypervolume plateaus).
- **Deb's constrained-domination:** ties between infeasible solutions are broken by total
  violation magnitude (`constraint_violation_magnitude`).
- **Closed bounds:** `[lower, upper]` everywhere.
- `try_new` constructors return `ConfigError`. Invalid configuration raises `ValueError` in
  Python instead of panicking.
- NSGA-III supports batch objectives and a GPU evaluator.
- `create_benchmark_problem` checks the objective count.

## 0.1.0 — 2026-08-31

- First release on crates.io and PyPI:
  - NSGA-II and NSGA-III
  - real, integer and binary variables
  - objective and decision-variable constraints
  - Rayon parallelism, batch objectives, and optional GPU evaluation (WGSL)
  - seeded runs, checkpoints and metrics
- Python bindings (abi3 wheels for CPython 3.9+).
