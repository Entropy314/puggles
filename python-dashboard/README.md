# puggles-dashboard

The optional visual add-on for [puggles](https://pypi.org/project/puggles/), a fast NSGA-II /
NSGA-III library with a Rust core. It adds a live browser dashboard, run replay, and side-by-side
comparison of runs.

```bash
pip install "puggles[dashboard]"     # puggles plus this add-on, at matching versions
```

The core `puggles` wheel has no visual code. With this add-on installed, `puggles.Dashboard`,
`puggles.show`, `puggles.load`, `puggles.reference_front` and `python -m puggles.dashboard` work.
They load on first use, so `import puggles` stays as fast as before. You can also import the
add-on directly as `puggles_dashboard`.

## Watch a run

Pass a dashboard as the callback. The page shows:

- the Pareto front in 2-D, 3-D, parallel coordinates or a scatter matrix
- decision variables and convergence
- a sortable table of solutions

```python
import puggles as pg

ga = pg.NSGAII(problem, seed=42)
dash = pg.Dashboard(ga, objective_names=["cost", "risk"])   # opens a tab
ga.run(20_000, callback=dash)                              # live; the page's Stop button ends the run
dash.save("run.pgrun.json.gz")                              # replay: python -m puggles.dashboard run.pgrun.json.gz
dash.export_html("run.html")                                # a standalone page you can share
dash.wait()                                                 # in a script: keep the page up until Ctrl+C
```

While a run is being watched, you can:

- pause it
- filter and pin solutions
- select the most balanced trade-off
- see where each generation's time goes

The page loads Plotly.js from a CDN, so viewing needs an internet connection.

**Speed.** The server and the recorder are native. When a dashboard is the `callback`, `run()`
records straight from the optimizer's Rust solutions and never builds Python objects per
generation. With a built-in (Rust) objective, the run also keeps the GIL released and its full
multithreaded speed. Population snapshots are sampled to `max_points` (default 2,000) members.

## Options

- **True front:** if it's known, pass it, and IGD becomes a convergence metric:
  `pg.Dashboard(ga, reference_front=pg.reference_front("dtlz2", 3))`.
- **Comparing runs:** `keep_runs=True` keeps each finished run overlaid on the next. Several
  seeds or algorithms then share one dashboard (see `examples/compare_runs.py`). `dash.compare(path)`
  adds a saved run, and so does `python -m puggles.dashboard a.pgrun.json b.pgrun.json`.
- **Goals:** set targets per objective in the page. With NSGA-III, *Steer* focuses the running
  optimization on them.
- **Groups and results:** call `set_group("NSGA-II")` before each run. The Results tab then
  compares groups across seeds with a significance test, and the convergence chart shows median
  and IQR bands.
- **Your own columns:** `columns={"weight": lambda x: ...}`. They appear in the table, colour-by,
  pins and the Drivers view, which shows which variables move each objective.
- **Remote:** `Dashboard(..., host="0.0.0.0")` requires a token, which is included in `dash.url`.
  Or tunnel the port: `ssh -L 8765:127.0.0.1:8765 host`.
- **Also:**
  - `dash.add_event("…")` marks the convergence chart.
  - The page offers notifications, a one-click HTML report, a shapes (colour-blind) mode, and
    keyboard shortcuts (`?`).
  - In Jupyter, a `Dashboard` as the last expression in a cell shows inline.

`pg.show(ga)` opens a finished optimizer's front. The header shows where evaluations run: CPU
threads, or the GPU and its device, with a live count of GPU evaluations. For GPU runs, build the
core with `maturin develop --release --features gpu` and pass a `pg.GpuProblem`; see
`examples/dashboard_demo.py --gpu`.

## Replay from the command line

```bash
python -m puggles.dashboard run.pgrun.json.gz                 # serve a saved run
python -m puggles.dashboard a.pgrun.json b.pgrun.json         # overlay runs
python -m puggles.dashboard run.pgrun.json --html report.html # standalone page, no server
```

The Rust crate (`puggles`, feature `plot`) writes the same `.pgrun.json` files.

## How it plugs into the core

`Dashboard.__puggles_recorder__()` returns a capsule holding a small, versioned native interface
(`puggles::recorder_abi` in the Rust crate). `NSGAII.run` / `NSGAIII.run` look for it on their
`callback`. If the interface doesn't match, for example after mixing versions, the core issues a
`RuntimeWarning` and falls back to calling the dashboard as an ordinary Python callback.
`puggles-dashboard` requires the `puggles` release with the same version number. That's what the
`[dashboard]` extra installs.

## Links

- Source, Rust API, and full guide: https://github.com/Entropy314/puggles
- Issues: https://github.com/Entropy314/puggles/issues

## License

MIT OR Apache-2.0
