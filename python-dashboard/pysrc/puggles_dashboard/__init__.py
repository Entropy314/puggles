"""Interactive browser dashboard for puggles runs — live while a GA runs, or replayed from a file.

The optional visual add-on for puggles: ``pip install "puggles[dashboard]"``. Import it as
``puggles.dashboard`` (or ``pg.Dashboard``) or directly as ``puggles_dashboard``.

    import puggles as pg

    ga = pg.NSGAII(problem, seed=42)
    dash = pg.Dashboard(ga, objective_names=["cost", "risk"])   # opens a browser tab
    ga.run(20_000, callback=dash)                              # watch it evolve
    dash.save("run.pgrun.json.gz")                              # replay: pg.dashboard.load(...)

The server and recorder are native (Rust); this module adds the conveniences around them.
Passed as ``callback=``, a Dashboard records straight from the optimizer's Rust solutions
through ``Dashboard.__puggles_recorder__`` — no GIL, no per-generation Python objects.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from ._native import Dashboard, __version__, reference_front

__all__ = ["Dashboard", "show", "load", "read_run_text", "reference_front"]


def show(
    ga: Any,
    *,
    objective_names: Optional[Sequence[str]] = None,
    variable_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    open_browser: bool = True,
    port: int = 0,
    block: Optional[bool] = None,
) -> Dashboard:
    """Open a finished optimizer's Pareto front and final population in the dashboard.

    ``block`` defaults to True in a script (so the page outlives it) and False in an
    interactive session or notebook.
    """
    archive = ga.get_archive()
    if not archive:
        raise ValueError("the optimizer has no archive yet — call run() first")
    dash = Dashboard(ga, objective_names=objective_names, variable_names=variable_names,
                     title=title, open_browser=open_browser, port=port)
    dash(archive, ga.get_population(), ga.nfe)
    dash.finish()
    return _maybe_block(dash, block)


def load(
    path: "str | Path",
    *,
    compare: Sequence["str | Path"] = (),
    open_browser: bool = True,
    port: int = 0,
    block: Optional[bool] = None,
) -> Dashboard:
    """Replay a run saved by ``Dashboard.save`` or the Rust crate (``.pgrun.json[.gz]``),
    with the runs in ``compare`` overlaid on it."""
    dash = Dashboard._replay(read_run_text(path), open_browser=False, port=port)
    for other in compare:
        dash.compare(other)
    if open_browser:
        import webbrowser
        webbrowser.open(dash.url)
    return _maybe_block(dash, block)


def read_run_text(path: "str | Path") -> str:
    """A run file's JSON text; gzip is detected by content, not by name."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def _maybe_block(dash: Dashboard, block: Optional[bool]) -> Dashboard:
    if block is None:
        block = not _interactive()
    if block:
        dash.wait()
    return dash


def _interactive() -> bool:
    return hasattr(sys, "ps1") or "IPython" in sys.modules or bool(sys.flags.interactive)
