"""Type stubs for the add-on's native module (re-exported by puggles_dashboard)."""

import os
import pathlib
from typing import Callable, Dict, List, Optional, Union

from puggles._puggles import GpuProblem, NSGAII, NSGAIII, Problem, Solution

__version__: str
_recorder_abi_version: int

class Dashboard:
    """Live browser dashboard for a run. Pass it as ``callback=`` to ``NSGAII.run`` / ``NSGAIII.run``.

    The server is native and keeps serving while the optimizer runs. Passed as ``callback=``,
    it records straight from the Rust solutions and marks the run finished when ``run()``
    returns. The page's Stop button ends the run after the current generation.
    """
    url: str
    state: str
    """waiting | running | stopping | stopped | done | replay"""
    frame_count: int
    def __init__(
        self,
        source: Union[Problem, GpuProblem, NSGAII, NSGAIII, None] = None,
        *,
        objective_names: Optional[List[str]] = None,
        variable_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        open_browser: bool = True,
        host: str = "127.0.0.1",
        port: int = 0,
        max_frames: int = 500,
        reference_front: Optional[List[List[float]]] = None,
        keep_runs: bool = False,
        group: Optional[str] = None,
        columns: Optional[Dict[str, Callable[[List[float]], float]]] = None,
        token: Union[str, bool, None] = None,
        max_points: int = 2000,
    ) -> None:
        """reference_front: the true Pareto front if known (see reference_front()); drawn in the
        objective view and used for IGD. keep_runs: keep each finished run as a comparison when
        the next starts on this dashboard. group: the first run's group for across-seed statistics
        (default: the algorithm). columns: your own per-solution columns, {name: f(variables)}.
        token: require an access token (a string, or True to generate one; on by default when
        host isn't loopback). max_points: population members kept per generation (0 = all)."""
    token: Optional[str]
    def set_group(self, group: str) -> None:
        """Group of the next run that starts (or of the first run), e.g. "NSGA-II, pop 100"."""
    def add_event(self, label: str, nfe: Optional[int] = None) -> None:
        """Mark a moment on the convergence chart (kept in saved runs)."""
    paused: bool
    comparisons: List[str]
    """Names of the runs overlaid on this one."""
    def __call__(self, archive: List[Solution], population: List[Solution], nfe: int) -> bool: ...
    def __puggles_recorder__(self) -> object:
        """The native recorder (a capsule) ``run()`` uses when this is its ``callback``."""
    def finish(self) -> None: ...
    def request_stop(self) -> None: ...
    def pause(self) -> None:
        """Hold the run after its next generation (the page's Pause button)."""
    def resume(self) -> None: ...
    def compare(self, source: Union[str, "os.PathLike[str]", "Dashboard"], name: Optional[str] = None) -> None:
        """Overlay another run: a saved .pgrun.json[.gz] path or another Dashboard."""
    def _repr_html_(self) -> str:
        """Jupyter: show the dashboard in the notebook cell."""
    def to_json(self) -> str: ...
    def save(self, path: Union[str, "os.PathLike[str]"]) -> "pathlib.Path":
        """``.pgrun.json``, or gzip-compressed when the name ends in ``.gz``."""
    def export_html(self, path: Union[str, "os.PathLike[str]"]) -> "pathlib.Path":
        """A self-contained replay page that opens without Python."""
    def wait(self) -> None:
        """Block until Ctrl+C or ``close()``, keeping the page alive after a script's run."""
    def close(self) -> None: ...
    def __enter__(self) -> "Dashboard": ...
    def __exit__(self, *exc: object) -> None: ...

def reference_front(name: str, number_of_objectives: int = 2, points: int = 200) -> List[List[float]]:
    """Points on a benchmark's true Pareto front: "zdt1", "zdt2" (2 objectives), "dtlz1"-"dtlz4"."""
