"""Fast multi-objective evolutionary optimization (NSGA-II / NSGA-III) with a Rust core.

The live dashboard is an optional add-on (``pip install "puggles[dashboard]"``); once it's
installed, ``puggles.Dashboard``, ``puggles.show``, ``puggles.load`` and
``puggles.reference_front`` come from it.
"""

from typing import TYPE_CHECKING

from ._puggles import *  # noqa: F401,F403  (the native extension: Problem, NSGAII, NSGAIII, ...)
from ._puggles import __version__  # noqa: F401

if TYPE_CHECKING:  # for editors and type checkers; at runtime these load on first use
    from puggles_dashboard import Dashboard, load, reference_front, show  # noqa: F401

# Served by the optional add-on, imported on first use so `import puggles` never loads it.
_FROM_DASHBOARD = ("Dashboard", "show", "load", "reference_front")


def __getattr__(name: str):
    if name in _FROM_DASHBOARD or name == "dashboard":
        import importlib

        dashboard = importlib.import_module(".dashboard", __name__)  # raises the install hint
        return dashboard if name == "dashboard" else getattr(dashboard, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_FROM_DASHBOARD))
