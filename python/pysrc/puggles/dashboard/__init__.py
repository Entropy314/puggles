"""The live browser dashboard — an optional add-on: ``pip install "puggles[dashboard]"``.

This module re-exports the add-on (the ``puggles_dashboard`` package), so ``puggles.dashboard``
and ``python -m puggles.dashboard RUN_FILE`` work once it's installed.
"""

try:
    from puggles_dashboard import *  # noqa: F401,F403
    from puggles_dashboard import __all__, __version__  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "puggles_dashboard":
        raise
    from .._missing import missing_dashboard

    raise missing_dashboard() from None
