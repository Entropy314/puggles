"""The error raised when the optional dashboard add-on is used but not installed."""

INSTALL_HINT = 'the dashboard is an optional add-on: pip install "puggles[dashboard]"'


def missing_dashboard() -> ImportError:
    return ImportError(INSTALL_HINT, name="puggles_dashboard")
