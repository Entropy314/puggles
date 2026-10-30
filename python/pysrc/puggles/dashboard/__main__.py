"""python -m puggles.dashboard RUN_FILE [MORE_RUNS...] — the add-on's CLI (see puggles_dashboard)."""

import sys

from puggles_dashboard.__main__ import main  # importing the package already gave the install hint

if __name__ == "__main__":
    sys.exit(main())
