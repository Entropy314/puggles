"""python -m puggles.dashboard RUN_FILE [MORE_RUNS...] — replay a saved run, with any further runs
overlaid for comparison, or convert them to a standalone page."""

import argparse
import sys
from pathlib import Path

from . import load


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m puggles.dashboard", description=__doc__)
    parser.add_argument("run_file", type=Path, help="a .pgrun.json or .pgrun.json.gz file")
    parser.add_argument("compare", type=Path, nargs="*", help="more runs to overlay for comparison")
    parser.add_argument("--port", type=int, default=0, help="port to serve on (default: any free port)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument("--html", type=Path, metavar="OUT",
                        help="write a self-contained replay page to OUT instead of serving")
    args = parser.parse_args(argv)

    if args.html:
        dash = load(args.run_file, compare=args.compare, open_browser=False, block=False)
        dash.export_html(args.html)
        dash.close()
        print(f"wrote {args.html}", file=sys.stderr)
        return 0
    load(args.run_file, compare=args.compare, open_browser=not args.no_browser, port=args.port, block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
