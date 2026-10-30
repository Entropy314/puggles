"""Fail unless every manifest carries the same release version.

The Rust crate, the core Python wheel and the dashboard add-on are released together, and the
two wheels talk through a native interface that only matches within a release — so their
versions, and the pins between them, must agree. Run from anywhere: python scripts/check_versions.py
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def toml(path):
    return tomllib.loads((ROOT / path).read_text())


def pin(requirements, name):
    for req in requirements:
        m = re.fullmatch(rf"{re.escape(name)}\s*==\s*(\S+)", req.strip())
        if m:
            return m.group(1)
    return None


crate = toml("Cargo.toml")["package"]["version"]
core_rs = toml("python/Cargo.toml")
core_py = toml("python/pyproject.toml")["project"]
addon_rs = toml("python-dashboard/Cargo.toml")
addon_py = toml("python-dashboard/pyproject.toml")["project"]

found = {
    "Cargo.toml [package].version": crate,
    "python/Cargo.toml [package].version": core_rs["package"]["version"],
    "python/Cargo.toml puggles dependency": core_rs["dependencies"]["puggles"]["version"],
    "python/pyproject.toml version": core_py["version"],
    "python/pyproject.toml [dashboard] extra pin": pin(core_py["optional-dependencies"]["dashboard"], "puggles-dashboard"),
    "python-dashboard/Cargo.toml [package].version": addon_rs["package"]["version"],
    "python-dashboard/Cargo.toml puggles dependency": addon_rs["dependencies"]["puggles"]["version"],
    "python-dashboard/pyproject.toml version": addon_py["version"],
    "python-dashboard/pyproject.toml puggles pin": pin(addon_py["dependencies"], "puggles"),
}
width = max(map(len, found))
for where, version in found.items():
    print(f"{where:<{width}}  {version}")
if len(set(found.values())) != 1:
    sys.exit(f"\nversion mismatch: every entry above must be {crate}")
print(f"\nall {len(found)} agree on {crate}")
