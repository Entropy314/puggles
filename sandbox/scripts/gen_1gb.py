#!/usr/bin/env python3
"""Fast numpy generator for a ~1 GB returns CSV (20 assets, single-factor model).

Matches the schema of returns_{1,10,100}mb.csv: `date` + A00..A19 columns.
Written in row chunks to keep memory bounded. Run:  python3 scripts/gen_1gb.py
"""
import os
import time
import numpy as np

N_ASSETS = 20
SEED = 42
TARGET_ROWS = 5_600_000          # ~191 bytes/row → ~1.07 GB
CHUNK = 200_000

rng = np.random.default_rng(SEED)
means = rng.normal(0.0004, 0.0002, N_ASSETS)
stds = rng.uniform(0.008, 0.025, N_ASSETS)
betas = rng.uniform(0.5, 1.5, N_ASSETS)

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
path = os.path.join(data_dir, "returns_1gb.csv")
header = "date," + ",".join(f"A{i:02d}" for i in range(N_ASSETS)) + "\n"

t0 = time.perf_counter()
written = 0
with open(path, "w", buffering=1 << 22) as f:
    f.write(header)
    while written < TARGET_ROWS:
        n = min(CHUNK, TARGET_ROWS - written)
        market = rng.normal(0.0, 0.01, (n, 1))            # shared factor per row
        idio = rng.normal(0.0, 1.0, (n, N_ASSETS)) * stds  # idiosyncratic
        rows = means + betas * market + idio               # (n, N_ASSETS)
        # Fake business-day dates (cheap, schema-compatible)
        idx = np.arange(written, written + n)
        years = 2000 + idx // 252
        doy = (idx % 252) + 1
        months = np.minimum(12, doy // 21 + 1)
        days = np.minimum(28, doy % 21 + 1)
        # Build lines
        buf = []
        for k in range(n):
            date = f"{years[k]}-{months[k]:02d}-{days[k]:02d}"
            buf.append(date + "," + ",".join(f"{v:.6f}" for v in rows[k]))
        f.write("\n".join(buf) + "\n")
        written += n
        if written % 1_000_000 < CHUNK:
            print(f"  {written:,} rows  ({os.path.getsize(path)/1e9:.2f} GB, {time.perf_counter()-t0:.0f}s)", flush=True)

size_gb = os.path.getsize(path) / 1e9
print(f"Done: {written:,} rows → {path}  ({size_gb:.2f} GB, {time.perf_counter()-t0:.0f}s)")
