#!/usr/bin/env python3
"""Generate synthetic return datasets targeting 1 MB, 10 MB, 100 MB.

Each CSV has N_ASSETS columns of daily log-returns drawn from a
correlated multivariate normal distribution, plus a date index column.

Output: sandbox/data/returns_{1mb,10mb,100mb}.csv
"""

import os
import random
import math

N_ASSETS = 20
SEED = 42

# Row budget to hit each target (row ≈ 11 + N_ASSETS*9 bytes ≈ 191 bytes)
TARGETS = [
    ("1mb",  5_500),
    ("10mb", 55_000),
    ("100mb", 549_000),
]

TICKERS = [f"A{i:02d}" for i in range(N_ASSETS)]

random.seed(SEED)

# Synthetic asset parameters
means   = [random.gauss(0.0004, 0.0002) for _ in range(N_ASSETS)]
stds    = [random.uniform(0.008, 0.025) for _ in range(N_ASSETS)]
# Simple factor model: each asset has market beta + idio noise
market_betas = [random.uniform(0.5, 1.5) for _ in range(N_ASSETS)]


def sample_row(rng_state):
    """Draw one row of correlated returns via a single-factor model."""
    market_shock = random.gauss(0.0, 0.01)
    return [
        random.gauss(means[i] + market_betas[i] * market_shock, stds[i])
        for i in range(N_ASSETS)
    ]


data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)

header = "date," + ",".join(TICKERS) + "\n"

for label, n_rows in TARGETS:
    path = os.path.join(data_dir, f"returns_{label}.csv")
    print(f"Generating {label} dataset ({n_rows:,} rows) → {path}")

    with open(path, "w", buffering=1 << 20) as f:
        f.write(header)
        for row_idx in range(n_rows):
            # Fake date: start 2000-01-03, increment by 1 business day
            year  = 2000 + row_idx // 252
            doy   = (row_idx % 252) + 1
            month = min(12, doy // 21 + 1)
            day   = min(28, doy % 21 + 1)
            date_str = f"{year}-{month:02d}-{day:02d}"
            vals = sample_row(row_idx)
            f.write(date_str + "," + ",".join(f"{v:.6f}" for v in vals) + "\n")

    size_mb = os.path.getsize(path) / 1e6
    print(f"  → {size_mb:.1f} MB written")

print("\nDone. Files in sandbox/data/:")
for label, _ in TARGETS:
    path = os.path.join(data_dir, f"returns_{label}.csv")
    print(f"  {os.path.getsize(path)/1e6:.1f} MB  returns_{label}.csv")
