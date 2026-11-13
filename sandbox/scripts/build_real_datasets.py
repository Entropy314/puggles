#!/usr/bin/env python3
"""Build the realistic benchmark datasets from REAL daily prices of 20 US large caps.

1. Download each ticker's full daily history (split/dividend-adjusted close) from Yahoo
   Finance's public chart endpoint into a raw cache (skipped when cached).
2. polars: inner-join the 20 series on date → data/real_prices.csv (the real price history),
   then pct_change → daily simple returns.
3. Write data/real_returns_{1kb,1mb,10mb,100mb,1gb}.csv with schema `date,scenario,<20 tickers>`:
     scenario 0     = the real history (1kb / 1mb are the most recent 5 / 5,500 real rows)
     scenario 1..K  = stationary-free block bootstrap of the real returns: each scenario
                      replays the real calendar with 20-day blocks of whole cross-sections
                      drawn at random — keeping real cross-asset correlation, fat tails and
                      short-range volatility clustering.
   Returns (not prices) are stored because compounding thousands of years of bootstrapped
   returns into a price level would overflow.

Run:  python3 -I scripts/build_real_datasets.py <raw_cache_dir>
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import polars as pl

TICKERS = ["AAPL", "MSFT", "IBM", "KO", "PG", "JNJ", "XOM", "CVX", "WMT", "PFE",
           "MRK", "INTC", "ORCL", "JPM", "BAC", "GE", "MCD", "DIS", "HD", "PEP"]
# Same row counts as the synthetic returns_{size}.csv files, so sizes line up.
SIZES = [("1kb", 5), ("1mb", 5_500), ("10mb", 55_000), ("100mb", 549_000), ("1gb", 5_600_000)]
BLOCK = 20
SEED = 42
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1=0&period2=9999999999&interval=1d"

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def fetch(raw_dir, t):
    path = os.path.join(raw_dir, f"{t}.json")
    if not os.path.exists(path):
        req = urllib.request.Request(URL.format(t), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(path, "wb") as f:
            f.write(r.read())
        time.sleep(1)  # be polite to the endpoint
    res = json.load(open(path))["chart"]["result"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    return (pl.DataFrame({"ts": res["timestamp"], t: adj}, schema={"ts": pl.Int64, t: pl.Float64})
            .with_columns(pl.from_epoch("ts", time_unit="s").dt.date().alias("date"))
            .drop("ts").drop_nulls().unique("date", keep="last"))


def main(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    prices = fetch(raw_dir, TICKERS[0])
    for t in TICKERS[1:]:
        prices = prices.join(fetch(raw_dir, t), on="date", how="inner")
    prices = prices.sort("date").select(["date", *TICKERS])
    prices.write_csv(os.path.join(DATA, "real_prices.csv"), float_precision=4)

    rets = (prices.with_columns(pl.col(TICKERS).pct_change())
            .slice(1)  # first row has no previous price
            .with_columns(pl.lit(0, pl.Int32).alias("scenario"))
            .select(["date", "scenario", *TICKERS]))
    R, dates, T = rets.select(TICKERS).to_numpy(), rets["date"], rets.height
    print(f"real history: {T:,} trading days {dates[0]} → {dates[-1]}  ({len(TICKERS)} tickers)")

    rng = np.random.default_rng(SEED)
    n_blocks = -(-T // BLOCK)

    def scenario(k):
        starts = rng.integers(0, T - BLOCK + 1, n_blocks)
        idx = (starts[:, None] + np.arange(BLOCK)).ravel()[:T]
        return pl.DataFrame({"date": dates, "scenario": pl.Series([k] * T, dtype=pl.Int32),
                             **{t: R[idx, i] for i, t in enumerate(TICKERS)}})

    for label, rows in SIZES:
        path = os.path.join(DATA, f"real_returns_{label}.csv")
        rng = np.random.default_rng(SEED)  # every size draws the same scenario sequence
        with open(path, "w") as f:
            if rows <= T:
                rets.tail(rows).write_csv(f, float_precision=6)
            else:
                rets.write_csv(f, float_precision=6)
                left, k = rows - T, 1
                while left > 0:
                    chunk = scenario(k).head(left)
                    chunk.write_csv(f, include_header=False, float_precision=6)
                    left -= chunk.height
                    k += 1
        print(f"  {label:>6}: {rows:>9,} rows → {os.path.getsize(path):>13,} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
