# NSGA-II Optimization Roadmap

> **Status (2026-08-27):** Quick wins + **Tier A (ENS-SS)** + **B2 (crossover clones)** + **B4
> (archive clones)** are **implemented** (66 tests pass, quality preserved; single-obj 55→15 ms,
> ZDT1 27→20 ms, DTLZ2 71→56 ms — see `sandbox/benchmarks.md` §5). **Deferred:** B1 (SmallVec —
> high churn: 59 sites/7 files + checkpoint serde), B3 (offspring-buffer reuse — negligible), and
> all of Tier C. The sections below are the original design; items above are done.

## Context

The §5 single-objective benchmark put rustypus at **2.75 µs/eval** vs genevo 0.43 and
genetic_algorithm 0.16 — ~6–17× more wall-time per evaluation. The cost is NSGA-II's
per-generation bookkeeping, not the objective. The **quick wins** (de-virtualize the dominance
comparison + an M=1 fast path — see `.claude/plans/for-this-i-want-glistening-avalanche.md`) are
the prerequisite and land first; this roadmap covers everything after, retaining exact
multi-objective correctness throughout.

**Prerequisite (assumed done):** `fast_non_dominated_sort` is generic (`<D: Dominance + ?Sized>`)
so `ParetoDominance::compare_solutions` inlines, and M=1 takes an O(N log N) fast path. Tier A below
reuses that inlined comparison.

Every change is guarded by the same suite: `cargo test` (66 tests — front membership via
`test_fast_non_dominated_sort` at [dominance.rs:301](src/dominance.rs#L301), single/multi runs,
`test_seeded_runs_are_reproducible` at [genetic_algorithms_v2.rs:729](src/genetic_algorithms_v2.rs#L729),
`test_nsga3_reproducible`), plus the benchmarks that measure quality: `bench_zdt1` (M=2, IGD),
`bench_dtlz` (M=3, HV/IGD via `bench_dtlz.py`), `bench_singleobj` (M=1, time + best f).

---

## A. The big one — ENS non-dominated sort (M ≥ 2)

The only change that closes the genuine multi-objective per-evaluation gap.

**What.** Replace the O(MN²) pairwise double loop in
[`fast_non_dominated_sort`](src/dominance.rs#L71) with **ENS-SS** (Efficient Non-dominated Sort,
sequential search). Same fronts out, far fewer comparisons.

**Where.** `src/dominance.rs` only. The 6 call sites
([genetic_algorithms_v2.rs:108,125,188](src/genetic_algorithms_v2.rs#L108),
[nsga3.rs:186,197,204](src/nsga3.rs#L186)) keep the same signature — internal change.

**How.**
1. Sort solution indices lexicographically by the **direction-adjusted** objectives (reuse the
   same negate-iff-`direction==-1` logic from [dominance.rs:37-43](src/dominance.rs#L37)), best-first.
2. Process in that order; for each solution, scan existing fronts (ENS-SS: first-to-last) and place
   it in the first front where **no current member dominates it** — using the now-inlined
   `dominance.compare_solutions`. A solution can only be dominated by earlier-sorted ones, so most
   comparisons are skipped.
3. Keep the M=1 fast path branch ahead of this (M=1 → sort-by-objective; M≥2 → ENS).

**Effort / risk.** Medium / medium — the ranking is the crate's correctness core. Front **membership**
stays identical (tests pass); **within-front order** differs from the naive BFS, so exact solutions
change but IGD/HV do not. Best-Order-Sort is a faster alternative but more code — start with ENS-SS.

**Verify.** `test_fast_non_dominated_sort` still 3 fronts of 2/2/1; `bench_zdt1` IGD and `bench_dtlz`
HV/IGD **statistically unchanged**; wall-time down (most at larger populations — at N≈200 the win is
moderate, it grows with pop size). Add a `criterion`-free timing print or reuse the bench harness.

---

## B. Constant-factor wins (stack on top; help every M, incl. cheap objectives)

### B1. Lighten the `Solution` objective/constraint vectors
**What.** `objective_fitness_values` and `constraint_values` are heap `Vec<f64>`
([core.rs:105-106](src/core.rs#L105)) — a fresh allocation on every `Solution::clone`. For typical
small M, store them inline.
**How.** Add a `smallvec` dep; change both fields to `SmallVec<[f64; 4]>`. Mechanically update read
sites (`objective_fitness_values[i]` indexing is unchanged; only the type at construction sites in
`core.rs`, `dominance.rs`, `metrics.rs`, operators, and the test literals).
**Effort / risk.** Medium-high (many touch points, but mechanical) / medium.
**Note.** Dropping the per-`Solution` `Arc<Problem>` ([core.rs:103](src/core.rs#L103)) would remove an
atomic refcount per clone, but it threads `&Problem` through evaluate/operators/sort — high churn for
a smaller gain than the Vec allocations. **Defer** the Arc removal; do the SmallVec part.
**Verify.** All tests (bit-identical numerics); `bench_singleobj`/`bench_zdt1` time down, quality equal.

### B2. Cut the redundant parent clones in crossover
**What.** `perform_crossover` ([crossover.rs:396-397](src/genetic_operators/crossover.rs#L396)) clones
both parents, **and** each operator's `crossover()` clones them again
([crossover.rs:56-57](src/genetic_operators/crossover.rs#L56), etc.) → ~4 `Solution` clones (each up to
3 Vec copies) per 2 children for an all-Real problem.
**How.** When the problem is a **single gene type** (the common all-Real case), return the operator's
children directly — skip the manager's re-clone + per-gene merge. For mixed-type problems, use the
first operator's children as the base and overwrite only other-typed genes (one fewer clone pair).
**Effort / risk.** Low-medium / low-medium — preserve mixed-type gene merge (guarded by
`test_nsgaii_mixed_types`).
**Verify.** `test_nsgaii_mixed_types`, reproducibility tests; `bench_zdt1` time down, IGD equal.

### B3. Reuse per-generation buffers
**What.** `iterate_n` reallocates the offspring/combined `Vec`s each generation
([genetic_algorithms_v2.rs:295-296](src/genetic_algorithms_v2.rs#L295)).
**How.** Keep a reusable scratch `Vec<Solution>` on the `NSGAII` struct; `clear()` + reuse instead of
allocating. `std::mem::take`/`swap` already used — extend the pattern.
**Effort / risk.** Low / low.
**Verify.** Tests unchanged; small time improvement, most visible on cheap objectives.

### B4. Cheaper / less-frequent `update_archive`
**What.** [`update_archive`](src/genetic_algorithms_v2.rs#L175) clones feasible solutions + front-0
**every generation** ([:176](src/genetic_algorithms_v2.rs#L176), [:190](src/genetic_algorithms_v2.rs#L190),
[:206](src/genetic_algorithms_v2.rs#L206)) and runs its own sort.
**How.** Cheapest: call it every K generations (and once at the end) instead of every generation.
Better: maintain the archive incrementally, merging only the new front-0 rather than re-sorting
feasible∪archive from scratch. Avoid the feasible-clone by ranking indices first, cloning only kept.
**Effort / risk.** Medium / medium — archive size/quality is correctness-sensitive; guarded by
`test_nsgaii_multithreaded_archive_nonempty` and the IGD/HV benches (archive feeds them).
**Verify.** `bench_zdt1` IGD and `bench_dtlz` HV/IGD unchanged; per-generation time down.

---

## C. Situational (only under specific conditions)

### C1. Fitness caching — **mostly already handled**
rustypus already skips re-evaluating unchanged solutions: `evaluate_population` only evaluates
`!evaluated` ([genetic_algorithms_v2.rs:456-459](src/genetic_algorithms_v2.rs#L456)) and survivors keep
`evaluated = true`. A gene-keyed cache would only add value by **deduplicating identical new
offspring** — niche. **Skip** unless profiling shows many duplicate genomes.

### C2. Parallelize the dominance sort (large populations only)
`rayon` is already a dep. The domination-count loop can be parallelized, but at N≈200 it's
overhead-bound — the sort is cheap in absolute terms. **Only** worth it for large populations
(N ≫ 1000). Effort low / risk low; gate behind a population-size threshold.

### C3. O(N) crowding distance for M=2
[`crowding_distance`](src/dominance.rs#L123) sorts each front per objective. For exactly M=2 a single
sorted pass suffices. Minor next to the sort — low priority. Effort low / risk low.

---

## Recommended order

1. **A (ENS sort)** — the headline multi-objective win; do it first on top of the quick wins.
2. **B2 (crossover clones)** and **B3 (buffer reuse)** — low-risk constant-factor, quick.
3. **B1 (SmallVec)** — bigger mechanical change; do once the above are validated.
4. **B4 (update_archive)** — needs care; validate IGD/HV closely.
5. **C** items only if profiling/scale justifies them.

After each tier: run `cargo test` + `cargo test --features gpu`, then `bench_singleobj` / `bench_zdt1`
/ `bench_dtlz`, and record the time delta with quality (IGD/HV/best-f) held constant. Refresh
`sandbox/benchmarks.md` §5 (and §1/§4 if multi-objective time moved) from the measured numbers.
