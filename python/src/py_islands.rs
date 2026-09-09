//! Island-model parallelism exposed to Python.
//!
//! Only usable with a pure-Rust objective (a built-in benchmark via `create_benchmark_problem`).
//! A Python callable is serialized by the GIL, so running islands in parallel around one would
//! add contention without adding throughput.

use pyo3::prelude::*;
use puggles::genetic_algorithms_v2::ExecutionMode;
use puggles::islands::{run_islands, IslandConfig};
use std::sync::Arc;

use crate::py_nsgaii::extract_store;
use crate::py_solution::PySolution;

/// Run several independent NSGA-II populations in parallel and return the merged Pareto front.
///
/// One large population can converge onto a single region of the front; several smaller ones
/// explore independently and are pooled at the end, trading per-island depth for coverage.
///
/// Args:
///     problem: A Problem with a **native Rust** objective (see create_benchmark_problem).
///              Python callables are rejected: the GIL would serialize the islands.
///     islands: Number of independent populations (default 4).
///     population_size: Population size per island (default 50).
///     max_nfe_per_island: Evaluation budget per island; total work is islands * this.
///     seed: Base seed. Island i uses seed + i, so a given seed reproduces the whole run.
///
/// Returns:
///     The non-dominated set of the pooled island archives.
#[pyfunction]
#[pyo3(signature = (problem, islands = 4, population_size = 50, max_nfe_per_island = 10_000, seed = None))]
pub fn run_island_model(
    py: Python<'_>,
    problem: PyObject,
    islands: usize,
    population_size: usize,
    max_nfe_per_island: usize,
    seed: Option<u64>,
) -> PyResult<Vec<PySolution>> {
    let store = extract_store(py, &problem)?;
    if store.uses_python_callable || store.uses_batch_callable {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_island_model needs a native Rust objective — a Python callable is serialized by \
             the GIL, so parallel islands would add contention without throughput. Build the \
             problem with create_benchmark_problem().",
        ));
    }
    if islands == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("islands must be > 0"));
    }

    let core_problem = Arc::clone(&store.problem);
    let config = IslandConfig {
        islands,
        population_size,
        max_nfe_per_island,
        seed,
        execution_mode: ExecutionMode::Sequential,
    };

    // No Python callable involved, so release the GIL for the whole parallel run.
    let front = py.allow_threads(move || run_islands(core_problem, config));
    Ok(front.iter().map(PySolution::from_core_solution).collect())
}
