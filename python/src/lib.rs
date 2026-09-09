use pyo3::prelude::*;

mod py_benchmarks;
mod py_islands;
mod py_metrics;
#[cfg(feature = "gpu")]
mod py_gpu;
mod py_nsga3;
mod py_nsgaii;
mod py_operators;
mod py_problem;
mod py_solution;
mod py_types;

#[pymodule]
fn puggles(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Kept in step with Cargo.toml automatically, so it can never drift from the wheel.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Core types
    m.add_class::<py_types::PyReal>()?;
    m.add_class::<py_types::PyInteger>()?;
    m.add_class::<py_types::PyBitBinary>()?;
    m.add_class::<py_problem::PyProblem>()?;
    m.add_class::<py_solution::PySolution>()?;
    m.add_class::<py_nsgaii::PyNSGAII>()?;
    m.add_class::<py_nsga3::PyNSGAIII>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<py_gpu::PyGpuProblem>()?;

    // Operator config
    m.add_class::<py_operators::PyCrossoverConfig>()?;
    m.add_class::<py_operators::PyMutationConfig>()?;

    // Benchmark functions
    m.add_function(wrap_pyfunction!(py_benchmarks::paraboloid_hyper_5, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::paraboloid_5, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::paraboloid_5_loc, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::simple_objective, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::xyz_objective, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::dtlz1, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::dtlz2, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::dtlz3, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::dtlz4, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::dtlz5, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::dtlz6, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::dtlz7, m)?)?;
    m.add_function(wrap_pyfunction!(py_benchmarks::create_benchmark_problem, m)?)?;

    // Front-quality metrics (minimization space)
    m.add_function(wrap_pyfunction!(py_metrics::hypervolume_2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_metrics::igd, m)?)?;
    m.add_function(wrap_pyfunction!(py_metrics::spacing, m)?)?;

    // Island-model parallelism
    m.add_function(wrap_pyfunction!(py_islands::run_island_model, m)?)?;

    Ok(())
}
