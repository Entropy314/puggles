use pyo3::prelude::*;
use puggles::benchmark_objective_functions;
use puggles::gatypes::{Real, SolutionDataTypes};

use crate::py_problem::{create_problem_from_fn, PyProblem};

#[pyfunction]
pub fn paraboloid_hyper_5(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::parabloid_hyper_5(&x)
}

#[pyfunction]
pub fn paraboloid_5(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::parabloid_5(&x)
}

#[pyfunction]
pub fn paraboloid_5_loc(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::parabloid_5_loc(&x)
}

#[pyfunction]
pub fn simple_objective(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::simple_objective(&x)
}

#[pyfunction]
pub fn xyz_objective(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::xyz_objective(&x)
}

#[pyfunction]
pub fn dtlz1(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::dtlz1(&x)
}

#[pyfunction]
pub fn dtlz2(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::dtlz2(&x)
}

#[pyfunction]
pub fn dtlz3(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::dtlz3(&x)
}

#[pyfunction]
#[pyo3(signature = (x, alpha=100.0))]
pub fn dtlz4(x: Vec<f64>, alpha: f64) -> Vec<f64> {
    benchmark_objective_functions::dtlz4(&x, alpha)
}

#[pyfunction]
pub fn dtlz5(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::dtlz5(&x)
}

#[pyfunction]
pub fn dtlz6(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::dtlz6(&x)
}

#[pyfunction]
pub fn dtlz7(x: Vec<f64>) -> Vec<f64> {
    benchmark_objective_functions::dtlz7(&x)
}

/// Create a Problem pre-configured with a built-in benchmark function.
///
/// This bypasses the Python callable overhead — the objective function runs entirely in Rust,
/// enabling true multi-threaded parallelism.
///
/// Args:
///     name: Benchmark name ("dtlz1"-"dtlz7", "paraboloid_5", "paraboloid_5_loc",
///           "paraboloid_hyper_5", "simple_objective", "xyz_objective").
///     solution_length: Number of decision variables.
///     number_of_objectives: Number of objectives returned by the function.
///     bounds: List of (lower, upper) tuples for each variable. All treated as Real.
///     direction: Optional list of -1/1 per objective (default: all -1 for minimize).
#[pyfunction]
#[pyo3(signature = (name, solution_length, number_of_objectives, bounds, direction=None))]
pub fn create_benchmark_problem(
    name: &str,
    solution_length: usize,
    number_of_objectives: usize,
    bounds: Vec<(f64, f64)>,
    direction: Option<Vec<i8>>,
) -> PyResult<PyProblem> {
    if bounds.len() != solution_length {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bounds length must match solution_length",
        ));
    }

    let obj_fn: fn(&Vec<f64>) -> Vec<f64> = match name {
        "dtlz1" => benchmark_objective_functions::dtlz1,
        "dtlz2" => benchmark_objective_functions::dtlz2,
        "dtlz3" => benchmark_objective_functions::dtlz3,
        "dtlz5" => benchmark_objective_functions::dtlz5,
        "dtlz6" => benchmark_objective_functions::dtlz6,
        "dtlz7" => benchmark_objective_functions::dtlz7,
        "paraboloid_5" => benchmark_objective_functions::parabloid_5,
        "paraboloid_5_loc" => benchmark_objective_functions::parabloid_5_loc,
        "paraboloid_hyper_5" => benchmark_objective_functions::parabloid_hyper_5,
        "simple_objective" => benchmark_objective_functions::simple_objective,
        "xyz_objective" => benchmark_objective_functions::xyz_objective,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown benchmark '{}'. Available: dtlz1-dtlz7, paraboloid_5, paraboloid_5_loc, \
                 paraboloid_hyper_5, simple_objective, xyz_objective",
                name
            )))
        }
    };

    let data_types: Vec<SolutionDataTypes> = bounds
        .iter()
        .map(|&(lo, hi)| SolutionDataTypes::Real(Real::new(Some(lo), Some(hi))))
        .collect();

    Ok(create_problem_from_fn(
        solution_length,
        number_of_objectives,
        data_types,
        obj_fn,
        direction,
    ))
}
