use pyo3::prelude::*;
use puggles::core::{EvalFn, Problem};
use puggles::gatypes::SolutionDataTypes;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::py_types::parse_data_types;

// Global unique ID counter for problems
static PROBLEM_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Single-evaluation callable registry
// ---------------------------------------------------------------------------
//
// ponytail: `EvalFn` stores a bare `fn` pointer, which cannot capture a Python
// callable. We bridge via a process-global registry keyed by problem id plus a
// trampoline `fn`. A single active problem id is set at the start of each
// `run()`, so sequential use of any number of problems is fine; concurrent GAs
// over *different* Python-callable problems in threads are unsupported (the GIL
// makes that pointless anyway). Upgrade path if ever needed: change core
// `EvalFn` to hold `Arc<dyn Fn + Send + Sync>` and store the callable directly.

fn callable_registry() -> &'static Mutex<HashMap<u64, Py<PyAny>>> {
    static INSTANCE: OnceLock<Mutex<HashMap<u64, Py<PyAny>>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn active_problem_id() -> &'static Mutex<u64> {
    static INSTANCE: OnceLock<Mutex<u64>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(0))
}

/// Trampoline function matching `fn(&Vec<f64>) -> Vec<f64>`.
/// Looks up the Python callable from the global registry and calls it via the GIL.
fn python_objective_trampoline(input: &Vec<f64>) -> Vec<f64> {
    let problem_id = *active_problem_id().lock().unwrap_or_else(|e| e.into_inner());
    let registry = callable_registry().lock().unwrap_or_else(|e| e.into_inner());
    let callable = registry
        .get(&problem_id)
        .expect("Python callable not found in registry — was the Problem dropped?");
    Python::with_gil(|py| {
        let py_list = pyo3::types::PyList::new(py, input).unwrap();
        let result = callable
            .call1(py, (py_list,))
            .expect("Python objective function raised an exception");
        result
            .extract::<Vec<f64>>(py)
            .expect("Python objective function must return a list of floats")
    })
}

/// Set the active problem ID before a run. Called by PyNSGAII.
pub fn set_active_problem_id(id: u64) {
    *active_problem_id().lock().unwrap_or_else(|e| e.into_inner()) = id;
}

// ---------------------------------------------------------------------------
// Batch-evaluation callable registry
// ---------------------------------------------------------------------------

fn batch_callable_registry() -> &'static Mutex<HashMap<u64, Py<PyAny>>> {
    static INSTANCE: OnceLock<Mutex<HashMap<u64, Py<PyAny>>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn active_batch_problem_id() -> &'static Mutex<u64> {
    static INSTANCE: OnceLock<Mutex<u64>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(0))
}

/// Batch trampoline: calls `f(population: List[List[float]]) -> List[List[float]]`.
/// The Python function can use multiprocessing / ProcessPoolExecutor internally.
fn python_batch_objective_trampoline(inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let problem_id = *active_batch_problem_id().lock().unwrap_or_else(|e| e.into_inner());
    let registry = batch_callable_registry().lock().unwrap_or_else(|e| e.into_inner());
    let callable = registry
        .get(&problem_id)
        .expect("Python batch callable not found in registry — was the Problem dropped?");
    Python::with_gil(|py| {
        // Build List[List[float]]
        let py_pop = pyo3::types::PyList::new(
            py,
            inputs.iter().map(|sol| pyo3::types::PyList::new(py, sol).unwrap()),
        )
        .unwrap();
        let result = callable
            .call1(py, (py_pop,))
            .expect("Python batch objective function raised an exception");
        result
            .extract::<Vec<Vec<f64>>>(py)
            .expect("Python batch objective function must return a list of list of floats")
    })
}

pub fn set_active_batch_problem_id(id: u64) {
    *active_batch_problem_id().lock().unwrap_or_else(|e| e.into_inner()) = id;
}

// ---------------------------------------------------------------------------
// ProblemStore — owns the Problem and its callables
// ---------------------------------------------------------------------------

pub struct ProblemStore {
    pub problem: Arc<Problem>,
    pub problem_id: u64,
    /// True if using a single-evaluation Python callable (trampoline).
    pub uses_python_callable: bool,
    /// True if using a batch Python callable (batch trampoline).
    pub uses_batch_callable: bool,
}

impl Drop for ProblemStore {
    fn drop(&mut self) {
        if self.uses_python_callable {
            callable_registry()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&self.problem_id);
        }
        if self.uses_batch_callable {
            batch_callable_registry()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&self.problem_id);
        }
    }
}

// ---------------------------------------------------------------------------
// PyProblem
// ---------------------------------------------------------------------------

#[pyclass(name = "Problem")]
pub struct PyProblem {
    pub(crate) store: ProblemStore,
}

#[pymethods]
impl PyProblem {
    /// Create a new optimization Problem.
    ///
    /// Args:
    ///     solution_length: Number of decision variables.
    ///     number_of_objectives: Number of objective functions.
    ///     solution_data_types: List of Real(...), Integer(...), or BitBinary() per variable.
    ///     objective_function: A callable `f(x: list[float]) -> list[float]`.
    ///                         Pass `None` when using `batch_objective_function`.
    ///     direction: Optional list of -1 (minimize) or 1 (maximize) per objective.
    ///     objective_constraints: Optional constraint bounds (None where unconstrained).
    ///     constraint_operands: Optional comparison operators ("<", ">", "<=", ">=", "==", "!=").
    ///     batch_objective_function: Optional batch callable
    ///                               `f(pop: list[list[float]]) -> list[list[float]]`.
    ///                               When provided, the GA calls this once per generation instead of
    ///                               `objective_function` per solution. Use Python multiprocessing
    ///                               (e.g. ProcessPoolExecutor) inside this function for parallelism.
    #[new]
    #[pyo3(signature = (
        solution_length,
        number_of_objectives,
        solution_data_types,
        objective_function = None,
        direction = None,
        objective_constraints = None,
        constraint_operands = None,
        batch_objective_function = None,
    ))]
    fn new(
        py: Python<'_>,
        solution_length: usize,
        number_of_objectives: usize,
        solution_data_types: Vec<PyObject>,
        objective_function: Option<PyObject>,
        direction: Option<Vec<i8>>,
        objective_constraints: Option<Vec<Option<f64>>>,
        constraint_operands: Option<Vec<Option<String>>>,
        batch_objective_function: Option<PyObject>,
    ) -> PyResult<Self> {
        let data_types: Vec<SolutionDataTypes> = parse_data_types(py, solution_data_types)?;
        let problem_id = PROBLEM_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        if let Some(batch_fn) = batch_objective_function {
            // Batch mode: register the batch callable, use the batch trampoline.
            // A no-op single-eval function is required by Problem::new but will never be called.
            batch_callable_registry()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(problem_id, batch_fn.into_pyobject(py).unwrap().into());

            fn placeholder(_: &Vec<f64>) -> Vec<f64> { Vec::new() }

            let mut problem = Problem::new(
                solution_length,
                number_of_objectives,
                objective_constraints,
                constraint_operands,
                direction,
                data_types,
                placeholder,
            );
            problem.eval_fn = EvalFn::Batch(python_batch_objective_trampoline);

            return Ok(PyProblem {
                store: ProblemStore {
                    problem: Arc::new(problem),
                    problem_id,
                    uses_python_callable: false,
                    uses_batch_callable: true,
                },
            });
        }

        // Single-evaluation mode (original behaviour).
        let obj_fn = objective_function.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Either objective_function or batch_objective_function must be provided",
            )
        })?;
        callable_registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(problem_id, obj_fn.into_pyobject(py).unwrap().into());

        let problem = Problem::new(
            solution_length,
            number_of_objectives,
            objective_constraints,
            constraint_operands,
            direction,
            data_types,
            python_objective_trampoline,
        );

        Ok(PyProblem {
            store: ProblemStore {
                problem: Arc::new(problem),
                problem_id,
                uses_python_callable: true,
                uses_batch_callable: false,
            },
        })
    }

    #[getter]
    fn solution_length(&self) -> usize {
        self.store.problem.solution_length
    }

    #[getter]
    fn number_of_objectives(&self) -> usize {
        self.store.problem.number_of_objectives
    }

    fn __repr__(&self) -> String {
        format!(
            "Problem(solution_length={}, number_of_objectives={})",
            self.store.problem.solution_length, self.store.problem.number_of_objectives
        )
    }
}

// ---------------------------------------------------------------------------
// create_problem_from_fn — used by benchmark factory (bypasses Python registry)
// ---------------------------------------------------------------------------

pub fn create_problem_from_fn(
    solution_length: usize,
    number_of_objectives: usize,
    data_types: Vec<SolutionDataTypes>,
    objective_function: fn(&Vec<f64>) -> Vec<f64>,
    direction: Option<Vec<i8>>,
) -> PyProblem {
    let problem_id = PROBLEM_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    PyProblem {
        store: ProblemStore {
            problem: Arc::new(Problem::new(
                solution_length,
                number_of_objectives,
                None,
                None,
                direction,
                data_types,
                objective_function,
            )),
            problem_id,
            uses_python_callable: false,
            uses_batch_callable: false,
        },
    }
}
