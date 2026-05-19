use pyo3::prelude::*;
use rustypus::core::{EvalFn, Problem};
use rustypus::gatypes::SolutionDataTypes;
use std::cell::RefCell;
use std::sync::Arc;

use crate::py_types::parse_data_types;

// ---------------------------------------------------------------------------
// Thread-local storage for active callables
// ---------------------------------------------------------------------------

thread_local! {
    static TLS_CALLABLE: RefCell<Option<Py<PyAny>>> = RefCell::new(None);
    static TLS_BATCH_CALLABLE: RefCell<Option<Py<PyAny>>> = RefCell::new(None);
}

fn python_objective_trampoline(input: &Vec<f64>) -> Vec<f64> {
    TLS_CALLABLE.with(|cell| {
        let borrow = cell.borrow();
        let callable = borrow.as_ref()
            .expect("No active Python callable — was activate_callable() called before run()?");
        Python::with_gil(|py| {
            let py_list = pyo3::types::PyList::new(py, input).unwrap();
            callable.call1(py, (py_list,))
                .expect("Python objective function raised an exception")
                .extract::<Vec<f64>>(py)
                .expect("Python objective function must return a list of floats")
        })
    })
}

fn python_batch_objective_trampoline(inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    TLS_BATCH_CALLABLE.with(|cell| {
        let borrow = cell.borrow();
        let callable = borrow.as_ref()
            .expect("No active batch callable — was activate_batch_callable() called before run()?");
        Python::with_gil(|py| {
            let py_pop = pyo3::types::PyList::new(
                py,
                inputs.iter().map(|sol| pyo3::types::PyList::new(py, sol).unwrap()),
            ).unwrap();
            callable.call1(py, (py_pop,))
                .expect("Python batch objective function raised an exception")
                .extract::<Vec<Vec<f64>>>(py)
                .expect("Python batch objective function must return a list of list of floats")
        })
    })
}

pub fn activate_callable(callable: Py<PyAny>) {
    TLS_CALLABLE.with(|c| *c.borrow_mut() = Some(callable));
}
pub fn deactivate_callable() {
    TLS_CALLABLE.with(|c| *c.borrow_mut() = None);
}
pub fn activate_batch_callable(callable: Py<PyAny>) {
    TLS_BATCH_CALLABLE.with(|c| *c.borrow_mut() = Some(callable));
}
pub fn deactivate_batch_callable() {
    TLS_BATCH_CALLABLE.with(|c| *c.borrow_mut() = None);
}

// ---------------------------------------------------------------------------
// ProblemStore — owns the Problem and its callables
// ---------------------------------------------------------------------------

pub struct ProblemStore {
    pub problem: Arc<Problem>,
    pub callable: Option<Py<PyAny>>,
    pub batch_callable: Option<Py<PyAny>>,
    #[cfg(feature = "gpu")]
    pub gpu_evaluator: Option<std::sync::Arc<rustypus::gpu_evaluator::GpuEvaluator>>,
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

        if let Some(batch_fn) = batch_objective_function {
            // Batch mode: use EvalFn::Batch with the batch trampoline.
            let problem = Arc::new(Problem {
                solution_length,
                number_of_objectives,
                objective_constraint: objective_constraints,
                objective_constraint_operands: constraint_operands,
                direction: direction.or_else(|| Some(vec![-1; number_of_objectives])),
                solution_data_types: data_types,
                eval_fn: EvalFn::Batch(python_batch_objective_trampoline),
            });

            return Ok(PyProblem {
                store: ProblemStore {
                    problem,
                    callable: None,
                    batch_callable: Some(batch_fn.into_pyobject(py).unwrap().into()),
                    #[cfg(feature = "gpu")]
                    gpu_evaluator: None,
                },
            });
        }

        // Single-evaluation mode (original behaviour).
        let obj_fn = objective_function.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Either objective_function or batch_objective_function must be provided",
            )
        })?;

        let problem = Arc::new(Problem::new(
            solution_length,
            number_of_objectives,
            objective_constraints,
            constraint_operands,
            direction,
            data_types,
            python_objective_trampoline,
        ));

        Ok(PyProblem {
            store: ProblemStore {
                problem,
                callable: Some(obj_fn.into_pyobject(py).unwrap().into()),
                batch_callable: None,
                #[cfg(feature = "gpu")]
                gpu_evaluator: None,
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
            callable: None,
            batch_callable: None,
            #[cfg(feature = "gpu")]
            gpu_evaluator: None,
        },
    }
}
