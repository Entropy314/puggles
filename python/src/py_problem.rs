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

static ACTIVE_PROBLEM_ID: AtomicU64 = AtomicU64::new(0);

/// The callable registered under `id`, as a new reference. The registry lock is held only for
/// the lookup, never across the call into Python: holding it there would order it against the
/// GIL (deadlocking a callable that drops a `Problem`, whose `Drop` takes the same lock).
fn registered(py: Python<'_>, registry: &Mutex<HashMap<u64, Py<PyAny>>>, id: u64) -> Option<Py<PyAny>> {
    registry.lock().unwrap_or_else(|e| e.into_inner()).get(&id).map(|f| f.clone_ref(py))
}

/// Trampoline function matching `fn(&Vec<f64>) -> Vec<f64>`.
/// Looks up the Python callable from the global registry and calls it via the GIL.
fn python_objective_trampoline(input: &Vec<f64>) -> Vec<f64> {
    let problem_id = ACTIVE_PROBLEM_ID.load(Ordering::Relaxed);
    Python::attach(|py| {
        let callable = registered(py, callable_registry(), problem_id)
            .expect("Python callable not found in registry — was the Problem dropped?");
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
    ACTIVE_PROBLEM_ID.store(id, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Batch-evaluation callable registry
// ---------------------------------------------------------------------------

fn batch_callable_registry() -> &'static Mutex<HashMap<u64, Py<PyAny>>> {
    static INSTANCE: OnceLock<Mutex<HashMap<u64, Py<PyAny>>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Problem ids whose batch objective takes a `(P, n)` float64 ndarray (`batch_numpy=True`).
fn batch_numpy_ids() -> &'static Mutex<std::collections::HashSet<u64>> {
    static INSTANCE: OnceLock<Mutex<std::collections::HashSet<u64>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

static ACTIVE_BATCH_PROBLEM_ID: AtomicU64 = AtomicU64::new(0);

/// Batch trampoline: calls `f(population) -> objectives` once per generation.
/// `population` is `List[List[float]]`, or one `(P, n)` float64 ndarray with `batch_numpy=True`
/// (a single contiguous copy instead of P·n Python floats). The result may be a list of lists
/// or a 2-D float64 array.
fn python_batch_objective_trampoline(inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    use numpy::{PyArray2, PyArrayMethods};
    let problem_id = ACTIVE_BATCH_PROBLEM_ID.load(Ordering::Relaxed);
    let as_numpy = batch_numpy_ids().lock().unwrap_or_else(|e| e.into_inner()).contains(&problem_id);
    Python::attach(|py| {
        let callable = registered(py, batch_callable_registry(), problem_id)
            .expect("Python batch callable not found in registry — was the Problem dropped?");
        let py_pop: Py<PyAny> = if as_numpy {
            PyArray2::from_vec2(py, inputs).expect("solutions have equal length").into_any().unbind()
        } else {
            pyo3::types::PyList::new(py, inputs.iter().map(|sol| pyo3::types::PyList::new(py, sol).unwrap()))
                .unwrap()
                .into_any()
                .unbind()
        };
        let result = callable
            .call1(py, (py_pop,))
            .expect("Python batch objective function raised an exception");
        let result = result.bind(py);
        // A list never needs numpy; anything else that is a float64 2-D array is read directly.
        if !result.is_instance_of::<pyo3::types::PyList>() {
            if let Ok(arr) = result.cast::<PyArray2<f64>>() {
                let view = arr.readonly();
                return view.as_array().rows().into_iter().map(|r| r.to_vec()).collect();
            }
        }
        result
            .extract::<Vec<Vec<f64>>>()
            .expect("Python batch objective function must return a list of lists or a 2-D float64 array")
    })
}

pub fn set_active_batch_problem_id(id: u64) {
    ACTIVE_BATCH_PROBLEM_ID.store(id, Ordering::Relaxed);
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
    /// A GIL-free compiled objective (`objective_cfunc`), if this problem uses one.
    pub cfunc: Option<CFunc>,
}

// ---------------------------------------------------------------------------
// GIL-free compiled objective (Numba @cfunc)
// ---------------------------------------------------------------------------
//
// ponytail: same single-active-problem model as ACTIVE_PROBLEM_ID — `activate()` publishes
// the function and data pointers right before a run, and `cfunc_trampoline` reads them without
// the GIL, so Rayon threads can call it. Concurrent runs over *different* cfunc problems are
// unsupported. Upgrade path: a core `EvalFn` that can carry a context pointer.

type CObjective = unsafe extern "C" fn(*const f64, i64, *mut f64, i64, *mut std::ffi::c_void);

static CFUNC_ADDR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static CFUNC_DATA: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static CFUNC_OUTPUTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub struct CFunc {
    addr: usize,
    data_ptr: usize,
    outputs: usize,
    /// Keeps the cfunc and its data alive (and the data pointer valid) as long as the Problem.
    _keep: (Py<PyAny>, Option<Py<PyAny>>),
}

impl CFunc {
    /// Validates the signature at this trust boundary: calling a pointer with the wrong
    /// signature would crash the process instead of raising.
    fn new(py: Python<'_>, cf: &Bound<'_, PyAny>, data: Option<Py<PyAny>>, outputs: usize) -> PyResult<Self> {
        let ct = py.import("ctypes")?;
        let dbl = ct.getattr("POINTER")?.call1((ct.getattr("c_double")?,))?;
        let i64t = ct.getattr("c_int64")?;
        let expected = pyo3::types::PyTuple::new(py, [&dbl, &i64t, &dbl, &i64t, &ct.getattr("c_void_p")?])?;
        let sig = cf.getattr("ctypes").map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("objective_cfunc must be a numba @cfunc")
        })?;
        let args = pyo3::types::PyTuple::new(py, sig.getattr("argtypes")?.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        if !args.eq(&expected)? || !sig.getattr("restype")?.is_none() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "objective_cfunc must have the C signature \
                 void(CPointer(float64), int64, CPointer(float64), int64, voidptr)",
            ));
        }
        let data_ptr = match &data {
            None => 0,
            Some(arr) => {
                let arr = arr.bind(py);
                let contiguous = arr.getattr("flags")?.get_item("C_CONTIGUOUS")?.extract::<bool>()?;
                if !contiguous {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "cfunc_data must be a C-contiguous NumPy array (np.ascontiguousarray)",
                    ));
                }
                arr.getattr("ctypes")?.getattr("data")?.extract::<usize>()?
            }
        };
        Ok(CFunc {
            addr: cf.getattr("address")?.extract::<usize>()?,
            data_ptr,
            outputs,
            _keep: (cf.clone().unbind(), data),
        })
    }

    /// Publish this cfunc for `cfunc_trampoline`. Called right before every run.
    pub fn activate(&self) {
        CFUNC_ADDR.store(self.addr, Ordering::Relaxed);
        CFUNC_DATA.store(self.data_ptr, Ordering::Relaxed);
        CFUNC_OUTPUTS.store(self.outputs, Ordering::Relaxed);
    }
}

/// Calls the active cfunc directly: no GIL, no Python objects.
fn cfunc_trampoline(x: &Vec<f64>) -> Vec<f64> {
    let mut out = vec![0.0; CFUNC_OUTPUTS.load(Ordering::Relaxed)];
    // SAFETY: the address came from a numba cfunc whose signature `CFunc::new` checked against
    // `CObjective`; the Problem keeps it and its data alive while it can run.
    unsafe {
        let f: CObjective = std::mem::transmute(CFUNC_ADDR.load(Ordering::Relaxed));
        f(x.as_ptr(), x.len() as i64, out.as_mut_ptr(), out.len() as i64,
          CFUNC_DATA.load(Ordering::Relaxed) as *mut std::ffi::c_void);
    }
    out
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
            batch_numpy_ids().lock().unwrap_or_else(|e| e.into_inner()).remove(&self.problem_id);
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
    ///     permutation: Encode solutions as permutations of 0..solution_length.
    ///     batch_numpy: Pass the population to `batch_objective_function` as one `(P, n)` float64
    ///                  NumPy array instead of a list of lists (needs numpy). Returning a 2-D
    ///                  float64 array is always accepted.
    ///     objective_cfunc: A Numba `@cfunc("void(CPointer(float64), int64, CPointer(float64),
    ///                      int64, voidptr)")` computing `out[0..m]` from `x[0..n]`. puggles calls
    ///                      it directly from Rust without the GIL, so `execution_mode=
    ///                      "multithreaded"` runs it on every core.
    ///     cfunc_data: Optional C-contiguous NumPy array passed to `objective_cfunc` as its
    ///                 `voidptr` argument (e.g. a dataset). Kept alive by the Problem.
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
        permutation = false,
        batch_numpy = false,
        objective_cfunc = None,
        cfunc_data = None,
    ))]
    fn new(
        py: Python<'_>,
        solution_length: usize,
        number_of_objectives: usize,
        solution_data_types: Vec<Py<PyAny>>,
        objective_function: Option<Py<PyAny>>,
        direction: Option<Vec<i8>>,
        objective_constraints: Option<Vec<Option<f64>>>,
        constraint_operands: Option<Vec<Option<String>>>,
        batch_objective_function: Option<Py<PyAny>>,
        permutation: bool,
        batch_numpy: bool,
        objective_cfunc: Option<Py<PyAny>>,
        cfunc_data: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let data_types: Vec<SolutionDataTypes> = parse_data_types(py, solution_data_types)?;
        let problem_id = PROBLEM_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        if let Some(cf) = objective_cfunc {
            let cfunc = CFunc::new(py, cf.bind(py), cfunc_data, number_of_objectives)?;
            let mut problem = Problem::try_new(
                solution_length,
                number_of_objectives,
                objective_constraints,
                constraint_operands,
                direction,
                data_types,
                cfunc_trampoline,
            )
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            if permutation {
                problem = problem.with_permutation_encoding();
            }
            return Ok(PyProblem {
                store: ProblemStore {
                    problem: Arc::new(problem),
                    problem_id,
                    uses_python_callable: false,
                    uses_batch_callable: false,
                    cfunc: Some(cfunc),
                },
            });
        }

        if let Some(batch_fn) = batch_objective_function {
            // Batch mode: register the batch callable, use the batch trampoline.
            // A no-op single-eval function is required by Problem::new but will never be called.
            batch_callable_registry()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(problem_id, batch_fn.into_pyobject(py).unwrap().into());
            if batch_numpy {
                batch_numpy_ids().lock().unwrap_or_else(|e| e.into_inner()).insert(problem_id);
            }

            fn placeholder(_: &Vec<f64>) -> Vec<f64> { Vec::new() }

            let mut problem = Problem::try_new(
                solution_length,
                number_of_objectives,
                objective_constraints,
                constraint_operands,
                direction,
                data_types,
                placeholder,
            )
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            problem.eval_fn = EvalFn::Batch(python_batch_objective_trampoline);
            if permutation {
                problem = problem.with_permutation_encoding();
            }

            return Ok(PyProblem {
                store: ProblemStore {
                    problem: Arc::new(problem),
                    problem_id,
                    uses_python_callable: false,
                    uses_batch_callable: true,
                    cfunc: None,
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

        let mut problem = Problem::try_new(
            solution_length,
            number_of_objectives,
            objective_constraints,
            constraint_operands,
            direction,
            data_types,
            python_objective_trampoline,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if permutation {
            problem = problem.with_permutation_encoding();
        }

        Ok(PyProblem {
            store: ProblemStore {
                problem: Arc::new(problem),
                problem_id,
                uses_python_callable: true,
                uses_batch_callable: false,
                cfunc: None,
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

    /// Per-objective direction: -1 = minimize, 1 = maximize.
    #[getter]
    fn direction(&self) -> Vec<i8> {
        direction_of(&self.store.problem)
    }

    /// Per-variable `(kind, lower_bound, upper_bound)`, kind one of "real", "integer", "binary".
    #[getter]
    fn variable_types(&self) -> Vec<VariableType> {
        variable_types_of(&self.store.problem)
    }

    /// True when this problem uses permutation encoding.
    #[getter]
    fn permutation(&self) -> bool {
        self.store.problem.encoding == puggles::core::Encoding::Permutation
    }

    fn __repr__(&self) -> String {
        format!(
            "Problem(solution_length={}, number_of_objectives={})",
            self.store.problem.solution_length, self.store.problem.number_of_objectives
        )
    }
}

/// A variable's `(kind, lower_bound, upper_bound)`.
pub(crate) type VariableType = (&'static str, Option<f64>, Option<f64>);

pub(crate) fn direction_of(p: &Problem) -> Vec<i8> {
    p.direction.clone().unwrap_or_else(|| vec![-1; p.number_of_objectives])
}

pub(crate) fn variable_types_of(p: &Problem) -> Vec<VariableType> {
    p.solution_data_types
        .iter()
        .map(|t| match t {
            SolutionDataTypes::Real(r) => ("real", r.lower_bound, r.upper_bound),
            SolutionDataTypes::Integer(i) => ("integer", i.lower_bound.map(|v| v as f64), i.upper_bound.map(|v| v as f64)),
            SolutionDataTypes::BitBinary(_) => ("binary", Some(0.0), Some(1.0)),
        })
        .collect()
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
            problem: Arc::new(
                Problem::new(
                    solution_length,
                    number_of_objectives,
                    None,
                    None,
                    direction,
                    data_types,
                    objective_function,
                ),
            ),
            problem_id,
            uses_python_callable: false,
            uses_batch_callable: false,
            cfunc: None,
        },
    }
}
