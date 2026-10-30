use pyo3::prelude::*;
use puggles::genetic_algorithms_v2::ExecutionMode;
use puggles::nsga3::NSGAIII;
use puggles::recorder_abi::{RecorderHandle, Snapshot};
use std::sync::Arc;

use crate::py_recorder::{native_recorder, run_watched_with_gil};
use crate::py_nsgaii::extract_store;
use crate::py_problem::set_active_problem_id;
use crate::py_solution::PySolution;

/// NSGA-III — reference-point many-objective GA. Preferred over NSGA-II for 3+ objectives,
/// where crowding distance diversifies poorly. Operators are the crate defaults (SBX +
/// polynomial mutation); operator configs are not exposed (NSGA-III owns them privately).
#[pyclass(name = "NSGAIII")]
pub struct PyNSGAIII {
    problem: PyObject,
    population_size: usize,
    divisions: usize,
    execution_mode: ExecutionMode,
    seed: Option<u64>,
    last_archive: Vec<PySolution>,
    last_population: Vec<PySolution>,
    last_nfe: usize,
}

#[pymethods]
impl PyNSGAIII {
    /// Create an NSGA-III optimizer.
    ///
    /// Args:
    ///     problem: The Problem to optimize (a batch objective is NOT supported here — use NSGAII).
    ///     population_size: Individuals per generation. 0 derives it from the reference-point count.
    ///     divisions: Reference-point density (e.g. 12 for 3 objectives → 91 points).
    ///     execution_mode: "sequential", "multithreaded", or "gpu" (default "sequential"). A
    ///                     Python-callable objective always runs Sequential (GIL).
    ///     seed: Optional u64 seeding every source of randomness, so the same seed reproduces
    ///           a run exactly. `None` draws fresh entropy. Pair it with "sequential" —
    ///           "multithreaded" is not deterministic even when seeded.
    #[new]
    #[pyo3(signature = (problem, population_size = 0, divisions = 12, execution_mode = "sequential", seed = None))]
    fn new(problem: PyObject, population_size: usize, divisions: usize, execution_mode: &str, seed: Option<u64>) -> Self {
        let mode = match execution_mode {
            "multithreaded" => ExecutionMode::MultiThreaded,
            "gpu" => ExecutionMode::GPU,
            _ => ExecutionMode::Sequential,
        };
        PyNSGAIII {
            problem,
            population_size,
            divisions,
            execution_mode: mode,
            seed,
            last_archive: Vec::new(),
            last_population: Vec::new(),
            last_nfe: 0,
        }
    }

    /// Run for up to `max_nfe` objective evaluations.
    ///
    /// Args:
    ///     max_nfe: Maximum number of objective function evaluations.
    ///     callback: Optional callable invoked after each generation.
    ///               Signature: `(archive, population, nfe) -> bool | None`
    ///               Return `False` to stop early. A `Dashboard` works here and is fastest.
    #[pyo3(signature = (max_nfe, callback = None))]
    fn run(&mut self, py: Python<'_>, max_nfe: usize, callback: Option<PyObject>) -> PyResult<()> {
        let store = extract_store(py, &self.problem)?;
        if store.uses_batch_callable {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "NSGA-III does not support a batch objective; use NSGAII for batch_objective_function.",
            ));
        }
        if store.gpu.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "NSGA-III evaluates one solution at a time and can't run a GpuProblem; use NSGAII.",
            ));
        }
        // A Python callable re-acquires the GIL per eval → force Sequential (see PyNSGAII).
        let mode = if store.uses_python_callable {
            ExecutionMode::Sequential
        } else {
            self.execution_mode
        };
        if store.uses_python_callable {
            set_active_problem_id(store.problem_id);
        }

        let pop_size = self.population_size;
        let divisions = self.divisions;
        let seed = self.seed;

        // A dashboard add-on's callback records natively (see PyNSGAII::run).
        let dashboard = native_recorder(py, callback.as_ref())?;
        let callback = if dashboard.is_some() { None } else { callback };

        let (archive, population, nfe) = if let Some(ref cb) = callback {
            // Hold the GIL to call back into Python each generation. A raised exception stops
            // the run and is re-raised once the GA has unwound.
            let mut ga = NSGAIII::new(Arc::clone(&store.problem), pop_size, divisions, mode);
            if let Some(s) = seed {
                ga = ga.with_seed(s);
            }
            let mut error: Option<PyErr> = None;
            ga.run_with(max_nfe, |ga| {
                let archive: Vec<PySolution> =
                    ga.get_archive().iter().map(PySolution::from_core_solution).collect();
                let population: Vec<PySolution> =
                    ga.population.iter().map(PySolution::from_core_solution).collect();
                match cb.call1(py, (archive, population, ga.get_nfe())) {
                    Ok(result) => !matches!(result.extract::<bool>(py), Ok(false)),
                    Err(e) => {
                        error = Some(e);
                        false
                    }
                }
            });
            if let Some(e) = error {
                return Err(e);
            }
            (ga.take_archive(), std::mem::take(&mut ga.population), ga.get_nfe())
        } else if store.uses_python_callable {
            // Hold the GIL: the objective trampoline calls back into Python.
            let mut ga = NSGAIII::new(Arc::clone(&store.problem), pop_size, divisions, mode);
            if let Some(s) = seed {
                ga = ga.with_seed(s);
            }
            match dashboard.as_ref() {
                Some(dash) => run_watched_with_gil(
                    py,
                    dash,
                    |on_gen| ga.run_with(max_nfe, |g| on_gen(g)),
                    |g: &mut NSGAIII| {
                        dash.apply_focus(g); // goals from the page steer NSGA-III
                        dash.record(&Snapshot::of_nsga3(g))
                    },
                )?,
                None => ga.run(max_nfe),
            }
            (ga.take_archive(), std::mem::take(&mut ga.population), ga.get_nfe())
        } else {
            // Pure-Rust objective: release the GIL for real parallelism.
            let problem = Arc::clone(&store.problem);
            py.allow_threads(move || {
                let mut ga = NSGAIII::new(problem, pop_size, divisions, mode);
                if let Some(s) = seed {
                    ga = ga.with_seed(s);
                }
                run_observed(&mut ga, max_nfe, dashboard.as_ref());
                (ga.take_archive(), std::mem::take(&mut ga.population), ga.get_nfe())
            })
        };

        self.last_archive = archive.into_iter().map(PySolution::from_owned).collect();
        self.last_population = population.into_iter().map(PySolution::from_owned).collect();
        self.last_nfe = nfe;
        Ok(())
    }

    /// Pareto-optimal archive from the last run.
    fn get_archive(&self) -> Vec<PySolution> {
        self.last_archive.clone()
    }

    /// Final population from the last run.
    fn get_population(&self) -> Vec<PySolution> {
        self.last_population.clone()
    }

    #[getter]
    fn nfe(&self) -> usize {
        self.last_nfe
    }

    /// The problem this optimizer was created with.
    #[getter]
    fn problem(&self, py: Python<'_>) -> PyObject {
        self.problem.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NSGAIII(population_size={}, divisions={}, nfe={})",
            self.population_size, self.divisions, self.last_nfe
        )
    }
}

/// `run`, recording each generation into `dashboard` when there is one (and applying goals
/// from its page, which steer NSGA-III).
fn run_observed(ga: &mut NSGAIII, max_nfe: usize, dashboard: Option<&RecorderHandle>) {
    match dashboard {
        Some(dash) => {
            ga.run_with(max_nfe, |ga| {
                dash.apply_focus(ga);
                dash.record_and_hold(&Snapshot::of_nsga3(ga))
            });
            dash.finish();
        }
        None => ga.run(max_nfe),
    }
}
