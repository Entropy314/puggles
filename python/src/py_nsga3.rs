use pyo3::prelude::*;
use rustypus::genetic_algorithms_v2::ExecutionMode;
use rustypus::nsga3::NSGAIII;
use std::sync::Arc;

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
    #[new]
    #[pyo3(signature = (problem, population_size = 0, divisions = 12, execution_mode = "sequential"))]
    fn new(problem: PyObject, population_size: usize, divisions: usize, execution_mode: &str) -> Self {
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
            last_archive: Vec::new(),
            last_population: Vec::new(),
            last_nfe: 0,
        }
    }

    /// Run for up to `max_nfe` objective evaluations.
    fn run(&mut self, py: Python<'_>, max_nfe: usize) -> PyResult<()> {
        let store = extract_store(py, &self.problem)?;
        if store.uses_batch_callable {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "NSGA-III does not support a batch objective; use NSGAII for batch_objective_function.",
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

        let (archive, population, nfe) = if store.uses_python_callable {
            // Hold the GIL: the objective trampoline calls back into Python.
            let mut ga = NSGAIII::new(Arc::clone(&store.problem), pop_size, divisions, mode);
            ga.run(max_nfe);
            (ga.get_archive().to_vec(), ga.population.clone(), ga.get_nfe())
        } else {
            // Pure-Rust objective: release the GIL for real parallelism.
            let problem = Arc::clone(&store.problem);
            py.allow_threads(move || {
                let mut ga = NSGAIII::new(problem, pop_size, divisions, mode);
                ga.run(max_nfe);
                (ga.get_archive().to_vec(), ga.population.clone(), ga.get_nfe())
            })
        };

        self.last_archive = archive.iter().map(PySolution::from_core_solution).collect();
        self.last_population = population.iter().map(PySolution::from_core_solution).collect();
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

    fn __repr__(&self) -> String {
        format!(
            "NSGAIII(population_size={}, divisions={}, nfe={})",
            self.population_size, self.divisions, self.last_nfe
        )
    }
}
