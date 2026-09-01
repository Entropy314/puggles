use pyo3::prelude::*;
use puggles::core::Problem;
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use puggles::genetic_operators::crossover::{
    ArithmeticCrossover, BlendCrossover, Crossover, CrossoverManager,
    DifferentialEvolutionCrossover, ParentCentricCrossover, SimulatedBinaryCrossover,
    UnimodalDistributionCrossover, UniformCrossover,
};
use puggles::genetic_operators::mutation::{
    BitFlipMutation, GaussianMutation, Mutation, MutationManager, PolynomialMutation,
    UniformMutation,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::py_gpu::PyGpuProblem;
use crate::py_operators::{PyCrossoverConfig, PyMutationConfig};
use crate::py_problem::{set_active_problem_id, PyProblem};
use crate::py_solution::PySolution;

// ---------------------------------------------------------------------------
// Extracted store metadata. The core NSGAII takes `Arc<Problem>`, so we just
// clone the Arc out of the problem object — no lifetimes, no raw pointers.
// ---------------------------------------------------------------------------

pub(crate) struct StoreData {
    pub(crate) problem: Arc<Problem>,
    pub(crate) problem_id: u64,
    pub(crate) uses_python_callable: bool,
    pub(crate) uses_batch_callable: bool,
}

pub(crate) fn extract_store(py: Python<'_>, obj: &PyObject) -> PyResult<StoreData> {
    if let Ok(p) = obj.extract::<Py<PyProblem>>(py) {
        let b = p.borrow(py);
        return Ok(StoreData {
            problem: Arc::clone(&b.store.problem),
            problem_id: b.store.problem_id,
            uses_python_callable: b.store.uses_python_callable,
            uses_batch_callable: b.store.uses_batch_callable,
        });
    }
    #[cfg(feature = "gpu")]
    if let Ok(p) = obj.extract::<Py<PyGpuProblem>>(py) {
        let b = p.borrow(py);
        return Ok(StoreData {
            problem: Arc::clone(&b.store.problem),
            problem_id: b.store.problem_id,
            uses_python_callable: b.store.uses_python_callable,
            uses_batch_callable: b.store.uses_batch_callable,
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "problem must be a Problem or GpuProblem instance",
    ))
}

// ---------------------------------------------------------------------------
// PyNSGAII
// ---------------------------------------------------------------------------

#[pyclass(name = "NSGAII")]
pub struct PyNSGAII {
    /// Accepts both `Problem` and `GpuProblem`. Held so its ProblemStore (and the
    /// callable registry entry) outlives every run.
    problem: PyObject,
    population_size: usize,
    execution_mode: ExecutionMode,
    crossover_config: Option<Py<PyCrossoverConfig>>,
    mutation_config: Option<Py<PyMutationConfig>>,
    num_threads: Option<usize>,
    seed: Option<u64>,
    last_archive: Vec<PySolution>,
    last_population: Vec<PySolution>,
    last_nfe: usize,
}

#[pymethods]
impl PyNSGAII {
    /// Create a new NSGA-II optimizer.
    ///
    /// Args:
    ///     problem: The Problem or GpuProblem to optimize.
    ///     population_size: Number of individuals per generation (default 100).
    ///     execution_mode: "sequential", "multithreaded", or "gpu" (default "sequential").
    ///                     A Python-callable objective always runs Sequential (the GIL
    ///                     serialises evaluations; parallel would only deadlock).
    ///     crossover_config: Optional CrossoverConfig to customise crossover operators.
    ///     mutation_config: Optional MutationConfig to customise mutation operators.
    ///     num_threads: Optional Rayon thread count (default: all logical CPUs).
    ///     seed: Optional u64 seeding every source of randomness (initial population,
    ///           crossover, mutation, selection). The same seed reproduces a run exactly.
    ///           `None` draws fresh entropy, so runs differ. Note that `multithreaded`
    ///           mode is not deterministic even with a seed, because Rayon interleaves
    ///           evaluations non-deterministically — pair `seed` with "sequential".
    #[new]
    #[pyo3(signature = (
        problem,
        population_size = 100,
        execution_mode = "sequential",
        crossover_config = None,
        mutation_config = None,
        num_threads = None,
        seed = None,
    ))]
    fn new(
        problem: PyObject,
        population_size: usize,
        execution_mode: &str,
        crossover_config: Option<Py<PyCrossoverConfig>>,
        mutation_config: Option<Py<PyMutationConfig>>,
        num_threads: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        let mode = match execution_mode {
            "multithreaded" => ExecutionMode::MultiThreaded,
            "gpu" => ExecutionMode::GPU,
            _ => ExecutionMode::Sequential,
        };
        PyNSGAII {
            problem,
            population_size,
            execution_mode: mode,
            crossover_config,
            mutation_config,
            num_threads,
            seed,
            last_archive: Vec::new(),
            last_population: Vec::new(),
            last_nfe: 0,
        }
    }

    /// Run the algorithm for up to `max_nfe` function evaluations.
    ///
    /// Args:
    ///     max_nfe: Maximum number of objective function evaluations.
    ///     callback: Optional callable invoked after each iteration.
    ///               Signature: `(archive, population, nfe) -> bool | None`
    ///               Return `False` to stop early.
    #[pyo3(signature = (max_nfe, callback = None))]
    fn run(&mut self, py: Python<'_>, max_nfe: usize, callback: Option<PyObject>) -> PyResult<()> {
        // Apply Rayon thread pool size (best-effort; ignored if already set)
        if let Some(n) = self.num_threads {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
        }

        let store = extract_store(py, &self.problem)?;

        // A Python callable re-acquires the GIL per evaluation. Forcing Sequential
        // avoids a rayon-worker-vs-GIL deadlock; the GIL serialises calls anyway.
        // Pure-Rust/GPU objectives keep their requested mode and true parallelism.
        let effective_mode = if store.uses_python_callable || store.uses_batch_callable {
            ExecutionMode::Sequential
        } else {
            self.execution_mode
        };

        if store.uses_python_callable {
            set_active_problem_id(store.problem_id);
        }
        if store.uses_batch_callable {
            crate::py_problem::set_active_batch_problem_id(store.problem_id);
        }

        // Build operator managers from optional configs
        let crossover_manager = self
            .crossover_config
            .as_ref()
            .map(|cfg| build_crossover_manager(&cfg.borrow(py)));
        let mutation_manager = self
            .mutation_config
            .as_ref()
            .map(|cfg| build_mutation_manager(&cfg.borrow(py), store.problem.solution_length));

        let needs_gil =
            store.uses_python_callable || store.uses_batch_callable || callback.is_some();

        if needs_gil {
            let mut ga = NSGAII::new(Arc::clone(&store.problem), self.population_size, effective_mode);
            if let Some(s) = self.seed {
                ga = ga.with_seed(s);
            }
            if let Some(cm) = crossover_manager {
                ga.crossover_manager = cm;
            }
            if let Some(mm) = mutation_manager {
                ga.mutation_manager = mm;
            }

            if let Some(ref cb) = callback {
                // Step-by-step loop with per-iteration callback.
                ga.initialize();
                // Evaluate the initial population (single-eval callables only).
                {
                    let pop = &mut ga.population;
                    for s in pop.iter_mut().filter(|s| !s.evaluated) {
                        s.evaluate();
                        ga.nfe.fetch_add(1, Ordering::Relaxed);
                    }
                }
                while ga.get_nfe() < max_nfe {
                    ga.iterate();
                    ga.update_archive();
                    let archive: Vec<PySolution> =
                        ga.get_archive().iter().map(PySolution::from_core_solution).collect();
                    let population: Vec<PySolution> =
                        ga.population.iter().map(PySolution::from_core_solution).collect();
                    let result = cb.call1(py, (archive, population, ga.get_nfe()))?;
                    if let Ok(false) = result.extract::<bool>(py) {
                        break;
                    }
                }
            } else {
                ga.run(max_nfe);
            }

            self.last_archive =
                ga.get_archive().iter().map(PySolution::from_core_solution).collect();
            self.last_population =
                ga.population.iter().map(PySolution::from_core_solution).collect();
            self.last_nfe = ga.get_nfe();
        } else {
            // Pure-Rust or GPU objective, no Python callback: release the GIL for true
            // Rayon parallelism. `Arc<Problem>` is Send + Sync, so it moves in directly.
            let problem = Arc::clone(&store.problem);
            let pop_size = self.population_size;
            let mode = effective_mode;
            let seed = self.seed;

            let (archive, population, nfe) = py.allow_threads(move || {
                let mut ga = NSGAII::new(problem, pop_size, mode);
                if let Some(s) = seed {
                    ga = ga.with_seed(s);
                }
                if let Some(cm) = crossover_manager {
                    ga.crossover_manager = cm;
                }
                if let Some(mm) = mutation_manager {
                    ga.mutation_manager = mm;
                }
                ga.run(max_nfe);
                let archive = ga.get_archive().to_vec();
                let population = ga.population.clone();
                let nfe = ga.get_nfe();
                (archive, population, nfe)
            });

            self.last_archive = archive.iter().map(PySolution::from_core_solution).collect();
            self.last_population =
                population.iter().map(PySolution::from_core_solution).collect();
            self.last_nfe = nfe;
        }

        Ok(())
    }

    /// Get the Pareto-optimal archive from the last run.
    fn get_archive(&self) -> Vec<PySolution> {
        self.last_archive.clone()
    }

    /// Get the final population from the last run.
    fn get_population(&self) -> Vec<PySolution> {
        self.last_population.clone()
    }

    /// Number of function evaluations performed in the last run.
    #[getter]
    fn nfe(&self) -> usize {
        self.last_nfe
    }

    fn __repr__(&self) -> String {
        format!(
            "NSGAII(population_size={}, execution_mode={:?}, nfe={})",
            self.population_size, self.execution_mode, self.last_nfe
        )
    }
}

// ---------------------------------------------------------------------------
// Config → manager builder helpers
// ---------------------------------------------------------------------------

/// `uniform_probability` is the per-gene *swap* probability — 1.0 swaps every gene, which
/// merely exchanges the parents. Defaults to the textbook 0.5.
fn build_crossover_manager(cfg: &PyCrossoverConfig) -> CrossoverManager {
    let mut mgr = CrossoverManager::new();

    let real: Box<dyn Crossover + Send> = match cfg.real_crossover.as_str() {
        "de" => Box::new(DifferentialEvolutionCrossover::new(
            Some(cfg.de_probability),
            Some(cfg.de_scaling_factor),
        )),
        "blend" => Box::new(BlendCrossover {
            probability: cfg.sbx_probability,
            alpha: cfg.blend_alpha,
        }),
        "pcx" => Box::new(ParentCentricCrossover {
            nparents: 2,
            noffspring: 2,
            eta: 0.1,
            zeta: 0.1,
        }),
        "undx" => Box::new(UnimodalDistributionCrossover {
            probability: cfg.sbx_probability,
            distribution_index: cfg.sbx_distribution_index,
            nparents: 2,
            zeta: 0.5,
            eta: 0.35,
        }),
        _ => Box::new(SimulatedBinaryCrossover::new(
            Some(cfg.sbx_probability),
            Some(cfg.sbx_distribution_index),
        )),
    };

    let swap = cfg.uniform_probability.unwrap_or(0.5);
    let integer: Box<dyn Crossover + Send> = match cfg.integer_crossover.as_str() {
        "arithmetic" => Box::new(ArithmeticCrossover { probability: swap }),
        _ => Box::new(UniformCrossover { probability: swap }),
    };

    let binary: Box<dyn Crossover + Send> = Box::new(UniformCrossover { probability: swap });

    mgr.set_default_real_crossover(real);
    mgr.set_default_integer_crossover(integer);
    mgr.set_default_binary_crossover(binary);
    mgr
}

/// Build a manager from a Python config. Starts from `MutationManager::new(n)` (the correct
/// NSGA-II defaults) and overrides only what the config explicitly names — `probability = None`
/// keeps the conventional 1/n rate rather than mutating every gene.
fn build_mutation_manager(cfg: &PyMutationConfig, n: usize) -> MutationManager {
    let mut mgr = MutationManager::new(n);
    let rate = cfg.probability.unwrap_or(1.0 / n.max(1) as f64);

    let real: Arc<dyn Mutation> = match cfg.real_mutation.as_str() {
        "gaussian" => Arc::new(GaussianMutation::new(Some(rate), Some(cfg.gaussian_std_dev))),
        "uniform" => Arc::new(UniformMutation { probability: rate }),
        _ => Arc::new(PolynomialMutation::new(
            Some(rate),
            Some(cfg.polynomial_distribution_index),
        )),
    };

    let integer: Arc<dyn Mutation> = match cfg.integer_mutation.as_str() {
        "uniform" => Arc::new(UniformMutation { probability: rate }),
        _ => Arc::new(PolynomialMutation::new(
            Some(rate),
            Some(cfg.polynomial_distribution_index),
        )),
    };

    mgr.set_default_real_mutation(real);
    mgr.set_default_integer_mutation(integer);
    mgr.set_default_binary_mutation(Arc::new(BitFlipMutation { probability: rate }));
    mgr
}
