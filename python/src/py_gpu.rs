/// Python bindings for GPU-accelerated problem evaluation via wgpu compute shaders.
///
/// Usage:
/// ```python
/// import puggles
///
/// shader = """
/// @group(0) @binding(0) var<storage, read> solutions: array<f32>;
/// @group(0) @binding(1) var<storage, read_write> objectives: array<f32>;
/// struct Params { solution_length: u32, num_objectives: u32, pop_size: u32 }
/// @group(0) @binding(2) var<uniform> params: Params;
///
/// @compute @workgroup_size(64)
/// fn main(@builtin(global_invocation_id) id: vec3<u32>) {
///     let sol_idx = id.x;
///     if sol_idx >= params.pop_size { return; }
///     let in_off  = sol_idx * params.solution_length;
///     let out_off = sol_idx * params.num_objectives;
///     var sum: f32 = 0.0;
///     for (var i: u32 = 0u; i < params.solution_length; i++) {
///         let x = solutions[in_off + i];
///         sum += x * x;
///     }
///     objectives[out_off] = sum;
/// }
/// """
///
/// problem = puggles.GpuProblem(
///     solution_length=10,
///     number_of_objectives=1,
///     solution_data_types=[puggles.Real(-5.0, 5.0)] * 10,
///     shader_wgsl=shader,
/// )
/// ga = puggles.NSGAII(problem, population_size=100, execution_mode="gpu")
/// ga.run(5000)
/// ```

use pyo3::prelude::*;
use puggles::core::{EvalFn, Problem};
use puggles::gpu_evaluator::GpuEvaluator;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::py_problem::ProblemStore;
use crate::py_types::parse_data_types;

static GPU_ID_COUNTER: AtomicU64 = AtomicU64::new(1_000_000);

/// Shared GPU evaluator stored globally so the batch trampoline can access it.
fn gpu_evaluator_store() -> &'static Mutex<Option<Arc<GpuEvaluator>>> {
    static INSTANCE: OnceLock<Mutex<Option<Arc<GpuEvaluator>>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(None))
}

/// Batch trampoline (Rust function pointer) for GPU evaluation.
fn gpu_batch_trampoline(inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let guard = gpu_evaluator_store().lock().unwrap();
    guard
        .as_ref()
        .expect("GpuEvaluator not initialised — create GpuProblem before calling run()")
        .evaluate_batch(inputs)
}

/// An optimization Problem whose objective function runs on the GPU via a WGSL compute shader.
#[pyclass(name = "GpuProblem")]
pub struct PyGpuProblem {
    pub(crate) store: ProblemStore,
}

#[pymethods]
impl PyGpuProblem {
    /// Create a GPU-accelerated optimization Problem.
    ///
    /// Args:
    ///     solution_length: Number of decision variables.
    ///     number_of_objectives: Number of objective functions.
    ///     solution_data_types: List of Real(...), Integer(...), or BitBinary() per variable.
    ///     shader_wgsl: WGSL compute shader implementing the objective function.
    ///                  Bindings: @binding(0) solutions (read), @binding(1) objectives (read_write),
    ///                            @binding(2) Params uniform { solution_length, num_objectives, pop_size }.
    ///     direction: Optional list of -1 (minimize) or 1 (maximize) per objective.
    #[new]
    #[pyo3(signature = (
        solution_length,
        number_of_objectives,
        solution_data_types,
        shader_wgsl,
        direction = None,
    ))]
    fn new(
        py: Python<'_>,
        solution_length: usize,
        number_of_objectives: usize,
        solution_data_types: Vec<PyObject>,
        shader_wgsl: &str,
        direction: Option<Vec<i8>>,
    ) -> PyResult<Self> {
        let data_types = parse_data_types(py, solution_data_types)?;

        // Initialise the wgpu evaluator (blocks on async GPU initialisation)
        let evaluator = Arc::new(GpuEvaluator::new_blocking(
            shader_wgsl,
            solution_length,
            number_of_objectives,
        ));
        *gpu_evaluator_store().lock().unwrap() = Some(Arc::clone(&evaluator));

        // Placeholder single-eval function — never called; GPU uses the batch trampoline.
        fn placeholder(_: &Vec<f64>) -> Vec<f64> { Vec::new() }

        let mut problem = Problem::new(
            solution_length,
            number_of_objectives,
            None,
            None,
            direction,
            data_types,
            placeholder,
        );
        problem.eval_fn = EvalFn::Batch(gpu_batch_trampoline);

        let problem_id = GPU_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(PyGpuProblem {
            store: ProblemStore {
                problem: Arc::new(problem),
                problem_id,
                uses_python_callable: false,
                // GPU trampoline is Rust-native; GIL release is safe in PyNSGAII::run
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
            "GpuProblem(solution_length={}, number_of_objectives={})",
            self.store.problem.solution_length, self.store.problem.number_of_objectives
        )
    }
}
