/// Python bindings for GPU-accelerated problem evaluation via wgpu compute shaders.
///
/// Usage:
/// ```python
/// import rustypus
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
/// problem = rustypus.GpuProblem(
///     solution_length=10,
///     number_of_objectives=1,
///     solution_data_types=[rustypus.Real(-5.0, 5.0)] * 10,
///     shader_wgsl=shader,
/// )
/// ga = rustypus.NSGAII(problem, population_size=100, execution_mode="gpu")
/// ga.run(5000)
/// ```

use pyo3::prelude::*;
use rustypus::core::{EvalFn, Problem};
use rustypus::gpu_evaluator::GpuEvaluator;
use std::cell::RefCell;
use std::sync::Arc;

use crate::py_problem::ProblemStore;
use crate::py_types::parse_data_types;

// ---------------------------------------------------------------------------
// Thread-local GPU evaluator
// ---------------------------------------------------------------------------

thread_local! {
    static TLS_GPU_EVALUATOR: RefCell<Option<Arc<GpuEvaluator>>> = RefCell::new(None);
}

fn gpu_batch_trampoline(inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    TLS_GPU_EVALUATOR.with(|cell| {
        let borrow = cell.borrow();
        borrow.as_ref()
            .expect("No active GPU evaluator — was activate_gpu_evaluator() called before run()?")
            .evaluate_batch(inputs)
    })
}

pub fn activate_gpu_evaluator(evaluator: Arc<GpuEvaluator>) {
    TLS_GPU_EVALUATOR.with(|c| *c.borrow_mut() = Some(evaluator));
}
pub fn deactivate_gpu_evaluator() {
    TLS_GPU_EVALUATOR.with(|c| *c.borrow_mut() = None);
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

        // Use EvalFn::Batch with the GPU batch trampoline.
        let problem = Arc::new(Problem {
            solution_length,
            number_of_objectives,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: direction.or_else(|| Some(vec![-1; number_of_objectives])),
            solution_data_types: data_types,
            eval_fn: EvalFn::Batch(gpu_batch_trampoline),
        });

        Ok(PyGpuProblem {
            store: ProblemStore {
                problem,
                callable: None,
                batch_callable: None,
                gpu_evaluator: Some(evaluator),
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
