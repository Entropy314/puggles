use pyo3::prelude::*;
use rustypus::core::Solution;

/// Owned solution exposed to Python. No lifetime parameters.
#[pyclass(name = "Solution")]
#[derive(Clone, Debug)]
pub struct PySolution {
    #[pyo3(get)]
    pub variables: Vec<f64>,
    #[pyo3(get)]
    pub objectives: Vec<f64>,
    #[pyo3(get)]
    pub constraints: Vec<f64>,
    #[pyo3(get)]
    pub evaluated: bool,
    #[pyo3(get)]
    pub feasible: bool,
    #[pyo3(get)]
    pub constraint_violation: usize,
}

#[pymethods]
impl PySolution {
    fn __repr__(&self) -> String {
        format!(
            "Solution(variables={:?}, objectives={:?}, feasible={})",
            self.variables, self.objectives, self.feasible
        )
    }
}

impl PySolution {
    pub fn from_core_solution(sol: &Solution) -> Self {
        PySolution {
            variables: sol.solution.clone(),
            objectives: sol.objective_fitness_values.clone(),
            constraints: sol.constraint_values.clone(),
            evaluated: sol.evaluated,
            feasible: sol.feasible,
            constraint_violation: sol.constraint_violation,
        }
    }
}
