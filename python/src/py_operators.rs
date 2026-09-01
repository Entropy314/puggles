use pyo3::prelude::*;

/// Configuration for crossover operators.
///
/// Selects which crossover strategy to use for each variable type.
/// Available real crossovers: "sbx" (default), "de", "blend", "pcx", "undx"
/// Available integer/binary crossovers: "uniform" (default), "arithmetic"
/// `uniform_probability` is the per-gene swap probability (default 0.5).
#[pyclass(name = "CrossoverConfig")]
#[derive(Clone, Debug)]
pub struct PyCrossoverConfig {
    #[pyo3(get, set)]
    pub real_crossover: String,
    #[pyo3(get, set)]
    pub integer_crossover: String,
    #[pyo3(get, set)]
    pub binary_crossover: String,
    #[pyo3(get, set)]
    pub sbx_probability: f64,
    #[pyo3(get, set)]
    pub sbx_distribution_index: f64,
    #[pyo3(get, set)]
    pub de_probability: f64,
    #[pyo3(get, set)]
    pub de_scaling_factor: f64,
    #[pyo3(get, set)]
    pub blend_alpha: f64,
    #[pyo3(get, set)]
    pub uniform_probability: Option<f64>,
}

#[pymethods]
impl PyCrossoverConfig {
    #[new]
    #[pyo3(signature = (
        real_crossover = "sbx",
        integer_crossover = "uniform",
        binary_crossover = "uniform",
        sbx_probability = 1.0,
        sbx_distribution_index = 20.0,
        de_probability = 0.9,
        de_scaling_factor = 0.8,
        blend_alpha = 0.5,
        uniform_probability = None,
    ))]
    fn new(
        real_crossover: &str,
        integer_crossover: &str,
        binary_crossover: &str,
        sbx_probability: f64,
        sbx_distribution_index: f64,
        de_probability: f64,
        de_scaling_factor: f64,
        blend_alpha: f64,
        uniform_probability: Option<f64>,
    ) -> Self {
        PyCrossoverConfig {
            real_crossover: real_crossover.to_string(),
            integer_crossover: integer_crossover.to_string(),
            binary_crossover: binary_crossover.to_string(),
            sbx_probability,
            sbx_distribution_index,
            de_probability,
            de_scaling_factor,
            blend_alpha,
            uniform_probability,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CrossoverConfig(real='{}', integer='{}', binary='{}')",
            self.real_crossover, self.integer_crossover, self.binary_crossover
        )
    }
}

/// Configuration for mutation operators.
///
/// Selects which mutation strategy to use for each variable type.
/// Available: "bitflip" (binary default), "uniform" (int/real default), "polynomial", "gaussian"
#[pyclass(name = "MutationConfig")]
#[derive(Clone, Debug)]
pub struct PyMutationConfig {
    #[pyo3(get, set)]
    pub real_mutation: String,
    #[pyo3(get, set)]
    pub integer_mutation: String,
    #[pyo3(get, set)]
    pub binary_mutation: String,
    /// Per-gene mutation rate. `None` = the conventional 1/n (n = number of decision variables).
    #[pyo3(get, set)]
    pub probability: Option<f64>,
    #[pyo3(get, set)]
    pub polynomial_distribution_index: f64,
    #[pyo3(get, set)]
    pub gaussian_std_dev: f64,
}

#[pymethods]
impl PyMutationConfig {
    #[new]
    #[pyo3(signature = (
        real_mutation = "polynomial",
        integer_mutation = "polynomial",
        binary_mutation = "bitflip",
        probability = None,
        polynomial_distribution_index = 20.0,
        gaussian_std_dev = 0.1,
    ))]
    fn new(
        real_mutation: &str,
        integer_mutation: &str,
        binary_mutation: &str,
        probability: Option<f64>,
        polynomial_distribution_index: f64,
        gaussian_std_dev: f64,
    ) -> Self {
        PyMutationConfig {
            real_mutation: real_mutation.to_string(),
            integer_mutation: integer_mutation.to_string(),
            binary_mutation: binary_mutation.to_string(),
            probability,
            polynomial_distribution_index,
            gaussian_std_dev,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MutationConfig(real='{}', integer='{}', binary='{}')",
            self.real_mutation, self.integer_mutation, self.binary_mutation
        )
    }
}
