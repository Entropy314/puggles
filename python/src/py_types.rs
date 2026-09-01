use pyo3::prelude::*;
use puggles::gatypes::{BitBinary, Integer, Real, SolutionDataTypes};

#[pyclass(name = "Real")]
#[derive(Clone, Debug)]
pub struct PyReal {
    #[pyo3(get)]
    pub lower_bound: f64,
    #[pyo3(get)]
    pub upper_bound: f64,
}

#[pymethods]
impl PyReal {
    #[new]
    fn new(lower_bound: f64, upper_bound: f64) -> Self {
        PyReal {
            lower_bound,
            upper_bound,
        }
    }

    fn __repr__(&self) -> String {
        format!("Real({}, {})", self.lower_bound, self.upper_bound)
    }
}

#[pyclass(name = "Integer")]
#[derive(Clone, Debug)]
pub struct PyInteger {
    #[pyo3(get)]
    pub lower_bound: i64,
    #[pyo3(get)]
    pub upper_bound: i64,
}

#[pymethods]
impl PyInteger {
    #[new]
    fn new(lower_bound: i64, upper_bound: i64) -> Self {
        PyInteger {
            lower_bound,
            upper_bound,
        }
    }

    fn __repr__(&self) -> String {
        format!("Integer({}, {})", self.lower_bound, self.upper_bound)
    }
}

#[pyclass(name = "BitBinary")]
#[derive(Clone, Debug)]
pub struct PyBitBinary;

#[pymethods]
impl PyBitBinary {
    #[new]
    fn new() -> Self {
        PyBitBinary
    }

    fn __repr__(&self) -> String {
        "BitBinary()".to_string()
    }
}

/// Parse a Python list of Real/Integer/BitBinary objects into Rust SolutionDataTypes
pub fn parse_data_types(py: Python<'_>, types: Vec<PyObject>) -> PyResult<Vec<SolutionDataTypes>> {
    types
        .into_iter()
        .map(|obj| {
            if let Ok(r) = obj.extract::<PyReal>(py) {
                Ok(SolutionDataTypes::Real(Real::new(
                    Some(r.lower_bound),
                    Some(r.upper_bound),
                )))
            } else if let Ok(i) = obj.extract::<PyInteger>(py) {
                Ok(SolutionDataTypes::Integer(Integer::new(
                    Some(i.lower_bound),
                    Some(i.upper_bound),
                )))
            } else if obj.extract::<PyBitBinary>(py).is_ok() {
                Ok(SolutionDataTypes::BitBinary(BitBinary::new()))
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected Real, Integer, or BitBinary",
                ))
            }
        })
        .collect()
}
