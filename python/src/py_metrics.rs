//! Pareto-front quality metrics, exposed so Python callers can score a run without
//! reimplementing them (or pulling in pymoo just for indicators).
//!
//! All of these assume **minimization**. Negate any maximized objective before calling.

use pyo3::prelude::*;
use puggles::metrics;

/// Exact 2-objective hypervolume (dominated area) relative to a reference (nadir) point.
///
/// Args:
///     front: List of [f1, f2] objective vectors (minimization space).
///     reference: Nadir point [r1, r2]; must be worse than every point on the front.
///
/// Returns the dominated area. Points not dominated by the reference are ignored.
#[pyfunction]
pub fn hypervolume_2d(front: Vec<Vec<f64>>, reference: (f64, f64)) -> f64 {
    metrics::hypervolume_2d(&front, [reference.0, reference.1])
}

/// Inverted Generational Distance: mean distance from each reference-set point to the nearest
/// point on `front`. Lower is better; 0 means the front covers the reference set.
#[pyfunction]
pub fn igd(front: Vec<Vec<f64>>, reference_set: Vec<Vec<f64>>) -> f64 {
    metrics::igd(&front, &reference_set)
}

/// Schott spacing: standard deviation of each point's nearest-neighbour distance. Lower means
/// a more uniform spread. Needs at least 2 points, otherwise returns 0.
#[pyfunction]
pub fn spacing(front: Vec<Vec<f64>>) -> f64 {
    metrics::spacing(&front)
}
