//! Pareto-front quality metrics.
//!
//! All functions take objective vectors (`&[Vec<f64>]`) and **assume minimization** —
//! negate any maximized objective before calling. Extract objective vectors from a
//! population/archive with [`objective_vectors`].

use crate::core::Solution;

/// Collect the objective vectors of a slice of solutions.
pub fn objective_vectors(solutions: &[Solution]) -> Vec<Vec<f64>> {
    solutions.iter().map(|s| s.objective_fitness_values.to_vec()).collect()
}

/// Exact 2-objective hypervolume (dominated area) relative to a reference (nadir) point,
/// for a **minimization** front. `reference` must be worse (larger) than every point on
/// each axis; points not dominated by the reference are ignored.
///
/// ponytail: 2 objectives only. Exact hypervolume for m>2 (WFG/HSO) is a separate
/// algorithm — add it when many-objective support lands.
pub fn hypervolume_2d(front: &[Vec<f64>], reference: [f64; 2]) -> f64 {
    let mut pts: Vec<[f64; 2]> = front
        .iter()
        .filter(|p| p.len() >= 2 && p[0] <= reference[0] && p[1] <= reference[1])
        .map(|p| [p[0], p[1]])
        .collect();
    if pts.is_empty() {
        return 0.0;
    }
    // Sort by f0 ascending; on a non-dominated set f1 then decreases.
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    let mut hv = 0.0;
    let mut prev_f1 = reference[1];
    for p in pts {
        if p[1] < prev_f1 {
            hv += (reference[0] - p[0]) * (prev_f1 - p[1]);
            prev_f1 = p[1];
        }
    }
    hv
}

/// Inverted Generational Distance: mean distance from each reference-set point to the
/// nearest point in `front`. Lower is better; 0 means the front covers the reference set.
pub fn igd(front: &[Vec<f64>], reference_set: &[Vec<f64>]) -> f64 {
    if front.is_empty() || reference_set.is_empty() {
        return f64::INFINITY;
    }
    let total: f64 = reference_set
        .iter()
        .map(|r| {
            front
                .iter()
                .map(|p| euclidean(p, r))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    total / reference_set.len() as f64
}

/// Spacing metric (Schott): standard deviation of each point's distance to its nearest
/// neighbour in the front. Lower means more uniform spread. Needs at least 2 points.
pub fn spacing(front: &[Vec<f64>]) -> f64 {
    let n = front.len();
    if n < 2 {
        return 0.0;
    }
    let d: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| euclidean(&front[i], &front[j]))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let mean = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|di| (di - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt()
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypervolume_2d() {
        // Two non-dominated points, reference (5,5): union area = 7 (worked by hand).
        let front = vec![vec![1.0, 4.0], vec![2.0, 3.0]];
        assert!((hypervolume_2d(&front, [5.0, 5.0]) - 7.0).abs() < 1e-9);
        // Single point rectangle.
        assert!((hypervolume_2d(&[vec![1.0, 1.0]], [3.0, 3.0]) - 4.0).abs() < 1e-9);
        // Empty / all-dominated-by-reference front → 0.
        assert_eq!(hypervolume_2d(&[], [5.0, 5.0]), 0.0);
        assert_eq!(hypervolume_2d(&[vec![9.0, 9.0]], [5.0, 5.0]), 0.0);
    }

    #[test]
    fn test_igd() {
        // Front sitting exactly on the reference set → IGD 0.
        let refset = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        assert!(igd(&refset, &refset) < 1e-12);
        // One reference point at distance 1 from the nearest front point.
        let front = vec![vec![0.0, 0.0]];
        let refs = vec![vec![0.0, 1.0]];
        assert!((igd(&front, &refs) - 1.0).abs() < 1e-9);
        assert_eq!(igd(&[], &refs), f64::INFINITY);
    }

    #[test]
    fn test_spacing() {
        // Evenly spaced collinear points → zero spacing variance.
        let front = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        assert!(spacing(&front) < 1e-9);
        assert_eq!(spacing(&[vec![0.0, 0.0]]), 0.0);
    }
}
