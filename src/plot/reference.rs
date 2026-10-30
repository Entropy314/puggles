//! True Pareto fronts of standard benchmarks, for [`RunMeta::reference_front`](super::RunMeta::reference_front):
//! the dashboard draws them and plots IGD (distance from the true front to the found one).
//!
//! All are for minimization, in objective space.

use crate::nsga3::das_dennis;

/// ZDT1: `f2 = 1 - sqrt(f1)`, `f1` in `[0, 1]`.
pub fn zdt1(points: usize) -> Vec<Vec<f64>> {
    line(points, |f1| 1.0 - f1.sqrt())
}

/// ZDT2: `f2 = 1 - f1²`, `f1` in `[0, 1]`.
pub fn zdt2(points: usize) -> Vec<Vec<f64>> {
    line(points, |f1| 1.0 - f1 * f1)
}

/// DTLZ1: the simplex `sum(f) = 0.5`, sampled on a Das–Dennis grid.
pub fn dtlz1(objectives: usize, divisions: usize) -> Vec<Vec<f64>> {
    das_dennis(objectives, divisions).into_iter().map(|p| p.into_iter().map(|v| v / 2.0).collect()).collect()
}

/// DTLZ2 (also DTLZ3 and DTLZ4): the unit sphere's positive orthant, `sum(f²) = 1`.
pub fn dtlz2(objectives: usize, divisions: usize) -> Vec<Vec<f64>> {
    das_dennis(objectives, divisions)
        .into_iter()
        .map(|p| {
            let norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            p.into_iter().map(|v| v / norm).collect()
        })
        .collect()
}

/// A front by benchmark name (`zdt1`, `zdt2`, `dtlz1`..`dtlz4`) with roughly `points` points.
pub fn by_name(name: &str, objectives: usize, points: usize) -> Option<Vec<Vec<f64>>> {
    let points = points.max(2);
    match name.to_ascii_lowercase().as_str() {
        "zdt1" if objectives == 2 => Some(zdt1(points)),
        "zdt2" if objectives == 2 => Some(zdt2(points)),
        "dtlz1" => Some(dtlz1(objectives, divisions_for(objectives, points))),
        "dtlz2" | "dtlz3" | "dtlz4" => Some(dtlz2(objectives, divisions_for(objectives, points))),
        _ => None,
    }
}

/// The largest Das–Dennis grid with at most `points` points (at least one division).
fn divisions_for(objectives: usize, points: usize) -> usize {
    let count = |d: usize| -> usize {
        // C(objectives + d - 1, d), computed incrementally to stay exact.
        (1..=d).fold(1usize, |acc, k| acc * (objectives - 1 + k) / k)
    };
    let mut d = 1;
    while count(d + 1) <= points && d < 200 {
        d += 1;
    }
    d
}

fn line(points: usize, f2: impl Fn(f64) -> f64) -> Vec<Vec<f64>> {
    let points = points.max(2);
    (0..points)
        .map(|i| {
            let f1 = i as f64 / (points - 1) as f64;
            vec![f1, f2(f1)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fronts_lie_on_their_surfaces() {
        for p in zdt1(50) {
            assert!((p[1] - (1.0 - p[0].sqrt())).abs() < 1e-12);
        }
        for p in dtlz2(5, 4) {
            assert!((p.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for p in dtlz1(3, 6) {
            assert!((p.iter().sum::<f64>() - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn by_name_sizes_the_grid() {
        let f = by_name("DTLZ2", 5, 200).unwrap();
        assert!(f.len() <= 200 && f.len() >= 70, "{}", f.len()); // 5 objectives: 6 divisions = 210 > 200, 5 = 126
        assert_eq!(by_name("zdt1", 2, 100).unwrap().len(), 100);
        assert!(by_name("zdt1", 3, 100).is_none());
        assert!(by_name("knapsack", 2, 100).is_none());
    }
}
