//! CPU map-reduce evaluation for objectives that scan a dataset.
//!
//! Evaluating one solution at a time streams the whole dataset once per solution, so a large
//! dataset makes evaluation memory-bandwidth bound (at 1 GB, 100 solutions read ~90 GB per
//! generation). [`RowScanEvaluator`] instead walks the rows once per generation: Rayon threads
//! take blocks of rows, and for each row every solution's accumulator is updated while the row is
//! still in cache. It is the CPU counterpart of `GpuEvaluator::with_partials`.
//!
//! Attach it like any device backend and run in `ExecutionMode::GPU` (meaning "use the attached
//! batch evaluator") or `ExecutionMode::Auto`.

use crate::core::BatchEvaluator;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Rows per Rayon task: big enough to amortize task overhead, small enough to balance.
const BLOCK_ROWS: usize = 4096;

/// Map-reduce evaluator over a row-major `rows × cols` f64 dataset.
///
/// - `prepare(solution)` runs once per solution per generation (e.g. normalize weights).
/// - `kernel(prepared, row, acc)` adds one row's contribution to that solution's `k` sums.
///   It runs rows × population times, so it is generic (inlined), not a function pointer.
/// - `finish(sums)` turns a solution's `k` summed values into its objectives.
pub struct RowScanEvaluator<P, K, F> {
    data: Arc<[f64]>,
    cols: usize,
    k: usize,
    prepare: P,
    kernel: K,
    finish: F,
    count: AtomicUsize,
}

impl<P, K, F> RowScanEvaluator<P, K, F>
where
    P: Fn(&[f64]) -> Vec<f64> + Send + Sync,
    K: Fn(&[f64], &[f64], &mut [f64]) + Send + Sync,
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    pub fn new(data: impl Into<Arc<[f64]>>, cols: usize, k: usize, prepare: P, kernel: K, finish: F) -> Self {
        let data = data.into();
        assert!(cols > 0 && data.len() % cols == 0, "data length must be a multiple of cols");
        RowScanEvaluator { data, cols, k, prepare, kernel, finish, count: AtomicUsize::new(0) }
    }

    pub fn rows(&self) -> usize {
        self.data.len() / self.cols
    }
}

impl<P, K, F> BatchEvaluator for RowScanEvaluator<P, K, F>
where
    P: Fn(&[f64]) -> Vec<f64> + Send + Sync,
    K: Fn(&[f64], &[f64], &mut [f64]) + Send + Sync,
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    fn evaluate_batch(&self, solutions: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.count.fetch_add(solutions.len(), Ordering::Relaxed);
        let prepared: Vec<Vec<f64>> = solutions.iter().map(|s| (self.prepare)(s)).collect();
        let (pop, k) = (prepared.len(), self.k);
        let sums = self
            .data
            .par_chunks(BLOCK_ROWS * self.cols)
            .fold(
                || vec![0.0; pop * k],
                |mut acc, block| {
                    for row in block.chunks_exact(self.cols) {
                        for (w, a) in prepared.iter().zip(acc.chunks_exact_mut(k)) {
                            (self.kernel)(w, row, a);
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0; pop * k],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            );
        sums.chunks_exact(k).map(|s| (self.finish)(s)).collect()
    }

    fn device_description(&self) -> String {
        format!("CPU row scan ({} threads)", rayon::current_num_threads())
    }

    fn evaluated_count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_per_solution_scan() {
        // 10_003 rows: not a multiple of BLOCK_ROWS, so the last block is partial.
        let (rows, cols) = (10_003, 5);
        let data: Vec<f64> = (0..rows * cols).map(|i| ((i * 37) % 101) as f64 / 101.0 - 0.5).collect();
        let sols: Vec<Vec<f64>> = (0..7).map(|p| (0..cols).map(|i| ((p + i) % 4) as f64 + 0.5).collect()).collect();

        let normalize = |x: &[f64]| {
            let t: f64 = x.iter().sum();
            x.iter().map(|v| v / t).collect()
        };
        let ev = RowScanEvaluator::new(
            data.clone(),
            cols,
            2,
            normalize,
            |w: &[f64], row: &[f64], acc: &mut [f64]| {
                let r: f64 = w.iter().zip(row).map(|(a, b)| a * b).sum();
                acc[0] += r;
                acc[1] += r * r;
            },
            move |s: &[f64]| vec![s[0] / rows as f64, s[1] / rows as f64],
        );
        let got = ev.evaluate_batch(&sols);

        for (sol, g) in sols.iter().zip(&got) {
            let w = normalize(sol);
            let (mut s, mut s2) = (0.0, 0.0);
            for row in data.chunks_exact(cols) {
                let r: f64 = w.iter().zip(row).map(|(a, b)| a * b).sum();
                s += r;
                s2 += r * r;
            }
            assert!((g[0] - s / rows as f64).abs() < 1e-12 && (g[1] - s2 / rows as f64).abs() < 1e-12);
        }
        assert_eq!(ev.evaluated_count(), sols.len());
    }
}
