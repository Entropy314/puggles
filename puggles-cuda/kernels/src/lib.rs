use cuda_std::prelude::*;

/// Rows per shared-memory tile; the launch must pass `TILE * n * 8` bytes of dynamic shared memory.
pub const TILE: usize = 64;

/// Portfolio partial sums, the same map-reduce contract as puggles' wgpu partials mode, in f64.
/// Block `b` owns a slice of the `rows × n` returns matrix and thread `p` owns solution `p`
/// (launch with ≥ pop threads per block). Each tile of rows is staged in shared memory and read by
/// every solution, so a generation reads the data once. Writes Σr and Σr² to
/// `out[(b * pop + p) * 2 ..]`, with r = Σ_i w_i·R_ti and w = x / Σx.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn portfolio_partials(sol: &[f64], data: &[f64], rows: usize, pop: usize, n: usize, out: *mut f64) {
    let blocks = thread::grid_dim_x() as usize;
    let b = thread::block_idx_x() as usize;
    let tid = thread::thread_idx_x() as usize;
    let threads = thread::block_dim_x() as usize;
    let per = rows.div_ceil(blocks);
    let lo = (b * per).min(rows);
    let hi = (lo + per).min(rows);
    let tile = cuda_std::shared::dynamic_shared_mem::<f64>();

    let live = tid < pop;
    let w = &sol[tid.min(pop - 1) * n..][..n];
    let total: f64 = w.iter().sum();
    let (mut s, mut s2) = (0.0f64, 0.0f64);
    let mut t0 = lo;
    // Every thread runs the same loop trip counts (they depend only on lo/hi), so the
    // barriers below are reached uniformly.
    while t0 < hi {
        let nrows = TILE.min(hi - t0);
        let mut j = tid;
        while j < nrows * n {
            unsafe { *tile.add(j) = data[t0 * n + j] };
            j += threads;
        }
        thread::sync_threads();
        if live {
            for r in 0..nrows {
                let mut x = 0.0f64;
                for i in 0..n {
                    x += w[i] * unsafe { *tile.add(r * n + i) };
                }
                x = if total > 1e-12 { x / total } else { 0.0 };
                s += x;
                s2 += x * x;
            }
        }
        thread::sync_threads();
        t0 += TILE;
    }
    if live {
        unsafe {
            *out.add((b * pop + tid) * 2) = s;
            *out.add((b * pop + tid) * 2 + 1) = s2;
        }
    }
}
