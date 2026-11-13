//! `CudaEvaluator`: a puggles `BatchEvaluator` that runs the Rust-written
//! `kernels::portfolio_partials` CUDA kernel, in f64. Attach it with
//! `NSGAII::new(.., ExecutionMode::GPU).with_batch_evaluator(CudaEvaluator::new(..)?)`.
//!
//! Written against Rust-CUDA @ 3a82c35 but NOT compiled or run by the author (macOS, no NVIDIA
//! GPU). Expect small API fixes on first build.

use cust::context::CurrentContext;
use cust::prelude::*;
use puggles::core::{reduce_partials, BatchEvaluator};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const TILE: usize = 64; // must match kernels::TILE
const THREADS: u32 = 128; // threads per block; the kernel needs ≥ pop threads

// Fields drop in declaration order, so `ctx` must come last: destroying the CUDA context before
// the module, stream and device buffers that live in it crashes the process (SIGSEGV) at exit.
struct Device {
    module: Module,
    stream: Stream,
    data: DeviceBuffer<f64>,
    ctx: Context,
}

pub struct CudaEvaluator {
    // ponytail: one Mutex around the CUDA objects; the GA calls evaluate_batch from one thread per
    // generation, so it never contends. Per-thread streams if concurrent evaluation is ever needed.
    dev: Mutex<Device>,
    rows: usize,
    n: usize,
    blocks: u32,
    finish: fn(&[f64]) -> Vec<f64>,
    name: String,
    count: AtomicUsize,
}

impl CudaEvaluator {
    /// Uploads the row-major `rows × n` matrix `data` once. `finish` turns one solution's summed
    /// (Σr, Σr²) into its objectives, as in puggles' wgpu partials mode.
    pub fn new(data: &[f64], n: usize, finish: fn(&[f64]) -> Vec<f64>) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = cust::quick_init()?;
        let name = cust::device::Device::get_device(0)?.name()?;
        let rows = data.len() / n;
        Ok(CudaEvaluator {
            dev: Mutex::new(Device {
                module: Module::from_ptx(PTX, &[])?,
                stream: Stream::new(StreamFlags::NON_BLOCKING, None)?,
                data: DeviceBuffer::from_slice(data)?,
                ctx,
            }),
            rows,
            n,
            // ~1k rows per block, capped so huge datasets still give a few hundred blocks.
            blocks: rows.div_ceil(1024).clamp(1, 256) as u32,
            finish,
            name,
            count: AtomicUsize::new(0),
        })
    }
}

impl BatchEvaluator for CudaEvaluator {
    fn evaluate_batch(&self, solutions: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let pop = solutions.len();
        assert!(pop <= THREADS as usize, "population {pop} exceeds the kernel's {THREADS} threads per block");
        self.count.fetch_add(pop, Ordering::Relaxed);
        let dev = self.dev.lock().unwrap();
        // CUDA contexts are per-thread; the caller's thread may not be the one that created it.
        CurrentContext::set_current(&dev.ctx).expect("set CUDA context");

        let flat: Vec<f64> = solutions.iter().flatten().copied().collect();
        let sol = DeviceBuffer::from_slice(&flat).expect("upload solutions");
        let len = self.blocks as usize * pop * 2;
        let out = DeviceBuffer::<f64>::zeroed(len).expect("allocate partials");
        let func = dev.module.get_function("portfolio_partials").expect("kernel in PTX");
        let shared_bytes = (TILE * self.n * std::mem::size_of::<f64>()) as u32;
        let stream = &dev.stream;
        unsafe {
            launch!(func<<<self.blocks, THREADS, shared_bytes, stream>>>(
                sol.as_device_ptr(),
                sol.len(),
                dev.data.as_device_ptr(),
                dev.data.len(),
                self.rows,
                pop,
                self.n,
                out.as_device_ptr(),
            ))
            .expect("launch portfolio_partials");
        }
        stream.synchronize().expect("synchronize");
        let mut raw = vec![0.0f64; len];
        out.copy_to(&mut raw).expect("download partials");
        reduce_partials(&raw, self.blocks as usize, pop, 2, self.finish)
    }

    fn device_description(&self) -> String {
        format!("{} (CUDA, Rust-CUDA kernels)", self.name)
    }

    fn evaluated_count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}
