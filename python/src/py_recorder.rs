//! Recording a run into an optional add-on's dashboard (`pip install "puggles[dashboard]"`).
//!
//! The visual code isn't in this wheel. An add-on's `Dashboard` offers a native recorder via
//! `__puggles_recorder__()` (a capsule; see `puggles::recorder_abi`), which `run()` feeds
//! straight from the Rust solutions with no GIL. Anything else passed as `callback=` is called
//! as a plain Python callable.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use puggles::genetic_algorithms_v2::NSGAII;
use puggles::recorder_abi::{Recorder, RecorderHandle, Snapshot, CAPSULE_NAME};
use std::time::Duration;

/// The native recorder `callback` offers, if any. One whose ABI doesn't match this build (a
/// mismatched add-on version) is reported with a warning and `callback` is then used as a plain
/// callable.
pub(crate) fn native_recorder(py: Python<'_>, callback: Option<&PyObject>) -> PyResult<Option<RecorderHandle>> {
    let Some(cb) = callback else { return Ok(None) };
    let cb = cb.bind(py);
    if !cb.hasattr("__puggles_recorder__")? {
        return Ok(None);
    }
    let offered = cb.call_method0("__puggles_recorder__")?;
    let refused = |why: String| -> PyResult<Option<RecorderHandle>> {
        let warning = py.get_type::<pyo3::exceptions::PyRuntimeWarning>();
        PyErr::warn(py, &warning, &std::ffi::CString::new(format!("{why}; recording through the Python callback instead")).unwrap_or_default(), 1)?;
        Ok(None)
    };
    let Ok(capsule) = offered.downcast::<PyCapsule>() else {
        return refused("__puggles_recorder__() didn't return a capsule".into());
    };
    if capsule.name()? != Some(CAPSULE_NAME) {
        return refused(format!("unknown recorder capsule {:?}", capsule.name()?));
    }
    // SAFETY: the capsule name promises a `Recorder` from an extension module, which Python
    // never unloads; `capsule` (and so the recorder) is alive for this call.
    match unsafe { RecorderHandle::from_raw(capsule.pointer() as *const Recorder) } {
        Ok(handle) => Ok(Some(handle)),
        Err(why) => refused(why),
    }
}

/// A GpuProblem's shader evaluator (always `None` without the `gpu` feature).
#[cfg(feature = "gpu")]
pub(crate) type GpuEval = Option<std::sync::Arc<puggles::gpu_evaluator::GpuEvaluator>>;
#[cfg(not(feature = "gpu"))]
pub(crate) type GpuEval = Option<std::convert::Infallible>;

/// Record a generation without waiting out a pause. A GpuProblem evaluates through a batch
/// objective, which the optimizer itself reports as CPU work; report its shader's device and
/// evaluation count instead.
pub(crate) fn record_without_hold(rec: &RecorderHandle, ga: &NSGAII, gpu: &GpuEval) {
    #[allow(unused_mut)]
    let mut snap = Snapshot::of_nsgaii(ga);
    #[cfg(feature = "gpu")]
    if let Some(gpu) = gpu {
        use puggles::genetic_algorithms_v2::{ExecutionInfo, ExecutionMode};
        snap.execution = Some(ExecutionInfo {
            requested: ExecutionMode::GPU,
            effective: ExecutionMode::GPU,
            cpu_threads: None,
            device: Some(gpu.device_description()),
            note: Some("GpuProblem: the shader evaluates each generation as one batch".into()),
        });
        snap.gpu_evaluations = Some(gpu.evaluated_count());
    }
    let _ = gpu;
    rec.record(&snap);
}

/// Record with the GIL released (pure-Rust objectives): a pause just blocks this thread.
pub(crate) fn record(rec: &RecorderHandle, ga: &NSGAII, gpu: &GpuEval) -> bool {
    record_without_hold(rec, ga, gpu);
    rec.hold()
}

/// Wait out a pause while holding the GIL: release it in short slices so other Python threads
/// run, and let Ctrl+C through (it stops the run and is re-raised).
pub(crate) fn hold_with_gil(py: Python<'_>, rec: &RecorderHandle) -> PyResult<bool> {
    loop {
        if let Some(keep_going) = py.allow_threads(|| rec.hold_for(Duration::from_millis(100))) {
            return Ok(keep_going);
        }
        if let Err(e) = py.check_signals() {
            rec.request_stop();
            return Err(e);
        }
    }
}

/// `run_with` for a Python-objective run (GIL held): record each generation, wait out pauses
/// with the GIL released, and surface a Ctrl+C raised while paused.
pub(crate) fn run_watched_with_gil<G>(
    py: Python<'_>,
    rec: &RecorderHandle,
    run: impl FnOnce(&mut dyn FnMut(&mut G) -> bool),
    record: impl Fn(&mut G),
) -> PyResult<()> {
    let mut error: Option<PyErr> = None;
    run(&mut |ga: &mut G| {
        record(ga);
        match hold_with_gil(py, rec) {
            Ok(keep_going) => keep_going,
            Err(e) => {
                error = Some(e);
                false
            }
        }
    });
    rec.finish();
    error.map_or(Ok(()), Err)
}
