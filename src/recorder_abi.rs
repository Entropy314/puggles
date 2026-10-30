//! The C ABI that lets an optimizer in one compiled library record into a dashboard compiled
//! into another (feature `recorder-abi`). Internal: it exists so the `puggles` Python wheel can
//! stay free of the visual code while the optional `puggles-dashboard` wheel still records
//! natively, with no GIL. Not a stable API; both sides must come from the same puggles release
//! (they check [`ABI_VERSION`] and the struct sizes, and refuse to connect otherwise).
//!
//! The recorder side hands out a [`Recorder`] (a vtable plus an opaque context), in Python
//! inside a capsule named [`CAPSULE_NAME`]. The optimizer side wraps it in a
//! [`RecorderHandle`] and sends each generation as a flat, `#[repr(C)]` [`Generation`].
//! With feature `plot` too, [`recorder_for`] builds a `Recorder` that feeds a
//! [`Dashboard`](crate::plot::Dashboard).

use crate::core::Solution;
use crate::genetic_algorithms_v2::{ExecutionInfo, ExecutionMode, GenerationTimes, NSGAII};
use crate::nsga3::NSGAIII;
use std::ffi::{c_void, CStr};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Bumped whenever [`Generation`] or [`RecorderVTable`] change.
pub const ABI_VERSION: u32 = 1;

/// The name of the Python capsule that carries a [`Recorder`].
pub const CAPSULE_NAME: &CStr = match CStr::from_bytes_with_nul(b"puggles.recorder.v1\0") {
    Ok(name) => name,
    Err(_) => panic!("capsule name must be NUL-terminated"),
};

/// A borrowed UTF-8 string; a null `ptr` means "none".
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RawStr {
    pub ptr: *const u8,
    pub len: usize,
}

impl RawStr {
    pub const NONE: RawStr = RawStr { ptr: std::ptr::null(), len: 0 };

    pub fn new(s: &str) -> Self {
        RawStr { ptr: s.as_ptr(), len: s.len() }
    }

    pub fn opt(s: Option<&str>) -> Self {
        s.map_or(RawStr::NONE, RawStr::new)
    }

    /// # Safety
    /// `ptr` (when not null) must point to `len` bytes of valid UTF-8 that outlive `'a`.
    pub unsafe fn get<'a>(self) -> Option<&'a str> {
        if self.ptr.is_null() {
            return None;
        }
        Some(std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.ptr, self.len)))
    }
}

/// One generation, as flat row-major arrays. Every pointer is valid only during the call it is
/// passed to.
#[repr(C)]
#[derive(Debug)]
pub struct Generation {
    /// `size_of::<Generation>()` on the sending side.
    pub struct_size: usize,
    pub nfe: usize,
    pub n_obj: usize,
    pub n_var: usize,
    pub archive_len: usize,
    /// `archive_len × n_obj`
    pub archive_objectives: *const f64,
    /// `archive_len × n_var`
    pub archive_variables: *const f64,
    /// `archive_len` flags, 0 or 1
    pub archive_feasible: *const u8,
    /// Population members sent (an even sample when `population_total` is larger).
    pub population_len: usize,
    pub population_total: usize,
    /// `population_len × n_obj`
    pub population_objectives: *const f64,
    pub population_feasible: *const u8,
    /// Constraints each member violates; null when the problem has no constraints.
    pub population_violations: *const usize,
    pub algorithm: RawStr,
    /// 1 when the `execution_*` fields are set.
    pub has_execution: u8,
    /// See [`mode_code`].
    pub execution_requested: u8,
    pub execution_effective: u8,
    /// -1 = none
    pub execution_cpu_threads: i64,
    pub execution_device: RawStr,
    pub execution_note: RawStr,
    /// -1 = none
    pub gpu_evaluations: i64,
    /// 1 when `times` is set: variation, evaluation, selection, archive (seconds).
    pub has_times: u8,
    pub times: [f64; 4],
    /// Seconds the sender spent packing this generation.
    pub pack_seconds: f64,
}

/// What a recorder can do. Every function must be thread-safe: the optimizer calls them from
/// whichever thread runs it, with no GIL held.
#[repr(C)]
pub struct RecorderVTable {
    pub abi_version: u32,
    /// `size_of::<RecorderVTable>()` on the recorder's side.
    pub struct_size: usize,
    /// Another reference to the context (released with `drop_ctx`).
    pub clone_ctx: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    pub drop_ctx: unsafe extern "C" fn(*mut c_void),
    /// Population members wanted per generation (0 = all).
    pub max_points: unsafe extern "C" fn(*mut c_void) -> usize,
    /// Record a generation without waiting out a pause.
    pub record: unsafe extern "C" fn(*mut c_void, *const Generation),
    /// Wait at most `millis` while paused: 1 = carry on, 0 = stopped, -1 = still paused.
    pub hold_for: unsafe extern "C" fn(*mut c_void, u64) -> i32,
    pub finish: unsafe extern "C" fn(*mut c_void),
    pub request_stop: unsafe extern "C" fn(*mut c_void),
    /// Take a pending steering request (marking the run steerable): 0 = none, 1 = focus
    /// (`n_obj` aspiration values written to `aspiration`, NaN = no preference, and the spread
    /// to `spread`), 2 = clear. The recorder marks it on the chart at `nfe`.
    pub take_focus: unsafe extern "C" fn(*mut c_void, nfe: usize, n_obj: usize, aspiration: *mut f64, spread: *mut f64) -> i32,
}

/// A recorder: its vtable and an owned reference to its context.
#[repr(C)]
#[derive(Debug)]
pub struct Recorder {
    pub vtable: *const RecorderVTable,
    pub ctx: *mut c_void,
}

/// The code an [`ExecutionMode`] travels as.
pub fn mode_code(mode: ExecutionMode) -> u8 {
    match mode {
        ExecutionMode::Sequential => 0,
        ExecutionMode::MultiThreaded => 1,
        ExecutionMode::GPU => 2,
    }
}

fn mode_from_code(code: u8) -> ExecutionMode {
    match code {
        0 => ExecutionMode::Sequential,
        2 => ExecutionMode::GPU,
        _ => ExecutionMode::MultiThreaded,
    }
}

/// Which of `n` members an even sample of at most `max` keeps (`None` = all).
fn sample_indices(n: usize, max: usize) -> Option<Vec<usize>> {
    (max != 0 && n > max).then(|| (0..max).map(|i| i * n / max).collect())
}

// -- the optimizer's side ---------------------------------------------------------------------

/// What the optimizer reports about one generation.
pub struct Snapshot<'a> {
    pub archive: &'a [Solution],
    pub population: &'a [Solution],
    pub nfe: usize,
    pub algorithm: &'static str,
    pub execution: Option<ExecutionInfo>,
    pub gpu_evaluations: Option<usize>,
    pub times: Option<GenerationTimes>,
}

impl<'a> Snapshot<'a> {
    pub fn of_nsgaii(ga: &'a NSGAII) -> Self {
        Snapshot {
            archive: ga.get_archive(),
            population: &ga.population,
            nfe: ga.get_nfe(),
            algorithm: "NSGAII",
            execution: Some(ga.execution_info()),
            gpu_evaluations: ga.gpu_evaluations(),
            times: Some(ga.last_generation_times()),
        }
    }

    pub fn of_nsga3(ga: &'a NSGAIII) -> Self {
        Snapshot {
            archive: ga.get_archive(),
            population: &ga.population,
            nfe: ga.get_nfe(),
            algorithm: "NSGAIII",
            execution: Some(ga.execution_info()),
            gpu_evaluations: ga.gpu_evaluations(),
            times: Some(ga.last_generation_times()),
        }
    }
}

/// Packing buffers, reused across generations (one allocation per run, not per generation).
#[derive(Default)]
struct Buffers {
    archive_objectives: Vec<f64>,
    archive_variables: Vec<f64>,
    archive_feasible: Vec<u8>,
    population_objectives: Vec<f64>,
    population_feasible: Vec<u8>,
    population_violations: Vec<usize>,
}

/// The optimizer's connection to a [`Recorder`]; releases its reference when dropped.
pub struct RecorderHandle {
    vtable: &'static RecorderVTable,
    ctx: *mut c_void,
    buffers: Mutex<Buffers>,
}

impl std::fmt::Debug for RecorderHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecorderHandle").field("ctx", &self.ctx).finish_non_exhaustive()
    }
}

// SAFETY: the vtable contract requires every function to be thread-safe, and the context is
// only ever used through them.
unsafe impl Send for RecorderHandle {}
unsafe impl Sync for RecorderHandle {}

impl RecorderHandle {
    /// Connect to `raw`, taking a reference of our own to its context.
    ///
    /// # Safety
    /// `raw` must point to a live [`Recorder`] whose vtable stays loaded for the rest of the
    /// process (a Python extension module's static, for instance).
    pub unsafe fn from_raw(raw: *const Recorder) -> Result<Self, String> {
        let raw = raw.as_ref().ok_or("null recorder")?;
        let vtable = raw.vtable.as_ref().ok_or("null recorder vtable")?;
        if vtable.abi_version != ABI_VERSION || vtable.struct_size != std::mem::size_of::<RecorderVTable>() {
            return Err(format!(
                "recorder ABI v{} (vtable {} bytes) doesn't match this build's v{ABI_VERSION} ({} bytes); \
                 install matching puggles and puggles-dashboard versions",
                vtable.abi_version,
                vtable.struct_size,
                std::mem::size_of::<RecorderVTable>()
            ));
        }
        let ctx = (vtable.clone_ctx)(raw.ctx);
        Ok(RecorderHandle { vtable, ctx, buffers: Mutex::new(Buffers::default()) })
    }

    /// Record a generation without waiting out a pause.
    pub fn record(&self, snap: &Snapshot<'_>) {
        let started = Instant::now();
        let max_points = unsafe { (self.vtable.max_points)(self.ctx) };
        let mut guard = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let b = &mut *guard;
        let (n_obj, n_var) = match snap.archive.first().or(snap.population.first()) {
            Some(s) => (s.problem.number_of_objectives, s.problem.solution_length),
            None => (0, 0),
        };

        b.archive_objectives.clear();
        b.archive_variables.clear();
        b.archive_feasible.clear();
        for s in snap.archive {
            b.archive_objectives.extend(objectives_of(s, n_obj));
            b.archive_variables.extend_from_slice(&s.solution);
            b.archive_feasible.push(s.feasible as u8);
        }

        let keep = sample_indices(snap.population.len(), max_points);
        let members: Box<dyn Iterator<Item = &Solution>> = match &keep {
            Some(k) => Box::new(k.iter().map(|&i| &snap.population[i])),
            None => Box::new(snap.population.iter()),
        };
        let constrained = snap.population.first().is_some_and(|s| s.problem.has_constraints());
        b.population_objectives.clear();
        b.population_feasible.clear();
        b.population_violations.clear();
        for s in members {
            b.population_objectives.extend(objectives_of(s, n_obj));
            b.population_feasible.push(s.feasible as u8);
            if constrained {
                b.population_violations.push(s.constraint_violation);
            }
        }

        let exec = snap.execution.as_ref();
        let times = snap.times.unwrap_or_default();
        let generation = Generation {
            struct_size: std::mem::size_of::<Generation>(),
            nfe: snap.nfe,
            n_obj,
            n_var,
            archive_len: snap.archive.len(),
            archive_objectives: b.archive_objectives.as_ptr(),
            archive_variables: b.archive_variables.as_ptr(),
            archive_feasible: b.archive_feasible.as_ptr(),
            population_len: b.population_feasible.len(),
            population_total: snap.population.len(),
            population_objectives: b.population_objectives.as_ptr(),
            population_feasible: b.population_feasible.as_ptr(),
            population_violations: if constrained { b.population_violations.as_ptr() } else { std::ptr::null() },
            algorithm: RawStr::new(snap.algorithm),
            has_execution: exec.is_some() as u8,
            execution_requested: exec.map_or(0, |e| mode_code(e.requested)),
            execution_effective: exec.map_or(0, |e| mode_code(e.effective)),
            execution_cpu_threads: exec.and_then(|e| e.cpu_threads).map_or(-1, |n| n as i64),
            execution_device: RawStr::opt(exec.and_then(|e| e.device.as_deref())),
            execution_note: RawStr::opt(exec.and_then(|e| e.note.as_deref())),
            gpu_evaluations: snap.gpu_evaluations.map_or(-1, |n| n as i64),
            has_times: snap.times.is_some() as u8,
            times: [times.variation, times.evaluation, times.selection, times.archive],
            pack_seconds: started.elapsed().as_secs_f64(),
        };
        unsafe { (self.vtable.record)(self.ctx, &generation) };
    }

    /// Wait at most `max` while the run is paused: `Some(keep_going)` once it may continue (or
    /// was stopped), `None` if it's still paused.
    pub fn hold_for(&self, max: Duration) -> Option<bool> {
        let millis = u64::try_from(max.as_millis()).unwrap_or(u64::MAX);
        match unsafe { (self.vtable.hold_for)(self.ctx, millis) } {
            1 => Some(true),
            0 => Some(false),
            _ => None,
        }
    }

    /// Block while the run is paused. Returns `false` once Stop was pressed.
    pub fn hold(&self) -> bool {
        loop {
            if let Some(keep_going) = self.hold_for(Duration::from_secs(3600)) {
                return keep_going;
            }
        }
    }

    /// Record, then wait out a pause; `false` once Stop was pressed.
    pub fn record_and_hold(&self, snap: &Snapshot<'_>) -> bool {
        self.record(snap);
        self.hold()
    }

    /// Mark the run finished.
    pub fn finish(&self) {
        unsafe { (self.vtable.finish)(self.ctx) }
    }

    /// End the run after the current generation, as the page's Stop button does.
    pub fn request_stop(&self) {
        unsafe { (self.vtable.request_stop)(self.ctx) }
    }

    /// Apply a pending steering request from the page to an NSGA-III run (before recording).
    pub fn apply_focus(&self, ga: &mut NSGAIII) {
        let m = ga.problem.number_of_objectives;
        let mut aspiration = vec![f64::NAN; m];
        let mut spread = 1.0;
        let kind = unsafe { (self.vtable.take_focus)(self.ctx, ga.get_nfe(), m, aspiration.as_mut_ptr(), &mut spread) };
        match kind {
            1 => ga.focus_on(&aspiration, spread),
            2 => ga.clear_focus(),
            _ => {}
        }
    }
}

impl Drop for RecorderHandle {
    fn drop(&mut self) {
        unsafe { (self.vtable.drop_ctx)(self.ctx) }
    }
}

fn objectives_of(s: &Solution, n_obj: usize) -> impl Iterator<Item = f64> + '_ {
    // An unevaluated solution has no objectives yet; send NaN (how the page spells "missing").
    (0..n_obj).map(move |o| s.objective_fitness_values.get(o).copied().unwrap_or(f64::NAN))
}

// -- the recorder's side ----------------------------------------------------------------------

impl Generation {
    /// `rows` rows of `width` values starting at `ptr`.
    ///
    /// # Safety
    /// Only during the call this generation was passed to.
    #[cfg(feature = "plot")]
    unsafe fn rows<'a>(ptr: *const f64, rows: usize, width: usize) -> impl Iterator<Item = &'a [f64]> {
        let all: &[f64] = if rows * width == 0 { &[] } else { std::slice::from_raw_parts(ptr, rows * width) };
        (0..rows).map(move |r| &all[r * width..(r + 1) * width])
    }

    /// # Safety
    /// `ptr` must point to `len` values (or `len` must be 0).
    #[cfg(feature = "plot")]
    unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
        if len == 0 || ptr.is_null() { &[] } else { std::slice::from_raw_parts(ptr, len) }
    }

    /// Where evaluations ran, if the sender said.
    ///
    /// # Safety
    /// Only during the call this generation was passed to.
    pub unsafe fn execution(&self) -> Option<ExecutionInfo> {
        (self.has_execution != 0).then(|| ExecutionInfo {
            requested: mode_from_code(self.execution_requested),
            effective: mode_from_code(self.execution_effective),
            cpu_threads: usize::try_from(self.execution_cpu_threads).ok(),
            device: self.execution_device.get().map(str::to_string),
            note: self.execution_note.get().map(str::to_string),
        })
    }
}

#[cfg(feature = "plot")]
mod dashboard {
    use super::*;
    use crate::plot::{Dashboard, FocusRequest, Frame, Timing};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// A [`Recorder`] that feeds `dash`. Release it with [`drop_recorder`].
    pub fn recorder_for(dash: Dashboard) -> Recorder {
        Recorder { vtable: &VTABLE, ctx: Box::into_raw(Box::new(dash)).cast() }
    }

    /// Release a [`Recorder`] made by [`recorder_for`].
    ///
    /// # Safety
    /// `recorder` must come from [`recorder_for`] and not be used afterwards.
    pub unsafe fn drop_recorder(recorder: Recorder) {
        drop_ctx(recorder.ctx);
    }

    /// The frame a [`Generation`] describes, plus the algorithm and execution it reports.
    ///
    /// # Safety
    /// Only during the call `g` was passed to.
    pub unsafe fn to_frame(g: &Generation) -> (Frame, String, Option<ExecutionInfo>) {
        let started = Instant::now();
        let archive_feasible = Generation::slice(g.archive_feasible, g.archive_len);
        let population_feasible = Generation::slice(g.population_feasible, g.population_len);
        let frame = Frame {
            nfe: g.nfe,
            objectives: Generation::rows(g.archive_objectives, g.archive_len, g.n_obj).map(<[f64]>::to_vec).collect(),
            variables: Generation::rows(g.archive_variables, g.archive_len, g.n_var).map(<[f64]>::to_vec).collect(),
            ok: archive_feasible.iter().map(|&f| f != 0).collect(),
            population_objectives: Generation::rows(g.population_objectives, g.population_len, g.n_obj)
                .map(<[f64]>::to_vec)
                .collect(),
            population_ok: population_feasible.iter().map(|&f| f != 0).collect(),
            population_violations: Generation::slice(g.population_violations, g.population_len).to_vec(),
            gpu_evals: usize::try_from(g.gpu_evaluations).ok(),
            population_total: (g.population_total > g.population_len).then_some(g.population_total),
            timing: (g.has_times != 0).then(|| Timing {
                generation: GenerationTimes {
                    variation: g.times[0],
                    evaluation: g.times[1],
                    selection: g.times[2],
                    archive: g.times[3],
                },
                record: g.pack_seconds + started.elapsed().as_secs_f64(),
            }),
            ..Default::default()
        };
        let algorithm = g.algorithm.get().unwrap_or("unknown").to_string();
        (frame, algorithm, g.execution())
    }

    static VTABLE: RecorderVTable = RecorderVTable {
        abi_version: ABI_VERSION,
        struct_size: std::mem::size_of::<RecorderVTable>(),
        clone_ctx,
        drop_ctx,
        max_points,
        record,
        hold_for,
        finish,
        request_stop,
        take_focus,
    };

    /// # Safety
    /// `ctx` is a live `Box<Dashboard>` from [`recorder_for`] or `clone_ctx`.
    unsafe fn dash<'a>(ctx: *mut c_void) -> &'a Dashboard {
        &*ctx.cast::<Dashboard>()
    }

    /// Never unwind into the caller (it may not be Rust); a panic just skips the call.
    fn guarded<T>(fallback: T, f: impl FnOnce() -> T) -> T {
        catch_unwind(AssertUnwindSafe(f)).unwrap_or(fallback)
    }

    unsafe extern "C" fn clone_ctx(ctx: *mut c_void) -> *mut c_void {
        Box::into_raw(Box::new(dash(ctx).clone())).cast()
    }

    unsafe extern "C" fn drop_ctx(ctx: *mut c_void) {
        if !ctx.is_null() {
            drop(Box::from_raw(ctx.cast::<Dashboard>()));
        }
    }

    unsafe extern "C" fn max_points(ctx: *mut c_void) -> usize {
        dash(ctx).max_points()
    }

    unsafe extern "C" fn record(ctx: *mut c_void, generation: *const Generation) {
        guarded((), || {
            let Some(g) = generation.as_ref() else { return };
            if g.struct_size != std::mem::size_of::<Generation>() {
                return; // from_raw already checked the versions; never misread a frame
            }
            let (frame, algorithm, execution) = to_frame(g);
            dash(ctx).record_frame_with_meta(frame, &algorithm, execution);
        })
    }

    unsafe extern "C" fn hold_for(ctx: *mut c_void, millis: u64) -> i32 {
        guarded(1, || match dash(ctx).hold_for(Duration::from_millis(millis)) {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        })
    }

    unsafe extern "C" fn finish(ctx: *mut c_void) {
        guarded((), || dash(ctx).finish())
    }

    unsafe extern "C" fn request_stop(ctx: *mut c_void) {
        guarded((), || dash(ctx).request_stop())
    }

    unsafe extern "C" fn take_focus(ctx: *mut c_void, nfe: usize, n_obj: usize, aspiration: *mut f64, spread: *mut f64) -> i32 {
        guarded(0, || {
            let d = dash(ctx);
            let Some(request) = d.take_focus_request() else { return 0 };
            d.add_event(nfe, request.event_label());
            match request {
                FocusRequest::Focus { aspiration: a, spread: s } => {
                    if !aspiration.is_null() {
                        let out = std::slice::from_raw_parts_mut(aspiration, n_obj);
                        for (o, v) in out.iter_mut().enumerate() {
                            *v = a.get(o).copied().unwrap_or(f64::NAN);
                        }
                    }
                    if !spread.is_null() {
                        *spread = s;
                    }
                    1
                }
                FocusRequest::Clear => 2,
            }
        })
    }
}

#[cfg(feature = "plot")]
pub use dashboard::{drop_recorder, recorder_for, to_frame};

#[cfg(all(test, feature = "plot"))]
mod tests {
    use super::*;
    use crate::core::Problem;
    use crate::gatypes::{Real, SolutionDataTypes};
    use crate::plot::{Dashboard, Frame, RunMeta};
    use std::sync::Arc;

    fn problem() -> Arc<Problem> {
        Arc::new(Problem::new(
            4, 2, None, None, Some(vec![-1, -1]),
            vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))); 4],
            |x| vec![x[0], 1.0 - x[0] + x[1..].iter().sum::<f64>()],
        ))
    }

    #[test]
    fn a_foreign_recording_matches_a_native_one() {
        let p = problem();
        let native = Dashboard::start(RunMeta::from_problem(&p), "127.0.0.1", 0).unwrap();
        let foreign = Dashboard::start(RunMeta::from_problem(&p), "127.0.0.1", 0).unwrap();
        foreign.set_max_points(15);
        native.set_max_points(15);

        let raw = recorder_for(foreign.clone());
        let handle = unsafe { RecorderHandle::from_raw(&raw) }.unwrap();
        let mut ga = NSGAII::new(Arc::clone(&p), 24, ExecutionMode::Sequential).with_seed(5);
        ga.run_with(600, |ga| {
            native.record_without_hold(ga);
            handle.record(&Snapshot::of_nsgaii(ga));
            true
        });
        handle.finish();
        drop(handle);
        unsafe { drop_recorder(raw) };

        let strip = |f: &Frame| {
            let mut f = f.clone();
            f.t = 0.0;
            f.timing = None;
            f
        };
        let (a, b) = (native.history(), foreign.history());
        assert_eq!(a.len(), b.len());
        for (x, y) in a.frames().zip(b.frames()) {
            assert_eq!(serde_json::to_string(&strip(x)).unwrap(), serde_json::to_string(&strip(y)).unwrap());
        }
        assert_eq!(a.meta.algorithm, b.meta.algorithm);
        assert_eq!(a.meta.execution, b.meta.execution);
        assert_eq!(b.latest().unwrap().population_total, Some(24), "the population was sampled");
        assert_eq!(foreign.state().as_str(), "done");
    }

    #[test]
    fn steering_and_stop_cross_the_abi() {
        let p = Arc::new(Problem::new(
            6, 3, None, None, Some(vec![-1, -1, -1]),
            vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))); 6],
            |x| vec![x[0], x[1], 2.0 - x[0] - x[1] + x[2..].iter().sum::<f64>()],
        ));
        let dash = Dashboard::start(RunMeta::from_problem(&p), "127.0.0.1", 0).unwrap();
        let raw = recorder_for(dash.clone());
        let handle = unsafe { RecorderHandle::from_raw(&raw) }.unwrap();
        unsafe { drop_recorder(raw) }; // the handle keeps its own reference

        let mut ga = NSGAIII::new(p, 24, 4, ExecutionMode::Sequential).with_seed(1);
        let mut gens = 0;
        ga.run_with(10_000, |ga| {
            gens += 1;
            if gens == 3 {
                // What the page's Goals panel does.
                dash.queue_focus(crate::plot::FocusRequest::Focus { aspiration: vec![0.1, f64::NAN, 0.5], spread: 0.25 });
            }
            handle.apply_focus(ga);
            if gens == 6 {
                handle.request_stop();
            }
            handle.record_and_hold(&Snapshot::of_nsga3(ga))
        });
        assert_eq!(gens, 6, "stop must end the run");
        let meta = dash.meta();
        assert_eq!(meta.steerable, Some(true));
        assert!(meta.events.iter().any(|e| e.label == "steered to goals (25% focus)"));
        assert_ne!(ga.reference_points(), crate::nsga3::das_dennis(3, 4).as_slice(), "focus was applied");
    }

    #[test]
    fn a_mismatched_vtable_is_refused() {
        let dash = Dashboard::start(RunMeta::default(), "127.0.0.1", 0).unwrap();
        let mut raw = recorder_for(dash);
        static BAD: std::sync::OnceLock<RecorderVTable> = std::sync::OnceLock::new();
        let good = unsafe { &*raw.vtable };
        let bad = BAD.get_or_init(|| RecorderVTable { abi_version: ABI_VERSION + 1, ..copy(good) });
        let real = raw.vtable;
        raw.vtable = bad;
        assert!(unsafe { RecorderHandle::from_raw(&raw) }.unwrap_err().contains("doesn't match"));
        raw.vtable = real;
        unsafe { drop_recorder(raw) };
    }

    fn copy(v: &RecorderVTable) -> RecorderVTable {
        RecorderVTable {
            abi_version: v.abi_version,
            struct_size: v.struct_size,
            clone_ctx: v.clone_ctx,
            drop_ctx: v.drop_ctx,
            max_points: v.max_points,
            record: v.record,
            hold_for: v.hold_for,
            finish: v.finish,
            request_stop: v.request_stop,
            take_focus: v.take_focus,
        }
    }
}
