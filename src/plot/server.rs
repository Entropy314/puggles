//! Live dashboard: a small HTTP server (std only) for the UI, a JSON API and a Server-Sent Events
//! stream of new generations.
//!
//! Routes: `GET /` (the page), `GET /api/meta`, `GET /api/history`, `GET /api/compare` (runs
//! shown alongside), `GET /api/events` (SSE: `meta`, `frame`, `state`, `reset`, `compare`),
//! `POST /api/stop`, `POST /api/pause`, `POST /api/resume`, `POST /api/goals`.
//!
//! Bound to a loopback address it only answers requests addressed to this machine. Bound to
//! anything else (e.g. `0.0.0.0` on a cluster node) it requires an access token: the URL carries
//! it once (`?token=…`), and the page then holds it in an HttpOnly cookie. The stream pushes at most ten frames a second
//! and only the newest one, so a fast optimizer never floods the browser. Recording costs a copy
//! of the archive and population objectives per generation; serving happens on other threads.

use super::history::{Frame, NamedRun, RunEvent, RunHistory, RunMeta, RunView};
use super::html::{open_in_browser, server_page};
use super::PlotError;
use crate::core::Solution;
use crate::genetic_algorithms_v2::ExecutionInfo;
use crate::nsga3::NSGAIII;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicUsize;
use serde::Serialize;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const PUSH_INTERVAL: Duration = Duration::from_millis(100);
const HEARTBEAT: Duration = Duration::from_secs(15);
const ACCEPT_POLL: Duration = Duration::from_millis(25);

/// Where a run is, as shown in the page header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunState {
    /// Serving, no generation recorded yet.
    Waiting,
    Running,
    /// Stop was pressed; the optimizer ends after the current generation.
    Stopping,
    Stopped,
    Done,
    /// Showing a saved run.
    Replay,
    /// Pause was pressed; the optimizer waits inside `record` until Resume.
    Paused,
}

impl RunState {
    pub fn as_str(self) -> &'static str {
        match self {
            RunState::Waiting => "waiting",
            RunState::Running => "running",
            RunState::Stopping => "stopping",
            RunState::Stopped => "stopped",
            RunState::Done => "done",
            RunState::Replay => "replay",
            RunState::Paused => "paused",
        }
    }
}

struct Shared {
    history: Mutex<RunHistory>,
    state: Mutex<RunState>,
    /// Bumped on every recorded generation; SSE connections poll it.
    version: AtomicU64,
    /// Bumped when a new run starts on the same dashboard.
    run_id: AtomicU64,
    /// Bumped when the run metadata changes (e.g. execution info arrives with the first record).
    meta_version: AtomicU64,
    /// Other runs shown alongside (other seeds, algorithms, CPU vs GPU, ...).
    comparisons: Mutex<Vec<(String, RunHistory)>>,
    compare_version: AtomicU64,
    /// Keep each finished run as a comparison when a new one starts on this dashboard.
    keep_runs: AtomicBool,
    paused: AtomicBool,
    stop: AtomicBool,
    closing: AtomicBool,
    host: String,
    port: u16,
    /// Required on every request when set (remote access).
    token: Mutex<Option<String>>,
    /// A steering change from the page, waiting for the optimizer to pick it up.
    focus: Mutex<Option<FocusRequest>>,
    steering: AtomicBool,
    /// Group for the next run that starts (so naming a run never relabels the finished one).
    pending_group: Mutex<Option<String>>,
    columns: Mutex<Option<Columns>>,
    /// Population members kept per frame (an even sample beyond this); 0 = all.
    max_points: AtomicUsize,
    /// The newest frame's JSON and the `version` it was serialized at, shared by every SSE
    /// connection so each new frame is serialized once, not once per open tab.
    latest_json: Mutex<(u64, Option<Arc<Vec<u8>>>)>,
}

/// Computes your columns' values from a solution's variables.
pub type ColumnFn = Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;

/// Your own per-solution columns: names, and a function from a solution's variables to values.
#[derive(Clone)]
pub struct Columns {
    pub names: Vec<String>,
    pub compute: ColumnFn,
}

/// A steering change requested from the page's Goals panel.
#[derive(Clone, Debug, PartialEq)]
pub enum FocusRequest {
    /// Concentrate on the aspiration levels (`NaN` = no preference); see [`NSGAIII::focus_on`].
    Focus { aspiration: Vec<f64>, spread: f64 },
    /// Back to the whole front.
    Clear,
}

impl FocusRequest {
    /// How applying this request is marked on the convergence chart.
    pub fn event_label(&self) -> String {
        match self {
            FocusRequest::Focus { spread, .. } => format!("steered to goals ({:.0}% focus)", spread * 100.0),
            FocusRequest::Clear => "steering off".to_string(),
        }
    }
}

impl Shared {
    fn history(&self) -> MutexGuard<'_, RunHistory> {
        self.history.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn state(&self) -> RunState {
        *self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn set_state(&self, s: RunState) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = s;
    }
    fn comparisons(&self) -> MutexGuard<'_, Vec<(String, RunHistory)>> {
        self.comparisons.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn token(&self) -> Option<String> {
        self.token.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]")
}

/// This machine's hostname (Unix: `/etc/hostname` or `$HOSTNAME`; Windows: `%COMPUTERNAME%`).
fn machine_name() -> String {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_else(|| "localhost".to_string())
}

/// 128 random bits as hex, from the OS-seeded CSPRNG.
fn new_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A live dashboard on `http://127.0.0.1:<port>/`. Cheap to clone (all clones share one server);
/// the server shuts down when the last clone is dropped or [`close`](Self::close) is called.
///
/// Feed it from [`NSGAII::run_with`](crate::genetic_algorithms_v2::NSGAII::run_with) or
/// [`NSGAIII::run_with`](crate::nsga3::NSGAIII::run_with): `|ga| dash.record(ga)` returns `false`
/// once the page's Stop button is pressed, which ends the run.
#[derive(Clone)]
pub struct Dashboard {
    inner: Arc<Inner>,
}

struct Inner {
    shared: Arc<Shared>,
    addr: SocketAddr,
    accept: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.shared.closing.store(true, Ordering::SeqCst);
        if let Some(h) = self.accept.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
    }
}

impl Dashboard {
    /// Serve a live dashboard for a run described by `meta`. `port` 0 picks a free port.
    pub fn start(meta: RunMeta, host: &str, port: u16) -> io::Result<Self> {
        Self::serve(RunHistory::new(meta), RunState::Waiting, host, port)
    }

    /// Serve a live dashboard that records into `history` (e.g. one with a custom `max_frames`).
    pub fn start_with(history: RunHistory, host: &str, port: u16) -> io::Result<Self> {
        Self::serve(history, RunState::Waiting, host, port)
    }

    /// Serve a finished or saved run for exploring.
    pub fn replay(history: RunHistory, host: &str, port: u16) -> io::Result<Self> {
        Self::serve(history, RunState::Replay, host, port)
    }

    fn serve(history: RunHistory, state: RunState, host: &str, port: u16) -> io::Result<Self> {
        let listener = TcpListener::bind((host, port))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let shared = Arc::new(Shared {
            history: Mutex::new(history),
            state: Mutex::new(state),
            version: AtomicU64::new(0),
            run_id: AtomicU64::new(0),
            meta_version: AtomicU64::new(0),
            comparisons: Mutex::new(Vec::new()),
            compare_version: AtomicU64::new(0),
            keep_runs: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            closing: AtomicBool::new(false),
            host: host.to_string(),
            port: addr.port(),
            // Secure by default: anything reachable from other machines needs the token.
            token: Mutex::new(if is_loopback(host) { None } else { Some(new_token()) }),
            focus: Mutex::new(None),
            steering: AtomicBool::new(false),
            pending_group: Mutex::new(None),
            columns: Mutex::new(None),
            max_points: AtomicUsize::new(2000),
            latest_json: Mutex::new((u64::MAX, None)),
        });
        let sh = Arc::clone(&shared);
        let accept = thread::Builder::new()
            .name("puggles-dashboard".into())
            .spawn(move || accept_loop(listener, sh))?;
        Ok(Dashboard { inner: Arc::new(Inner { shared, addr, accept: Mutex::new(Some(accept)) }) })
    }

    /// The page's address, including `?token=…` when one is required.
    pub fn url(&self) -> String {
        let a = self.inner.addr;
        let host = if a.ip().is_unspecified() {
            machine_name() // bound to every interface: give a name other machines can use
        } else if a.is_ipv6() {
            format!("[{}]", a.ip())
        } else {
            a.ip().to_string()
        };
        let token = self.inner.shared.token().map(|t| format!("?token={t}")).unwrap_or_default();
        format!("http://{host}:{}/{token}", a.port())
    }

    /// Require an access token on every request (generated if `None`), e.g. to share a
    /// dashboard bound to `127.0.0.1` through a tunnel. Returns the token.
    pub fn require_token(&self, token: Option<String>) -> String {
        let token = token.unwrap_or_else(new_token);
        *self.inner.shared.token.lock().unwrap_or_else(|e| e.into_inner()) = Some(token.clone());
        token
    }

    /// The access token, if one is required.
    pub fn token(&self) -> Option<String> {
        self.inner.shared.token()
    }

    pub fn addr(&self) -> SocketAddr {
        self.inner.addr
    }

    /// Print the URL to stderr and open it in the default browser (best effort).
    pub fn open_browser(&self) -> &Self {
        eprintln!("puggles dashboard: {}", self.url());
        let _ = open_in_browser(&self.url());
        self
    }

    // -- recording --------------------------------------------------------------------------

    /// Record the optimizer's current generation. Returns `false` once Stop was pressed.
    ///
    /// While the page has the run paused this blocks (after recording) until Resume or Stop.
    pub fn record<R: RunView + ?Sized>(&self, ga: &R) -> bool {
        self.record_without_hold(ga);
        self.hold()
    }

    /// [`record`](Self::record) without waiting out a pause — follow it with
    /// [`hold_for`](Self::hold_for), e.g. to release a lock (like Python's GIL) while paused.
    pub fn record_without_hold<R: RunView + ?Sized>(&self, ga: &R) {
        let frame = Frame::from_run_sampled(ga, self.max_points());
        self.record_frame_with_meta(frame, ga.algorithm(), ga.execution());
    }

    /// Record a generation built elsewhere — e.g. from an optimizer in another process or
    /// library — exactly as [`record_without_hold`](Self::record_without_hold) records a
    /// [`RunView`]: `algorithm` and `execution` are what it would report. The population is
    /// thinned to [`max_points`](Self::max_points) if it isn't already.
    pub fn record_frame_with_meta(&self, mut frame: Frame, algorithm: &str, execution: Option<ExecutionInfo>) {
        let sh = &self.inner.shared;
        frame.thin_population(sh.max_points.load(Ordering::SeqCst));
        let columns = sh.columns.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some(c) = columns {
            let mut out: BTreeMap<String, Vec<f64>> = c.names.iter().map(|n| (n.clone(), Vec::new())).collect();
            for x in &frame.variables {
                let values = (c.compute)(x);
                for (k, name) in c.names.iter().enumerate() {
                    out.get_mut(name).expect("named column").push(values.get(k).copied().unwrap_or(f64::NAN));
                }
            }
            frame.extra = Some(out);
        }
        let meta_changed = {
            let mut h = sh.history();
            // Archive the finished run before its metadata turns into the new run's.
            keep_finished_run(sh, &mut h, frame.nfe);
            let mut changed = h.update_meta_parts(algorithm, execution);
            if h.is_empty() || h.starts_new_run(frame.nfe) {
                if let Some(g) = sh.pending_group.lock().unwrap_or_else(|e| e.into_inner()).take() {
                    h.meta.group = Some(g);
                    changed = true;
                }
            }
            if sh.keep_runs.load(Ordering::SeqCst) {
                let label = Some(format!("run {} · {}", sh.comparisons().len() + 1, algorithm));
                if h.meta.run_label != label {
                    h.meta.run_label = label;
                    changed = true;
                }
            }
            changed
        };
        if meta_changed {
            sh.meta_version.fetch_add(1, Ordering::SeqCst);
        }
        self.push_frame(frame, true);
    }

    /// [`record`](Self::record) from raw parts.
    pub fn record_solutions(&self, archive: &[Solution], population: &[Solution], nfe: usize) -> bool {
        self.record_frame(Frame::from_solutions(archive, population, nfe))
    }

    /// Record a generation built elsewhere (its `gen` and `t` are assigned here).
    ///
    /// While the page has the run paused this blocks (after recording) until Resume or Stop.
    pub fn record_frame(&self, frame: Frame) -> bool {
        self.push_frame(frame, false);
        self.hold()
    }

    /// [`record_frame`](Self::record_frame) without waiting out a pause.
    pub fn record_frame_without_hold(&self, frame: Frame) {
        self.push_frame(frame, false);
    }

    /// Block while the run is paused. Returns `false` once Stop was pressed.
    pub fn hold(&self) -> bool {
        loop {
            if let Some(keep_going) = self.hold_for(Duration::from_secs(3600)) {
                return keep_going;
            }
        }
    }

    /// Wait at most `max` while the run is paused: `Some(keep_going)` once it may continue (or
    /// was stopped), `None` if it's still paused.
    pub fn hold_for(&self, max: Duration) -> Option<bool> {
        let sh = &self.inner.shared;
        let started = Instant::now();
        loop {
            let stop = sh.stop.load(Ordering::SeqCst);
            if stop || !sh.paused.load(Ordering::SeqCst) || sh.closing.load(Ordering::SeqCst) {
                if !stop {
                    sh.set_state(RunState::Running);
                }
                return Some(!stop);
            }
            sh.set_state(RunState::Paused);
            let left = max.saturating_sub(started.elapsed());
            if left.is_zero() {
                return None;
            }
            thread::sleep(left.min(Duration::from_millis(50)));
        }
    }

    fn push_frame(&self, frame: Frame, finished_run_kept: bool) {
        let sh = &self.inner.shared;
        let new_run = {
            let mut h = sh.history();
            if !finished_run_kept {
                keep_finished_run(sh, &mut h, frame.nfe);
            }
            h.record_frame(frame)
        };
        if new_run {
            sh.run_id.fetch_add(1, Ordering::SeqCst);
            sh.stop.store(false, Ordering::SeqCst);
        }
        sh.version.fetch_add(1, Ordering::SeqCst);
    }

    /// Hold the run after its next generation (what the page's Pause button does).
    pub fn pause(&self) {
        set_paused(&self.inner.shared, true);
    }

    pub fn resume(&self) {
        set_paused(&self.inner.shared, false);
    }

    pub fn is_paused(&self) -> bool {
        self.inner.shared.paused.load(Ordering::SeqCst)
    }

    /// Show another run alongside this one: its front and convergence are overlaid.
    pub fn compare_with(&self, name: impl Into<String>, history: RunHistory) {
        let sh = &self.inner.shared;
        sh.comparisons().push((name.into(), history));
        sh.compare_version.fetch_add(1, Ordering::SeqCst);
    }

    /// When on, each finished run is kept as a comparison when the next one starts, so running
    /// several optimizers (seeds, algorithms, CPU vs GPU) on one dashboard overlays them all.
    pub fn keep_runs(&self, on: bool) -> &Self {
        self.inner.shared.keep_runs.store(on, Ordering::SeqCst);
        self
    }

    /// The runs shown alongside this one.
    pub fn comparisons(&self) -> Vec<(String, RunHistory)> {
        self.inner.shared.comparisons().clone()
    }

    /// Name the group of the next run that starts (or of the first run), e.g. "NSGA-II, pop 100";
    /// runs sharing a group are summarized together across seeds. Without one, the page groups
    /// runs by algorithm.
    pub fn set_group(&self, group: impl Into<String>) {
        *self.inner.shared.pending_group.lock().unwrap_or_else(|e| e.into_inner()) = Some(group.into());
    }

    /// Add your own per-solution columns, computed from each front solution's variables every
    /// recorded generation; they appear in the table, color-by, pins, drivers and CSV.
    pub fn set_columns(&self, names: Vec<String>, compute: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static) {
        *self.inner.shared.columns.lock().unwrap_or_else(|e| e.into_inner()) = Some(Columns { names, compute: Arc::new(compute) });
    }

    /// Population members kept per recorded generation (default 2,000; an even sample beyond
    /// that, so huge populations stay fast to stream and draw). 0 keeps all.
    pub fn set_max_points(&self, max: usize) {
        self.inner.shared.max_points.store(max, Ordering::SeqCst);
    }

    /// Queue a steering change as the page's Goals panel does (for tests).
    #[cfg(test)]
    pub(crate) fn queue_focus(&self, request: FocusRequest) {
        *self.inner.shared.focus.lock().unwrap_or_else(|e| e.into_inner()) = Some(request);
    }

    /// The steering change waiting from the page's Goals panel, if any (taken once).
    pub fn take_focus_request(&self) -> Option<FocusRequest> {
        self.mark_steerable();
        self.inner.shared.focus.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Apply a pending steering change to an NSGA-III run; call it in `run_with` before `record`:
    /// `ga.run_with(n, |ga| { dash.apply_focus(ga); dash.record(ga) })`. Enables the page's Steer option.
    pub fn apply_focus(&self, ga: &mut NSGAIII) {
        let Some(request) = self.take_focus_request() else { return };
        match &request {
            FocusRequest::Focus { aspiration, spread } => ga.focus_on(aspiration, *spread),
            FocusRequest::Clear => ga.clear_focus(),
        }
        self.add_event(ga.get_nfe(), request.event_label());
    }

    /// Mark a moment on the convergence chart (kept in saved runs), e.g. a parameter change.
    pub fn add_event(&self, nfe: usize, label: impl Into<String>) {
        let sh = &self.inner.shared;
        sh.history().meta.events.push(RunEvent { nfe, label: label.into() });
        sh.meta_version.fetch_add(1, Ordering::SeqCst);
    }

    fn mark_steerable(&self) {
        let sh = &self.inner.shared;
        let mut h = sh.history();
        if h.meta.steerable != Some(true) {
            h.meta.steerable = Some(true);
            sh.meta_version.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Mark the run finished in the page. Call after `run_with` returns.
    pub fn finish(&self) {
        let sh = &self.inner.shared;
        sh.set_state(if sh.stop.load(Ordering::SeqCst) { RunState::Stopped } else { RunState::Done });
    }

    /// What the page's Stop button does: the next [`record`](Self::record) returns `false`.
    pub fn request_stop(&self) {
        request_stop(&self.inner.shared);
    }

    pub fn state(&self) -> RunState {
        self.inner.shared.state()
    }

    /// A snapshot of everything recorded so far. This copies every kept frame; for a single
    /// figure use [`frame_count`](Self::frame_count), [`latest_nfe`](Self::latest_nfe) or
    /// [`meta`](Self::meta).
    pub fn history(&self) -> RunHistory {
        self.inner.shared.history().clone()
    }

    /// Generations kept for replay.
    pub fn frame_count(&self) -> usize {
        self.inner.shared.history().len()
    }

    /// Evaluations at the newest recorded generation.
    pub fn latest_nfe(&self) -> Option<usize> {
        self.inner.shared.history().latest().map(|f| f.nfe)
    }

    /// The run's metadata (a copy).
    pub fn meta(&self) -> RunMeta {
        self.inner.shared.history().meta.clone()
    }

    /// Population members kept per recorded generation (0 = all); see [`set_max_points`](Self::set_max_points).
    pub fn max_points(&self) -> usize {
        self.inner.shared.max_points.load(Ordering::SeqCst)
    }

    pub fn to_json(&self) -> String {
        self.inner.shared.history().to_json()
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<PathBuf, PlotError> {
        self.inner.shared.history().save_json(path)
    }

    /// A self-contained replay page of the run so far, with any comparisons.
    pub fn save_html(&self, path: impl AsRef<Path>) -> Result<PathBuf, PlotError> {
        std::fs::write(path.as_ref(), self.to_html())?;
        Ok(path.as_ref().to_path_buf())
    }

    pub fn to_html(&self) -> String {
        let sh = &self.inner.shared;
        sh.history().to_html_compared(&sh.comparisons())
    }

    // -- lifetime ---------------------------------------------------------------------------

    /// Block until [`close`](Self::close) is called from another thread (or the process ends),
    /// keeping the page alive after the run.
    pub fn wait(&self) {
        while !self.is_closed() {
            thread::sleep(Duration::from_millis(200));
        }
    }

    pub fn close(&self) {
        self.inner.shared.closing.store(true, Ordering::SeqCst);
    }

    pub fn is_closed(&self) -> bool {
        self.inner.shared.closing.load(Ordering::SeqCst)
    }
}

fn request_stop(sh: &Shared) {
    let state = sh.state();
    if matches!(state, RunState::Waiting | RunState::Running | RunState::Paused) {
        sh.stop.store(true, Ordering::SeqCst);
        sh.set_state(RunState::Stopping);
    }
}

/// With `keep_runs` on, a generation that starts a new run first files the finished one away.
fn keep_finished_run(sh: &Shared, h: &mut RunHistory, nfe: usize) {
    if sh.keep_runs.load(Ordering::SeqCst) && h.starts_new_run(nfe) && !h.is_empty() {
        let mut comparisons = sh.comparisons();
        let name = h.meta.run_label.clone().unwrap_or_else(|| {
            format!("run {}", comparisons.len() + 1)
                + &h.meta.algorithm.as_ref().map(|a| format!(" · {a}")).unwrap_or_default()
        });
        comparisons.push((name, h.clone()));
        sh.compare_version.fetch_add(1, Ordering::SeqCst);
    }
}

fn set_paused(sh: &Shared, paused: bool) {
    sh.paused.store(paused, Ordering::SeqCst);
    // The recorder flips the state when it actually blocks or continues; reflect a resume of an
    // idle (waiting) run right away.
    if !paused && sh.state() == RunState::Paused {
        sh.set_state(RunState::Running);
    }
}

// -- HTTP ------------------------------------------------------------------------------------

fn accept_loop(listener: TcpListener, sh: Arc<Shared>) {
    while !sh.closing.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                // Accepted sockets inherit non-blocking mode on some platforms (macOS, Windows).
                if stream.set_nonblocking(false).is_err() {
                    continue;
                }
                let sh = Arc::clone(&sh);
                let _ = thread::Builder::new().name("puggles-dashboard-conn".into()).spawn(move || {
                    let _ = handle(stream, &sh);
                });
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => thread::sleep(ACCEPT_POLL),
            Err(_) => thread::sleep(ACCEPT_POLL),
        }
    }
}

struct Request {
    method: String,
    path: String,
    query_token: Option<String>,
    host: Option<String>,
    origin: Option<String>,
    cookie: Option<String>,
    body: Vec<u8>,
}

const MAX_BODY: usize = 64 * 1024;

fn read_request(stream: &TcpStream) -> io::Result<Request> {
    let mut reader = BufReader::new(stream.take((16 * 1024 + MAX_BODY) as u64));
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let path = path.split('#').next().unwrap_or("/").to_string();
    let query_token = query.split('&').find_map(|kv| kv.strip_prefix("token=")).map(str::to_string);
    let (mut host, mut origin, mut cookie, mut length) = (None, None, None, 0usize);
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let value = value.trim().to_string();
            match name.trim().to_ascii_lowercase().as_str() {
                "host" => host = Some(value),
                "origin" => origin = Some(value),
                "cookie" => cookie = Some(value),
                "content-length" => length = value.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    let mut body = vec![0u8; length.min(MAX_BODY)];
    reader.read_exact(&mut body)?;
    Ok(Request { method, path, query_token, host, origin, cookie, body })
}

fn cookie_name(sh: &Shared) -> String {
    format!("puggles_token_{}", sh.port) // per port: several dashboards can share a host
}

/// Compare without leaking how many leading characters matched.
fn same_secret(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// With a token: the request must carry it (cookie, or `?token=` on the page). Without one: only
/// answer requests addressed to this machine. Either way refuse cross-site writes — guards
/// against DNS rebinding and other pages poking the Stop button.
fn allowed(req: &Request, sh: &Shared) -> bool {
    let strip_port = |h: &str| -> String {
        h.rsplit_once(':').map_or(h, |(n, port)| if port.chars().all(|c| c.is_ascii_digit()) { n } else { h }).to_string()
    };
    if let Some(token) = sh.token() {
        let name = cookie_name(sh);
        let from_cookie = req.cookie.as_deref().and_then(|c| {
            c.split(';').find_map(|kv| kv.trim().strip_prefix(name.as_str()).and_then(|v| v.strip_prefix('=')).map(str::to_string))
        });
        let ok = from_cookie.iter().chain(req.query_token.iter()).any(|t| same_secret(t, &token));
        if !ok {
            return false;
        }
        if req.method != "GET" {
            if let (Some(o), Some(h)) = (&req.origin, &req.host) {
                let o = o.strip_prefix("http://").or_else(|| o.strip_prefix("https://")).unwrap_or(o);
                if o != h {
                    return false;
                }
            }
        }
        return true;
    }
    let local = |h: &str| {
        let name = strip_port(h);
        matches!(name.as_str(), "127.0.0.1" | "localhost" | "[::1]") || name == sh.host
    };
    if let Some(h) = &req.host {
        if !local(h) {
            return false;
        }
    }
    if req.method != "GET" {
        if let Some(o) = &req.origin {
            let o = o.strip_prefix("http://").unwrap_or(o);
            if !local(o) {
                return false;
            }
        }
    }
    true
}

fn handle(mut stream: TcpStream, sh: &Shared) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let req = read_request(&stream)?;
    if !allowed(&req, sh) {
        let msg: &[u8] = if sh.token().is_some() {
            b"Forbidden: open the dashboard with the URL it printed (it carries ?token=...)"
        } else {
            b"Forbidden"
        };
        return respond(&mut stream, "403 Forbidden", "text/plain", msg);
    }
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => {
            let cookie = match (&req.query_token, sh.token()) {
                (Some(t), Some(_)) => {
                    format!("Set-Cookie: {}={t}; Path=/; HttpOnly; SameSite=Strict\r\n", cookie_name(sh))
                }
                _ => String::new(),
            };
            respond_with(&mut stream, "200 OK", "text/html; charset=utf-8", &cookie, server_page().as_bytes())
        }
        ("POST", "/api/goals") => {
            #[derive(serde::Deserialize)]
            struct Goals {
                goals: Vec<Option<f64>>,
                #[serde(default)]
                steer: bool,
                #[serde(default = "default_spread")]
                spread: f64,
            }
            fn default_spread() -> f64 {
                0.25
            }
            let Ok(g) = serde_json::from_slice::<Goals>(&req.body) else {
                return respond(&mut stream, "400 Bad Request", "text/plain", b"expected {goals: [...], steer, spread}");
            };
            {
                let mut h = sh.history();
                let any = g.goals.iter().any(Option::is_some);
                h.meta.goals = if any { Some(g.goals.clone()) } else { None };
                sh.meta_version.fetch_add(1, Ordering::SeqCst);
            }
            let wants = g.steer && g.goals.iter().any(Option::is_some);
            let request = if wants {
                Some(FocusRequest::Focus { aspiration: g.goals.iter().map(|v| v.unwrap_or(f64::NAN)).collect(), spread: g.spread })
            } else if sh.steering.load(Ordering::SeqCst) {
                Some(FocusRequest::Clear)
            } else {
                None
            };
            sh.steering.store(wants, Ordering::SeqCst);
            if let Some(r) = request {
                *sh.focus.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
            }
            respond(&mut stream, "200 OK", "application/json", b"{\"ok\":true}")
        }
        ("GET", "/api/meta") => {
            let body = serde_json::to_vec(&sh.history().meta)?;
            respond(&mut stream, "200 OK", "application/json", &body)
        }
        ("GET", "/api/history") => {
            #[derive(Serialize)]
            struct History<'a> {
                state: &'static str,
                run_id: u64,
                frames: Vec<&'a Frame>,
            }
            let body = {
                let h = sh.history();
                serde_json::to_vec(&History {
                    state: sh.state().as_str(),
                    run_id: sh.run_id.load(Ordering::SeqCst),
                    frames: h.frames().collect(),
                })?
            };
            respond(&mut stream, "200 OK", "application/json", &body)
        }
        ("GET", "/api/compare") => {
            let body = {
                let comparisons = sh.comparisons();
                let runs: Vec<NamedRun<'_>> = comparisons.iter().map(|(n, h)| NamedRun::new(n, h)).collect();
                serde_json::to_vec(&runs)?
            };
            respond(&mut stream, "200 OK", "application/json", &body)
        }
        ("GET", "/api/events") => events(stream, sh),
        ("POST", "/api/stop") | ("POST", "/api/pause") | ("POST", "/api/resume") => {
            match req.path.as_str() {
                "/api/stop" => request_stop(sh),
                "/api/pause" => set_paused(sh, true),
                _ => set_paused(sh, false),
            }
            let body = format!(
                "{{\"state\":\"{}\",\"paused\":{}}}",
                sh.state().as_str(),
                sh.paused.load(Ordering::SeqCst)
            );
            respond(&mut stream, "200 OK", "application/json", body.as_bytes())
        }
        ("GET", _) | ("POST", _) => respond(&mut stream, "404 Not Found", "text/plain", b"Not Found"),
        _ => respond(&mut stream, "405 Method Not Allowed", "text/plain", b"Method Not Allowed"),
    }
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) -> io::Result<()> {
    respond_with(stream, status, content_type, "", body)
}

fn respond_with(stream: &mut TcpStream, status: &str, content_type: &str, extra_headers: &str, body: &[u8]) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n{extra_headers}Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// The newest frame as JSON, serialized once per `version` however many tabs are open.
fn latest_frame_json(sh: &Shared, version: u64) -> io::Result<Option<Arc<Vec<u8>>>> {
    let mut cache = sh.latest_json.lock().unwrap_or_else(|e| e.into_inner());
    if cache.0 != version {
        let json = sh.history().latest().map(serde_json::to_vec).transpose()?;
        *cache = (version, json.map(Arc::new));
    }
    Ok(cache.1.clone())
}

/// Stream events until the tab closes (a write fails) or the dashboard closes.
fn events(mut stream: TcpStream, sh: &Shared) -> io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n\
          X-Accel-Buffering: no\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()?;
    // Start "behind" so a client always gets the newest frame on connect.
    let mut seen_version = u64::MAX;
    let mut seen_run = sh.run_id.load(Ordering::SeqCst);
    let mut seen_meta = u64::MAX;
    let mut seen_compare = u64::MAX;
    let mut seen_paused: Option<bool> = None;
    let mut seen_state: Option<RunState> = None;
    let mut last_write = Instant::now();
    while !sh.closing.load(Ordering::SeqCst) {
        let mut out = Vec::new();
        let run = sh.run_id.load(Ordering::SeqCst);
        if run != seen_run {
            seen_run = run;
            out.extend(format!("event: reset\ndata: {{\"run_id\":{run}}}\n\n").into_bytes());
        }
        let compare_version = sh.compare_version.load(Ordering::SeqCst);
        if compare_version != seen_compare {
            seen_compare = compare_version;
            let count = sh.comparisons().len();
            out.extend(format!("event: compare\ndata: {{\"count\":{count}}}\n\n").into_bytes());
        }
        let paused = sh.paused.load(Ordering::SeqCst);
        if seen_paused != Some(paused) {
            seen_paused = Some(paused);
            out.extend(format!("event: paused\ndata: {{\"paused\":{paused}}}\n\n").into_bytes());
        }
        let meta_version = sh.meta_version.load(Ordering::SeqCst);
        if meta_version != seen_meta {
            seen_meta = meta_version;
            out.extend_from_slice(b"event: meta\ndata: ");
            out.extend(serde_json::to_vec(&sh.history().meta)?);
            out.extend_from_slice(b"\n\n");
        }
        let version = sh.version.load(Ordering::SeqCst);
        if version != seen_version {
            seen_version = version;
            if let Some(json) = latest_frame_json(sh, version)? {
                out.extend_from_slice(b"event: frame\ndata: ");
                out.extend_from_slice(&json);
                out.extend_from_slice(b"\n\n");
            }
        }
        let state = sh.state();
        if seen_state != Some(state) {
            seen_state = Some(state);
            out.extend(format!("event: state\ndata: {{\"state\":\"{}\"}}\n\n", state.as_str()).into_bytes());
        }
        if out.is_empty() && last_write.elapsed() > HEARTBEAT {
            out.extend_from_slice(b": keep-alive\n\n");
        }
        if !out.is_empty() {
            stream.write_all(&out)?;
            stream.flush()?;
            last_write = Instant::now();
        }
        thread::sleep(PUSH_INTERVAL);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Problem;
    use crate::gatypes::{Real, SolutionDataTypes};
    use crate::genetic_algorithms_v2::{ExecutionMode, NSGAII};

    fn get(dash: &Dashboard, method: &str, path: &str, extra: &str) -> String {
        let mut s = TcpStream::connect(dash.addr()).unwrap();
        write!(s, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{extra}\r\n").unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn serves_the_page_api_and_stop() {
        let problem = Arc::new(Problem::new(
            2, 2, None, None, Some(vec![-1, -1]),
            vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))); 2],
            |x| vec![x[0], 1.0 - x[0] + x[1]],
        ));
        let dash = Dashboard::start(RunMeta::from_problem(&problem).title("test"), "127.0.0.1", 0).unwrap();
        assert_eq!(dash.state(), RunState::Waiting);

        let page = get(&dash, "GET", "/", "");
        assert!(page.starts_with("HTTP/1.1 200"));
        assert!(page.contains("window.PugglesPlot"));
        assert!(get(&dash, "GET", "/api/meta", "").contains("\"title\":\"test\""));
        assert!(get(&dash, "GET", "/nope", "").starts_with("HTTP/1.1 404"));
        assert!(get(&dash, "POST", "/api/stop", "Origin: http://evil.example\r\n").starts_with("HTTP/1.1 403"));

        let mut ga = NSGAII::new(Arc::clone(&problem), 10, ExecutionMode::Sequential).with_seed(4);
        let mut stopped_at = None;
        ga.run_with(100_000, |ga| {
            if ga.get_nfe() >= 100 && stopped_at.is_none() {
                let reply = get(&dash, "POST", "/api/stop", "Origin: http://127.0.0.1\r\n");
                assert!(reply.contains("stopping"), "{reply}");
                stopped_at = Some(ga.get_nfe());
            }
            dash.record(ga)
        });
        dash.finish();
        assert_eq!(dash.state(), RunState::Stopped);
        assert!(ga.get_nfe() <= stopped_at.unwrap() + 10, "Stop must end the run at the next generation");

        let hist = get(&dash, "GET", "/api/history", "");
        assert!(hist.contains("\"state\":\"stopped\""));
        assert!(hist.contains("\"F\":[["));
        assert_eq!(dash.history().meta.algorithm.as_deref(), Some("NSGAII"));
        let exec = dash.history().meta.execution.expect("execution info recorded");
        assert_eq!(exec.effective, ExecutionMode::Sequential);
        assert_eq!(exec.cpu_threads, Some(1));
        assert!(get(&dash, "GET", "/api/meta", "").contains("\"execution\":{\"requested\":\"Sequential\""));
    }

    #[test]
    fn streams_frames_over_sse() {
        let dash = Dashboard::start(RunMeta::default(), "127.0.0.1", 0).unwrap();
        let mut s = TcpStream::connect(dash.addr()).unwrap();
        write!(s, "GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        dash.record_frame(Frame { nfe: 10, objectives: vec![vec![1.0, 2.0]], ..Default::default() });
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        let deadline = Instant::now() + Duration::from_secs(5);
        while !String::from_utf8_lossy(&buf).contains("event: frame") && Instant::now() < deadline {
            let n = s.read(&mut chunk).unwrap();
            buf.extend_from_slice(&chunk[..n]);
        }
        let text = String::from_utf8_lossy(&buf);
        assert!(text.contains("text/event-stream"));
        assert!(text.contains("event: frame\ndata: {\"gen\":1,\"nfe\":10"), "{text}");
        assert!(text.contains("event: meta\ndata: {"), "meta is sent on connect: {text}");
        dash.close();
    }

    fn toy_problem() -> Arc<Problem> {
        Arc::new(Problem::new(
            2, 2, None, None, Some(vec![-1, -1]),
            vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))); 2],
            |x| vec![x[0], 1.0 - x[0] + x[1]],
        ))
    }

    #[test]
    fn pause_holds_the_run_until_resume() {
        let problem = toy_problem();
        let dash = Dashboard::start(RunMeta::from_problem(&problem), "127.0.0.1", 0).unwrap();
        dash.pause();
        let d = dash.clone();
        let p = Arc::clone(&problem);
        let worker = thread::spawn(move || {
            let mut ga = NSGAII::new(p, 10, ExecutionMode::Sequential).with_seed(1);
            ga.run_with(200, |ga| d.record(ga));
            d.finish();
            ga.get_nfe()
        });
        thread::sleep(Duration::from_millis(300));
        assert_eq!(dash.state(), RunState::Paused);
        assert_eq!(dash.history().len(), 1, "paused after the first generation");
        assert!(get(&dash, "POST", "/api/resume", "").contains("\"paused\":false"));
        assert_eq!(worker.join().unwrap(), 200);
        assert_eq!(dash.state(), RunState::Done);
    }

    #[test]
    fn keep_runs_turns_earlier_runs_into_comparisons() {
        let problem = toy_problem();
        let dash = Dashboard::start(RunMeta::from_problem(&problem), "127.0.0.1", 0).unwrap();
        dash.keep_runs(true);
        for seed in 0..2 {
            let mut ga = NSGAII::new(Arc::clone(&problem), 10, ExecutionMode::Sequential).with_seed(seed);
            ga.run_with(100, |ga| dash.record(ga));
            let mut ga3 = crate::nsga3::NSGAIII::new(Arc::clone(&problem), 12, 4, ExecutionMode::Sequential).with_seed(seed);
            ga3.run_with(100, |ga| dash.record(ga));
        }
        let names: Vec<String> = dash.comparisons().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["run 1 · NSGAII", "run 2 · NSGAIII", "run 3 · NSGAII"]);
        assert_eq!(dash.history().meta.run_label.as_deref(), Some("run 4 · NSGAIII"));
        assert!(get(&dash, "GET", "/api/compare", "").contains("\"name\":\"run 2 · NSGAIII\""));
        let html = dash.to_html();
        assert!(html.contains("\"comparisons\":[{\"name\":\"run 1 · NSGAII\""));

        let latest = dash.history().latest().cloned().unwrap();
        let timing = latest.timing.expect("NSGA-II frames carry phase timings");
        assert!(timing.generation.evaluation >= 0.0 && timing.record >= 0.0);
    }

    fn post_json(dash: &Dashboard, path: &str, body: &str) -> String {
        get(dash, "POST", path, &format!("Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()))
    }

    #[test]
    fn token_is_required_when_set() {
        let dash = Dashboard::start(RunMeta::default(), "127.0.0.1", 0).unwrap();
        assert!(dash.token().is_none() && !dash.url().contains("token"));
        let token = dash.require_token(Some("s3cret".into()));
        assert!(dash.url().ends_with("/?token=s3cret"));
        assert!(get(&dash, "GET", "/api/meta", "").starts_with("HTTP/1.1 403"));
        let page = get(&dash, "GET", &format!("/?token={token}"), "");
        assert!(page.starts_with("HTTP/1.1 200"));
        let cookie = format!("puggles_token_{}=s3cret", dash.addr().port());
        assert!(page.contains(&format!("Set-Cookie: {cookie}; Path=/; HttpOnly; SameSite=Strict")));
        assert!(get(&dash, "GET", "/api/meta", &format!("Cookie: {cookie}\r\n")).starts_with("HTTP/1.1 200"));
        assert!(get(&dash, "GET", "/api/meta", "Cookie: puggles_token_1=nope\r\n").starts_with("HTTP/1.1 403"));
        // Bound beyond loopback, a token is on by default.
        let open = Dashboard::start(RunMeta::default(), "0.0.0.0", 0).unwrap();
        assert!(open.token().is_some_and(|t| t.len() == 32));
    }

    #[test]
    fn goals_reach_meta_and_steer_nsga3() {
        let problem = toy_problem();
        let dash = Dashboard::start(RunMeta::from_problem(&problem), "127.0.0.1", 0).unwrap();
        assert!(post_json(&dash, "/api/goals", "{\"goals\":[0.2,null],\"steer\":true,\"spread\":0.3}").contains("ok"));
        assert_eq!(dash.history().meta.goals, Some(vec![Some(0.2), None]));
        let req = dash.take_focus_request().expect("steering request queued");
        match req {
            FocusRequest::Focus { aspiration, spread } => {
                assert_eq!(aspiration[0], 0.2);
                assert!(aspiration[1].is_nan());
                assert_eq!(spread, 0.3);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(dash.history().meta.steerable, Some(true));
        post_json(&dash, "/api/goals", "{\"goals\":[null,null]}");
        assert_eq!(dash.take_focus_request(), Some(FocusRequest::Clear));
        assert_eq!(dash.history().meta.goals, None);
        assert!(post_json(&dash, "/api/goals", "not json").starts_with("HTTP/1.1 400"));

        let mut ga = crate::nsga3::NSGAIII::new(Arc::clone(&problem), 12, 6, ExecutionMode::Sequential).with_seed(1);
        post_json(&dash, "/api/goals", "{\"goals\":[0.1,0.9],\"steer\":true}");
        ga.run_with(300, |ga| {
            dash.apply_focus(ga);
            dash.record(ga)
        });
        assert!(ga.reference_points() != crate::nsga3::das_dennis(2, 6).as_slice(), "focus applied");
        let events = dash.history().meta.events;
        assert_eq!(events.len(), 1);
        assert!(events[0].label.starts_with("steered to goals"));
    }

    #[test]
    fn columns_thinning_and_groups() {
        let problem = toy_problem();
        let dash = Dashboard::start(RunMeta::from_problem(&problem), "127.0.0.1", 0).unwrap();
        dash.set_columns(vec!["sum".into(), "first".into()], |x| vec![x.iter().sum(), x[0]]);
        dash.set_max_points(7);
        dash.set_group("baseline");
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential).with_seed(2);
        ga.run_with(200, |ga| dash.record(ga));
        let h = dash.history();
        let f = h.latest().unwrap();
        let extra = f.extra.as_ref().unwrap();
        assert_eq!(extra["sum"].len(), f.variables.len());
        assert!((extra["sum"][0] - f.variables[0].iter().sum::<f64>()).abs() < 1e-12);
        assert_eq!(f.population_objectives.len(), 7);
        assert_eq!(f.population_total, Some(20));
        assert_eq!(h.meta.group.as_deref(), Some("baseline"));

        // A group named between runs applies to the next run, not the finished one.
        dash.keep_runs(true);
        dash.set_group("tuned");
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential).with_seed(3);
        ga.run_with(200, |ga| dash.record(ga));
        assert_eq!(dash.comparisons()[0].1.meta.group.as_deref(), Some("baseline"));
        assert_eq!(dash.history().meta.group.as_deref(), Some("tuned"));
    }
}
