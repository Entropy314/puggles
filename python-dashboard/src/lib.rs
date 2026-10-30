//! puggles-dashboard: the live browser dashboard for puggles runs, as an optional add-on
//! wheel. The core `puggles` wheel has no visual code; it records into this one through the
//! native interface in `puggles::recorder_abi` (see `Dashboard.__puggles_recorder__`).

use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule};
use puggles::plot::{Dashboard, Frame, RunHistory, RunMeta};
use puggles::recorder_abi::{drop_recorder, recorder_for, Recorder, CAPSULE_NAME};
use std::time::Duration;

/// Live browser dashboard for a run. Pass it as `callback=` to `NSGAII.run` / `NSGAIII.run`.
///
/// Args:
///     source: The Problem (for objective directions and variable bounds), or the
///         NSGAII/NSGAIII optimizer. Optional — the page infers what's missing.
///     objective_names: Labels for the objectives (default f1, f2, ...).
///     variable_names: Labels for the decision variables (default x1, x2, ...).
///     title: Page title.
///     open_browser: Open a browser tab now.
///     host, port: Where to serve; port=0 picks a free port.
///     max_frames: Generations kept for replay; longer runs are evenly thinned.
///     reference_front: Points on the true Pareto front, if known (see `reference_front()`);
///         drawn in the objective view and used for the IGD convergence metric.
///     keep_runs: Keep each finished run as a comparison when the next one starts, so several
///         optimizers (seeds, algorithms, CPU vs GPU) run on one dashboard are overlaid.
///     group: Group of the first run, e.g. "NSGA-II, pop 100" (default: the algorithm). Runs in
///         a group are summarized together across seeds; `set_group()` names the next run's.
///     columns: Your own per-solution columns, ``{"name": f(variables) -> float}``, computed
///         for each front solution every recorded generation (table, color, pins, drivers, CSV).
///     token: Require an access token: a string, or True to generate one. On by default when
///         `host` isn't a loopback address (e.g. "0.0.0.0" on a cluster node); `url` includes it.
///     max_points: Population members kept per generation (an even sample beyond; 0 = all).
///
/// The server is native: it keeps serving while the optimizer runs (no GIL contention), and
/// passed as `callback=` the dashboard records straight from the Rust solutions. The run is
/// marked finished in the page when `run()` returns. The page's Stop button ends the run after
/// the current generation. In a script, call `wait()` at the end to keep the page alive.
#[pyclass(name = "Dashboard", module = "puggles_dashboard")]
pub struct PyDashboard {
    pub(crate) inner: Dashboard,
}

#[pymethods]
impl PyDashboard {
    #[new]
    #[pyo3(signature = (
        source = None, *, objective_names = None, variable_names = None, title = None,
        open_browser = true, host = "127.0.0.1", port = 0, max_frames = 500,
        reference_front = None, keep_runs = false, group = None, columns = None, token = None,
        max_points = 2000
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        source: Option<PyObject>,
        objective_names: Option<Vec<String>>,
        variable_names: Option<Vec<String>>,
        title: Option<String>,
        open_browser: bool,
        host: &str,
        port: u16,
        max_frames: usize,
        reference_front: Option<Vec<Vec<f64>>>,
        keep_runs: bool,
        group: Option<String>,
        columns: Option<Bound<'_, pyo3::types::PyDict>>,
        token: Option<Bound<'_, PyAny>>,
        max_points: usize,
    ) -> PyResult<Self> {
        if max_frames < 2 {
            return Err(PyValueError::new_err("max_frames must be at least 2"));
        }
        let mut meta = meta_from_source(py, source.as_ref())?;
        meta.objective_names = objective_names.or(meta.objective_names);
        meta.variable_names = variable_names.or(meta.variable_names);
        meta.title = title.or(meta.title);
        meta.reference_front = reference_front;
        let history = RunHistory::new(meta).with_max_frames(max_frames);
        let inner = Dashboard::start_with(history, host, port).map_err(|e| PyOSError::new_err(e.to_string()))?;
        inner.keep_runs(keep_runs);
        inner.set_max_points(max_points);
        if let Some(g) = group {
            inner.set_group(g);
        }
        if let Some(cols) = columns {
            set_python_columns(&inner, &cols)?;
        }
        if let Some(t) = token {
            if let Ok(text) = t.extract::<String>() {
                inner.require_token(Some(text));
            } else if t.is_truthy()? {
                inner.require_token(None);
            }
        }
        let dash = PyDashboard { inner };
        dash.announce(py, open_browser)?;
        Ok(dash)
    }

    /// Mark a moment on the convergence chart (kept in saved runs), e.g. "raised mutation".
    /// `nfe` defaults to the latest recorded generation's evaluation count.
    #[pyo3(signature = (label, nfe = None))]
    fn add_event(&self, label: String, nfe: Option<usize>) {
        let nfe = nfe.unwrap_or_else(|| self.inner.latest_nfe().unwrap_or(0));
        self.inner.add_event(nfe, label);
    }

    /// Group of the next run that starts (or of the first run), e.g. "NSGA-II, pop 100".
    fn set_group(&self, group: String) {
        self.inner.set_group(group);
    }

    /// The access token, if one is required (it's also in `url`).
    #[getter]
    fn token(&self) -> Option<String> {
        self.inner.token()
    }

    /// Serve a saved run (the JSON text of a .pgrun.json file) for replay.
    #[staticmethod]
    #[pyo3(signature = (run_json, *, open_browser = true, host = "127.0.0.1", port = 0))]
    fn _replay(py: Python<'_>, run_json: &str, open_browser: bool, host: &str, port: u16) -> PyResult<Self> {
        let history = RunHistory::from_json(run_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = Dashboard::replay(history, host, port).map_err(|e| PyOSError::new_err(e.to_string()))?;
        let dash = PyDashboard { inner };
        dash.announce(py, open_browser)?;
        Ok(dash)
    }

    /// A self-contained replay page for a saved run's JSON text.
    #[staticmethod]
    fn _standalone_html(run_json: &str) -> PyResult<String> {
        let history = RunHistory::from_json(run_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(history.to_html())
    }

    #[getter]
    fn url(&self) -> String {
        self.inner.url()
    }

    /// waiting | running | stopping | stopped | done | replay
    #[getter]
    fn state(&self) -> &'static str {
        self.inner.state().as_str()
    }

    /// Number of generations kept for replay.
    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    /// Record one generation; returns False once Stop was pressed. This is the generic
    /// callback form, for when the dashboard is called from your own callback. Solutions are
    /// anything with ``objectives``, ``variables`` and ``feasible`` (``puggles.Solution``).
    fn __call__(
        &self,
        py: Python<'_>,
        archive: Vec<Bound<'_, PyAny>>,
        population: Vec<Bound<'_, PyAny>>,
        nfe: usize,
    ) -> PyResult<bool> {
        let field = |s: &Bound<'_, PyAny>, name: &str| -> PyResult<Vec<f64>> { s.getattr(name)?.extract() };
        let feasible = |s: &Bound<'_, PyAny>| -> PyResult<bool> { s.getattr("feasible")?.extract() };
        let total = population.len();
        let keep = puggles::plot::sample_indices(total, self.inner.max_points());
        let population: Vec<&Bound<'_, PyAny>> = match &keep {
            Some(k) => k.iter().map(|&i| &population[i]).collect(),
            None => population.iter().collect(),
        };
        let frame = Frame {
            nfe,
            objectives: archive.iter().map(|s| field(s, "objectives")).collect::<PyResult<_>>()?,
            variables: archive.iter().map(|s| field(s, "variables")).collect::<PyResult<_>>()?,
            ok: archive.iter().map(feasible).collect::<PyResult<_>>()?,
            population_objectives: population.iter().map(|s| field(s, "objectives")).collect::<PyResult<_>>()?,
            population_ok: population.iter().map(|s| feasible(s)).collect::<PyResult<_>>()?,
            population_total: keep.is_some().then_some(total),
            ..Default::default()
        };
        self.inner.record_frame_without_hold(frame);
        hold_with_gil(py, &self.inner)
    }

    /// The native recorder `NSGAII.run` / `NSGAIII.run` use when this dashboard is their
    /// ``callback`` — so a run records straight from the Rust solutions, without the GIL.
    fn __puggles_recorder__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        PyCapsule::new_with_destructor(
            py,
            SendRecorder(recorder_for(self.inner.clone())),
            Some(CAPSULE_NAME.to_owned()),
            // SAFETY: made by `recorder_for` just above, and the capsule is its only owner.
            |recorder, _| unsafe { drop_recorder(recorder.0) },
        )
    }

    /// Mark the run finished in the page (done automatically when passed as `callback=`).
    fn finish(&self) {
        self.inner.finish();
    }

    /// End the run after the current generation, as the page's Stop button does.
    fn request_stop(&self) {
        self.inner.request_stop();
    }

    /// Hold the run after its next generation (as the page's Pause button does).
    fn pause(&self) {
        self.inner.pause();
    }

    fn resume(&self) {
        self.inner.resume();
    }

    #[getter]
    fn paused(&self) -> bool {
        self.inner.is_paused()
    }

    /// Overlay another run: a saved `.pgrun.json[.gz]` path, or another Dashboard.
    #[pyo3(signature = (source, name = None))]
    fn compare(&self, py: Python<'_>, source: &Bound<'_, PyAny>, name: Option<String>) -> PyResult<()> {
        let (history, default_name) = if let Ok(other) = source.downcast::<PyDashboard>() {
            let h = other.borrow().inner.history(); // compare() needs the whole run
            let n = h.meta.title.clone().unwrap_or_else(|| "run".into());
            (h, n)
        } else {
            let path: std::path::PathBuf = source.extract()?;
            let text = read_run_text(py, &path)?;
            let h = RunHistory::from_json(&text).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let stem = path.file_name().and_then(|f| f.to_str()).unwrap_or("run");
            let stem = stem.trim_end_matches(".gz").trim_end_matches(".json").trim_end_matches(".pgrun");
            (h, stem.to_string())
        };
        let ours = self.inner.meta().n_obj;
        if let (Some(a), Some(b)) = (ours, history.meta.n_obj) {
            if a != b {
                return Err(PyValueError::new_err(format!(
                    "can't compare: that run has {b} objectives, this one has {a}"
                )));
            }
        }
        self.inner.compare_with(name.unwrap_or(default_name), history);
        Ok(())
    }

    /// Names of the runs shown alongside this one.
    #[getter]
    fn comparisons(&self) -> Vec<String> {
        self.inner.comparisons().into_iter().map(|(n, _)| n).collect()
    }

    /// Jupyter: show the live dashboard in the notebook cell.
    fn _repr_html_(&self) -> String {
        format!(
            "<iframe src=\"{}\" style=\"width:100%;height:760px;border:1px solid rgba(127,127,127,.25);border-radius:8px\" title=\"puggles dashboard\"></iframe>",
            self.inner.url()
        )
    }

    /// The recorded run as .pgrun.json text.
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// Save the run: `.pgrun.json`, or gzip-compressed when the name ends in `.gz`.
    /// Replay it with `puggles.dashboard.load(path)` or `python -m puggles.dashboard path`.
    fn save(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<std::path::PathBuf> {
        let json = self.inner.to_json();
        let bytes: Vec<u8> = if path.extension().is_some_and(|e| e == "gz") {
            let compressed = py.import("gzip")?.call_method1("compress", (PyBytes::new(py, json.as_bytes()),))?;
            compressed.extract()?
        } else {
            json.into_bytes()
        };
        std::fs::write(&path, bytes).map_err(|e| PyOSError::new_err(e.to_string()))?;
        Ok(path)
    }

    /// Write a self-contained replay page (with any comparisons) you can share without Python.
    fn export_html(&self, path: std::path::PathBuf) -> PyResult<std::path::PathBuf> {
        self.inner.save_html(&path).map_err(|e| PyOSError::new_err(e.to_string()))
    }

    /// Block until Ctrl+C (or `close()`), keeping the page alive after a script's run.
    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        eprintln!("puggles dashboard: {} — press Ctrl+C to exit", self.inner.url());
        let result = loop {
            if self.inner.is_closed() {
                break Ok(());
            }
            py.allow_threads(|| std::thread::sleep(Duration::from_millis(100)));
            match py.check_signals() {
                Ok(()) => {}
                Err(e) if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) => break Ok(()),
                Err(e) => break Err(e),
            }
        };
        self.inner.close();
        result
    }

    /// Stop serving.
    fn close(&self) {
        self.inner.close();
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(&self, _exc_type: PyObject, _exc: PyObject, _tb: PyObject) {
        self.inner.close();
    }

    fn __repr__(&self) -> String {
        format!(
            "Dashboard(url={:?}, state={:?}, frames={})",
            self.inner.url(),
            self.inner.state().as_str(),
            self.inner.frame_count()
        )
    }
}

impl PyDashboard {
    fn announce(&self, py: Python<'_>, open_browser: bool) -> PyResult<()> {
        eprintln!("puggles dashboard: {}", self.inner.url());
        if open_browser {
            // Python's webbrowser knows the platform's quirks (WSL, $BROWSER, ...).
            py.import("webbrowser")?.call_method1("open", (self.inner.url(),))?;
        }
        Ok(())
    }
}

/// `{"name": f(variables) -> float}` as dashboard columns. Each call takes the GIL; a callable
/// that raises reports its traceback once and yields NaN.
fn set_python_columns(dash: &Dashboard, cols: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
    let mut names = Vec::new();
    let mut funcs: Vec<PyObject> = Vec::new();
    for (k, v) in cols.iter() {
        if !v.is_callable() {
            return Err(PyValueError::new_err(format!("columns[{k}] must be a callable f(variables) -> float")));
        }
        names.push(k.extract::<String>()?);
        funcs.push(v.unbind());
    }
    let reported = std::sync::atomic::AtomicBool::new(false);
    dash.set_columns(names, move |x: &[f64]| {
        Python::with_gil(|py| {
            funcs
                .iter()
                .map(|f| match f.call1(py, (x.to_vec(),)).and_then(|r| r.extract::<f64>(py)) {
                    Ok(v) => v,
                    Err(e) => {
                        if !reported.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            e.print(py);
                        }
                        f64::NAN
                    }
                })
                .collect()
        })
    });
    Ok(())
}

/// A run file's JSON text; gzip is detected by content, not by name.
fn read_run_text(py: Python<'_>, path: &std::path::Path) -> PyResult<String> {
    let raw = std::fs::read(path).map_err(|e| PyOSError::new_err(format!("{}: {e}", path.display())))?;
    let raw = if raw.starts_with(&[0x1f, 0x8b]) {
        py.import("gzip")?.call_method1("decompress", (PyBytes::new(py, &raw),))?.extract::<Vec<u8>>()?
    } else {
        raw
    };
    String::from_utf8(raw).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Points on a benchmark's true Pareto front, for `Dashboard(reference_front=...)`.
///
/// Args:
///     name: "zdt1", "zdt2", "dtlz1", "dtlz2", "dtlz3" or "dtlz4".
///     number_of_objectives: Objective count (ZDT problems have 2).
///     points: Roughly how many points to return.
#[pyfunction]
#[pyo3(signature = (name, number_of_objectives = 2, points = 200))]
pub fn reference_front(name: &str, number_of_objectives: usize, points: usize) -> PyResult<Vec<Vec<f64>>> {
    puggles::plot::reference::by_name(name, number_of_objectives, points).ok_or_else(|| {
        PyValueError::new_err(format!(
            "no known front for {name:?} with {number_of_objectives} objectives \
             (have zdt1, zdt2 [2 objectives], dtlz1-dtlz4)"
        ))
    })
}

/// Metadata from a Problem/GpuProblem, or from an optimizer (its `problem` plus its class name).
/// Read through attributes, so this wheel needs nothing from the core one but its public API.
fn meta_from_source(py: Python<'_>, source: Option<&PyObject>) -> PyResult<RunMeta> {
    let Some(source) = source else { return Ok(RunMeta::default()) };
    let bound = source.bind(py);
    if bound.hasattr("variable_types")? {
        return meta_from_problem(bound);
    }
    if bound.hasattr("get_archive")? {
        let mut meta = match bound.getattr("problem") {
            Ok(problem) if problem.hasattr("variable_types")? => meta_from_problem(&problem)?,
            _ => RunMeta::default(),
        };
        meta.algorithm = Some(bound.get_type().name()?.to_string());
        return Ok(meta);
    }
    Err(PyTypeError::new_err("Dashboard source must be a Problem, GpuProblem, NSGAII or NSGAIII"))
}

/// What `RunMeta::from_problem` reports, from a Python `Problem`'s attributes.
fn meta_from_problem(problem: &Bound<'_, PyAny>) -> PyResult<RunMeta> {
    let types: Vec<(String, Option<f64>, Option<f64>)> = problem.getattr("variable_types")?.extract()?;
    Ok(RunMeta {
        n_obj: Some(problem.getattr("number_of_objectives")?.extract()?),
        n_var: Some(problem.getattr("solution_length")?.extract()?),
        direction: Some(problem.getattr("direction")?.extract()?),
        variable_types: Some(types),
        ..Default::default()
    })
}

/// Wait out a pause while holding the GIL: release it in short slices so other Python threads
/// run, and let Ctrl+C through (it stops the run and is re-raised).
fn hold_with_gil(py: Python<'_>, dash: &Dashboard) -> PyResult<bool> {
    loop {
        if let Some(keep_going) = py.allow_threads(|| dash.hold_for(Duration::from_millis(100))) {
            return Ok(keep_going);
        }
        if let Err(e) = py.check_signals() {
            dash.request_stop();
            return Err(e);
        }
    }
}

/// A [`Recorder`] can travel in a capsule: its context is a thread-safe `Dashboard`.
#[repr(transparent)]
struct SendRecorder(Recorder);

// SAFETY: the recorder's vtable functions are thread-safe and its context is a `Dashboard`
// (Send + Sync).
unsafe impl Send for SendRecorder {}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("_recorder_abi_version", puggles::recorder_abi::ABI_VERSION)?;
    m.add_class::<PyDashboard>()?;
    m.add_function(wrap_pyfunction!(reference_front, m)?)?;
    Ok(())
}
