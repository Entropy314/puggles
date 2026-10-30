//! Recording a run generation by generation, and the `.pgrun.json` run-file format.
//!
//! A run file is `{"format": "puggles-run", "version": 1, "meta": {..}, "frames": [..]}`; the
//! Python package reads and writes the same format. Each frame holds the archive's objectives
//! (`F`), variables (`X`) and feasibility (`ok`), plus the population's objectives (`PF`) and
//! feasibility (`Pok`). Non-finite numbers are written as `null` and read back as NaN.

use super::html::standalone_html;
use super::PlotError;
use crate::core::{Problem, Solution};
use crate::gatypes::SolutionDataTypes;
use crate::genetic_algorithms_v2::{ExecutionInfo, GenerationTimes, NSGAII};
use crate::nsga3::NSGAIII;
use serde::{Deserialize, Deserializer, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const RUN_FORMAT: &str = "puggles-run";
pub const RUN_VERSION: u32 = 1;

/// What a recorder needs from an optimizer. Implemented for [`NSGAII`] and [`NSGAIII`].
pub trait RunView {
    fn archive(&self) -> &[Solution];
    fn population(&self) -> &[Solution];
    fn nfe(&self) -> usize;
    fn algorithm(&self) -> &'static str;
    /// Where evaluations run (shown in the dashboard header).
    fn execution(&self) -> Option<ExecutionInfo> {
        None
    }
    /// Solutions evaluated on the GPU so far, if a GPU evaluator is attached.
    fn gpu_evaluations(&self) -> Option<usize> {
        None
    }
    /// Time the last generation spent in each phase.
    fn generation_times(&self) -> Option<GenerationTimes> {
        None
    }
}

impl RunView for NSGAII {
    fn archive(&self) -> &[Solution] {
        self.get_archive()
    }
    fn population(&self) -> &[Solution] {
        &self.population
    }
    fn nfe(&self) -> usize {
        self.get_nfe()
    }
    fn algorithm(&self) -> &'static str {
        "NSGAII"
    }
    fn execution(&self) -> Option<ExecutionInfo> {
        Some(self.execution_info())
    }
    fn gpu_evaluations(&self) -> Option<usize> {
        NSGAII::gpu_evaluations(self)
    }
    fn generation_times(&self) -> Option<GenerationTimes> {
        Some(self.last_generation_times())
    }
}

impl RunView for NSGAIII {
    fn archive(&self) -> &[Solution] {
        self.get_archive()
    }
    fn population(&self) -> &[Solution] {
        &self.population
    }
    fn nfe(&self) -> usize {
        self.get_nfe()
    }
    fn algorithm(&self) -> &'static str {
        "NSGAIII"
    }
    fn execution(&self) -> Option<ExecutionInfo> {
        Some(self.execution_info())
    }
    fn generation_times(&self) -> Option<GenerationTimes> {
        Some(self.last_generation_times())
    }
}

/// A variable's `(kind, lower, upper)`; kind is `real`, `integer` or `binary`.
pub type VariableType = (String, Option<f64>, Option<f64>);

/// Describes the problem to the UI. Every field is optional; the UI infers what's missing.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct RunMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_obj: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_var: Option<usize>,
    /// Per objective: -1 = minimize, 1 = maximize.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<Vec<i8>>,
    /// Per variable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variable_types: Option<Vec<VariableType>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective_names: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variable_names: Option<Vec<String>>,
    /// Where evaluations run; filled in from the optimizer on the first recorded generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionInfo>,
    /// Runs with the same group (e.g. "NSGA-II, pop 100") are summarized together across seeds;
    /// the page falls back to the algorithm name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// Aspiration levels per objective (`None` = no preference), set from the page's Goals panel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub goals: Option<Vec<Option<f64>>>,
    /// Things that happened during the run, e.g. steering changes, drawn on the convergence chart.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<RunEvent>,
    /// Whether the run reads goal changes back (the page offers "Steer" only then).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steerable: Option<bool>,
    /// What to call the current run next to compared ones (set by a `keep_runs` dashboard).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_label: Option<String>,
    /// Points on the true Pareto front, in objective space, when it's known (see
    /// [`reference`](super::reference)). The page draws it and plots IGD against it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_front: Option<Vec<Vec<f64>>>,
}

impl RunMeta {
    /// Objective count and directions, and each variable's type and bounds.
    pub fn from_problem(problem: &Problem) -> Self {
        let types = problem
            .solution_data_types
            .iter()
            .map(|t| match t {
                SolutionDataTypes::Real(r) => ("real".to_string(), r.lower_bound, r.upper_bound),
                SolutionDataTypes::Integer(i) => (
                    "integer".to_string(),
                    i.lower_bound.map(|v| v as f64),
                    i.upper_bound.map(|v| v as f64),
                ),
                SolutionDataTypes::BitBinary(_) => ("binary".to_string(), Some(0.0), Some(1.0)),
            })
            .collect();
        RunMeta {
            n_obj: Some(problem.number_of_objectives),
            n_var: Some(problem.solution_length),
            direction: Some(problem.direction.clone().unwrap_or_else(|| vec![-1; problem.number_of_objectives])),
            variable_types: Some(types),
            ..Default::default()
        }
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn algorithm(mut self, algorithm: impl Into<String>) -> Self {
        self.algorithm = Some(algorithm.into());
        self
    }

    pub fn objective_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.objective_names = Some(names.into_iter().map(Into::into).collect());
        self
    }

    pub fn variable_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.variable_names = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// Summarize this run with others of the same group (across seeds).
    pub fn group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// The true Pareto front, when known, e.g. `reference::dtlz2(5, 6)`.
    pub fn reference_front(mut self, points: Vec<Vec<f64>>) -> Self {
        self.reference_front = Some(points);
        self
    }
}

/// A labelled moment in a run, at an evaluation count.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RunEvent {
    pub nfe: usize,
    pub label: String,
}

/// Seconds a generation spent in each phase, plus the recorder's own cost.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
pub struct Timing {
    #[serde(flatten)]
    pub generation: GenerationTimes,
    /// Copying the generation into the recorder (the dashboard's overhead).
    pub record: f64,
}

/// One recorded generation.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct Frame {
    pub gen: usize,
    pub nfe: usize,
    /// Seconds since the first recorded generation.
    #[serde(default)]
    pub t: f64,
    #[serde(rename = "F", default, deserialize_with = "de_matrix")]
    pub objectives: Vec<Vec<f64>>,
    #[serde(rename = "X", default, deserialize_with = "de_matrix")]
    pub variables: Vec<Vec<f64>>,
    #[serde(default)]
    pub ok: Vec<bool>,
    #[serde(rename = "PF", default, deserialize_with = "de_matrix")]
    pub population_objectives: Vec<Vec<f64>>,
    #[serde(rename = "Pok", default)]
    pub population_ok: Vec<bool>,
    /// Solutions evaluated on the GPU so far, when a GPU evaluator is attached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_evals: Option<usize>,
    /// Constraints each population member violates (only for constrained problems).
    #[serde(rename = "Pcv", default, skip_serializing_if = "Vec::is_empty")]
    pub population_violations: Vec<usize>,
    /// Where this generation's time went.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<Timing>,
    /// Your own per-solution columns (e.g. cost breakdowns), aligned with the archive (`F`/`X`).
    #[serde(rename = "E", default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<std::collections::BTreeMap<String, Vec<f64>>>,
    /// The full population size, when `PF`/`Pok`/`Pcv` hold an even sample of a larger one.
    #[serde(rename = "Pn", default, skip_serializing_if = "Option::is_none")]
    pub population_total: Option<usize>,
}

impl Frame {
    /// A frame from an archive and population; `gen` and `t` are filled in by the recorder.
    pub fn from_solutions(archive: &[Solution], population: &[Solution], nfe: usize) -> Self {
        Frame::from_solutions_sampled(archive, population, nfe, 0)
    }

    /// [`from_solutions`](Self::from_solutions) keeping an even sample of at most `max_points`
    /// population members (0 = all), copying only those — the same result as
    /// [`thin_population`](Self::thin_population) afterwards, without copying the rest first.
    pub fn from_solutions_sampled(archive: &[Solution], population: &[Solution], nfe: usize, max_points: usize) -> Self {
        let keep = sample_indices(population.len(), max_points);
        let pick = |i: usize| &population[keep.as_ref().map_or(i, |k| k[i])];
        let kept = keep.as_ref().map_or(population.len(), Vec::len);
        Frame {
            gen: 0,
            nfe,
            t: 0.0,
            objectives: archive.iter().map(|s| s.objective_fitness_values.to_vec()).collect(),
            variables: archive.iter().map(|s| s.solution.clone()).collect(),
            ok: archive.iter().map(|s| s.feasible).collect(),
            population_objectives: (0..kept).map(|i| pick(i).objective_fitness_values.to_vec()).collect(),
            population_ok: (0..kept).map(|i| pick(i).feasible).collect(),
            gpu_evals: None,
            population_violations: if population.first().is_some_and(|s| s.problem.has_constraints()) {
                (0..kept).map(|i| pick(i).constraint_violation).collect()
            } else {
                Vec::new()
            },
            timing: None,
            extra: None,
            population_total: keep.map(|_| population.len()),
        }
    }

    /// Keep an even sample of at most `max` population members (the archive is untouched).
    pub fn thin_population(&mut self, max: usize) {
        let n = self.population_objectives.len();
        if max == 0 || n <= max {
            return;
        }
        let keep = sample_indices(n, max).expect("n > max");
        let pick = |v: &[bool]| keep.iter().filter_map(|&i| v.get(i).copied()).collect::<Vec<bool>>();
        self.population_objectives = keep.iter().map(|&i| std::mem::take(&mut self.population_objectives[i])).collect();
        self.population_ok = pick(&self.population_ok);
        if !self.population_violations.is_empty() {
            self.population_violations = keep.iter().filter_map(|&i| self.population_violations.get(i).copied()).collect();
        }
        self.population_total = Some(n);
    }

    /// A frame of the optimizer's current generation: solutions, GPU evaluation count, phase
    /// timings, and how long building the frame itself took.
    pub fn from_run<R: RunView + ?Sized>(ga: &R) -> Self {
        Frame::from_run_sampled(ga, 0)
    }

    /// [`from_run`](Self::from_run) keeping at most `max_points` population members (0 = all).
    pub fn from_run_sampled<R: RunView + ?Sized>(ga: &R, max_points: usize) -> Self {
        let started = Instant::now();
        let mut frame = Frame {
            gpu_evals: ga.gpu_evaluations(),
            ..Frame::from_solutions_sampled(ga.archive(), ga.population(), ga.nfe(), max_points)
        };
        frame.timing = ga
            .generation_times()
            .map(|generation| Timing { generation, record: started.elapsed().as_secs_f64() });
        frame
    }
}

/// Which of `n` members an even sample of at most `max` keeps (`None` = all of them).
pub fn sample_indices(n: usize, max: usize) -> Option<Vec<usize>> {
    (max != 0 && n > max).then(|| (0..max).map(|i| i * n / max).collect())
}

/// `null` (how JSON spells a non-finite number) reads back as NaN.
fn de_matrix<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Vec<f64>>, D::Error> {
    let rows: Vec<Vec<Option<f64>>> = Deserialize::deserialize(d)?;
    Ok(rows
        .into_iter()
        .map(|r| r.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        .collect())
}

/// A run, generation by generation.
///
/// Keeps at most `max_frames` frames (default 500). When full it drops every other frame and
/// from then on keeps every 2nd (then 4th, ...) generation, so the whole run stays covered at
/// bounded memory. The newest generation is always available through [`latest`](Self::latest).
#[derive(Clone, Debug)]
pub struct RunHistory {
    pub meta: RunMeta,
    frames: Vec<Frame>,
    /// The newest generation when the stride skipped it.
    pending: Option<Frame>,
    max_frames: usize,
    stride: usize,
    gen: usize,
    last_nfe: Option<usize>,
    t0: Option<Instant>,
}

impl RunHistory {
    pub fn new(meta: RunMeta) -> Self {
        RunHistory {
            meta,
            frames: Vec::new(),
            pending: None,
            max_frames: 500,
            stride: 1,
            gen: 0,
            last_nfe: None,
            t0: None,
        }
    }

    pub fn with_max_frames(mut self, max_frames: usize) -> Self {
        assert!(max_frames >= 2, "max_frames must be at least 2");
        self.max_frames = max_frames;
        self
    }

    /// Record the optimizer's current generation. Use from `run_with`:
    /// `ga.run_with(n, |ga| { history.record(ga); true })`.
    /// Returns `true` when this started a new run (the evaluation count went backwards).
    pub fn record<R: RunView + ?Sized>(&mut self, ga: &R) -> bool {
        self.update_meta(ga);
        self.record_frame(Frame::from_run(ga))
    }

    /// Fill in what the optimizer knows about itself. Returns `true` if anything changed.
    pub(crate) fn update_meta<R: RunView + ?Sized>(&mut self, ga: &R) -> bool {
        self.update_meta_parts(ga.algorithm(), ga.execution())
    }

    /// [`update_meta`](Self::update_meta) from the parts a [`RunView`] would report.
    pub(crate) fn update_meta_parts(&mut self, algorithm: &str, execution: Option<ExecutionInfo>) -> bool {
        let mut changed = false;
        if self.meta.algorithm.as_deref() != Some(algorithm) {
            self.meta.algorithm = Some(algorithm.to_string());
            changed = true;
        }
        if execution.is_some() && self.meta.execution != execution {
            self.meta.execution = execution;
            changed = true;
        }
        changed
    }

    /// Record a generation built elsewhere; its `gen` and `t` are assigned here.
    /// Returns `true` when this started a new run.
    pub fn record_frame(&mut self, mut frame: Frame) -> bool {
        let now = Instant::now();
        let new_run = self.starts_new_run(frame.nfe);
        if new_run {
            *self = RunHistory::new(std::mem::take(&mut self.meta)).with_max_frames(self.max_frames);
        }
        let t0 = *self.t0.get_or_insert(now);
        self.gen += 1;
        self.last_nfe = Some(frame.nfe);
        frame.gen = self.gen;
        frame.t = (now - t0).as_secs_f64();
        if self.gen % self.stride == 0 {
            self.frames.push(frame);
            self.pending = None;
            if self.frames.len() >= self.max_frames {
                let kept: Vec<Frame> = std::mem::take(&mut self.frames).into_iter().step_by(2).collect();
                self.frames = kept;
                self.stride *= 2;
            }
        } else {
            self.pending = Some(frame);
        }
        new_run
    }

    /// Whether a generation at `nfe` evaluations would start a new run (the count went backwards).
    pub fn starts_new_run(&self, nfe: usize) -> bool {
        self.last_nfe.is_some_and(|n| nfe < n)
    }

    /// Append a frame as-is (keeps its `gen` and `t`) — for building a history by hand.
    pub fn push_frame(&mut self, frame: Frame) {
        self.gen = frame.gen;
        self.last_nfe = Some(frame.nfe);
        self.frames.push(frame);
        self.pending = None;
    }

    /// Recorded frames, plus the newest generation if the stride skipped it.
    pub fn frames(&self) -> impl Iterator<Item = &Frame> {
        self.frames.iter().chain(self.pending.iter())
    }

    pub fn len(&self) -> usize {
        self.frames.len() + usize::from(self.pending.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn latest(&self) -> Option<&Frame> {
        self.pending.as_ref().or(self.frames.last())
    }

    // -- files ------------------------------------------------------------------------------

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.file(None)).expect("a run always serializes")
    }

    pub fn from_json(text: &str) -> Result<Self, PlotError> {
        #[derive(Deserialize)]
        struct RunFile {
            format: Option<String>,
            #[serde(default)]
            version: u32,
            #[serde(default)]
            meta: RunMeta,
            #[serde(default)]
            frames: Vec<Frame>,
        }
        let file: RunFile = serde_json::from_str(text)?;
        if file.format.as_deref() != Some(RUN_FORMAT) {
            return Err(PlotError::InvalidRun(format!("format is {:?}, expected {RUN_FORMAT:?}", file.format)));
        }
        if file.version > RUN_VERSION {
            return Err(PlotError::InvalidRun(format!(
                "run-file version {} is newer than this puggles reads ({RUN_VERSION})",
                file.version
            )));
        }
        let mut history = RunHistory::new(file.meta).with_max_frames(file.frames.len().max(2) + 1);
        for frame in file.frames {
            history.push_frame(frame);
        }
        Ok(history)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<PathBuf, PlotError> {
        std::fs::write(path.as_ref(), self.to_json())?;
        Ok(path.as_ref().to_path_buf())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, PlotError> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// A self-contained replay page (the full dashboard, no server needed).
    pub fn to_html(&self) -> String {
        self.to_html_compared(&[])
    }

    /// A replay page with other runs overlaid for comparison (e.g. other seeds or algorithms).
    pub fn to_html_compared(&self, others: &[(String, RunHistory)]) -> String {
        let mut file = self.file(Some("run"));
        file.comparisons = others.iter().map(|(name, h)| NamedRun::new(name, h)).collect();
        standalone_html(&serde_json::to_string(&file).expect("a run always serializes"))
    }

    pub fn save_html(&self, path: impl AsRef<Path>) -> Result<PathBuf, PlotError> {
        std::fs::write(path.as_ref(), self.to_html())?;
        Ok(path.as_ref().to_path_buf())
    }

    pub(crate) fn file(&self, kind: Option<&'static str>) -> RunFileRef<'_> {
        RunFileRef {
            kind,
            format: RUN_FORMAT,
            version: RUN_VERSION,
            meta: &self.meta,
            frames: self.frames().collect(),
            comparisons: Vec::new(),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct RunFileRef<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    format: &'static str,
    version: u32,
    meta: &'a RunMeta,
    frames: Vec<&'a Frame>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) comparisons: Vec<NamedRun<'a>>,
}

/// Another run shown alongside the main one.
#[derive(Serialize)]
pub(crate) struct NamedRun<'a> {
    name: &'a str,
    meta: &'a RunMeta,
    frames: Vec<&'a Frame>,
}

impl<'a> NamedRun<'a> {
    pub(crate) fn new(name: &'a str, history: &'a RunHistory) -> Self {
        NamedRun { name, meta: &history.meta, frames: history.frames().collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gatypes::Real;
    use crate::genetic_algorithms_v2::ExecutionMode;
    use std::sync::Arc;

    fn problem() -> Arc<Problem> {
        Arc::new(Problem::new(
            3,
            2,
            None,
            None,
            Some(vec![-1, -1]),
            (0..3).map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0)))).collect(),
            |x| vec![x[0], 1.0 - x[0].sqrt() + x[1] + x[2]],
        ))
    }

    #[test]
    fn records_every_generation_and_thins_when_full() {
        let p = problem();
        let mut ga = NSGAII::new(Arc::clone(&p), 10, ExecutionMode::Sequential).with_seed(1);
        let mut h = RunHistory::new(RunMeta::from_problem(&p)).with_max_frames(8);
        let mut gens = 0;
        ga.run_with(10 * 41, |ga| {
            h.record(ga);
            gens += 1;
            true
        });
        assert!(h.len() <= 9, "history must stay bounded, got {}", h.len());
        assert_eq!(h.latest().unwrap().gen, gens, "latest generation is always kept");
        assert_eq!(h.meta.algorithm.as_deref(), Some("NSGAII"));
        let gens_seen: Vec<usize> = h.frames().map(|f| f.gen).collect();
        assert!(gens_seen.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn json_round_trip_keeps_frames_and_non_finite_values() {
        let p = problem();
        let mut h = RunHistory::new(RunMeta::from_problem(&p).objective_names(["cost", "risk"]));
        let mut ga = NSGAII::new(Arc::clone(&p), 10, ExecutionMode::Sequential).with_seed(2);
        ga.run_with(60, |ga| {
            h.record(ga);
            true
        });
        let mut frame = h.latest().unwrap().clone();
        frame.objectives[0][0] = f64::INFINITY;
        frame.gen += 1;
        h.push_frame(frame);

        let back = RunHistory::from_json(&h.to_json()).unwrap();
        assert_eq!(back.meta, h.meta);
        assert_eq!(back.len(), h.len());
        assert!(back.latest().unwrap().objectives[0][0].is_nan(), "non-finite becomes null becomes NaN");
        assert!(RunHistory::from_json(r#"{"format":"other"}"#).is_err());
        assert!(RunHistory::from_json(r#"{"format":"puggles-run","version":99}"#).is_err());
    }

    #[test]
    fn a_new_run_resets_the_history() {
        let mut h = RunHistory::new(RunMeta::default());
        assert!(!h.record_frame(Frame { nfe: 100, ..Default::default() }));
        assert!(!h.record_frame(Frame { nfe: 200, ..Default::default() }));
        assert!(h.record_frame(Frame { nfe: 100, ..Default::default() }));
        assert_eq!(h.len(), 1);
        assert_eq!(h.latest().unwrap().gen, 1);
    }
}
