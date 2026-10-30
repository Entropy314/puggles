//! A ggplot-style builder: data + aesthetic mappings + geometric layers + labels/scales/theme,
//! rendered to an interactive HTML figure (see `web/plot.js` for the spec it produces).
//!
//! Data from a slice of solutions has one row per solution and the columns `f1..fm`
//! (objectives), `x1..xn` (variables), `feasible`, and `rank` (non-dominated front, 1 = best).
//! Data from a [`RunHistory`] has one row per recorded generation: `gen`, `nfe`, `time`,
//! `front_size`, and `f1..fm` (best value of each objective on the front). [`Field`] names
//! these columns; [`Field::Col`] names any column of a hand-built [`PlotData`].

use super::history::RunHistory;
use super::html::{open_in_browser, standalone_html};
use super::PlotError;
use crate::core::Solution;
use crate::dominance::{fast_non_dominated_sort, ParetoDominance};
use serde::{Serialize, Serializer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A cell of [`PlotData`]. Non-finite numbers are written as `null`.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Num(f64),
    Bool(bool),
    Text(String),
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Num(v) if v.is_finite() => s.serialize_f64(*v),
            Value::Num(_) => s.serialize_none(),
            Value::Bool(b) => s.serialize_bool(*b),
            Value::Text(t) => s.serialize_str(t),
        }
    }
}

macro_rules! value_from_num {
    ($($t:ty),*) => {$(
        impl From<$t> for Value {
            fn from(v: $t) -> Self {
                Value::Num(v as f64)
            }
        }
    )*};
}
value_from_num!(f64, f32, i64, i32, u64, u32, usize);

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Text(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Text(v)
    }
}

/// A column of the plot's data.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Field {
    /// Objective `i` (0-based), column `f{i+1}`.
    Obj(usize),
    /// Decision variable `j` (0-based), column `x{j+1}`.
    Var(usize),
    Feasible,
    /// Non-dominated front, 1 = Pareto front.
    Rank,
    Gen,
    Nfe,
    /// Seconds since the run started.
    Time,
    FrontSize,
    /// Any column by name.
    Col(String),
}

impl Field {
    pub fn column(&self) -> String {
        match self {
            Field::Obj(i) => format!("f{}", i + 1),
            Field::Var(j) => format!("x{}", j + 1),
            Field::Feasible => "feasible".into(),
            Field::Rank => "rank".into(),
            Field::Gen => "gen".into(),
            Field::Nfe => "nfe".into(),
            Field::Time => "time".into(),
            Field::FrontSize => "front_size".into(),
            Field::Col(name) => name.clone(),
        }
    }
}

impl From<&str> for Field {
    fn from(name: &str) -> Self {
        Field::Col(name.to_string())
    }
}
impl From<String> for Field {
    fn from(name: String) -> Self {
        Field::Col(name)
    }
}

/// Aesthetic mappings: which column drives each visual channel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Aes {
    pub x: Option<Field>,
    pub y: Option<Field>,
    /// Setting `z` makes a point/line layer 3-D.
    pub z: Option<Field>,
    /// Numeric columns get a continuous scale, booleans and text a categorical one.
    pub color: Option<Field>,
}

/// Start an empty set of aesthetic mappings.
pub fn aes() -> Aes {
    Aes::default()
}

impl Aes {
    pub fn x(mut self, f: impl Into<Field>) -> Self {
        self.x = Some(f.into());
        self
    }
    pub fn y(mut self, f: impl Into<Field>) -> Self {
        self.y = Some(f.into());
        self
    }
    pub fn z(mut self, f: impl Into<Field>) -> Self {
        self.z = Some(f.into());
        self
    }
    pub fn color(mut self, f: impl Into<Field>) -> Self {
        self.color = Some(f.into());
        self
    }
    /// `self`, with every mapping set in `over` replacing ours.
    fn merged(&self, over: &Aes) -> Aes {
        Aes {
            x: over.x.clone().or_else(|| self.x.clone()),
            y: over.y.clone().or_else(|| self.y.clone()),
            z: over.z.clone().or_else(|| self.z.clone()),
            color: over.color.clone().or_else(|| self.color.clone()),
        }
    }
}

/// Columns of equal length, in insertion order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlotData {
    columns: Vec<(String, Vec<Value>)>,
}

impl PlotData {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or replace) a column.
    pub fn column<V: Into<Value>>(mut self, name: impl Into<String>, values: impl IntoIterator<Item = V>) -> Self {
        let name = name.into();
        let values: Vec<Value> = values.into_iter().map(Into::into).collect();
        match self.columns.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = values,
            None => self.columns.push((name, values)),
        }
        self
    }

    pub fn get(&self, name: &str) -> Option<&[Value]> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice())
    }

    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|(n, _)| n.as_str())
    }

    pub fn len(&self) -> usize {
        self.columns.first().map_or(0, |(_, v)| v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One row per solution: `f1..`, `x1..`, `feasible`, `rank`.
    pub fn from_solutions(solutions: &[Solution]) -> Self {
        let m = solutions.first().map_or(0, |s| s.objective_fitness_values.len());
        let n = solutions.first().map_or(0, |s| s.solution.len());
        let mut data = PlotData::new();
        for i in 0..m {
            data = data.column(Field::Obj(i).column(), solutions.iter().map(|s| s.objective_fitness_values[i]));
        }
        for j in 0..n {
            data = data.column(Field::Var(j).column(), solutions.iter().map(|s| s.solution[j]));
        }
        data = data.column("feasible", solutions.iter().map(|s| s.feasible));
        let mut rank = vec![0usize; solutions.len()];
        for (r, front) in fast_non_dominated_sort(solutions, &ParetoDominance).iter().enumerate() {
            for &i in front {
                rank[i] = r + 1;
            }
        }
        data.column("rank", rank)
    }

    /// One row per recorded generation: `gen`, `nfe`, `time`, `front_size`, best `f1..`.
    pub fn from_history(history: &RunHistory) -> Self {
        let frames: Vec<_> = history.frames().collect();
        let m = history.meta.n_obj.unwrap_or_else(|| {
            frames.iter().find_map(|f| f.objectives.first().or(f.population_objectives.first())).map_or(0, |r| r.len())
        });
        let dirs = history.meta.direction.clone().unwrap_or_else(|| vec![-1; m]);
        let mut data = PlotData::new()
            .column("gen", frames.iter().map(|f| f.gen))
            .column("nfe", frames.iter().map(|f| f.nfe))
            .column("time", frames.iter().map(|f| f.t))
            .column("front_size", frames.iter().map(|f| f.objectives.len()));
        for i in 0..m {
            let maximize = dirs.get(i).copied().unwrap_or(-1) == 1;
            let best = frames.iter().map(|f| {
                let rows = if f.objectives.is_empty() { &f.population_objectives } else { &f.objectives };
                let vals = rows.iter().filter_map(|r| r.get(i).copied()).filter(|v| v.is_finite());
                let b = if maximize { vals.fold(f64::NEG_INFINITY, f64::max) } else { vals.fold(f64::INFINITY, f64::min) };
                if b.is_finite() { b } else { f64::NAN }
            });
            data = data.column(Field::Obj(i).column(), best);
        }
        data
    }
}

impl From<&[Solution]> for PlotData {
    fn from(s: &[Solution]) -> Self {
        PlotData::from_solutions(s)
    }
}
impl From<&Vec<Solution>> for PlotData {
    fn from(s: &Vec<Solution>) -> Self {
        PlotData::from_solutions(s)
    }
}
impl From<&RunHistory> for PlotData {
    fn from(h: &RunHistory) -> Self {
        PlotData::from_history(h)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Geom {
    Point,
    Line,
    Histogram,
    Parallel(Vec<Field>),
    Splom(Vec<Field>),
}

#[derive(Clone, Debug)]
struct Layer {
    geom: Geom,
    aes: Aes,
    params: Params,
}

#[derive(Serialize, Clone, Debug, Default)]
struct Params {
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    alpha: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bins: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dims: Option<Vec<String>>,
}

#[derive(Serialize, Clone, Debug, Default)]
struct Labs {
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    x: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    y: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    z: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    color: Option<String>,
}

#[derive(Serialize, Clone, Debug, Default)]
struct Scales {
    #[serde(skip_serializing_if = "Option::is_none")]
    x: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    y: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    color: Option<&'static str>,
}

/// A figure under construction. Start with [`ggplot`].
#[derive(Clone, Debug)]
pub struct Plot {
    data: PlotData,
    aes: Aes,
    layers: Vec<Layer>,
    labs: Labs,
    scales: Scales,
    theme: &'static str,
    misuse: Option<String>,
}

/// Start a plot of `data`: a slice of solutions, a [`RunHistory`], or a [`PlotData`].
pub fn ggplot(data: impl Into<PlotData>) -> Plot {
    Plot {
        data: data.into(),
        aes: Aes::default(),
        layers: Vec::new(),
        labs: Labs::default(),
        scales: Scales::default(),
        theme: "auto",
        misuse: None,
    }
}

static SHOW_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl Plot {
    /// Default mappings for every layer (merged with any set earlier).
    pub fn aes(mut self, aes: Aes) -> Self {
        self.aes = self.aes.merged(&aes);
        self
    }

    fn layer(mut self, geom: Geom) -> Self {
        self.layers.push(Layer { geom, aes: Aes::default(), params: Params::default() });
        self
    }

    /// Scatter plot (2-D, or 3-D when `z` is mapped).
    pub fn geom_point(self) -> Self {
        self.layer(Geom::Point)
    }

    /// Line through the points in `x` order (or data order in 3-D).
    pub fn geom_line(self) -> Self {
        self.layer(Geom::Line)
    }

    /// Histogram of `x`; set the bin count with [`bins`](Self::bins).
    pub fn geom_histogram(self) -> Self {
        self.layer(Geom::Histogram)
    }

    /// Parallel coordinates over `dims`, colored by `color` if mapped — for 4+ objectives.
    pub fn geom_parallel<I, F>(self, dims: I) -> Self
    where
        I: IntoIterator<Item = F>,
        F: Into<Field>,
    {
        self.layer(Geom::Parallel(dims.into_iter().map(Into::into).collect()))
    }

    /// Scatter-plot matrix over `dims`.
    pub fn geom_splom<I, F>(self, dims: I) -> Self
    where
        I: IntoIterator<Item = F>,
        F: Into<Field>,
    {
        self.layer(Geom::Splom(dims.into_iter().map(Into::into).collect()))
    }

    fn last_layer(&mut self, what: &str) -> Option<&mut Layer> {
        if self.layers.is_empty() && self.misuse.is_none() {
            self.misuse = Some(format!("{what}() applies to the latest geom_*, but none was added yet"));
        }
        self.layers.last_mut()
    }

    /// Mappings for the latest layer only, overriding the plot's (ggplot's `geom_point(aes(..))`).
    pub fn layer_aes(mut self, aes: Aes) -> Self {
        if let Some(l) = self.last_layer("layer_aes") {
            l.aes = l.aes.merged(&aes);
        }
        self
    }

    /// Marker size in pixels, for the latest layer.
    pub fn size(mut self, px: f64) -> Self {
        if let Some(l) = self.last_layer("size") {
            l.params.size = Some(px);
        }
        self
    }

    /// Opacity in `[0, 1]`, for the latest layer.
    pub fn alpha(mut self, alpha: f64) -> Self {
        if let Some(l) = self.last_layer("alpha") {
            l.params.alpha = Some(alpha.clamp(0.0, 1.0));
        }
        self
    }

    /// Line width in pixels, for the latest layer.
    pub fn width(mut self, px: f64) -> Self {
        if let Some(l) = self.last_layer("width") {
            l.params.width = Some(px);
        }
        self
    }

    /// Bin count, for the latest (histogram) layer.
    pub fn bins(mut self, bins: usize) -> Self {
        if let Some(l) = self.last_layer("bins") {
            l.params.bins = Some(bins);
        }
        self
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.labs.title = Some(title.into());
        self
    }
    pub fn xlab(mut self, label: impl Into<String>) -> Self {
        self.labs.x = Some(label.into());
        self
    }
    pub fn ylab(mut self, label: impl Into<String>) -> Self {
        self.labs.y = Some(label.into());
        self
    }
    pub fn zlab(mut self, label: impl Into<String>) -> Self {
        self.labs.z = Some(label.into());
        self
    }
    /// Title of the color legend / color bar.
    pub fn color_lab(mut self, label: impl Into<String>) -> Self {
        self.labs.color = Some(label.into());
        self
    }

    pub fn scale_x_log10(mut self) -> Self {
        self.scales.x = Some("log");
        self
    }
    pub fn scale_y_log10(mut self) -> Self {
        self.scales.y = Some("log");
        self
    }
    /// Single-hue light-to-dark color scale (the default for numeric colors).
    pub fn scale_color_sequential(mut self) -> Self {
        self.scales.color = Some("sequential");
        self
    }
    /// Two-hue scale around a neutral midpoint, for signed values.
    pub fn scale_color_diverging(mut self) -> Self {
        self.scales.color = Some("diverging");
        self
    }
    /// Treat a numeric color column (e.g. `Rank`) as categories.
    pub fn scale_color_categorical(mut self) -> Self {
        self.scales.color = Some("categorical");
        self
    }

    pub fn theme_light(mut self) -> Self {
        self.theme = "light";
        self
    }
    pub fn theme_dark(mut self) -> Self {
        self.theme = "dark";
        self
    }
    /// Follow the viewer's system setting (the default).
    pub fn theme_auto(mut self) -> Self {
        self.theme = "auto";
        self
    }

    // -- output -----------------------------------------------------------------------------

    /// The plot spec as JSON (what the page renders). Validates every layer.
    pub fn to_json(&self) -> Result<String, PlotError> {
        Ok(serde_json::to_string(&self.spec()?)?)
    }

    /// A self-contained HTML page with the figure.
    pub fn to_html(&self) -> Result<String, PlotError> {
        #[derive(Serialize)]
        struct Embedded<'a> {
            kind: &'static str,
            spec: Spec<'a>,
        }
        Ok(standalone_html(&serde_json::to_string(&Embedded { kind: "plot", spec: self.spec()? })?))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf, PlotError> {
        let html = self.to_html()?;
        std::fs::write(path.as_ref(), html)?;
        Ok(path.as_ref().to_path_buf())
    }

    /// Write the figure to a temporary file and open it in the browser. Returns the file path.
    pub fn show(&self) -> Result<PathBuf, PlotError> {
        let n = SHOW_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("puggles-plot-{}-{n}.html", std::process::id()));
        self.save(&path)?;
        open_in_browser(&path.to_string_lossy())?;
        Ok(path)
    }

    fn spec(&self) -> Result<Spec<'_>, PlotError> {
        if let Some(m) = &self.misuse {
            return Err(PlotError::InvalidLayer(m.clone()));
        }
        if self.layers.is_empty() {
            return Err(PlotError::InvalidLayer("add at least one geom_* layer".into()));
        }
        let mut used: BTreeMap<String, &[Value]> = BTreeMap::new();
        let mut resolve = |f: &Option<Field>| -> Result<Option<String>, PlotError> {
            let Some(f) = f else { return Ok(None) };
            let name = f.column();
            let values = self.data.get(&name).ok_or_else(|| self.unknown(f, &name))?;
            used.insert(name.clone(), values);
            Ok(Some(name))
        };
        let mut layers = Vec::with_capacity(self.layers.len());
        for (k, layer) in self.layers.iter().enumerate() {
            let a = self.aes.merged(&layer.aes);
            let geom = match &layer.geom {
                Geom::Point => "point",
                Geom::Line => "line",
                Geom::Histogram => "histogram",
                Geom::Parallel(_) => "parallel",
                Geom::Splom(_) => "splom",
            };
            let need = |ok: bool, what: &str| {
                if ok { Ok(()) } else { Err(PlotError::InvalidLayer(format!("layer {} (geom_{geom}) needs {what}", k + 1))) }
            };
            let mut params = layer.params.clone();
            match &layer.geom {
                Geom::Point | Geom::Line => need(a.x.is_some() && a.y.is_some(), "x and y mapped in aes()")?,
                Geom::Histogram => need(a.x.is_some(), "x mapped in aes()")?,
                Geom::Parallel(dims) | Geom::Splom(dims) => {
                    need(dims.len() >= 2, "at least two dimensions")?;
                    let mut names = Vec::with_capacity(dims.len());
                    for d in dims {
                        names.push(resolve(&Some(d.clone()))?.expect("dimension is set"));
                    }
                    params.dims = Some(names);
                }
            }
            let (x, y, z) = match &layer.geom {
                Geom::Point | Geom::Line => (resolve(&a.x)?, resolve(&a.y)?, resolve(&a.z)?),
                Geom::Histogram => (resolve(&a.x)?, None, None),
                Geom::Parallel(_) | Geom::Splom(_) => (None, None, None),
            };
            let color = resolve(&a.color)?;
            layers.push(SpecLayer { geom, aes: SpecAes { x, y, z, color }, params });
        }
        Ok(Spec {
            version: 1,
            data: SpecData { columns: used },
            layers,
            labs: &self.labs,
            scales: &self.scales,
            theme: self.theme,
        })
    }

    fn unknown(&self, field: &Field, name: &str) -> PlotError {
        let available: Vec<&str> = self.data.column_names().collect();
        let shown = if available.len() > 14 {
            format!("{}, ... ({} columns)", available[..12].join(", "), available.len())
        } else {
            available.join(", ")
        };
        PlotError::UnknownField(format!("{field:?} (column \"{name}\") is not in the data; it has: {shown}"))
    }
}

#[derive(Serialize)]
struct Spec<'a> {
    version: u32,
    data: SpecData<'a>,
    layers: Vec<SpecLayer>,
    labs: &'a Labs,
    scales: &'a Scales,
    theme: &'static str,
}

#[derive(Serialize)]
struct SpecData<'a> {
    columns: BTreeMap<String, &'a [Value]>,
}

#[derive(Serialize)]
struct SpecLayer {
    geom: &'static str,
    aes: SpecAes,
    params: Params,
}

#[derive(Serialize)]
struct SpecAes {
    #[serde(skip_serializing_if = "Option::is_none")]
    x: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    y: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    z: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    color: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::Field::*;
    use crate::core::Problem;
    use crate::gatypes::{Real, SolutionDataTypes};
    use crate::genetic_algorithms_v2::{ExecutionMode, NSGAII};
    use crate::plot::{RunHistory, RunMeta};
    use std::sync::Arc;

    fn front() -> (Vec<Solution>, RunHistory) {
        let problem = Arc::new(Problem::new(
            3, 2, None, None, Some(vec![-1, -1]),
            vec![SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))); 3],
            |x| vec![x[0], 1.0 - x[0].sqrt() + x[1] + x[2]],
        ));
        let mut ga = NSGAII::new(Arc::clone(&problem), 20, ExecutionMode::Sequential).with_seed(9);
        let mut h = RunHistory::new(RunMeta::from_problem(&problem));
        ga.run_with(400, |ga| {
            h.record(ga);
            true
        });
        (ga.population.clone(), h)
    }

    fn spec(p: &Plot) -> serde_json::Value {
        serde_json::from_str(&p.to_json().unwrap()).unwrap()
    }

    #[test]
    fn point_layer_spec_carries_only_used_columns() {
        let (pop, _) = front();
        let p = ggplot(&pop).aes(aes().x(Obj(0)).y(Obj(1)).color(Var(2))).geom_point().size(6.0).title("front");
        let s = spec(&p);
        let cols = s["data"]["columns"].as_object().unwrap();
        assert_eq!(cols.keys().collect::<Vec<_>>(), ["f1", "f2", "x3"]);
        assert_eq!(cols["f1"].as_array().unwrap().len(), pop.len());
        assert_eq!(s["layers"][0]["geom"], "point");
        assert_eq!(s["layers"][0]["aes"]["color"], "x3");
        assert_eq!(s["layers"][0]["params"]["size"], 6.0);
        assert_eq!(s["labs"]["title"], "front");
    }

    #[test]
    fn rank_feasible_and_history_columns() {
        let (pop, history) = front();
        let data = PlotData::from_solutions(&pop);
        let ranks: Vec<f64> = data.get("rank").unwrap().iter().map(|v| match v { Value::Num(n) => *n, _ => 0.0 }).collect();
        assert!(ranks.iter().all(|&r| r >= 1.0) && ranks.contains(&1.0));
        assert!(matches!(data.get("feasible").unwrap()[0], Value::Bool(true)));

        let s = spec(&ggplot(&history).aes(aes().x(Nfe).y(Obj(1))).geom_line());
        let best: Vec<f64> = s["data"]["columns"]["f2"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(best.len(), history.len());
        assert!(best.last() <= best.first(), "a minimized objective's best on the front should not get worse");
    }

    #[test]
    fn layer_aes_overrides_plot_aes_and_dims_resolve() {
        let (pop, _) = front();
        let p = ggplot(&pop)
            .aes(aes().x(Obj(0)).y(Obj(1)))
            .geom_point()
            .geom_point().layer_aes(aes().y(Var(0)))
            .geom_parallel([Obj(0), Obj(1), Var(1)]);
        let s = spec(&p);
        assert_eq!(s["layers"][0]["aes"]["y"], "f2");
        assert_eq!(s["layers"][1]["aes"]["y"], "x1");
        assert_eq!(s["layers"][2]["params"]["dims"], serde_json::json!(["f1", "f2", "x2"]));
    }

    #[test]
    fn invalid_plots_are_errors_not_panics() {
        let (pop, _) = front();
        let unknown = ggplot(&pop).aes(aes().x(Obj(0)).y(Obj(5))).geom_point().to_json().unwrap_err();
        assert!(matches!(unknown, PlotError::UnknownField(ref m) if m.contains("f6") && m.contains("f1")), "{unknown}");
        assert!(matches!(ggplot(&pop).aes(aes().x(Obj(0))).geom_point().to_json(), Err(PlotError::InvalidLayer(_))));
        assert!(matches!(ggplot(&pop).to_json(), Err(PlotError::InvalidLayer(_))));
        assert!(matches!(ggplot(&pop).size(3.0).geom_point().to_json(), Err(PlotError::InvalidLayer(_))));
        assert!(matches!(ggplot(&pop).geom_parallel([Obj(0)]).to_json(), Err(PlotError::InvalidLayer(_))));
    }

    #[test]
    fn hand_built_data_and_html() {
        let data = PlotData::new().column("a", [1.0, 2.0, f64::NAN]).column("b", [3, 4, 5]).column("g", ["p", "q", "p"]);
        let p = ggplot(data).aes(aes().x("a").y("b").color("g")).geom_point().scale_color_categorical();
        let s = spec(&p);
        assert_eq!(s["data"]["columns"]["a"][2], serde_json::Value::Null, "NaN is written as null");
        let html = p.to_html().unwrap();
        assert!(html.contains("\"kind\":\"plot\""));
        let path = std::env::temp_dir().join(format!("puggles-test-{}.html", std::process::id()));
        p.save(&path).unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("window.PUGGLES_DATA"));
        let _ = std::fs::remove_file(path);
    }
}
