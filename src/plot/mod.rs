//! Interactive plots and a live dashboard, rendered in the browser (feature `plot`).
//!
//! Everything visual is a small web page (Plotly.js, loaded from a pinned CDN URL) compiled
//! into the crate; Rust only produces data. Three entry points:
//!
//! - [`ggplot`] — a grammar-of-graphics builder for one figure, written to a self-contained
//!   HTML file:
//!   ```no_run
//!   # use puggles::plot::{ggplot, aes, Field::*};
//!   # fn demo(archive: &[puggles::core::Solution]) -> Result<(), puggles::plot::PlotError> {
//!   ggplot(archive)
//!       .aes(aes().x(Obj(0)).y(Obj(1)).color(Var(2)))
//!       .geom_point()
//!       .title("Pareto front")
//!       .show()?;
//!   # Ok(()) }
//!   ```
//! - [`Dashboard`] — a live view of a running optimization on `http://127.0.0.1:<port>/`:
//!   ```no_run
//!   # use puggles::plot::{Dashboard, RunMeta};
//!   # fn demo(mut ga: puggles::genetic_algorithms_v2::NSGAII) -> std::io::Result<()> {
//!   let dash = Dashboard::start(RunMeta::from_problem(&ga.problem), "127.0.0.1", 0)?;
//!   dash.open_browser();
//!   ga.run_with(20_000, |ga| dash.record(ga));
//!   dash.finish();
//!   dash.wait(); // keep serving until the process is interrupted
//!   # Ok(()) }
//!   ```
//! - [`RunHistory`] — record a run without a server, then [`save_html`](RunHistory::save_html)
//!   (offline replay page) or [`save_json`](RunHistory::save_json) (`python -m puggles.dashboard`
//!   reads it).

mod ggplot;
mod history;
mod html;
pub mod reference;
mod server;

pub use ggplot::{aes, ggplot, Aes, Field, Plot, PlotData, Value};
pub use history::{sample_indices, Frame, RunEvent, RunHistory, RunMeta, RunView, Timing, VariableType, RUN_FORMAT, RUN_VERSION};
pub use html::{open_in_browser, standalone_html};
pub use server::{ColumnFn, Columns, Dashboard, FocusRequest, RunState};

use std::fmt;

/// Errors from building, validating, or writing a plot or run file.
#[derive(Debug)]
pub enum PlotError {
    /// An aesthetic or dimension names data the plot doesn't have.
    UnknownField(String),
    /// A layer is missing something it needs (e.g. `geom_point` without `y`).
    InvalidLayer(String),
    /// A run file that isn't one, or is from a newer puggles.
    InvalidRun(String),
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for PlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlotError::UnknownField(m) => write!(f, "unknown field: {m}"),
            PlotError::InvalidLayer(m) => write!(f, "invalid layer: {m}"),
            PlotError::InvalidRun(m) => write!(f, "invalid run file: {m}"),
            PlotError::Io(e) => write!(f, "{e}"),
            PlotError::Json(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PlotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PlotError::Io(e) => Some(e),
            PlotError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PlotError {
    fn from(e: std::io::Error) -> Self {
        PlotError::Io(e)
    }
}

impl From<serde_json::Error> for PlotError {
    fn from(e: serde_json::Error) -> Self {
        PlotError::Json(e)
    }
}
