//! Checkpoint / resume for a running optimization.
//!
//! A `Problem` holds a function pointer (the objective), which can't be serialized, so a
//! checkpoint stores only the evolved **data** — decision variables and evaluation results.
//! On resume you supply the same `Problem`; [`GaState`] rehydrates against it.
//!
//! [`GaState`] derives `serde::{Serialize, Deserialize}`, so pick any format:
//!
//! ```ignore
//! let json = serde_json::to_string(&ga.save_state())?;
//! // ... later ...
//! ga.load_state(serde_json::from_str(&json)?);
//! ```

use crate::core::{Problem, Solution};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// One solution's evolved data, without its `Problem`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SolutionRecord {
    pub solution: Vec<f64>,
    pub objective_fitness_values: Vec<f64>,
    pub constraint_values: Vec<f64>,
    pub constraint_violation: usize,
    pub feasible: bool,
    pub evaluated: bool,
}

impl SolutionRecord {
    pub fn from_solution(s: &Solution) -> Self {
        SolutionRecord {
            solution: s.solution.clone(),
            objective_fitness_values: s.objective_fitness_values.clone(),
            constraint_values: s.constraint_values.clone(),
            constraint_violation: s.constraint_violation,
            feasible: s.feasible,
            evaluated: s.evaluated,
        }
    }

    pub fn into_solution(self, problem: Arc<Problem>) -> Solution {
        Solution {
            problem,
            solution: self.solution,
            objective_fitness_values: self.objective_fitness_values,
            constraint_values: self.constraint_values,
            constraint_violation: self.constraint_violation,
            feasible: self.feasible,
            evaluated: self.evaluated,
        }
    }
}

/// Serializable snapshot of a run: population, archive, and evaluations spent.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GaState {
    pub population: Vec<SolutionRecord>,
    pub archive: Vec<SolutionRecord>,
    pub nfe: usize,
}
