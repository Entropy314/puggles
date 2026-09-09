use crate::gatypes::SolutionDataTypes;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use smallvec::SmallVec;
use std::sync::Arc;

/// Objective/constraint value vectors — inline for the small objective counts (≤ 4) typical of
/// multi-objective work, so a `Solution` clone allocates nothing for them (genes stay a `Vec`).
pub type ObjVec = SmallVec<[f64; 4]>;

/// An invalid-configuration error from a `try_new` constructor.
///
/// The panicking `new` constructors are thin wrappers over these, so Rust callers can keep the
/// terse form while the Python bindings turn the same failure into a clean `ValueError` instead
/// of surfacing a Rust panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(String);

impl ConfigError {
    pub fn new(message: impl Into<String>) -> Self {
        ConfigError(message.into())
    }
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// How a solution vector is structured, which decides how it is generated and varied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Encoding {
    /// Each gene is independent and typed by its `SolutionDataTypes` entry. Crossover and
    /// mutation act gene by gene. The default.
    #[default]
    PerGene,
    /// The solution is a permutation of `0..solution_length`. Genes are not independent — the
    /// whole vector must stay a permutation — so variation uses order crossover and swap
    /// mutation instead of the per-gene operators.
    Permutation,
}

/// Evaluation function discriminant: single-solution or batch.
#[derive(Clone, Debug)]
pub enum EvalFn {
    Single(fn(&Vec<f64>) -> Vec<f64>),
    Batch(fn(&Vec<Vec<f64>>) -> Vec<Vec<f64>>),
}

#[derive(Debug, Clone)]
pub struct Problem {
    pub solution_length: usize,
    pub number_of_objectives: usize,
    pub objective_constraint: Option<Vec<Option<f64>>>, // Upper or Lower bound for the objective function eg. [10, 20]
    pub objective_constraint_operands: Option<Vec<Option<String>>>, // Operands for Greater than or less than the objective constraint eg. ["<", ">"]
    pub direction: Option<Vec<i8>>, // Defaults vector to -1 with length of number_of_objectives eg. [-1, -1]
    pub solution_data_types: Vec<SolutionDataTypes>,     // solution type is a vector of the solution types eg. [BitBinary, Integer(lower_bound:Some(10), upper_bound:Some(20)), Real(lower_bound:Some(1.0), upper_bound:Some(20.0))]
    pub eval_fn: EvalFn,
    /// Decision-variable constraints `g(x) <= 0` (feasible when `<= 0`). Each violated
    /// `g` adds one to `constraint_violation`, so feasible solutions dominate infeasible
    /// ones just like the objective-bound constraints. `None` = unconstrained.
    pub variable_constraints: Option<Vec<fn(&Vec<f64>) -> f64>>,
    /// How the solution vector is structured. Defaults to [`Encoding::PerGene`]; set
    /// [`Encoding::Permutation`] via [`Problem::with_permutation_encoding`] for ordering
    /// problems (TSP, scheduling).
    pub encoding: Encoding,
}

impl Problem {
    /// Fallible constructor. Returns [`ConfigError`] instead of panicking, so callers that
    /// build a `Problem` from untrusted input (notably the Python bindings) can report a clean
    /// error rather than unwinding a panic across the FFI boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        solution_length: usize,
        number_of_objectives: usize,
        objective_constraint: Option<Vec<Option<f64>>>, //number_of_objectives
        objective_constraint_operands: Option<Vec<Option<String>>>, //number_of_objectives
        direction: Option<Vec<i8>>,
        solution_data_types: Vec<SolutionDataTypes>,// Vec of Binary or Integer or Real
        objective_function: fn(&Vec<f64>) -> Vec<f64>
    ) -> Result<Self, ConfigError> {
        if solution_length != solution_data_types.len() {
            return Err(ConfigError::new("solution_length does not match solution_data_types length"));
        }

        // Check if lengths match number_of_objectives
        if let Some(ref constraints) = objective_constraint {
            if constraints.len() != number_of_objectives {
                return Err(ConfigError::new("objective_constraint length does not match number_of_objectives"));
            }
        }

        if let Some(ref operands) = objective_constraint_operands {
            if operands.len() != number_of_objectives {
                return Err(ConfigError::new("objective_constraint_operands length does not match number_of_objectives"));
            }
        }

        // Bounds and operands are only meaningful together — reject half a configuration at
        // construction rather than at the first evaluation.
        match (&objective_constraint, &objective_constraint_operands) {
            (Some(_), None) => {
                return Err(ConfigError::new(
                    "objective_constraint is set but objective_constraint_operands is None — supply both or neither",
                ))
            }
            (None, Some(_)) => {
                return Err(ConfigError::new(
                    "objective_constraint_operands is set but objective_constraint is None — supply both or neither",
                ))
            }
            _ => {}
        }

        if let Some(ref operands) = objective_constraint_operands {
            for op in operands.iter().flatten() {
                if !matches!(op.as_str(), "<" | ">" | "<=" | ">=" | "==" | "!=") {
                    return Err(ConfigError::new(format!(
                        "Invalid operand: {op} (expected one of <, >, <=, >=, ==, !=)"
                    )));
                }
            }
        }

        let direction: Option<Vec<i8>> = direction.or_else(|| Some(vec![-1; number_of_objectives]));

        if let Some(ref dirs) = direction {
            if dirs.len() != number_of_objectives {
                return Err(ConfigError::new("direction length does not match number_of_objectives"));
            }
            if let Some(bad) = dirs.iter().find(|d| **d != 1 && **d != -1) {
                return Err(ConfigError::new(format!(
                    "direction entries must be 1 (maximize) or -1 (minimize), got {bad}"
                )));
            }
        }

        Ok(Problem {
            solution_length,
            number_of_objectives,
            objective_constraint,
            objective_constraint_operands,
            direction,
            solution_data_types,
            variable_constraints: None,
            eval_fn: EvalFn::Single(objective_function),
            encoding: Encoding::PerGene,
        })
    }

    /// Panicking constructor. Use [`Problem::try_new`] to handle invalid configuration as an error.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        solution_length: usize,
        number_of_objectives: usize,
        objective_constraint: Option<Vec<Option<f64>>>, //number_of_objectives
        objective_constraint_operands: Option<Vec<Option<String>>>, //number_of_objectives
        direction: Option<Vec<i8>>,
        solution_data_types: Vec<SolutionDataTypes>,// Vec of Binary or Integer or Real
        objective_function: fn(&Vec<f64>) -> Vec<f64>
    ) -> Self {
        Self::try_new(
            solution_length,
            number_of_objectives,
            objective_constraint,
            objective_constraint_operands,
            direction,
            solution_data_types,
            objective_function,
        )
        .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Switch this problem to permutation encoding: every solution is a permutation of
    /// `0..solution_length`. Builder-style.
    ///
    /// The `solution_data_types` entries are ignored for generation under this encoding (the
    /// value space is fixed by the permutation itself), and variation switches to order
    /// crossover + swap mutation so offspring stay valid permutations.
    pub fn with_permutation_encoding(mut self) -> Self {
        self.encoding = Encoding::Permutation;
        self
    }

    pub fn generate_solution(&self, rng: &mut SmallRng) -> Vec<f64> {
        if self.encoding == Encoding::Permutation {
            let mut perm: Vec<f64> = (0..self.solution_length).map(|i| i as f64).collect();
            perm.shuffle(rng);
            return perm;
        }
        let mut solution: Vec<f64> = Vec::new();
        for solution_type in &self.solution_data_types {
            match solution_type {
                SolutionDataTypes::BitBinary(binary) => {
                    solution.push(binary.generate_value(rng).unwrap() as f64);
                }
                SolutionDataTypes::Integer(integer) => {
                    solution.push(integer.generate_value(rng).unwrap() as f64);
                }
                SolutionDataTypes::Real(real) => {
                    solution.push(real.generate_value(rng).unwrap());
                }
            }
        }
        solution
    }

    /// Attach decision-variable constraints `g(x) <= 0` (feasible when `<= 0`). Builder-style.
    pub fn with_variable_constraints(mut self, constraints: Vec<fn(&Vec<f64>) -> f64>) -> Self {
        self.variable_constraints = Some(constraints);
        self
    }

    /// Whether *any* constraint source is configured — objective bounds or decision-variable
    /// `g(x) <= 0`. The single place that answers this: dominance and every non-dominated sort
    /// gate constraint handling on it, so a problem with only `variable_constraints` is still
    /// sorted feasibility-first.
    pub fn has_constraints(&self) -> bool {
        self.objective_constraint.as_ref().is_some_and(|c| !c.is_empty())
            || self.variable_constraints.as_ref().is_some_and(|g| !g.is_empty())
    }
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub problem: Arc<Problem>,
    pub solution: Vec<f64>, // Derived from Problem.solution_data_types
    pub objective_fitness_values: ObjVec,
    pub constraint_values: ObjVec,
    pub evaluated: bool, // default false
    pub constraint_violation: usize, // default 0
    /// Total magnitude of constraint violation — how *far* outside the feasible region this
    /// solution sits, summed over every violated constraint (0.0 when feasible).
    ///
    /// `constraint_violation` counts how many constraints are broken; this measures by how
    /// much. Deb's constrained-domination breaks a tie between two infeasible solutions on
    /// total violation, which a bare count cannot do.
    pub constraint_violation_magnitude: f64, // default 0.0
    pub feasible: bool
}


impl Solution {
    pub fn new(problem: Arc<Problem>) -> Self {
        // Convenience constructor: non-deterministic. For reproducible runs the GA
        // generates the initial population from its own seeded RNG (see NSGAII).
        let mut rng = SmallRng::from_entropy();
        let solution = problem.generate_solution(&mut rng);
        // create vectore of length number_of_objectives
        let objective_fitness_values: ObjVec = ObjVec::new();
        let constraint_values: ObjVec = ObjVec::new();
        let evaluated: bool = false;
        let constraint_violation = 0;
        let feasible = false;

        Solution {
            problem,
            solution,
            objective_fitness_values,
            constraint_values,
            evaluated,
            constraint_violation,
            constraint_violation_magnitude: 0.0,
            feasible,
        }
    }

    pub fn evaluate_constraints(&mut self) -> Vec<f64> {
        let mut constraint_values: Vec<f64> = Vec::new();
        // Bounds and operands are only meaningful together. Half a configuration used to be
        // skipped silently, dropping every constraint without a word — refuse it instead.
        // `Problem::try_new` rejects this up front; a hand-built struct literal can still
        // reach here, so the guard stays.
        match (&self.problem.objective_constraint, &self.problem.objective_constraint_operands) {
            (Some(_), None) => panic!(
                "objective_constraint is set but objective_constraint_operands is None — \
                 supply both (e.g. Some(vec![Some(\"<\".into())])) or neither"
            ),
            (None, Some(_)) => panic!(
                "objective_constraint_operands is set but objective_constraint is None — \
                 supply both or neither"
            ),
            _ => {}
        }
        if let (Some(constraints), Some(operands)) =
            (&self.problem.objective_constraint, &self.problem.objective_constraint_operands)
        {
            for i in 0..constraints.len() {
                // A `None` bound or operand means "no constraint on this objective" — the
                // whole point of the `Option` elements. Treat it as satisfied.
                let (Some(bound), Some(op)) = (constraints[i], operands[i].as_deref()) else {
                    constraint_values.push(1.0);
                    continue;
                };
                let obj = self.objective_fitness_values[i];
                let satisfied = match op {
                    "<" => obj < bound,
                    ">" => obj > bound,
                    "<=" => obj <= bound,
                    ">=" => obj >= bound,
                    "==" => obj == bound,
                    "!=" => obj != bound,
                    other => panic!("Invalid operand: {}", other),
                };
                constraint_values.push(satisfied as i8 as f64);
            }
        }
        constraint_values
    }

    pub fn calculate_constraint_violation(&mut self) -> usize {
        let mut constraint_violation = 0;
        for constraint_value in &self.constraint_values {
            if *constraint_value == 0.0 {
                constraint_violation += 1;
            }
        }
        // Decision-variable constraints g(x) <= 0: each violated g adds one.
        // Folded here so all evaluation paths (sequential, batch, GPU) count it.
        if let Some(gs) = &self.problem.variable_constraints {
            for g in gs {
                if g(&self.solution) > 0.0 {
                    constraint_violation += 1;
                }
            }
        }
        constraint_violation
    }

    /// Total magnitude of constraint violation: how far outside the feasible region this
    /// solution sits, summed over every violated constraint. 0.0 when feasible.
    ///
    /// Objective bounds contribute their overshoot (`obj - bound` for `<`/`<=`, `bound - obj`
    /// for `>`/`>=`, `|obj - bound|` for `==`). A violated `!=` has no natural distance — the
    /// objective *equals* a value it must not — so it contributes a flat 1.0.
    /// Decision-variable constraints contribute `max(0, g(x))`.
    pub fn calculate_constraint_violation_magnitude(&self) -> f64 {
        let mut total = 0.0;

        if let (Some(constraints), Some(operands)) =
            (&self.problem.objective_constraint, &self.problem.objective_constraint_operands)
        {
            for i in 0..constraints.len() {
                let (Some(bound), Some(op)) = (constraints[i], operands[i].as_deref()) else {
                    continue;
                };
                let obj = self.objective_fitness_values[i];
                total += match op {
                    "<" | "<=" => (obj - bound).max(0.0),
                    ">" | ">=" => (bound - obj).max(0.0),
                    "==" => (obj - bound).abs(),
                    // Equality when inequality was required: no distance exists, count it once.
                    "!=" => if obj == bound { 1.0 } else { 0.0 },
                    other => panic!("Invalid operand: {}", other),
                };
            }
        }

        if let Some(gs) = &self.problem.variable_constraints {
            for g in gs {
                total += g(&self.solution).max(0.0);
            }
        }

        total
    }

    pub fn is_feasible(&self) -> bool {
        self.constraint_violation == 0
    }

    pub fn evaluate(&mut self) {
        self.objective_fitness_values = match self.problem.eval_fn {
            EvalFn::Single(f) => f(&self.solution).into(),
            EvalFn::Batch(_) => panic!(
                "Solution::evaluate() called on a batch-mode Problem. \
                 Use the GA's population evaluation instead \
                 (NSGAII::evaluate_population / NSGAIII::run)."
            ),
        };
        self.evaluated = true;
        self.constraint_values = self.evaluate_constraints().into();
        self.constraint_violation = self.calculate_constraint_violation();
        self.constraint_violation_magnitude = self.calculate_constraint_violation_magnitude();
        self.feasible = self.is_feasible();
    }

}



// Write Unit Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gatypes::{BitBinary, Integer, Real};
    use crate::benchmark_objective_functions::parabloid_5_loc;

    #[test]
    fn test_bit_binary() {
        let mut rng = SmallRng::seed_from_u64(0);
        let bit_binary = BitBinary::new();
        let value = bit_binary.generate_value(&mut rng).unwrap();
        assert!(value == 0 || value == 1);
    }

    #[test]
    fn test_integer() {
        let mut rng = SmallRng::seed_from_u64(0);
        let integer = Integer::new(Some(10), Some(20));
        let value = integer.generate_value(&mut rng).unwrap();
        assert!((10..=20).contains(&value)); // bounds are closed on both ends
    }

    #[test]
    fn test_real() {
        let mut rng = SmallRng::seed_from_u64(0);
        let real = Real::new(Some(10.0), Some(20.0));
        let value = real.generate_value(&mut rng).unwrap();
        assert!((10.0..=20.0).contains(&value)); // bounds are closed on both ends
    }

    #[test]
    fn test_problem_initialization() {
        let solution_data_types = vec![
            SolutionDataTypes::BitBinary(BitBinary::new()),
            SolutionDataTypes::Integer(Integer::new(Some(10), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(10.0), Some(20.0))),
            SolutionDataTypes::Integer(Integer::new(Some(10), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(10.0), Some(20.0))),
        ];

        let problem = Problem::new(
            5,
            1,
            None,
            None,
            None,
            solution_data_types,
            parabloid_5_loc,
        );


        assert_eq!(problem.solution_length, 5);
        assert_eq!(problem.number_of_objectives, 1);


    }

    #[test]
    fn test_generate_solution() {
        let solution_data_types = vec![
            SolutionDataTypes::BitBinary(BitBinary::new()),
            SolutionDataTypes::Integer(Integer::new(Some(10), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(10.0), Some(20.0))),
            SolutionDataTypes::Integer(Integer::new(Some(-100), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(-10.0), Some(20.0))),
        ];

        let problem = Problem::new(
            5,
            1,
            None,
            None,
            None,
            solution_data_types,
            parabloid_5_loc,
        );

        let mut rng = SmallRng::seed_from_u64(0);
        let solution: Vec<f64> = problem.generate_solution(&mut rng);
        println!("{:?}", solution);
        assert_eq!(solution.len(), 5);

    }

    #[test]
    fn test_solution_evaluation() {
        let solution_data_types = vec![
            SolutionDataTypes::BitBinary(BitBinary::new()),
            SolutionDataTypes::Integer(Integer::new(Some(10), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(10.0), Some(20.0))),
            SolutionDataTypes::Integer(Integer::new(Some(-100), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(-10.0), Some(20.0))),
        ];

        let problem = Arc::new(Problem::new(
            5,
            1,
            None,
            None,
            None,
            solution_data_types,
            parabloid_5_loc,
        ));

        let mut solution = Solution::new(Arc::clone(&problem));
        solution.evaluate();

        assert!(solution.evaluated);
        assert_eq!(solution.objective_fitness_values.len(), 1);
    }

    #[test]
    fn test_solution_feasibility() {
        let solution_data_types = vec![
            SolutionDataTypes::BitBinary(BitBinary::new()),
            SolutionDataTypes::Integer(Integer::new(Some(10), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(10.0), Some(20.0))),
            SolutionDataTypes::Integer(Integer::new(Some(-100), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(-10.0), Some(20.0))),
        ];

        let problem = Arc::new(Problem::new(
            5,
            1,
            None,
            None,
            None,
            solution_data_types,
            parabloid_5_loc,
        ));

        let mut solution = Solution::new(Arc::clone(&problem));
        solution.evaluate();

        assert!(solution.is_feasible());
    }

    #[test]
    fn test_solution_constraint_violation() {
        // parabloid_5_loc([1,2,3,4,5]) = 0  → satisfies < 15 → violation count 0
        // parabloid_5_loc([10,10,10,10,10]) = 285 → violates < 15 → violation count 1
        let solution_data_types = vec![
            SolutionDataTypes::Real(Real::new(Some(0.0), Some(20.0))),
            SolutionDataTypes::Real(Real::new(Some(0.0), Some(20.0))),
            SolutionDataTypes::Real(Real::new(Some(0.0), Some(20.0))),
            SolutionDataTypes::Real(Real::new(Some(0.0), Some(20.0))),
            SolutionDataTypes::Real(Real::new(Some(0.0), Some(20.0))),
        ];

        let problem = Arc::new(Problem::new(
            5,
            1,
            Some(vec![Some(15.0)]),
            Some(vec![Some("<".to_string())]),
            None,
            solution_data_types,
            parabloid_5_loc,
        ));

        // Solution at the optimum — objective = 0, satisfies < 15
        let mut feasible = Solution::new(Arc::clone(&problem));
        feasible.solution = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        feasible.evaluate();
        assert_eq!(feasible.constraint_violation, 0, "solution at optimum should have no violations");

        // Solution far from optimum — objective = 285, violates < 15
        let mut infeasible = Solution::new(Arc::clone(&problem));
        infeasible.solution = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        infeasible.evaluate();
        assert_eq!(infeasible.constraint_violation, 1, "solution with objective 285 should violate < 15");
        assert!(!infeasible.feasible);
    }


    #[test]
    fn test_variable_constraints() {
        fn sphere(x: &Vec<f64>) -> Vec<f64> { vec![x.iter().map(|v| v * v).sum()] }
        // g(x) = sum(x) - 5 <= 0  → feasible when the weights sum to at most 5.
        let problem = Arc::new(
            Problem::new(
                2, 1, None, None, Some(vec![-1]),
                vec![
                    SolutionDataTypes::Real(Real::new(Some(0.0), Some(10.0))),
                    SolutionDataTypes::Real(Real::new(Some(0.0), Some(10.0))),
                ],
                sphere,
            )
            .with_variable_constraints(vec![|x: &Vec<f64>| x.iter().sum::<f64>() - 5.0]),
        );

        let mut feasible = Solution::new(Arc::clone(&problem));
        feasible.solution = vec![1.0, 1.0]; // sum 2 <= 5
        feasible.evaluate();
        assert_eq!(feasible.constraint_violation, 0);
        assert!(feasible.feasible);

        let mut infeasible = Solution::new(Arc::clone(&problem));
        infeasible.solution = vec![4.0, 4.0]; // sum 8 > 5
        infeasible.evaluate();
        assert_eq!(infeasible.constraint_violation, 1);
        assert!(!infeasible.feasible);
    }

    #[test]
    fn test_problem_with_constraints() {
        let solution_data_types = vec![
            SolutionDataTypes::BitBinary(BitBinary::new()),
            SolutionDataTypes::Integer(Integer::new(Some(10), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(10.0), Some(20.0))),
            SolutionDataTypes::Integer(Integer::new(Some(-100), Some(20))),
            SolutionDataTypes::Real(Real::new(Some(-10.0), Some(20.0))),
        ];

        let problem = Problem::new(
            5,
            1,
            Some(vec![Some(15.0)]),
            Some(vec![Some("<".to_string())]),
            None,
            solution_data_types,
            parabloid_5_loc,
        );

        assert_eq!(problem.solution_length, 5);
        assert_eq!(problem.number_of_objectives, 1);
        assert_eq!(problem.objective_constraint.as_ref().unwrap().len(), 1);
        assert_eq!(problem.objective_constraint_operands.as_ref().unwrap().len(), 1);
    }
}
