// import SolutionTypes, BitBinary, Integer, Real from  gatypes.rs
// use crate::gatypes::{SolutionType, BitBinary, Integer, Real};
use crate::constraints::ComparisonFunctions;

use crate::gatypes::SolutionDataTypes;
use std::sync::Arc;

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
}

impl Problem {
    pub fn new(
        solution_length: usize,
        number_of_objectives: usize,
        objective_constraint: Option<Vec<Option<f64>>>, //number_of_objectives
        objective_constraint_operands: Option<Vec<Option<String>>>, //number_of_objectives
        direction: Option<Vec<i8>>,
        solution_data_types: Vec<SolutionDataTypes>,// Vec of Binary or Integer or Real
        objective_function: fn(&Vec<f64>) -> Vec<f64>
    ) -> Self {
        // If solution_length != solution_data_types.len() panic
        if solution_length != solution_data_types.len() {
            panic!("solution_length does not match solution_data_types length");
        }

        // Check if lengths match number_of_objectives
        if let Some(ref constraints) = objective_constraint {
            if constraints.len() != number_of_objectives {
                panic!("objective_constraint length does not match number_of_objectives");
            }
        }

        if let Some(ref operands) = objective_constraint_operands {
            if operands.len() != number_of_objectives {
                panic!("objective_constraint_operands length does not match number_of_objectives");
            }
        }

        let direction: Option<Vec<i8>> = direction.or_else(|| Some(vec![-1; number_of_objectives]));

        if let Some(ref dirs) = direction {
            if dirs.len() != number_of_objectives {
                panic!("direction length does not match number_of_objectives");
            }
        }

        Problem {
            solution_length,
            number_of_objectives,
            objective_constraint,
            objective_constraint_operands,
            direction,
            solution_data_types,
            eval_fn: EvalFn::Single(objective_function),
        }
    }

    pub fn solution_length(&self) -> &usize {
        &self.solution_length
    }

    pub fn number_of_objectives(&self) -> &usize {
        &self.number_of_objectives
    }

    pub fn objective_constraint(&self) -> &Option<Vec<Option<f64>>> {
        &self.objective_constraint
    }

    pub fn objective_constraint_operands(&self) -> &Option<Vec<Option<String>>> {
        // Check if Operatnds are are <, >, <=, >=, ==, !=
        let operands = &self.objective_constraint_operands;
        if operands.is_some() {
            let operands  = operands.as_ref().unwrap();
            for operand in operands {
                let operand = operand.as_ref().unwrap();
                if operand != "<" && operand != ">" && operand != "<=" && operand != ">=" && operand != "==" && operand != "!=" {
                    panic!("Invalid operand: {}", operand);
                }
            }
        }

        &self.objective_constraint_operands
    }

    pub fn direction(&self) -> &Option<Vec<i8>> {
        &self.direction
    }

    pub fn solution_data_types(&self) -> &Vec<SolutionDataTypes> {
        &self.solution_data_types
    }

    pub fn objective_function(&self) -> fn(&Vec<f64>) -> Vec<f64> {
        match self.eval_fn {
            EvalFn::Single(f) => f,
            EvalFn::Batch(_) => panic!("objective_function() called on a batch-mode Problem"),
        }
    }

    pub fn generate_solution(&self) -> Vec<f64> {
        let mut solution: Vec<f64> = Vec::new();
        for solution_type in &self.solution_data_types {
            match solution_type {
                SolutionDataTypes::BitBinary(binary) => {
                    solution.push(binary.generate_value().unwrap() as f64);
                }
                SolutionDataTypes::Integer(integer) => {
                    solution.push(integer.generate_value().unwrap() as f64);
                }
                SolutionDataTypes::Real(real) => {
                    solution.push(real.generate_value().unwrap());
                }
            }
        }
        solution
    }
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub problem: Arc<Problem>,
    pub solution: Vec<f64>, // Derived from Problem.solution_data_types
    pub objective_fitness_values: Vec<f64>,
    pub constraint_values: Vec<f64>,
    pub evaluated: bool, // default false
    pub constraint_violation: usize, // default 0
    pub feasible: bool
}


impl Solution {
    pub fn new(problem: Arc<Problem>) -> Self {
        let solution = problem.generate_solution();
        // create vectore of length number_of_objectives
        let objective_fitness_values: Vec<f64> = Vec::with_capacity(*problem.number_of_objectives());
        let constraint_values: Vec<f64> = Vec::with_capacity(*problem.number_of_objectives());
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
            feasible,
        }
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    pub fn solution(&self) -> &Vec<f64> {
        &self.solution
    }

    pub fn objective_fitness_values(&self) -> &Vec<f64> {
        &self.objective_fitness_values
    }

    pub fn constraint_values(&self) -> &Vec<f64> {
        &self.constraint_values
    }

    pub fn evaluated(&self) -> &bool {
        &self.evaluated
    }

    pub fn constraint_violation(&self) -> &usize {
        &self.constraint_violation
    }

    pub fn feasible(&self) -> &bool {
        &self.feasible
    }

    pub fn evaluate_constraints(&mut self) -> Vec<f64> {
        let mut constraint_values: Vec<f64> = Vec::new();
        let objective_constraint = self.problem.objective_constraint();
        let objective_constraint_operands = self.problem.objective_constraint_operands();
        // println!("OBJECTIVE CONSTRAINT: {:?}", objective_constraint);
        // println!("OBJECTIVE CONSTRAINT OPERANDS: {:?}", objective_constraint_operands);
        if objective_constraint.is_some() && objective_constraint_operands.is_some() {
            let objective_constraint: &Vec<Option<f64>> = &objective_constraint.as_ref().unwrap();
            let objective_constraint_operands: &Vec<Option<String>> = &objective_constraint_operands.as_ref().unwrap();
            for _i in 0..objective_constraint.len() {
                let operand: &Option<String> = &objective_constraint_operands[_i];
                let constraint: &Option<f64> = &objective_constraint[_i];
                let obj_value: &f64 = &self.objective_fitness_values[_i];
                // println!("OBJ VALUE: {:?}", obj_value);
                let comparison_fn= ComparisonFunctions::new();// operand.as_ref().unwrap();
                // println!("OPERAND: {:?}", operand);
                let comparison_fn = comparison_fn.functions.get(operand.as_ref().unwrap()).unwrap();
                let constraint_value = comparison_fn.compare(*obj_value, constraint.unwrap());
                // println!("CONSTRAINT VALUE: {:?}", constraint_value);
                constraint_values.push(constraint_value as i8 as f64);
            }
        }
        constraint_values
    }

    pub fn calculate_constraint_violation(&mut self) -> usize {
        let mut constraint_violation = 0;
        let constraint_values: &Vec<f64> = self.constraint_values();
        for constraint_value in constraint_values {
            if *constraint_value == 0.0 {
                constraint_violation += 1;
            }
        }
        constraint_violation
    }

    pub fn is_feasible(&mut self) -> bool {
        let constraint_violation = self.constraint_violation();
        if *constraint_violation == 0 {
            true
        } else {
            false
        }
    }

    pub fn evaluate(&mut self) {
        self.objective_fitness_values = match self.problem.eval_fn {
            EvalFn::Single(f) => f(&self.solution),
            EvalFn::Batch(_) => panic!(
                "Solution::evaluate() called on a batch-mode Problem. \
                 Use evaluate_population() instead."
            ),
        };
        self.evaluated = true;
        self.constraint_values = self.evaluate_constraints();
        self.constraint_violation = self.calculate_constraint_violation();
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
        let bit_binary = BitBinary::new();
        let value = bit_binary.generate_value().unwrap();
        assert!(value == 0 || value == 1);
    }

    #[test]
    fn test_integer() {
        let integer = Integer::new(Some(10), Some(20));
        let value = integer.generate_value().unwrap();
        assert!(value >= 10 && value < 20);
    }

    #[test]
    fn test_real() {
        let real = Real::new(Some(10.0), Some(20.0));
        let value = real.generate_value().unwrap();
        assert!(value >= 10.0 && value < 20.0);
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


        assert_eq!(*problem.solution_length(), 5);
        assert_eq!(*problem.number_of_objectives(), 1);


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

        let solution: Vec<f64> = problem.generate_solution();
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
        assert_eq!(solution.objective_fitness_values().len(), 1);
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

        assert_eq!(*problem.solution_length(), 5);
        assert_eq!(*problem.number_of_objectives(), 1);
        assert_eq!(problem.objective_constraint().as_ref().unwrap().len(), 1);
        assert_eq!(problem.objective_constraint_operands().as_ref().unwrap().len(), 1);
    }
}
