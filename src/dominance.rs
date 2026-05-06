use crate::core::{Problem, Solution};


#[derive(Debug)]
pub enum DominanceEnum  {
    ParetoDominance,
    EpsilonDominance,
    AttributeDominance
}

pub trait Dominance: Send + Sync {
    fn compare_solutions(&self, solution_1: &Solution, solution_2: &Solution) -> i32;
}

#[derive(Debug)]
pub struct ParetoDominance;

impl Dominance for ParetoDominance {
    /// Returns -1 if solution_1 dominates, 1 if solution_2 dominates, 0 if non-dominated
    fn compare_solutions(&self, solution_1: &Solution, solution_2: &Solution) -> i32 {
        let problem: &Problem = &solution_1.problem;
        let n_objectives = *problem.number_of_objectives();

        // Constraint-based dominance: fewer violations wins
        if let Some(ref constraints) = &problem.objective_constraint {
            if !constraints.is_empty() && solution_1.constraint_violation != solution_2.constraint_violation {
                return match (solution_1.constraint_violation, solution_2.constraint_violation) {
                    (0, _) => -1,
                    (_, 0) => 1,
                    (v1, v2) if v1 < v2 => -1,
                    (v1, v2) if v1 > v2 => 1,
                    _ => 0,
                };
            }
        }

        let mut is_solution_1_better = false;
        let mut is_solution_2_better = false;

        for i in 0..n_objectives {
            let mut obj_1 = solution_1.objective_fitness_values[i];
            let mut obj_2 = solution_2.objective_fitness_values[i];

            if let Some(direction) = problem.direction() {
                // For minimization (direction == -1), negate so "lower is better" becomes "higher is better"
                if direction[i] == -1 {
                    obj_1 = -obj_1;
                    obj_2 = -obj_2;
                }
            }

            if obj_1 < obj_2 {
                is_solution_2_better = true;
                if is_solution_1_better {
                    return 0; // non-dominated
                }
            } else if obj_1 > obj_2 {
                is_solution_1_better = true;
                if is_solution_2_better {
                    return 0; // non-dominated
                }
            }
        }

        if is_solution_1_better == is_solution_2_better {
            0 // equal on all objectives
        } else if is_solution_1_better {
            -1
        } else {
            1
        }
    }
}

/// Fast non-dominated sorting (Deb et al., 2002).
/// Returns a vector of fronts, where each front is a vector of indices into `population`.
/// Front 0 is the Pareto-optimal set.
pub fn fast_non_dominated_sort(population: &[Solution], dominance: &dyn Dominance) -> Vec<Vec<usize>> {
    let n = population.len();
    // domination_count[i] = number of solutions that dominate solution i
    let mut domination_count = vec![0usize; n];
    // dominated_set[i] = set of solution indices that solution i dominates
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut front_0: Vec<usize> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let flag = dominance.compare_solutions(&population[i], &population[j]);
            if flag == -1 {
                // i dominates j
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if flag == 1 {
                // j dominates i
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            front_0.push(i);
        }
    }

    fronts.push(front_0);

    let mut current_front = 0;
    while !fronts[current_front].is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &i in &fronts[current_front] {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        if next_front.is_empty() {
            break;
        }
        fronts.push(next_front);
        current_front += 1;
    }

    fronts
}

/// Crowding distance assignment for a single front.
/// Returns a vector of crowding distances indexed by position in `front_indices`.
pub fn crowding_distance(population: &[Solution], front_indices: &[usize]) -> Vec<f64> {
    let n = front_indices.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let mut distances = vec![0.0f64; n];
    let n_objectives = *population[front_indices[0]].problem.number_of_objectives();

    for m in 0..n_objectives {
        // Sort front by objective m
        let mut sorted_local: Vec<usize> = (0..n).collect();
        sorted_local.sort_by(|&a, &b| {
            let va = population[front_indices[a]].objective_fitness_values[m];
            let vb = population[front_indices[b]].objective_fitness_values[m];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let f_min = population[front_indices[sorted_local[0]]].objective_fitness_values[m];
        let f_max = population[front_indices[sorted_local[n - 1]]].objective_fitness_values[m];
        let range = f_max - f_min;

        // Boundary solutions get infinite distance
        distances[sorted_local[0]] = f64::INFINITY;
        distances[sorted_local[n - 1]] = f64::INFINITY;

        if range > f64::EPSILON {
            for i in 1..(n - 1) {
                let prev = population[front_indices[sorted_local[i - 1]]].objective_fitness_values[m];
                let next = population[front_indices[sorted_local[i + 1]]].objective_fitness_values[m];
                distances[sorted_local[i]] += (next - prev) / range;
            }
        }
    }

    distances
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Problem, Solution};
    use crate::gatypes::{SolutionDataTypes, Real, BitBinary, Integer};
    use crate::benchmark_objective_functions::{parabloid_5, parabloid_hyper_5, simple_objective};

    #[test]
    fn test_pareto_dominance_sol1_dominates() {
        let problem = Problem {
            solution_length: 5,
            number_of_objectives: 1,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![1]), // maximize
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            objective_function: parabloid_5,
            batch_objective_function: None,
        };

        let mut sol1 = Solution {
            problem: &problem,
            solution: vec![10.0, 10.0, 10.0, 10.0, 10.0],
            objective_fitness_values: vec![],
            constraint_values: vec![],
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let mut sol2 = Solution {
            problem: &problem,
            solution: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            objective_fitness_values: vec![],
            constraint_values: vec![],
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };

        sol1.evaluate();
        sol2.evaluate();

        let result = ParetoDominance.compare_solutions(&sol1, &sol2);
        assert_eq!(result, -1); // sol1 has higher value, maximizing => sol1 dominates
    }

    #[test]
    fn test_pareto_dominance_sol2_dominates() {
        let problem = Problem {
            solution_length: 5,
            number_of_objectives: 1,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1]), // minimize
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            objective_function: parabloid_5,
            batch_objective_function: None,
        };

        let mut sol1 = Solution {
            problem: &problem,
            solution: vec![10.0, 10.0, 10.0, 10.0, 10.0],
            objective_fitness_values: vec![],
            constraint_values: vec![],
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let mut sol2 = Solution {
            problem: &problem,
            solution: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            objective_fitness_values: vec![],
            constraint_values: vec![],
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };

        sol1.evaluate();
        sol2.evaluate();

        let result = ParetoDominance.compare_solutions(&sol1, &sol2);
        assert_eq!(result, 1); // sol2 has lower value, minimizing => sol2 dominates
    }

    #[test]
    fn test_pareto_dominance_non_dominated() {
        let problem = Problem {
            solution_length: 2,
            number_of_objectives: 2,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1]), // minimize both
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            objective_function: |x| vec![x[0], x[1]],
            batch_objective_function: None,
        };

        // sol1 is better on obj0, sol2 is better on obj1 => non-dominated
        let sol1 = Solution {
            problem: &problem,
            solution: vec![1.0, 10.0],
            objective_fitness_values: vec![1.0, 10.0],
            constraint_values: vec![],
            constraint_violation: 0,
            feasible: true,
            evaluated: true,
        };
        let sol2 = Solution {
            problem: &problem,
            solution: vec![10.0, 1.0],
            objective_fitness_values: vec![10.0, 1.0],
            constraint_values: vec![],
            constraint_violation: 0,
            feasible: true,
            evaluated: true,
        };

        let result = ParetoDominance.compare_solutions(&sol1, &sol2);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_fast_non_dominated_sort() {
        let problem = Problem {
            solution_length: 2,
            number_of_objectives: 2,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1]),
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            objective_function: |x| vec![x[0], x[1]],
            batch_objective_function: None,
        };

        let population = vec![
            // Front 0: Pareto-optimal
            Solution { problem: &problem, solution: vec![1.0, 5.0], objective_fitness_values: vec![1.0, 5.0], constraint_values: vec![], constraint_violation: 0, feasible: true, evaluated: true },
            Solution { problem: &problem, solution: vec![5.0, 1.0], objective_fitness_values: vec![5.0, 1.0], constraint_values: vec![], constraint_violation: 0, feasible: true, evaluated: true },
            // Front 1: dominated by front 0
            Solution { problem: &problem, solution: vec![3.0, 6.0], objective_fitness_values: vec![3.0, 6.0], constraint_values: vec![], constraint_violation: 0, feasible: true, evaluated: true },
            // Front 1: dominated by front 0
            Solution { problem: &problem, solution: vec![6.0, 3.0], objective_fitness_values: vec![6.0, 3.0], constraint_values: vec![], constraint_violation: 0, feasible: true, evaluated: true },
            // Front 2: dominated by front 1
            Solution { problem: &problem, solution: vec![10.0, 10.0], objective_fitness_values: vec![10.0, 10.0], constraint_values: vec![], constraint_violation: 0, feasible: true, evaluated: true },
        ];

        let fronts = fast_non_dominated_sort(&population, &ParetoDominance);

        assert_eq!(fronts.len(), 3);
        assert_eq!(fronts[0].len(), 2); // front 0: indices 0,1
        assert_eq!(fronts[1].len(), 2); // front 1: indices 2,3
        assert_eq!(fronts[2].len(), 1); // front 2: index 4
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let problem = Problem {
            solution_length: 2,
            number_of_objectives: 2,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1, -1]),
            solution_data_types: vec![
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
                SolutionDataTypes::Real(Real::new(Some(0.), Some(100.))),
            ],
            objective_function: |x| vec![x[0], x[1]],
            batch_objective_function: None,
        };

        let population = vec![
            Solution { problem: &problem, solution: vec![1.0, 5.0], objective_fitness_values: vec![1.0, 5.0], constraint_values: vec![], constraint_violation: 0, feasible: true, evaluated: true },
            Solution { problem: &problem, solution: vec![3.0, 3.0], objective_fitness_values: vec![3.0, 3.0], constraint_values: vec![], constraint_violation: 0, feasible: true, evaluated: true },
            Solution { problem: &problem, solution: vec![5.0, 1.0], objective_fitness_values: vec![5.0, 1.0], constraint_values: vec![], constraint_violation: 0, feasible: true, evaluated: true },
        ];

        let front = vec![0, 1, 2];
        let distances = crowding_distance(&population, &front);

        // boundary solutions should have infinite distance
        assert_eq!(distances[0], f64::INFINITY);
        assert_eq!(distances[2], f64::INFINITY);
        // middle solution should have finite distance
        assert!(distances[1].is_finite());
        assert!(distances[1] > 0.0);
    }
}
