use crate::dominance::{ParetoDominance, Dominance};
use crate::core::Solution;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub trait Selector: Send + Sync {
    fn select_index(&mut self, population: &[Solution], n: usize) -> Vec<usize>;
}

#[derive(Debug)]
pub struct TournamentSelector {
    tournament_size: usize,
    dominance: ParetoDominance,
    rng: StdRng,
}

impl TournamentSelector {
    pub fn new(tournament_size: usize, dominance: ParetoDominance, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(seed_value) => StdRng::seed_from_u64(seed_value),
            None => StdRng::from_entropy(),
        };
        TournamentSelector { tournament_size, dominance, rng }
    }

    pub fn select_one<'a>(&mut self, population: &[&'a Solution]) -> &'a Solution {
        let mut winner_idx = self.rng.gen_range(0..population.len());
        for _ in 0..self.tournament_size {
            let challenger_idx = self.rng.gen_range(0..population.len());
            let flag = self.dominance.compare_solutions(population[challenger_idx], population[winner_idx]);
            if flag < 0 {
                winner_idx = challenger_idx;
            }
        }
        population[winner_idx]
    }

    pub fn select<'a>(&mut self, n: usize, population: &[&'a Solution]) -> Vec<&'a Solution> {
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            results.push(self.select_one(population));
        }
        results
    }
}

impl Selector for TournamentSelector {
    fn select_index(&mut self, population: &[Solution], n: usize) -> Vec<usize> {
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            let mut winner_idx = self.rng.gen_range(0..population.len());
            for _ in 0..self.tournament_size {
                let challenger_idx = self.rng.gen_range(0..population.len());
                let flag = self.dominance.compare_solutions(&population[challenger_idx], &population[winner_idx]);
                if flag < 0 {
                    winner_idx = challenger_idx;
                }
            }
            results.push(winner_idx);
        }
        results
    }
}

impl Default for TournamentSelector {
    fn default() -> Self {
        TournamentSelector::new(2, ParetoDominance, Some(1234))
    }
}

/// Tournament selector that uses rank and crowding distance (NSGA-II style).
/// Prefers lower rank; ties broken by higher crowding distance.
#[derive(Debug)]
pub struct CrowdingTournamentSelector {
    tournament_size: usize,
    rng: StdRng,
}

impl CrowdingTournamentSelector {
    pub fn new(tournament_size: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        CrowdingTournamentSelector { tournament_size, rng }
    }

    /// Select `n` indices from population using rank and crowding distance arrays.
    /// `ranks[i]` is the non-domination front index (0 = best).
    /// `crowding[i]` is the crowding distance (higher = more diverse).
    pub fn select_indices(
        &mut self,
        n: usize,
        ranks: &[usize],
        crowding: &[f64],
    ) -> Vec<usize> {
        let pop_len = ranks.len();
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            let mut winner = self.rng.gen_range(0..pop_len);
            for _ in 1..self.tournament_size {
                let challenger = self.rng.gen_range(0..pop_len);
                if crowding_compare(ranks[challenger], crowding[challenger], ranks[winner], crowding[winner]) {
                    winner = challenger;
                }
            }
            results.push(winner);
        }
        results
    }
}

/// Returns true if (rank_a, crowd_a) is preferred over (rank_b, crowd_b).
/// Lower rank wins; if equal rank, higher crowding distance wins.
#[inline]
pub fn crowding_compare(rank_a: usize, crowd_a: f64, rank_b: usize, crowd_b: f64) -> bool {
    rank_a < rank_b || (rank_a == rank_b && crowd_a > crowd_b)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::core::{EvalFn, Solution, Problem};
    use crate::gatypes::{SolutionDataTypes, BitBinary, Integer, Real};
    use crate::benchmark_objective_functions::{parabloid_5_loc, parabloid_hyper_5};

    fn setup_problem(objective_function: fn(&Vec<f64>) -> Vec<f64>, direction: Vec<i8>) -> Problem {
        Problem {
            solution_length: 5,
            number_of_objectives: 1,
            objective_constraint: Some(vec![Some(10.0)]),
            objective_constraint_operands: Some(vec![Some("<".to_string())]),
            direction: Some(direction),
            solution_data_types: vec![
                SolutionDataTypes::BitBinary(BitBinary::new()),
                SolutionDataTypes::Integer(Integer::new(Some(10), Some(20))),
                SolutionDataTypes::Real(Real::new(Some(10.0), Some(20.0))),
                SolutionDataTypes::Integer(Integer::new(Some(-100), Some(20))),
                SolutionDataTypes::Real(Real::new(Some(-10.0), Some(20.0))),
            ],
            eval_fn: EvalFn::Single(objective_function),
        }
    }

    fn setup_solutions(problem: Arc<Problem>) -> Vec<Solution> {
        vec![
            Solution {
                problem: Arc::clone(&problem),
                solution: vec![1.0, 2.0, -3.0, 4.0, 5.0],
                objective_fitness_values: Vec::new(),
                constraint_values: Vec::new(),
                constraint_violation: 0,
                feasible: false,
                evaluated: false,
            },
            Solution {
                problem: Arc::clone(&problem),
                solution: vec![12.0, 10.0, -3.0, 4.0, 5.0],
                objective_fitness_values: Vec::new(),
                constraint_values: Vec::new(),
                constraint_violation: 0,
                feasible: false,
                evaluated: false,
            },
            Solution {
                problem: Arc::clone(&problem),
                solution: vec![-22.0, 12.0, -3.0, 1.0, 5.0],
                objective_fitness_values: Vec::new(),
                constraint_values: Vec::new(),
                constraint_violation: 0,
                feasible: false,
                evaluated: false,
            },
            Solution {
                problem: Arc::clone(&problem),
                solution: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                objective_fitness_values: Vec::new(),
                constraint_values: Vec::new(),
                constraint_violation: 0,
                feasible: false,
                evaluated: false,
            },
        ]
    }

    fn evaluate_solutions(solutions: &mut Vec<Solution>) {
        for solution in solutions.iter_mut() {
            solution.evaluate();
        }
    }

    #[test]
    fn test_tournament_selector_single_objective_maximize() {
        let problem = Arc::new(setup_problem(parabloid_5_loc, vec![1]));
        let mut solutions = setup_solutions(Arc::clone(&problem));
        evaluate_solutions(&mut solutions);

        let population: Vec<&Solution> = solutions.iter().collect();
        let mut tournament_selector = TournamentSelector::default();
        let winner = tournament_selector.select_one(&population);

        println!("Winner: {:?}", winner);
    }

    #[test]
    fn test_tournament_selector_single_objective_minimize() {
        let problem = Arc::new(setup_problem(parabloid_5_loc, vec![-1]));
        let mut solutions = setup_solutions(Arc::clone(&problem));
        evaluate_solutions(&mut solutions);

        let population: Vec<&Solution> = solutions.iter().collect();
        let mut tournament_selector = TournamentSelector::default();
        let winner = tournament_selector.select_one(&population);

        println!("Winner: {:?}", winner);
    }

    #[test]
    fn test_selection_function() {
        let problem = Arc::new(setup_problem(parabloid_hyper_5, vec![-1, -1, -1, -1, -1]));
        let mut solutions = setup_solutions(Arc::clone(&problem));
        evaluate_solutions(&mut solutions);

        let population: Vec<&Solution> = solutions.iter().collect();
        let mut tournament_selector = TournamentSelector::default();
        let winners = tournament_selector.select(2, &population);

        println!("Winners: {:?}", winners);
    }

    #[test]
    fn test_crowding_tournament_selector() {
        let ranks = vec![0, 0, 1, 1, 2];
        let crowding = vec![f64::INFINITY, 0.5, f64::INFINITY, 0.3, 0.1];

        let mut selector = CrowdingTournamentSelector::new(2, Some(42));
        let selected = selector.select_indices(3, &ranks, &crowding);

        assert_eq!(selected.len(), 3);
        // All selected indices should be valid
        for &idx in &selected {
            assert!(idx < 5);
        }
    }

    #[test]
    fn test_crowding_compare() {
        // Lower rank wins
        assert!(crowding_compare(0, 0.5, 1, 1.0));
        assert!(!crowding_compare(1, 1.0, 0, 0.5));
        // Same rank, higher crowding wins
        assert!(crowding_compare(0, 1.0, 0, 0.5));
        assert!(!crowding_compare(0, 0.5, 0, 1.0));
    }
}
