use crate::core::Solution;
use crate::gatypes::SolutionDataTypes;
use std::collections::HashMap;
use std::sync::Arc;
use rand::Rng;
use rand::rngs::SmallRng;

/// Trait for mutation operations. `rng` is threaded through so a seeded run is reproducible.
pub trait Mutation: Send + Sync {
    fn mutate(&self, parent: &Solution, index: usize, rng: &mut SmallRng) -> f64;
}
/// MutationManager to manage and apply mutations
pub struct MutationManager {
    default_mutations: HashMap<&'static str, Arc<dyn Mutation>>,
}

impl MutationManager {
    /// Standard NSGA-II defaults for a problem with `solution_length` decision variables:
    /// polynomial mutation (eta=20) for Real and Integer genes and bit-flip for BitBinary,
    /// all at the conventional per-gene rate `1/n`.
    ///
    /// The rate has to be `1/n`, not 1.0: mutating every gene each generation replaces the
    /// whole genome with a fresh random one, which is random search, not evolution.
    pub fn new(solution_length: usize) -> Self {
        let rate = 1.0 / solution_length.max(1) as f64;
        let polynomial: Arc<dyn Mutation> = Arc::new(PolynomialMutation::new(Some(rate), Some(20.0)));
        let mut default_mutations: HashMap<&'static str, Arc<dyn Mutation>> = HashMap::new();
        default_mutations.insert("BitBinary", Arc::new(BitFlipMutation { probability: rate }));
        default_mutations.insert("Real", Arc::clone(&polynomial));
        default_mutations.insert("Integer", polynomial);

        Self { default_mutations }
    }

    pub fn set_default_real_mutation(&mut self, mutation: Arc<dyn Mutation>) {
        self.default_mutations.insert("Real", mutation);
    }

    pub fn set_default_integer_mutation(&mut self, mutation: Arc<dyn Mutation>) {
        self.default_mutations.insert("Integer", mutation);
    }

    pub fn set_default_binary_mutation(&mut self, mutation: Arc<dyn Mutation>) {
        self.default_mutations.insert("BitBinary", mutation);
    }

    pub fn mutate(&self, parent: &Solution, rng: &mut SmallRng) -> Solution {
        let mut child = parent.clone();
        for (i, solution_type) in parent.problem.solution_data_types.iter().enumerate() {
            let key = match solution_type {
                SolutionDataTypes::BitBinary(_) => "BitBinary",
                SolutionDataTypes::Real(_) => "Real",
                SolutionDataTypes::Integer(_) => "Integer",
            };
            if let Some(mutation) = self.default_mutations.get(key) {
                child.solution[i] = mutation.mutate(parent, i, rng);
            }
        }
        child.feasible = false;
        child.evaluated = false;
        child
    }
}


/// Helper trait to retrieve bounds for solution data types
pub trait SolutionTypeBounds {
    fn get_lower_bound(&self) -> Option<f64>;
    fn get_upper_bound(&self) -> Option<f64>;
}

impl SolutionTypeBounds for SolutionDataTypes {
    fn get_lower_bound(&self) -> Option<f64> {
        match self {
            SolutionDataTypes::Real(real) => real.lower_bound,
            SolutionDataTypes::Integer(integer) => integer.lower_bound.map(|v| v as f64),
            _ => None,
        }
    }

    fn get_upper_bound(&self) -> Option<f64> {
        match self {
            SolutionDataTypes::Real(real) => real.upper_bound,
            SolutionDataTypes::Integer(integer) => integer.upper_bound.map(|v| v as f64),
            _ => None,
        }
    }
}

/// Bit Flip Mutation
pub struct BitFlipMutation {
    pub probability: f64,
}

impl Default for BitFlipMutation {
    /// Conservative standalone default. The GA uses `MutationManager::new(n)`, which sets the
    /// conventional `1/n` rate instead.
    fn default() -> Self {
        Self { probability: 0.1 }
    }
}

impl Mutation for BitFlipMutation {
    fn mutate(&self, parent: &Solution, index: usize, rng: &mut SmallRng) -> f64 {
        if rng.gen::<f64>() < self.probability {
            1.0 - parent.solution[index]
        } else {
            parent.solution[index]
        }
    }
}

/// Uniform Mutation
pub struct UniformMutation {
    pub probability: f64,
}

impl Default for UniformMutation {
    fn default() -> Self {
        Self { probability: 1.0 }
    }
}

impl Mutation for UniformMutation {
    fn mutate(&self, parent: &Solution, index: usize, rng: &mut SmallRng) -> f64 {
        match &parent.problem.solution_data_types[index] {
            SolutionDataTypes::Integer(integer) => {
                let lower_bound = integer.lower_bound.unwrap_or(i64::MIN) as f64;
                let upper_bound = integer.upper_bound.unwrap_or(i64::MAX) as f64;

                if rng.gen::<f64>() < self.probability {
                    rng.gen_range(lower_bound..=upper_bound).round() // Ensures result is an integer
                } else {
                    parent.solution[index]
                }
            }
            SolutionDataTypes::Real(real) => {
                let lower_bound = real.lower_bound.unwrap_or(f64::MIN);
                let upper_bound = real.upper_bound.unwrap_or(f64::MAX);

                if rng.gen::<f64>() < self.probability {
                    rng.gen_range(lower_bound..=upper_bound)
                } else {
                    parent.solution[index]
                }
            }
            _ => parent.solution[index], // No mutation for other types
        }
    }
}

pub struct PolynomialMutation {
    pub probability: f64,
    pub distribution_index: f64,
}

impl PolynomialMutation {
    pub fn new(probability: Option<f64>, distribution_index: Option<f64>) -> Self {
        Self {
            probability: probability.unwrap_or(1.0),
            distribution_index: distribution_index.unwrap_or(20.0),
        }
    }
}

impl Mutation for PolynomialMutation {
    fn mutate(&self, parent: &Solution, index: usize, rng: &mut SmallRng) -> f64 {
        match &parent.problem.solution_data_types[index] {
            SolutionDataTypes::Integer(integer) => {
                let lower_bound = integer.lower_bound.unwrap_or(i64::MIN) as f64;
                let upper_bound = integer.upper_bound.unwrap_or(i64::MAX) as f64;

                if rng.gen::<f64>() < self.probability {
                    let u = rng.gen::<f64>();
                    let dx = upper_bound - lower_bound;
                    let delta = if u < 0.5 {
                        let bl = (parent.solution[index] - lower_bound) / dx;
                        (2.0 * u + (1.0 - 2.0 * u) * (1.0 - bl).powf(self.distribution_index + 1.0))
                            .powf(1.0 / (self.distribution_index + 1.0))
                            - 1.0
                    } else {
                        let bu = (upper_bound - parent.solution[index]) / dx;
                        (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - bu).powf(self.distribution_index + 1.0))
                            .powf(1.0 / (self.distribution_index + 1.0))
                            - 1.0
                    };
                    (parent.solution[index] + delta * dx).round().clamp(lower_bound, upper_bound)
                } else {
                    parent.solution[index]
                }
            }
            SolutionDataTypes::Real(real) => {
                let lower_bound = real.lower_bound.unwrap_or(f64::MIN);
                let upper_bound = real.upper_bound.unwrap_or(f64::MAX);

                if rng.gen::<f64>() < self.probability {
                    let u = rng.gen::<f64>();
                    let dx = upper_bound - lower_bound;
                    let delta = if u < 0.5 {
                        let bl = (parent.solution[index] - lower_bound) / dx;
                        (2.0 * u + (1.0 - 2.0 * u) * (1.0 - bl).powf(self.distribution_index + 1.0))
                            .powf(1.0 / (self.distribution_index + 1.0))
                            - 1.0
                    } else {
                        let bu = (upper_bound - parent.solution[index]) / dx;
                        (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - bu).powf(self.distribution_index + 1.0))
                            .powf(1.0 / (self.distribution_index + 1.0))
                            - 1.0
                    };
                    (parent.solution[index] + delta * dx).clamp(lower_bound, upper_bound)
                } else {
                    parent.solution[index]
                }
            }
            _ => parent.solution[index], // No mutation for other types
        }
    }
}

/// Gaussian Mutation
pub struct GaussianMutation {
    pub probability: f64,
    pub standard_deviation: f64,
}

impl GaussianMutation {
    pub fn new(probability: Option<f64>, standard_deviation: Option<f64>) -> Self {
        Self {
            probability: probability.unwrap_or(1.0),
            standard_deviation: standard_deviation.unwrap_or(0.1),
        }
    }
}

impl Mutation for GaussianMutation {
    fn mutate(&self, parent: &Solution, index: usize, rng: &mut SmallRng) -> f64 {
        let lower_bound = parent.problem.solution_data_types[index]
            .get_lower_bound()
            .unwrap_or(f64::MIN);
        let upper_bound = parent.problem.solution_data_types[index]
            .get_upper_bound()
            .unwrap_or(f64::MAX);

        if rng.gen::<f64>() < self.probability {
            // Box-Muller: two uniforms -> one standard normal. `rand`'s uniform is [0,1), so
            // clamp u1 off zero before the log.
            let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
            let u2: f64 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (parent.solution[index] + z * self.standard_deviation).clamp(lower_bound, upper_bound)
        } else {
            parent.solution[index]
        }
    }
}
// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    fn test_rng() -> SmallRng { SmallRng::seed_from_u64(0) }
    use std::sync::Arc;
    use crate::core::{EvalFn, Problem, Solution};
    use crate::gatypes::{SolutionDataTypes, Real, Integer, BitBinary};

    fn setup_problem() -> Problem {
        Problem {
            solution_length: 5,
            number_of_objectives: 1,
            objective_constraint: Some(vec![Some(10.0)]),
            objective_constraint_operands: Some(vec![Some("<".to_string())]),
            direction: Some(vec![1]),
            solution_data_types: vec![
                SolutionDataTypes::BitBinary(BitBinary::new()),
                SolutionDataTypes::Integer(Integer::new(Some(-2000), Some(2000))),
                SolutionDataTypes::Real(Real::new(Some(-100.0), Some(1000.0))),
                SolutionDataTypes::Real(Real::new(Some(-100.0), Some(1000.0))),
                SolutionDataTypes::Real(Real::new(Some(-100.0), Some(1000.0))),
            ],
            variable_constraints: None,
            eval_fn: EvalFn::Single(|x| vec![x.iter().sum()]),
        }
    }

    fn setup_solution(problem: Arc<Problem>) -> Solution {
        Solution {
            problem,
            solution: vec![1.0, 10.0, 10.0, 10.0, 10.0],
            objective_fitness_values: Default::default(),
            constraint_values: Default::default(),
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        }
    }


    /// The defaults must mutate *sparingly* — around 1/n of the genes per call. Mutating every
    /// gene rebuilds the genome from scratch each generation, which is random search; that was
    /// the original defect here, and the old version of this test asserted it.
    #[test]
    fn test_default_mutation_manager() {
        let problem = Arc::new(setup_problem());
        let parent = setup_solution(Arc::clone(&problem));
        let mutation_manager = MutationManager::new(problem.solution_length);
        let mut rng = test_rng();

        let n = problem.solution_length; // 5 genes -> expected rate 1/5
        let trials = 400;
        let mut changed = 0usize;
        for _ in 0..trials {
            let child = mutation_manager.mutate(&parent, &mut rng);
            assert!(child.solution[0] == 0.0 || child.solution[0] == 1.0, "binary gene stays a bit");
            changed += (0..n).filter(|&i| child.solution[i] != parent.solution[i]).count();
        }
        let rate = changed as f64 / (trials * n) as f64;
        assert!(
            rate > 0.05 && rate < 0.45,
            "per-gene mutation rate {rate:.3} should sit near 1/n = {:.3}, not ~0 or ~1",
            1.0 / n as f64
        );
    }

    #[test]
    fn test_bit_flip_mutation() {
        let problem = Arc::new(setup_problem());
        let mutation = BitFlipMutation{probability: 1.0};
        let parent1 = setup_solution(Arc::clone(&problem));
        let mut parent2 = setup_solution(Arc::clone(&problem));
        parent2.solution[0] = 0.0;
        let child1 = mutation.mutate(&parent1, 0, &mut test_rng());
        let child2 = mutation.mutate(&parent2, 0, &mut test_rng());
        assert!(child1 == 0.0);
        assert!(child2 == 1.0);
    }
    #[test]
    fn test_polynomial_mutation_with_integer_and_real() {
        let problem = Arc::new(setup_problem());

        let parent = setup_solution(Arc::clone(&problem));


        let mutation = PolynomialMutation::new(Some(1.0), Some(20.0));
        let child_solution_0 = mutation.mutate(&parent, 1, &mut test_rng()); // Integer mutation
        let child_solution_1 = mutation.mutate(&parent, 2, &mut test_rng()); // Real mutation
        let child_solution_2 = mutation.mutate(&parent, 3, &mut test_rng()); // Real mutation
        let child_solution_3 = mutation.mutate(&parent, 4, &mut test_rng()); // Real mutation

        assert!(child_solution_0 >= -2000.0 && child_solution_0 <= 2000.0);
        assert!(child_solution_1 >= -100.0 && child_solution_1 <= 1000.0);
        assert!(child_solution_2 >= -100.0 && child_solution_2 <= 1000.0);
        assert!(child_solution_3 >= -100.0 && child_solution_3 <= 1000.0);
        // assert new solution is mutated
        assert!(child_solution_0 != parent.solution[1]);
        assert!(child_solution_1 != parent.solution[2]);
        assert!(child_solution_2 != parent.solution[3]);
        assert!(child_solution_3 != parent.solution[4]);

    }

    #[test]
    fn test_uniform_mutation_with_integer_and_real() {
        let problem = Arc::new(setup_problem());

        let parent = setup_solution(Arc::clone(&problem));

        let mutation = UniformMutation::default();
        let child_solution_0 = mutation.mutate(&parent, 1, &mut test_rng()); // Integer mutation
        let child_solution_1 = mutation.mutate(&parent, 2, &mut test_rng()); // Real mutation
        let child_solution_2 = mutation.mutate(&parent, 3, &mut test_rng()); // Real mutation
        let child_solution_3 = mutation.mutate(&parent, 4, &mut test_rng()); // Real mutation

        assert!(child_solution_0 >= -2000.0 && child_solution_0 <= 2000.0);
        assert!(child_solution_1 >= -100.0 && child_solution_1 <= 1000.0);
        assert!(child_solution_2 >= -100.0 && child_solution_2 <= 1000.0);
        assert!(child_solution_3 >= -100.0 && child_solution_3 <= 1000.0);
        // assert new solution is mutated
        assert!(child_solution_0 != parent.solution[1]);
        assert!(child_solution_1 != parent.solution[2]);
        assert!(child_solution_2 != parent.solution[3]);
        assert!(child_solution_3 != parent.solution[4]);

    }

    #[test]
    fn test_gaussian_mutation_with_integer_and_real() {
        let problem = Arc::new(setup_problem());

        let parent = setup_solution(Arc::clone(&problem));

        let mutation = GaussianMutation::new(Some(1.0), Some(0.1));
        let child_solution_0 = mutation.mutate(&parent, 1, &mut test_rng()); // Integer mutation
        let child_solution_1 = mutation.mutate(&parent, 2, &mut test_rng()); // Real mutation
        let child_solution_2 = mutation.mutate(&parent, 3, &mut test_rng()); // Real mutation
        let child_solution_3 = mutation.mutate(&parent, 4, &mut test_rng()); // Real mutation

        assert!(child_solution_0 >= -2000.0 && child_solution_0 <= 2000.0);
        assert!(child_solution_1 >= -100.0 && child_solution_1 <= 1000.0);
        assert!(child_solution_2 >= -100.0 && child_solution_2 <= 1000.0);
        assert!(child_solution_3 >= -100.0 && child_solution_3 <= 1000.0);
        // assert new solution is mutated
        assert!(child_solution_0 != parent.solution[1]);
        assert!(child_solution_1 != parent.solution[2]);
        assert!(child_solution_2 != parent.solution[3]);
        assert!(child_solution_3 != parent.solution[4]);
    }


}
