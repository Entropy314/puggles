use crate::core::Solution;
use crate::gatypes::SolutionDataTypes;
use rand::Rng;
use rand::rngs::SmallRng;

/// Trait for crossover operations. `rng` is threaded through so a seeded run is reproducible.
pub trait Crossover {
    fn crossover(&self, parent1: &Solution, parent2: &Solution, rng: &mut SmallRng) -> (Solution, Solution);
}

/// Simulated Binary Crossover (SBX) operator
pub struct SimulatedBinaryCrossover {
    pub probability: f64,
    pub distribution_index: f64,
}

impl SimulatedBinaryCrossover {
    pub fn new(probability: Option<f64>, distribution_index: Option<f64>) -> Self {
        Self {
            probability: probability.unwrap_or(1.0),
            distribution_index: distribution_index.unwrap_or(20.0),
        }
    }

    fn sbx_crossover(&self, x1: f64, x2: f64, lower: f64, upper: f64, rng: &mut SmallRng) -> (f64, f64) {
        if (x2 - x1).abs() > f64::EPSILON {
            let y1 = x1.min(x2);
            let y2 = x1.max(x2);

            let beta = 1.0 + 2.0 * (y1 - lower) / (y2 - y1);
            let alpha = 2.0 - beta.powf(-(self.distribution_index + 1.0));
            let rand = rng.gen::<f64>();

            let betaq = if rand <= 1.0 / alpha {
                (rand * alpha).powf(1.0 / (self.distribution_index + 1.0))
            } else {
                (1.0 / (2.0 - rand * alpha)).powf(1.0 / (self.distribution_index + 1.0))
            };

            let mut c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
            let mut c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

            if rng.gen::<bool>() {
                std::mem::swap(&mut c1, &mut c2);
            }

            (c1.clamp(lower, upper), c2.clamp(lower, upper))
        } else {
            (x1, x2)
        }
    }
}

impl Crossover for SimulatedBinaryCrossover {
    fn crossover(&self, parent1: &Solution, parent2: &Solution, rng: &mut SmallRng) -> (Solution, Solution) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();
        for (i, solution_type) in parent1.problem.solution_data_types.iter().enumerate() {
            let random_number: f64 = rng.gen();
            match solution_type {
                SolutionDataTypes::Real(real) => {
                    if random_number < self.probability {
                        let lower_bound = real.lower_bound.unwrap_or(f64::MIN);
                        let upper_bound = real.upper_bound.unwrap_or(f64::MAX);
                        let (c1, c2) = self.sbx_crossover(child1.solution[i], child2.solution[i], lower_bound, upper_bound, rng);
                        child1.solution[i] = c1;
                        child2.solution[i] = c2;
                        }
                    }
            _ => {}

            }
        }

        child1.evaluated = false;
        child1.feasible = false;
        child2.evaluated = false;
        child2.feasible = false;
        (child1, child2)
    }
}

pub struct DifferentialEvolutionCrossover {
    pub probability: f64,
    pub scaling_factor: f64,
}

impl DifferentialEvolutionCrossover {
    pub fn new(probability: Option<f64>, scaling_factor: Option<f64>) -> Self {
        Self {
            probability: probability.unwrap_or(0.9),
            scaling_factor: scaling_factor.unwrap_or(0.8),
        }
    }
}

impl Crossover for DifferentialEvolutionCrossover {
    fn crossover(&self, parent1: &Solution, parent2: &Solution, rng: &mut SmallRng) -> (Solution, Solution) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();

        for (i, solution_type) in parent1.problem.solution_data_types.iter().enumerate() {
            if let SolutionDataTypes::Real(real) = solution_type {
                if rng.gen::<f64>() < self.probability {
                    let lower = real.lower_bound.unwrap_or(f64::MIN);
                    let upper = real.upper_bound.unwrap_or(f64::MAX);
                    let mut c1 = parent1.solution[i] + self.scaling_factor * (parent2.solution[i] - parent1.solution[i]);
                    let mut c2 = parent2.solution[i] + self.scaling_factor * (parent1.solution[i] - parent2.solution[i]);

                    c1 = c1.clamp(lower, upper);
                    c2 = c2.clamp(lower, upper);

                    child1.solution[i] = c1;
                    child2.solution[i] = c2;
                }
            }
        }

        child1.evaluated = false;
        child1.feasible = false;
        child2.evaluated = false;
        child2.feasible = false;

        (child1, child2)
    }
}

/// Parent-Centric Crossover operator
pub struct ParentCentricCrossover {
    pub nparents: usize,
    pub noffspring: usize,
    pub eta: f64,
    pub zeta: f64
}

impl Crossover for ParentCentricCrossover {
    fn crossover(&self, parent1: &Solution, parent2: &Solution, _rng: &mut SmallRng) -> (Solution, Solution) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();

        for (i, solution_type) in parent1.problem.solution_data_types.iter().enumerate() {
            match solution_type {
                SolutionDataTypes::Real(real) => {
                    let lower = real.lower_bound.unwrap_or(f64::MIN);
                    let upper = real.upper_bound.unwrap_or(f64::MAX);
                    let mut sum = 0.0;
                    for _ in 0..self.nparents {
                        sum += parent1.solution[i] + parent2.solution[i];
                    }
                    let avg = sum / self.nparents as f64;
                    let mut c1 = parent1.solution[i] + self.eta * (avg - parent1.solution[i]);
                    let mut c2 = parent2.solution[i] + self.eta * (avg - parent2.solution[i]);

                    c1 = c1.clamp(lower, upper);
                    c2 = c2.clamp(lower, upper);

                    child1.solution[i] = c1;
                    child2.solution[i] = c2;
                }
                SolutionDataTypes::Integer(integer) => {
                    let lower = integer.lower_bound.unwrap_or(i64::MIN) as f64;
                    let upper = integer.upper_bound.unwrap_or(i64::MAX) as f64;
                    let mut sum = 0.0;
                    for _ in 0..self.nparents {
                        sum += parent1.solution[i] + parent2.solution[i];
                    }
                    let avg = sum / self.nparents as f64;
                    let mut c1 = parent1.solution[i] + self.eta * (avg - parent1.solution[i]);
                    let mut c2 = parent2.solution[i] + self.eta * (avg - parent2.solution[i]);

                    c1 = c1.round().clamp(lower, upper);
                    c2 = c2.round().clamp(lower, upper);

                    child1.solution[i] = c1;
                    child2.solution[i] = c2;
                }
                _ => {}
            }
        }

        child1.evaluated = false;
        child1.feasible = false;
        child2.evaluated = false;
        child2.feasible = false;

        (child1, child2)
    }
}

// Unimodal Distribution Crossover (UDX) operator
pub struct UnimodalDistributionCrossover {
    pub probability: f64,
    pub distribution_index: f64,
    pub nparents: usize,
    pub zeta: f64,
    pub eta: f64

}

impl Crossover for UnimodalDistributionCrossover {
    fn crossover(&self, parent1: &Solution, parent2: &Solution, _rng: &mut SmallRng) -> (Solution, Solution) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();
        for (i, solution_type) in parent1.problem.solution_data_types.iter().enumerate() {
            if let SolutionDataTypes::Real(real) = solution_type {
                let lower = real.lower_bound.unwrap_or(f64::MIN);
                let upper = real.upper_bound.unwrap_or(f64::MAX);
                let mut sum = 0.0;
                for _j in 0..self.nparents {
                    sum += parent1.solution[i] + parent2.solution[i];
                }
                let avg = sum / self.nparents as f64;
                let mut c1 = parent1.solution[i] + self.eta * (avg - parent1.solution[i]);
                let mut c2 = parent2.solution[i] + self.eta * (avg - parent2.solution[i]);

                c1 = c1.clamp(lower, upper);
                c2 = c2.clamp(lower, upper);

                child1.solution[i] = c1;
                child2.solution[i] = c2;
            }
        }

        child1.evaluated = false;
        child1.feasible = false;
        child2.evaluated = false;
        child2.feasible = false;
        (child1, child2)
    }
}

// Blend Crossover (BLX) operator
pub struct BlendCrossover {
    pub probability: f64,
    pub alpha: f64,
}

impl Crossover for BlendCrossover {
    fn crossover(&self, parent1: &Solution, parent2: &Solution, _rng: &mut SmallRng) -> (Solution, Solution) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();
        for (i, solution_type) in parent1.problem.solution_data_types.iter().enumerate() {
            if let SolutionDataTypes::Real(real) = solution_type {
                let lower = real.lower_bound.unwrap_or(f64::MIN);
                let upper = real.upper_bound.unwrap_or(f64::MAX);
                let mut c1 = 0.5 * (1.0 + self.alpha) * parent1.solution[i] + 0.5 * (1.0 - self.alpha) * parent2.solution[i];
                let mut c2 = 0.5 * (1.0 + self.alpha) * parent2.solution[i] + 0.5 * (1.0 - self.alpha) * parent1.solution[i];

                c1 = c1.clamp(lower, upper);
                c2 = c2.clamp(lower, upper);

                child1.solution[i] = c1;
                child2.solution[i] = c2;
            }
        }

        child1.evaluated = false;
        child1.feasible = false;
        child2.evaluated = false;
        child2.feasible = false;
        (child1, child2)
    }
}

// Uniform Crossover for integer types
pub struct UniformCrossover {
    pub probability: f64,
}

impl Crossover for UniformCrossover {
    fn crossover(&self, parent1: &Solution, parent2: &Solution, rng: &mut SmallRng) -> (Solution, Solution) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();

        for (i, solution_type) in parent1.problem.solution_data_types.iter().enumerate() {
            match solution_type {
                SolutionDataTypes::Integer(integer) => {
                    let lower = integer.lower_bound.unwrap_or(i64::MIN);
                    let upper = integer.upper_bound.unwrap_or(i64::MAX);
                    for j in 0..32 {
                        if rng.gen::<f64>() < self.probability {
                            let mask = 1 << j;
                            let c1 = (parent1.solution[i] as i64 & mask) | (parent2.solution[i] as i64 & !mask);
                            let c2 = (parent2.solution[i] as i64 & mask) | (parent1.solution[i] as i64 & !mask);
                            child1.solution[i] = c1.clamp(lower, upper) as f64;
                            child2.solution[i] = c2.clamp(lower, upper) as f64;
                        }
                    }
                }
                SolutionDataTypes::BitBinary(_) => {
                    if rng.gen::<f64>() < self.probability {
                        let c1 = (parent1.solution[i] as i64) ^ (1); // Flip the bit for child1
                        let c2 = (parent2.solution[i] as i64) ^ (1); // Flip the bit for child2
                        child1.solution[i] = c1 as f64;
                        child2.solution[i] = c2 as f64;
                    }
                }
                _ => {}
            }
        }

        child1.evaluated = false;
        child1.feasible = false;
        child2.evaluated = false;
        child2.feasible = false;
        (child1, child2)
    }
}

// Arithmetic Crossover for integer types
pub struct ArithmeticCrossover {
    pub probability: f64,
}

impl Crossover for ArithmeticCrossover {
    fn crossover(&self, parent1: &Solution, parent2: &Solution, _rng: &mut SmallRng) -> (Solution, Solution) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();
        for (i, solution_type) in parent1.problem.solution_data_types.iter().enumerate() {
            if let SolutionDataTypes::Integer(integer) = solution_type {
                let lower = integer.lower_bound.unwrap_or(i64::MIN);
                let upper = integer.upper_bound.unwrap_or(i64::MAX);
                let c1 = (parent1.solution[i] + parent2.solution[i]) / 2.;
                let c2 = (parent1.solution[i] + parent2.solution[i]) / 2.;
                child1.solution[i] = c1.clamp(lower as f64, upper as f64);
                child2.solution[i] = c2.clamp(lower as f64, upper as f64);
            }
        }

        child1.evaluated = false;
        child1.feasible = false;
        child2.evaluated = false;
        child2.feasible = false;
        (child1, child2)
    }
}



/// CrossoverManager to manage and apply different crossover operations
pub struct CrossoverManager {
    default_real_crossover: Box<dyn Crossover + Send>,
    default_integer_crossover: Box<dyn Crossover + Send>,
    default_binary_crossover: Box<dyn Crossover + Send>,
}

impl CrossoverManager {
    /// Creates a new `CrossoverManager` with default crossovers
    pub fn new() -> Self {
        Self {
            default_real_crossover: Box::new(SimulatedBinaryCrossover::new(None, None)),
            default_integer_crossover: Box::new(UniformCrossover { probability: 1.0 }),
            default_binary_crossover: Box::new(UniformCrossover { probability: 1.0 }),
        }
    }

    /// Sets the default crossover for Real types
    pub fn set_default_real_crossover(&mut self, crossover: Box<dyn Crossover + Send>) {
        self.default_real_crossover = crossover;
    }

    /// Sets the default crossover for Integer types
    pub fn set_default_integer_crossover(&mut self, crossover: Box<dyn Crossover + Send>) {
        self.default_integer_crossover = crossover;
    }

    /// Sets the default crossover for BitBinary types
    pub fn set_default_binary_crossover(&mut self, crossover: Box<dyn Crossover + Send>) {
        self.default_binary_crossover = crossover;
    }

    /// Performs crossover on the given parents and returns the children
    pub fn perform_crossover(
        &self,
        parent1: &Solution,
        parent2: &Solution,
        rng: &mut SmallRng,
    ) -> Vec<Solution> {
        let types = &parent1.problem.solution_data_types;

        // Each operator only touches genes of its own type (others pass through unchanged),
        // so call each *once* on the whole pair and take each gene from the operator that
        // owns its type. Previously this ran a full crossover per gene — O(D²) work and ~2D
        // full-solution clones per pair; now it is O(D) with at most 3 operator calls.
        let need = |f: fn(&SolutionDataTypes) -> bool| types.iter().any(f);
        let real = if need(|t| matches!(t, SolutionDataTypes::Real(_))) {
            Some(self.default_real_crossover.crossover(parent1, parent2, rng))
        } else { None };
        let integer = if need(|t| matches!(t, SolutionDataTypes::Integer(_))) {
            Some(self.default_integer_crossover.crossover(parent1, parent2, rng))
        } else { None };
        let binary = if need(|t| matches!(t, SolutionDataTypes::BitBinary(_))) {
            Some(self.default_binary_crossover.crossover(parent1, parent2, rng))
        } else { None };

        // Single gene type (the common case — all-Real/Integer/Binary problems): the operator
        // already produced the two children as parent clones with its own genes crossed, so return
        // them directly — no second parent-clone pair + per-gene merge. RNG use is unchanged, so
        // this is bit-identical to the merge path (~2 Solution clones/pair instead of ~4).
        let present = real.is_some() as u8 + integer.is_some() as u8 + binary.is_some() as u8;
        let (mut child1, mut child2) = if present == 1 {
            real.or(integer).or(binary).expect("one gene type present")
        } else {
            // Mixed types (rare): merge each operator's own-typed genes into a parent-clone base.
            let mut c1 = parent1.clone();
            let mut c2 = parent2.clone();
            for (i, solution_type) in types.iter().enumerate() {
                let res = match solution_type {
                    SolutionDataTypes::Real(_) => real.as_ref(),
                    SolutionDataTypes::Integer(_) => integer.as_ref(),
                    SolutionDataTypes::BitBinary(_) => binary.as_ref(),
                };
                if let Some((rc1, rc2)) = res {
                    c1.solution[i] = rc1.solution[i];
                    c2.solution[i] = rc2.solution[i];
                }
            }
            (c1, c2)
        };

        child1.evaluated = false;
        child1.feasible = false;
        child2.evaluated = false;
        child2.feasible = false;

        vec![child1, child2]
    }
}


// Unit tests for SimulatedBinaryCrossover
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::sync::Arc;

    fn test_rng() -> SmallRng {
        SmallRng::seed_from_u64(0)
    }
    use crate::core::{EvalFn, Problem, Solution};
    use crate::gatypes::{SolutionDataTypes, Real, BitBinary, Integer};

    fn setup_problem() -> Problem {
        Problem {
            solution_length: 5,
            number_of_objectives: 1,
            objective_constraint: Some(vec![Some(10.0)]),
            objective_constraint_operands: Some(vec![Some("<".to_string())]),
            direction: Some(vec![1]),
            solution_data_types: vec![
                SolutionDataTypes::BitBinary(BitBinary::new()),
                SolutionDataTypes::Integer(Integer::new(Some(10), Some(2000))),
                SolutionDataTypes::Real(Real::new(Some(10.0), Some(1000.0))),
                SolutionDataTypes::Real(Real::new(Some(10.0), Some(1000.0))),
                SolutionDataTypes::Real(Real::new(Some(10.0), Some(1000.0))),
            ],
            variable_constraints: None,
            eval_fn: EvalFn::Single(|x| vec![x.iter().sum()]),
        }
    }
    // Create solutions
    fn setup_solutions(problem: Arc<Problem>) -> Vec<Solution> {
        vec![
            Solution {
                problem: Arc::clone(&problem),
                solution: vec![1.0, 10.0, 10.0, 10.0, 10.0],
                objective_fitness_values: Default::default(),
                constraint_values: Default::default(),
                constraint_violation: 0,
                feasible: false,
                evaluated: false,
            },
            Solution {
                problem: Arc::clone(&problem),
                solution: vec![1.0, 20.0, 20.0, 20.0, 20.0],
                objective_fitness_values: Default::default(),
                constraint_values: Default::default(),
                constraint_violation: 0,
                feasible: false,
                evaluated: false,
            },
            Solution {
                problem: Arc::clone(&problem),
                solution: vec![0.0, 15.0, 15.0, 15.0, 15.0],
                objective_fitness_values: Default::default(),
                constraint_values: Default::default(),
                constraint_violation: 0,
                feasible: false,
                evaluated: false,
            },
        ]
    }

    #[test]
    fn test_sbx_crossover() {
        let problem = Arc::new(setup_problem());
        let solutions = setup_solutions(Arc::clone(&problem));
        let sbx = SimulatedBinaryCrossover::new(None, None);
        let (_child1, _child2) = sbx.crossover(&solutions[0], &solutions[2], &mut test_rng());
    }

    #[test]
    fn test_sbx_crossover_same_values() {
        let sbx = SimulatedBinaryCrossover::new(None, None);
        let (c1, c2) = sbx.sbx_crossover(10.0, 10.0, 0.0, 30.0, &mut test_rng());
        assert_eq!(c1, 10.0);
        assert_eq!(c2, 10.0);
    }

    #[test]
    fn test_sbx_crossover_same_values_with_bounds() {
        let sbx = SimulatedBinaryCrossover::new(None, None);
        let (c1, c2) = sbx.sbx_crossover(10.0, 10.0, 10.0, 10.0, &mut test_rng());
        assert_eq!(c1, 10.0);
        assert_eq!(c2, 10.0);
    }

    #[test]
    fn test_uniform_crossover_with_bitbinary_and_integer() {
        let problem = Arc::new(setup_problem());
        let parent1 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![1.0, 10.0, 10.0, 10.0, 10.0],
            objective_fitness_values: Default::default(),
            constraint_values: Default::default(),
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let parent2 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![0.0, 20.0, 20.0, 20.0, 20.0],
            objective_fitness_values: Default::default(),
            constraint_values: Default::default(),
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let uc = UniformCrossover { probability: 1.0 };
        let (child1, child2) = uc.crossover(&parent1, &parent2, &mut test_rng());
        assert_ne!(child1.solution, parent1.solution);
        assert_ne!(child2.solution, parent2.solution);
    }

    #[test]
    fn test_parent_centric_crossover() {
        let problem = Arc::new(setup_problem());
        let parent1 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![1.0, 10.0, 20.0, 30.0, 40.0],
            objective_fitness_values: Default::default(),
            constraint_values: Default::default(),
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let parent2 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![0.0, 60.0, 70.5, 80.2, 90.3],
            objective_fitness_values: Default::default(),
            constraint_values: Default::default(),
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let pcc = ParentCentricCrossover { nparents: 2, noffspring: 2, eta: 0.5, zeta: 0.5 };
        let (child1, child2) = pcc.crossover(&parent1, &parent2, &mut test_rng());
        assert_ne!(child1.solution, parent1.solution);
        assert_ne!(child2.solution, parent2.solution);
    }

    #[test]
    fn test_unimodal_distribution_crossover() {
        let problem = Arc::new(setup_problem());
        let parent1 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![1.0, 10.0, 20.0, 30.0, 40.0],
            objective_fitness_values: Default::default(),
            constraint_values: Default::default(),
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let parent2 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![0.0, 60.0, 70.5, 80.2, 90.3],
            objective_fitness_values: Default::default(),
            constraint_values: Default::default(),
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let udc = UnimodalDistributionCrossover { probability: 1.0, distribution_index: 20.0, nparents: 2, zeta: 0.5, eta: 0.5 };
        let (child1, child2) = udc.crossover(&parent1, &parent2, &mut test_rng());
        assert_ne!(child1.solution, parent1.solution);
        assert_ne!(child2.solution, parent2.solution);
    }

    #[test]
    fn test_default_crossover_manager() {
        let problem = Arc::new(setup_problem());
        let parent1 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![1.0, 10.0, 20.0, 30.0, 40.0],
            objective_fitness_values: Default::default(),
            constraint_values: Default::default(),
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let parent2 = Solution {
            problem: Arc::clone(&problem),
            solution: vec![0.0, 60.0, 70.5, 80.2, 90.3],
            objective_fitness_values: Default::default(),
            constraint_values: Default::default(),
            constraint_violation: 0,
            feasible: false,
            evaluated: false,
        };
        let cm = CrossoverManager::new();
        let children = cm.perform_crossover(&parent1, &parent2, &mut test_rng());
        assert_ne!(children[0].solution, parent1.solution);
        assert_ne!(children[1].solution, parent2.solution);
    }
}
