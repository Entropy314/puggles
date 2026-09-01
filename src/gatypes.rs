// import random crate
use rand::Rng;
use rand::rngs::SmallRng;


// Create Enum for types called SolutionType
#[derive(Debug, Clone)]
pub enum SolutionDataTypes {
    Real(Real),
    Integer(Integer),
    BitBinary(BitBinary)
}

#[derive(Debug, Clone)]
pub struct BitBinary {
}

// create a method on BitBinary that randomly genrates a 1 or 0 for value
impl BitBinary {
    pub fn new() -> Self {
        Self { }
    }

    pub fn generate_value(&self, rng: &mut SmallRng) -> Option<i64> {
        Some(rng.gen_range(0..2))
    }
}


// Create an Integer object where the lower and upper bounds are optional parameters
#[derive(Debug, Clone)]
pub struct Integer {
    pub lower_bound: Option<i64>,
    pub upper_bound: Option<i64>
}

// create a method on Integer that randomly genrates a number between the lower and upper bounds
impl Integer {
    // When creating a new Integer object, the lower and upper bounds are optional parameters check if lower < upper with a panic
    pub fn new(lower_bound: Option<i64>, upper_bound: Option<i64>) -> Self {
        if lower_bound.is_some() && upper_bound.is_some() {
            if lower_bound.unwrap() > upper_bound.unwrap() {
                panic!("Lower bound must be less than upper bound");
            } else if lower_bound.unwrap() == upper_bound.unwrap() {
                panic!("Lower bound must not be equal to upper bound");
            } 
        }
        Self {
            lower_bound,
            upper_bound
        }
    }
    // Create Generate Value Method
    pub fn generate_value(&self, rng: &mut SmallRng) -> Option<i64> {
        Some(rng.gen_range(self.lower_bound.unwrap_or(i64::MIN)..self.upper_bound.unwrap_or(i64::MAX)))
    }
}

#[derive(Debug, Clone)]
pub struct Real {
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>
}

// create a method on Real that randomly genrates a number between the lower and upper bounds
impl Real {
    // When creating a new Real object, the lower and upper bounds are optional parameters check if lower < upper with a panic
    pub fn new(lower_bound: Option<f64>, upper_bound: Option<f64>) -> Self {
        if lower_bound.is_some() && upper_bound.is_some() {
            if lower_bound.unwrap() > upper_bound.unwrap() {
                panic!("Lower bound must be less than upper bound");
            } else if lower_bound.unwrap() == upper_bound.unwrap() {
                panic!("Lower bound must not be equal to upper bound");
            }
        }
        Self {
            lower_bound,
            upper_bound
        }
    }

    pub fn generate_value(&self, rng: &mut SmallRng) -> Option<f64> {
        Some(rng.gen_range(self.lower_bound.unwrap_or(f64::MIN)..self.upper_bound.unwrap_or(f64::MAX)))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_bit_binary_generation() {
        let mut rng = SmallRng::seed_from_u64(0);
        let bit_binary = BitBinary::new();
        for _ in 0..100 {
            let value = bit_binary.generate_value(&mut rng).unwrap();
            assert!(value == 0 || value == 1);
        }
    }

    #[test]
    fn test_integer_generation_with_bounds() {
        let mut rng = SmallRng::seed_from_u64(0);
        let integer = Integer::new(Some(10), Some(20));
        for _ in 0..100 {
            let value = integer.generate_value(&mut rng).unwrap();
            assert!(value >= 10 && value < 20);
        }
    }

    #[test]
    fn test_real_generation_with_bounds() {
        let mut rng = SmallRng::seed_from_u64(0);
        let real = Real::new(Some(10.0), Some(20.0));
        for _ in 0..100 {
            let value = real.generate_value(&mut rng).unwrap();
            assert!(value >= 10.0 && value < 20.0);
        }
    }

    #[test]
    fn test_integer_generation_without_bounds() {
        let mut rng = SmallRng::seed_from_u64(0);
        let integer = Integer::new(None, None);
        // Any i64 is in range, so the only meaningful property is that generation succeeds
        // and does not keep returning the same value.
        let values: Vec<i64> = (0..100).map(|_| integer.generate_value(&mut rng).unwrap()).collect();
        assert!(values.iter().any(|&v| v != values[0]), "unbounded generation is not random");
    }

    #[test]
    #[should_panic(expected = "Lower bound must be less than upper bound")]
    fn test_integer_invalid_bounds() {
        Integer::new(Some(20), Some(10));
    }

    #[test]
    #[should_panic(expected = "Lower bound must be less than upper bound")]
    fn test_real_invalid_bounds() {
        Real::new(Some(20.0), Some(10.0));
    }
}