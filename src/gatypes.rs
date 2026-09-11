use rand::Rng;
use rand::rngs::SmallRng;

use crate::core::ConfigError;

/// The decision-variable types a `Problem` can be built from.
///
/// Bounds are **inclusive on both ends**: `Integer::new(Some(0), Some(10))` generates and
/// mutates within `0..=10`. Every operator clamps to the same closed interval, so the value
/// space is identical whether a gene came from initial generation, crossover, or mutation.
#[derive(Debug, Clone)]
pub enum SolutionDataTypes {
    Real(Real),
    Integer(Integer),
    BitBinary(BitBinary)
}

#[derive(Debug, Clone)]
pub struct BitBinary {
}

impl BitBinary {
    pub fn new() -> Self {
        Self { }
    }

    pub fn generate_value(&self, rng: &mut SmallRng) -> Option<i64> {
        Some(rng.gen_range(0..2))
    }
}

impl Default for BitBinary {
    fn default() -> Self {
        Self::new()
    }
}

/// An integer decision variable over the **closed** interval `[lower_bound, upper_bound]`.
/// `None` means unbounded (the full `i64` range).
#[derive(Debug, Clone)]
pub struct Integer {
    pub lower_bound: Option<i64>,
    pub upper_bound: Option<i64>
}

impl Integer {
    /// Fallible constructor. Validates the *effective* bounds, so a one-sided `None` that
    /// collapses onto the `i64` extreme (e.g. `Some(i64::MAX)` with `None`) is rejected here
    /// rather than panicking later inside `rand`.
    pub fn try_new(lower_bound: Option<i64>, upper_bound: Option<i64>) -> Result<Self, ConfigError> {
        let lo = lower_bound.unwrap_or(i64::MIN);
        let hi = upper_bound.unwrap_or(i64::MAX);
        if lo > hi {
            return Err(ConfigError::new("Lower bound must be less than upper bound"));
        }
        if lo == hi {
            return Err(ConfigError::new("Lower bound must not be equal to upper bound"));
        }
        Ok(Self { lower_bound, upper_bound })
    }

    /// Panicking constructor. Use [`Integer::try_new`] to handle invalid bounds as an error.
    pub fn new(lower_bound: Option<i64>, upper_bound: Option<i64>) -> Self {
        Self::try_new(lower_bound, upper_bound).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Draw uniformly from the **closed** interval `[lower, upper]`.
    pub fn generate_value(&self, rng: &mut SmallRng) -> Option<i64> {
        Some(rng.gen_range(self.lower_bound.unwrap_or(i64::MIN)..=self.upper_bound.unwrap_or(i64::MAX)))
    }
}

/// A real decision variable over the **closed** interval `[lower_bound, upper_bound]`.
///
/// Both bounds must be finite and span a finite width. `None` is rejected at construction:
/// an unbounded real range overflows `rand`'s uniform sampler, so failing fast with a clear
/// message beats a `range overflow` panic from deep inside a dependency.
#[derive(Debug, Clone)]
pub struct Real {
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>
}

impl Real {
    /// Fallible constructor. Requires finite bounds spanning a finite, non-empty width.
    pub fn try_new(lower_bound: Option<f64>, upper_bound: Option<f64>) -> Result<Self, ConfigError> {
        let (lo, hi) = match (lower_bound, upper_bound) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => {
                return Err(ConfigError::new(
                    "Real requires finite lower and upper bounds — an unbounded real range \
                     overflows the uniform sampler. Supply Some(lower), Some(upper).",
                ))
            }
        };
        if !lo.is_finite() || !hi.is_finite() {
            return Err(ConfigError::new("Real bounds must be finite (no NaN or infinity)"));
        }
        if !(hi - lo).is_finite() {
            return Err(ConfigError::new("Real bounds span a non-finite width — narrow the range"));
        }
        if lo > hi {
            return Err(ConfigError::new("Lower bound must be less than upper bound"));
        }
        if lo == hi {
            return Err(ConfigError::new("Lower bound must not be equal to upper bound"));
        }
        Ok(Self { lower_bound, upper_bound })
    }

    /// Panicking constructor. Use [`Real::try_new`] to handle invalid bounds as an error.
    pub fn new(lower_bound: Option<f64>, upper_bound: Option<f64>) -> Self {
        Self::try_new(lower_bound, upper_bound).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Draw uniformly from the **closed** interval `[lower, upper]`.
    pub fn generate_value(&self, rng: &mut SmallRng) -> Option<f64> {
        // try_new guarantees both bounds are Some and finite.
        Some(rng.gen_range(self.lower_bound.unwrap()..=self.upper_bound.unwrap()))
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
            assert!((10..=20).contains(&value));
        }
    }

    /// Bounds are closed, so the upper bound must be reachable by generation — the same value
    /// space mutation and crossover clamp into. Before this was inclusive, mutation could
    /// produce a gene the initial population never could.
    #[test]
    fn test_integer_generation_reaches_upper_bound() {
        let mut rng = SmallRng::seed_from_u64(0);
        let integer = Integer::new(Some(0), Some(3));
        let seen: Vec<i64> = (0..500).map(|_| integer.generate_value(&mut rng).unwrap()).collect();
        assert!(seen.contains(&3), "closed upper bound must be generatable");
        assert!(seen.contains(&0), "closed lower bound must be generatable");
        assert!(seen.iter().all(|v| (0..=3).contains(v)));
    }

    #[test]
    fn test_real_generation_with_bounds() {
        let mut rng = SmallRng::seed_from_u64(0);
        let real = Real::new(Some(10.0), Some(20.0));
        for _ in 0..100 {
            let value = real.generate_value(&mut rng).unwrap();
            assert!((10.0..=20.0).contains(&value));
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

    /// One-sided `None` used to skip validation entirely, then panic inside `rand` with
    /// "range overflow" when the effective bounds collapsed to a single point.
    #[test]
    fn test_integer_one_sided_none_is_validated() {
        let err = Integer::try_new(Some(i64::MAX), None).unwrap_err();
        assert!(err.to_string().contains("must not be equal"));
    }

    /// Unbounded reals used to panic inside `rand` on the first `generate_value`.
    #[test]
    fn test_real_requires_finite_bounds() {
        assert!(Real::try_new(None, None).is_err());
        assert!(Real::try_new(Some(0.0), None).is_err());
        assert!(Real::try_new(Some(f64::NEG_INFINITY), Some(0.0)).is_err());
        assert!(Real::try_new(Some(f64::NAN), Some(1.0)).is_err());
        // A finite range that still overflows f64 subtraction is rejected too.
        assert!(Real::try_new(Some(f64::MIN), Some(f64::MAX)).is_err());
    }

    #[test]
    fn test_try_new_ok() {
        assert!(Integer::try_new(Some(0), Some(5)).is_ok());
        assert!(Real::try_new(Some(-1.0), Some(1.0)).is_ok());
    }
}
