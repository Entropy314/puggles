use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

/// Tournament selector that uses rank and crowding distance (NSGA-II style).
/// Prefers lower rank; ties broken by higher crowding distance.
#[derive(Debug)]
pub struct CrowdingTournamentSelector {
    tournament_size: usize,
    rng: SmallRng,
}

impl CrowdingTournamentSelector {
    pub fn new(tournament_size: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
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
