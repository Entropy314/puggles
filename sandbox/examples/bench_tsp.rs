//! Permutation-encoded GA comparison — puggles vs genevo vs genetic_algorithm, on a TSP tour.
//!
//! §5 (bench_singleobj) compares the three on a real-valued problem. Permutation encoding
//! (added in `src/core.rs::Encoding::Permutation`) is a different representation entirely —
//! genes are a permutation of `0..N`, so whole-vector operators (order crossover, swap
//! mutation) replace the per-gene ones. This puts the same three libraries on that ground:
//! minimize the length of a closed tour over N random cities.
//!
//! Every library uses its own permutation machinery:
//!   - puggles: `Problem::with_permutation_encoding()` → OX1 crossover + swap mutation.
//!   - genevo: a custom `GenomeBuilder<Vec<usize>>` + its built-in `OrderOneCrossover` (OX1)
//!     and `SwapOrderMutator` (`genevo::operator::prelude`), the same OX1 family puggles uses.
//!   - genetic_algorithm: `UniqueGenotype` — its docs say it "does not support gene or point
//!     crossover" for permutation genomes, only `CrossoverClone`/`CrossoverRejuvenate` (no
//!     recombination at all). So this library's TSP search is swap-mutation + selection only —
//!     a real difference in what's being compared, not a tuning gap.
//!
//! APPLES-TO-APPLES: same equal-NFE-budget approach as bench_singleobj (see that file for why
//! generation count isn't a fair unit — genetic_algorithm caches fitness).
//!
//! Run:  cargo run --release --example bench_tsp

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

const N_CITIES: usize = 30;
const POP: usize = 60;
const NFE: usize = 60_000;
const RUNS: usize = 10;

/// Counts every real tour-length evaluation, across all libraries, so the budget is comparable.
static EVALS: AtomicUsize = AtomicUsize::new(0);
fn reset_evals() {
    EVALS.store(0, Ordering::Relaxed);
}
fn evals() -> usize {
    EVALS.load(Ordering::Relaxed)
}

// ponytail: a fixed splitmix64 stream instead of a `rand` dependency — sandbox doesn't
// otherwise need one, and this only has to be deterministic, not high-quality.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

static CITIES: LazyLock<Vec<(f64, f64)>> = LazyLock::new(|| {
    let mut rng = SplitMix64(7);
    (0..N_CITIES).map(|_| (rng.next_f64() * 100.0, rng.next_f64() * 100.0)).collect()
});

/// Total length of the closed tour visiting `order` (a permutation of `0..N_CITIES`).
fn tour_length(order: &[usize]) -> f64 {
    EVALS.fetch_add(1, Ordering::Relaxed);
    (0..order.len())
        .map(|i| {
            let (ax, ay) = CITIES[order[i]];
            let (bx, by) = CITIES[order[(i + 1) % order.len()]];
            ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
        })
        .sum()
}

fn stats(xs: &[f64]) -> (f64, f64) {
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let std = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64).sqrt();
    (mean, std)
}

struct Row {
    ms: f64,
    ms_std: f64,
    best: f64,
    evals: f64,
}

// ── puggles (NSGA-II, permutation encoding) ──────────────────────────────────
fn bench_puggles() -> Row {
    use puggles::core::{EvalFn, Problem};
    use puggles::gatypes::{Integer, SolutionDataTypes};
    use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
    use std::sync::Arc;

    fn obj(x: &Vec<f64>) -> Vec<f64> {
        let order: Vec<usize> = x.iter().map(|&v| v as usize).collect();
        vec![tour_length(&order)]
    }

    let types: Vec<SolutionDataTypes> = (0..N_CITIES)
        .map(|_| SolutionDataTypes::Integer(Integer::new(Some(0), Some(N_CITIES as i64 - 1))))
        .collect();
    let problem = Arc::new(
        Problem {
            solution_length: N_CITIES,
            number_of_objectives: 1,
            objective_constraint: None,
            objective_constraint_operands: None,
            direction: Some(vec![-1]),
            solution_data_types: types,
            variable_constraints: None,
            encoding: puggles::core::Encoding::PerGene,
            eval_fn: EvalFn::Single(obj),
        }
        .with_permutation_encoding(),
    );

    NSGAII::new(Arc::clone(&problem), POP, ExecutionMode::Sequential).run(NFE); // warm-up

    let (mut times, mut bests, mut nevals) = (Vec::new(), Vec::new(), Vec::new());
    for run in 0..RUNS {
        reset_evals();
        let mut ga =
            NSGAII::new(Arc::clone(&problem), POP, ExecutionMode::Sequential).with_seed(run as u64);
        let t0 = Instant::now();
        ga.run(NFE);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        nevals.push(evals() as f64);
        bests.push(
            ga.get_archive()
                .iter()
                .chain(ga.population.iter())
                .map(|s| s.objective_fitness_values[0])
                .fold(f64::INFINITY, f64::min),
        );
    }
    let (ms, ms_std) = stats(&times);
    Row { ms, ms_std, best: stats(&bests).0, evals: stats(&nevals).0 }
}

// ── genevo — custom permutation genome + its built-in OX1 crossover/swap mutation ─
fn bench_genevo() -> Row {
    use genevo::operator::prelude::*;
    use genevo::population::build_population;
    use genevo::prelude::*;
    use genevo::random::SliceRandom;

    type Genome = Vec<usize>;

    struct TourGenomeBuilder;
    impl GenomeBuilder<Genome> for TourGenomeBuilder {
        fn build_genome<R: Rng>(&self, _: usize, rng: &mut R) -> Genome {
            let mut order: Genome = (0..N_CITIES).collect();
            order.shuffle(rng);
            order
        }
    }

    #[derive(Clone, Debug)]
    struct TourFitness;
    impl FitnessFunction<Genome, i64> for TourFitness {
        fn fitness_of(&self, g: &Genome) -> i64 {
            (-tour_length(g) * 1000.0) as i64 // genevo maximizes → encode minimization
        }
        fn average(&self, values: &[i64]) -> i64 {
            values.iter().sum::<i64>() / values.len() as i64
        }
        fn highest_possible_fitness(&self) -> i64 {
            0
        }
        fn lowest_possible_fitness(&self) -> i64 {
            -1_000_000_000
        }
    }

    let run_once = || -> f64 {
        let initial_population: Population<Genome> = build_population()
            .with_genome_builder(TourGenomeBuilder)
            .of_size(POP)
            .uniform_at_random();

        let mut sim = simulate(
            genetic_algorithm()
                .with_evaluation(TourFitness)
                .with_selection(MaximizeSelector::new(0.7, 2))
                .with_crossover(OrderOneCrossover::new())
                .with_mutation(SwapOrderMutator::new(0.05))
                .with_reinsertion(ElitistReinserter::new(TourFitness, true, 0.7))
                .with_initial_population(initial_population)
                .build(),
        )
        .until(GenerationLimit::new(u64::MAX)) // never the real stop; we break on NFE
        .build();

        let mut best_fit = i64::MIN;
        loop {
            match sim.step() {
                Ok(SimResult::Intermediate(step)) => {
                    best_fit = best_fit.max(step.result.best_solution.solution.fitness);
                    if evals() >= NFE {
                        break;
                    }
                }
                Ok(SimResult::Final(step, _, _, _)) => {
                    best_fit = best_fit.max(step.result.best_solution.solution.fitness);
                    break;
                }
                Err(_) => break,
            }
        }
        -(best_fit as f64) / 1000.0
    };

    reset_evals();
    run_once(); // warm-up

    let (mut times, mut bests, mut nevals) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..RUNS {
        reset_evals();
        let t0 = Instant::now();
        let best = run_once();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        nevals.push(evals() as f64);
        bests.push(best);
    }
    let (ms, ms_std) = stats(&times);
    Row { ms, ms_std, best: stats(&bests).0, evals: stats(&nevals).0 }
}

// ── genetic_algorithm — UniqueGenotype: swap-mutation only, no crossover available ─
fn bench_genetic_algorithm() -> Row {
    use genetic_algorithm::strategy::evolve::prelude::*;

    #[derive(Clone, Debug)]
    struct TourFitness;
    impl Fitness for TourFitness {
        type Genotype = UniqueGenotype<usize>;
        fn calculate_for_chromosome(
            &mut self,
            chromosome: &FitnessChromosome<Self>,
            _genotype: &FitnessGenotype<Self>,
        ) -> Option<FitnessValue> {
            Some((tour_length(&chromosome.genes) * 1000.0) as FitnessValue)
        }
    }

    let run_gens = |gens: usize, seed: u64| -> f64 {
        let genotype = UniqueGenotype::builder()
            .with_allele_list((0..N_CITIES).collect())
            .build()
            .unwrap();
        let evolve = Evolve::builder()
            .with_genotype(genotype)
            .with_fitness(TourFitness)
            .with_fitness_ordering(FitnessOrdering::Minimize)
            .with_target_population_size(POP)
            .with_max_generations(gens)
            .with_select(SelectTournament::new(0.5, 0.02, 4))
            .with_crossover(CrossoverRejuvenate::new(0.8)) // no gene/point crossover for Unique
            .with_mutate(MutateSingleGene::new(0.2))
            .with_rng_seed_from_u64(seed)
            .call()
            .unwrap();
        evolve.best_fitness_score().unwrap() as f64 / 1000.0
    };

    const PROBE_GENS: usize = 200;
    reset_evals();
    run_gens(PROBE_GENS, 0);
    let per_gen = (evals() as f64 / PROBE_GENS as f64).max(1.0);
    let gens = ((NFE as f64 / per_gen).round() as usize).max(1);

    run_gens(gens, 0); // warm-up at the calibrated budget

    let (mut times, mut bests, mut nevals) = (Vec::new(), Vec::new(), Vec::new());
    for run in 0..RUNS {
        reset_evals();
        let t0 = Instant::now();
        let best = run_gens(gens, run as u64);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        nevals.push(evals() as f64);
        bests.push(best);
    }
    let (ms, ms_std) = stats(&times);
    Row { ms, ms_std, best: stats(&bests).0, evals: stats(&nevals).0 }
}

fn main() {
    println!("\n# TSP tour (permutation encoding) — {N_CITIES} cities  pop={POP}  budget={NFE} evals  {RUNS} runs");
    println!(
        "{:<20} {:>10} {:>9} {:>10} {:>13}",
        "library", "ms/run", "±std", "evals", "best tour"
    );
    let hr = "─".repeat(66);
    println!("{hr}");

    for (name, r) in [
        ("puggles", bench_puggles()),
        ("genevo", bench_genevo()),
        ("genetic_algorithm", bench_genetic_algorithm()),
    ] {
        println!(
            "{:<20} {:>10.1} {:>9.1} {:>10.0} {:>13.2}",
            name, r.ms, r.ms_std, r.evals, r.best
        );
        println!("RESULT\t{name}\t{:.1}\t{:.1}\t{:.0}\t{:.2}", r.ms, r.ms_std, r.evals, r.best);
    }

    println!("{hr}");
    println!("# equal budget: ~{NFE} real tour-length evaluations each (shared atomic counter)");
    println!("# best tour = mean shortest tour length reached (lower = better)");
}
