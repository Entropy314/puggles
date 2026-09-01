//! Single-objective GA comparison — puggles vs genevo vs genetic_algorithm.
//!
//! genevo and genetic_algorithm are SINGLE-objective GA frameworks (scalar fitness);
//! they cannot produce Pareto fronts, so they can't run the multi-objective ZDT1/DTLZ2
//! benchmarks. This puts all three on equal footing: minimize the Rastrigin function
//! (continuous, N-dim, global minimum 0 at the origin).
//!
//! APPLES-TO-APPLES: all three get the SAME population size and the SAME budget of
//! *actual objective-function evaluations* (NFE) — enforced by a shared atomic counter
//! inside `rastrigin`, not by generation count. This matters because genetic_algorithm
//! caches fitness (unchanged chromosomes aren't re-evaluated), so an equal *generation*
//! budget would let it do far fewer real evaluations than the others. genevo is stepped
//! manually until it hits NFE; genetic_algorithm's generation count is calibrated to NFE.
//! Operators still differ per library (no crate exposes an identical set) — this measures
//! each library at a fixed evaluation budget, not an identical algorithm.
//!
//! Run:  cargo run --release --example bench_singleobj

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const N: usize = 10; // dimensions
const POP: usize = 100;
const NFE: usize = 20_000; // equal evaluation budget for every library
const RUNS: usize = 15; // more runs → stable means (Rastrigin is multi-modal; puggles +
                        // genetic_algorithm are seeded per run for reproducibility)
const BOUND: f64 = 5.12;

/// Counts every real objective evaluation, across all libraries, so the budget is comparable.
static EVALS: AtomicUsize = AtomicUsize::new(0);
fn reset_evals() {
    EVALS.store(0, Ordering::Relaxed);
}
fn evals() -> usize {
    EVALS.load(Ordering::Relaxed)
}

/// Rastrigin: f(x) = 10n + Σ(xᵢ² − 10·cos(2π·xᵢ)).  Global min f=0 at x=0.
fn rastrigin(x: &[f64]) -> f64 {
    EVALS.fetch_add(1, Ordering::Relaxed);
    use std::f64::consts::PI;
    10.0 * x.len() as f64
        + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos()).sum::<f64>()
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

// ── puggles (NSGA-II with a single objective) ────────────────────────────────
fn bench_puggles() -> Row {
    use puggles::core::{EvalFn, Problem};
    use puggles::gatypes::{Real, SolutionDataTypes};
    use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
    use std::sync::Arc;

    fn obj(x: &Vec<f64>) -> Vec<f64> {
        vec![rastrigin(x)]
    }

    let types: Vec<SolutionDataTypes> = (0..N)
        .map(|_| SolutionDataTypes::Real(Real::new(Some(-BOUND), Some(BOUND))))
        .collect();
    let problem = Arc::new(Problem {
        solution_length: N,
        number_of_objectives: 1,
        objective_constraint: None,
        objective_constraint_operands: None,
        direction: Some(vec![-1]),
        solution_data_types: types,
        variable_constraints: None,
        eval_fn: EvalFn::Single(obj),
    });

    NSGAII::new(Arc::clone(&problem), POP, ExecutionMode::Sequential).run(NFE); // warm-up

    let (mut times, mut bests, mut nevals) = (Vec::new(), Vec::new(), Vec::new());
    for run in 0..RUNS {
        reset_evals();
        let mut ga =
            NSGAII::new(Arc::clone(&problem), POP, ExecutionMode::Sequential).with_seed(run as u64);
        let t0 = Instant::now();
        ga.run(NFE); // run() counts its own NFE → ~NFE real evaluations
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

// ── genevo — stepped manually until it reaches NFE evaluations ─────────────────
fn bench_genevo() -> Row {
    use genevo::operator::prelude::*;
    use genevo::population::{build_population, ValueEncodedGenomeBuilder};
    use genevo::prelude::*;

    type Genome = Vec<f64>;

    #[derive(Clone, Debug)]
    struct RastriginFitness;
    impl FitnessFunction<Genome, i64> for RastriginFitness {
        fn fitness_of(&self, g: &Genome) -> i64 {
            (-rastrigin(g) * 1000.0) as i64 // genevo maximizes → encode minimization
        }
        fn average(&self, values: &[i64]) -> i64 {
            values.iter().sum::<i64>() / values.len() as i64
        }
        fn highest_possible_fitness(&self) -> i64 {
            0
        }
        fn lowest_possible_fitness(&self) -> i64 {
            -100_000_000
        }
    }

    // Run one search, stopping the manual step loop once NFE real evaluations are spent.
    let run_once = || -> f64 {
        let initial_population: Population<Genome> = build_population()
            .with_genome_builder(ValueEncodedGenomeBuilder::new(N, -BOUND, BOUND))
            .of_size(POP)
            .uniform_at_random();

        let mut sim = simulate(
            genetic_algorithm()
                .with_evaluation(RastriginFitness)
                .with_selection(MaximizeSelector::new(0.7, 2))
                .with_crossover(MultiPointCrossBreeder::new(3))
                .with_mutation(RandomValueMutator::new(0.1, -BOUND, BOUND))
                .with_reinsertion(ElitistReinserter::new(RastriginFitness, true, 0.7))
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

// ── genetic_algorithm — generation count calibrated to NFE (it caches fitness) ─
fn bench_genetic_algorithm() -> Row {
    use genetic_algorithm::genotype::MutationType;
    use genetic_algorithm::strategy::evolve::prelude::*;

    #[derive(Clone, Debug)]
    struct RastriginFitness;
    impl Fitness for RastriginFitness {
        type Genotype = RangeGenotype<f64>;
        fn calculate_for_chromosome(
            &mut self,
            chromosome: &FitnessChromosome<Self>,
            _genotype: &FitnessGenotype<Self>,
        ) -> Option<FitnessValue> {
            Some((rastrigin(&chromosome.genes) * 1000.0) as FitnessValue)
        }
    }

    let run_gens = |gens: usize, seed: u64| -> f64 {
        let genotype = RangeGenotype::builder()
            .with_genes_size(N)
            .with_allele_range(-BOUND..=BOUND)
            .with_mutation_type(MutationType::Range(0.5))
            .build()
            .unwrap();
        let evolve = Evolve::builder()
            .with_genotype(genotype)
            .with_fitness(RastriginFitness)
            .with_fitness_ordering(FitnessOrdering::Minimize)
            .with_target_population_size(POP)
            .with_max_generations(gens)
            .with_select(SelectTournament::new(0.5, 0.02, 4))
            .with_crossover(CrossoverUniform::new(0.7, 0.8))
            .with_mutate(MutateSingleGene::new(0.2))
            .with_rng_seed_from_u64(seed)
            .call()
            .unwrap();
        evolve.best_fitness_score().unwrap() as f64 / 1000.0
    };

    // Calibrate: measure evaluations per generation (caching makes this < POP), then
    // pick the generation count that spends ≈ NFE real evaluations.
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
    println!("\n# Rastrigin (single-objective) — N={N}  pop={POP}  budget={NFE} evals  {RUNS} runs");
    println!(
        "{:<20} {:>10} {:>9} {:>10} {:>13}",
        "library", "ms/run", "±std", "evals", "best f (→0)"
    );
    let hr = "─".repeat(66);
    println!("{hr}");

    for (name, r) in [
        ("puggles", bench_puggles()),
        ("genevo", bench_genevo()),
        ("genetic_algorithm", bench_genetic_algorithm()),
    ] {
        println!(
            "{:<20} {:>10.1} {:>9.1} {:>10.0} {:>13.4}",
            name, r.ms, r.ms_std, r.evals, r.best
        );
        println!("RESULT\t{name}\t{:.1}\t{:.1}\t{:.0}\t{:.4}", r.ms, r.ms_std, r.evals, r.best);
    }

    println!("{hr}");
    println!("# equal budget: ~{NFE} real objective evaluations each (shared atomic counter)");
    println!("# best f = mean best Rastrigin reached (lower = better; global optimum = 0)");
}
