# Graph Report - .  (2026-07-15)

## Corpus Check
- Corpus is ~16,483 words - fits in a single context window. You may not need a graph.

## Summary
- 318 nodes · 774 edges · 14 communities
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 51 edges (avg confidence: 0.8)
- Token cost: 159,124 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_NSGA-II Engine|NSGA-II Engine]]
- [[_COMMUNITY_Crossover Operators|Crossover Operators]]
- [[_COMMUNITY_Mutation Operators|Mutation Operators]]
- [[_COMMUNITY_NSGA-III Many-Objective|NSGA-III Many-Objective]]
- [[_COMMUNITY_Problem & Solution Core|Problem & Solution Core]]
- [[_COMMUNITY_Benchmark Objectives|Benchmark Objectives]]
- [[_COMMUNITY_Solution Data Types|Solution Data Types]]
- [[_COMMUNITY_GPU Evaluator|GPU Evaluator]]
- [[_COMMUNITY_Pareto Dominance & Sorting|Pareto Dominance & Sorting]]
- [[_COMMUNITY_Crowding Tournament Selector|Crowding Tournament Selector]]
- [[_COMMUNITY_Quality Metrics|Quality Metrics]]
- [[_COMMUNITY_Checkpoint Serialization|Checkpoint Serialization]]

## God Nodes (most connected - your core abstractions)
1. `NSGAII` - 47 edges
2. `NSGAIII` - 37 edges
3. `Solution` - 20 edges
4. `Problem` - 17 edges
5. `fast_non_dominated_sort()` - 15 edges
6. `CrossoverManager` - 15 edges
7. `MutationManager` - 15 edges
8. `Crossover` - 14 edges
9. `Mutation` - 13 edges
10. `GpuEvaluator` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Default operators adapt per variable type` --rationale_for--> `MutationManager`  [INFERRED]
  GUIDE.md → src/genetic_operators/mutation.rs
- `Constraint-based Pareto dominance` --conceptually_related_to--> `ParetoDominance`  [INFERRED]
  GUIDE.md → src/dominance.rs
- `ExecutionMode (Sequential/MultiThreaded/GPU with CPU fallback)` --references--> `ExecutionMode`  [INFERRED]
  GUIDE.md → src/genetic_algorithms_v2.rs
- `NSGA-II algorithm` --references--> `NSGAII`  [INFERRED]
  GUIDE.md → src/genetic_algorithms_v2.rs
- `Default operators adapt per variable type` --rationale_for--> `CrossoverManager`  [INFERRED]
  GUIDE.md → src/genetic_operators/crossover.rs

## Import Cycles
- 1-file cycle: `src/checkpoint.rs -> src/checkpoint.rs`
- 1-file cycle: `src/core.rs -> src/core.rs`
- 1-file cycle: `src/genetic_operators/crossover.rs -> src/genetic_operators/crossover.rs`
- 1-file cycle: `src/genetic_operators/mutation.rs -> src/genetic_operators/mutation.rs`
- 1-file cycle: `src/genetic_algorithms_v2.rs -> src/genetic_algorithms_v2.rs`
- 1-file cycle: `src/gatypes.rs -> src/gatypes.rs`
- 1-file cycle: `src/genetic_operators/selectors.rs -> src/genetic_operators/selectors.rs`
- 1-file cycle: `src/metrics.rs -> src/metrics.rs`
- 1-file cycle: `src/nsga3.rs -> src/nsga3.rs`

## Hyperedges (group relationships)
- **NSGA-II generational evolution loop** — src_genetic_algorithms_v2_nsgaii, src_dominance_fast_non_dominated_sort, src_dominance_crowding_distance, genetic_operators_selectors_crowdingtournamentselector, genetic_operators_crossover_crossovermanager, genetic_operators_mutation_mutationmanager [INFERRED 0.85]
- **Pareto-front quality metrics** — src_metrics_objective_vectors, src_metrics_hypervolume_2d, src_metrics_igd, src_metrics_spacing [INFERRED 0.85]
- **Typed solution/gene generation** — src_core_problem, src_gatypes_solutiondatatypes, src_gatypes_bitbinary, src_gatypes_integer, src_gatypes_real [INFERRED 0.85]
- **Crossover operator strategy pattern** — genetic_operators_crossover_crossovermanager, genetic_operators_crossover_crossover, genetic_operators_crossover_simulatedbinarycrossover, genetic_operators_crossover_uniformcrossover [INFERRED 0.85]
- **Mutation operator strategy pattern** — genetic_operators_mutation_mutationmanager, genetic_operators_mutation_mutation, genetic_operators_mutation_bitflipmutation, genetic_operators_mutation_uniformmutation [INFERRED 0.85]
- **Per-variable-type operator dispatch** — genetic_operators_crossover_crossovermanager, genetic_operators_mutation_mutationmanager, src_gatypes_solutiondatatypes [INFERRED 0.75]

## Communities (14 total, 0 thin omitted)

### Community 0 - "NSGA-II Engine"
Cohesion: 0.10
Nodes (36): Duration, GaState, GpuEvaluator, crowding_distance(), ExecutionMode, NSGAII, Arc, AtomicUsize (+28 more)

### Community 1 - "Crossover Operators"
Cohesion: 0.12
Nodes (29): Box, ArithmeticCrossover, BlendCrossover, Crossover, CrossoverManager, DifferentialEvolutionCrossover, ParentCentricCrossover, setup_problem() (+21 more)

### Community 2 - "Mutation Operators"
Cohesion: 0.13
Nodes (26): BitFlipMutation, GaussianMutation, Mutation, MutationManager, PolynomialMutation, setup_problem(), setup_solution(), SolutionDataTypes (+18 more)

### Community 3 - "NSGA-III Many-Objective"
Cohesion: 0.16
Nodes (20): ExecutionMode, associate(), das_dennis(), dtlz_like(), normalize(), NSGAIII, Arc, AtomicUsize (+12 more)

### Community 4 - "Problem & Solution Core"
Cohesion: 0.17
Nodes (20): SolutionDataTypes, EvalFn, Problem, Arc, Option, Self, SmallRng, Vec (+12 more)

### Community 5 - "Benchmark Objectives"
Cohesion: 0.15
Nodes (24): NSGA-II algorithm, dtlz1(), dtlz2(), dtlz3(), dtlz4(), dtlz5(), dtlz6(), dtlz7() (+16 more)

### Community 6 - "Solution Data Types"
Cohesion: 0.23
Nodes (13): BitBinary, Integer, Real, Option, Self, SmallRng, SolutionDataTypes, test_bit_binary_generation() (+5 more)

### Community 7 - "GPU Evaluator"
Cohesion: 0.16
Nodes (13): BindGroupLayout, Buffer, ComputePipeline, Device, Queue, Batch evaluation (EvalFn::Batch bypasses per-solution parallelism), ExecutionMode (Sequential/MultiThreaded/GPU with CPU fallback), as_bytes() (+5 more)

### Community 8 - "Pareto Dominance & Sorting"
Cohesion: 0.19
Nodes (13): Constraint-based Pareto dominance, Dominance, fast_non_dominated_sort(), ParetoDominance, Send, Solution, Sync, Vec (+5 more)

### Community 9 - "Crowding Tournament Selector"
Cohesion: 0.27
Nodes (7): crowding_compare(), CrowdingTournamentSelector, test_crowding_tournament_selector(), Option, Self, SmallRng, Vec

### Community 10 - "Quality Metrics"
Cohesion: 0.29
Nodes (7): euclidean(), hypervolume_2d(), igd(), objective_vectors(), Solution, Vec, spacing()

### Community 11 - "Checkpoint Serialization"
Cohesion: 0.29
Nodes (7): GaState, Arc, Problem, Self, Solution, Vec, SolutionRecord

## Knowledge Gaps
- **26 isolated node(s):** `Self`, `Problem`, `Send`, `Sync`, `Default` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `NSGAII` connect `NSGA-II Engine` to `Crossover Operators`, `Mutation Operators`, `NSGA-III Many-Objective`, `Problem & Solution Core`, `Benchmark Objectives`, `GPU Evaluator`, `Pareto Dominance & Sorting`, `Crowding Tournament Selector`, `Quality Metrics`, `Checkpoint Serialization`?**
  _High betweenness centrality (0.498) - this node is a cross-community bridge._
- **Why does `NSGAIII` connect `NSGA-III Many-Objective` to `NSGA-II Engine`, `Crossover Operators`, `Mutation Operators`, `Problem & Solution Core`, `Pareto Dominance & Sorting`, `Crowding Tournament Selector`?**
  _High betweenness centrality (0.216) - this node is a cross-community bridge._
- **Why does `CrossoverManager` connect `Crossover Operators` to `NSGA-II Engine`, `NSGA-III Many-Objective`, `Solution Data Types`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `NSGAII` (e.g. with `NSGA-II algorithm` and `EvalFn`) actually correct?**
  _`NSGAII` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `NSGAIII` (e.g. with `ParetoDominance` and `ExecutionMode`) actually correct?**
  _`NSGAIII` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Solution` (e.g. with `Crossover` and `Mutation`) actually correct?**
  _`Solution` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Problem` (e.g. with `EvalFn` and `SolutionDataTypes`) actually correct?**
  _`Problem` has 3 INFERRED edges - model-reasoned connections that need verification._