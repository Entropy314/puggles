import os
import pathlib
from typing import Callable, Dict, List, Optional, Tuple, Union

__version__: str

class Real:
    lower_bound: float
    upper_bound: float
    def __init__(self, lower_bound: float, upper_bound: float) -> None: ...

class Integer:
    lower_bound: int
    upper_bound: int
    def __init__(self, lower_bound: int, upper_bound: int) -> None: ...

class BitBinary:
    def __init__(self) -> None: ...

class Solution:
    variables: List[float]
    objectives: List[float]
    constraints: List[float]
    evaluated: bool
    feasible: bool
    constraint_violation: int
    constraint_violation_magnitude: float

# callback(archive, population, nfe) -> False to stop early. A dashboard from the optional
# add-on (`pip install "puggles[dashboard]"`) works here too: it offers a native recorder through
# `__puggles_recorder__()`, which `run()` uses instead of calling it from Python each generation.
GenerationCallback = Callable[[List["Solution"], List["Solution"], int], Optional[bool]]

class Problem:
    solution_length: int
    number_of_objectives: int
    direction: List[int]
    """Per objective: -1 = minimize, 1 = maximize."""
    variable_types: List[Tuple[str, Optional[float], Optional[float]]]
    """Per variable: (kind, lower_bound, upper_bound); kind is "real", "integer" or "binary"."""
    permutation: bool
    """True when the problem uses permutation encoding (``permutation=True``)."""
    def __init__(
        self,
        solution_length: int,
        number_of_objectives: int,
        solution_data_types: List[Union[Real, Integer, BitBinary]],
        objective_function: Optional[Callable[[List[float]], List[float]]] = None,
        direction: Optional[List[int]] = None,
        objective_constraints: Optional[List[Optional[float]]] = None,
        constraint_operands: Optional[List[Optional[str]]] = None,
        batch_objective_function: Optional[
            Callable[[List[List[float]]], List[List[float]]]
        ] = None,
        permutation: bool = False,
    ) -> None: ...

class GpuProblem:
    """
    Optimization problem whose objective function runs on the GPU via a WGSL compute shader.

    Only in builds with the ``gpu`` feature (``maturin develop --release --features gpu``); the
    PyPI wheels are CPU-only, so check ``hasattr(puggles, "GpuProblem")``.

    The shader binds, in ``@group(0)``:
      @binding(0)  var<storage, read>        solutions  : array<f32>  # flat input
      @binding(1)  var<storage, read_write>  objectives : array<f32>  # flat output
      @binding(2)  var<uniform>              params     : Params       # { solution_length, num_objectives, pop_size }
      @binding(3)  var<storage, read>        data       : array<f32>  # optional auxiliary data (empty from Python)

    Each invocation (global_invocation_id.x) processes one solution.
    """
    solution_length: int
    number_of_objectives: int
    direction: List[int]
    variable_types: List[Tuple[str, Optional[float], Optional[float]]]
    def __init__(
        self,
        solution_length: int,
        number_of_objectives: int,
        solution_data_types: List[Union[Real, Integer, BitBinary]],
        shader_wgsl: str,
        direction: Optional[List[int]] = None,
    ) -> None: ...

class NSGAII:
    nfe: int
    problem: Union[Problem, GpuProblem]
    def __init__(
        self,
        problem: Union[Problem, GpuProblem],
        population_size: int = 100,
        execution_mode: str = "sequential",
        crossover_config: Optional["CrossoverConfig"] = None,
        mutation_config: Optional["MutationConfig"] = None,
        num_threads: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None: ...
    def run(
        self,
        max_nfe: int,
        callback: Optional[GenerationCallback] = None,
    ) -> None:
        """Called as callback(archive, population, nfe) after each generation; return False to stop."""
    def get_archive(self) -> List[Solution]: ...
    def get_population(self) -> List[Solution]: ...
    def archive_hypervolume(self, reference: Tuple[float, float]) -> float:
        """Hypervolume of the last run's archive against a nadir reference point.

        Computed in minimization space (maximized objectives are negated). 2 objectives only.
        """
        ...
    def save_state(self) -> str:
        """Serialize the last run (population, archive, nfe) to a JSON string.

        The objective function is not stored. To resume: rebuild the same Problem,
        call load_state(), then run() with the cumulative budget.
        Raises RuntimeError if called before the first run().
        """
        ...
    def load_state(self, state: str) -> None:
        """Restore a checkpoint from save_state(); applied on the next run()."""
        ...
    def run_until_hv_converged(
        self,
        max_nfe: int,
        reference: Tuple[float, float],
        patience: int = 20,
        epsilon: float = 1e-6,
    ) -> int:
        """Run until the budget is spent or archive hypervolume plateaus. 2 objectives only.

        Diversity-aware alternative to a fixed budget: hypervolume responds to the front
        filling in, not just extending. Returns the number of generations run.
        """
        ...

class NSGAIII:
    """Reference-point many-objective GA. Prefer over NSGAII for 3+ objectives.
    Supports batch objectives and a GpuProblem (gpu builds)."""
    nfe: int
    problem: Union[Problem, GpuProblem]
    def __init__(
        self,
        problem: Union[Problem, GpuProblem],
        population_size: int = 0,
        divisions: int = 12,
        execution_mode: str = "sequential",
        seed: Optional[int] = None,
    ) -> None: ...
    def run(self, max_nfe: int, callback: Optional[GenerationCallback] = None) -> None:
        """Called as callback(archive, population, nfe) after each generation; return False to stop."""
    def get_archive(self) -> List[Solution]: ...
    def get_population(self) -> List[Solution]: ...

class CrossoverConfig:
    real_crossover: str
    integer_crossover: str
    binary_crossover: str
    sbx_probability: float
    sbx_distribution_index: float
    de_probability: float
    de_scaling_factor: float
    blend_alpha: float
    uniform_probability: Optional[float]
    def __init__(
        self,
        real_crossover: str = "sbx",
        integer_crossover: str = "uniform",
        binary_crossover: str = "uniform",
        sbx_probability: float = 1.0,
        sbx_distribution_index: float = 20.0,
        de_probability: float = 0.9,
        de_scaling_factor: float = 0.8,
        blend_alpha: float = 0.5,
        uniform_probability: Optional[float] = None,
    ) -> None: ...

class MutationConfig:
    real_mutation: str
    integer_mutation: str
    binary_mutation: str
    probability: Optional[float]
    polynomial_distribution_index: float
    gaussian_std_dev: float
    def __init__(
        self,
        real_mutation: str = "polynomial",
        integer_mutation: str = "polynomial",
        binary_mutation: str = "bitflip",
        probability: Optional[float] = None,
        polynomial_distribution_index: float = 20.0,
        gaussian_std_dev: float = 0.1,
    ) -> None: ...

# Benchmark functions
def paraboloid_hyper_5(x: List[float]) -> List[float]: ...
def paraboloid_5(x: List[float]) -> List[float]: ...
def paraboloid_5_loc(x: List[float]) -> List[float]: ...
def simple_objective(x: List[float]) -> List[float]: ...
def xyz_objective(x: List[float]) -> List[float]: ...
def dtlz1(x: List[float]) -> List[float]: ...
def dtlz2(x: List[float]) -> List[float]: ...
def dtlz3(x: List[float]) -> List[float]: ...
def dtlz4(x: List[float], alpha: float = 100.0) -> List[float]: ...
def dtlz5(x: List[float]) -> List[float]: ...
def dtlz6(x: List[float]) -> List[float]: ...
def dtlz7(x: List[float]) -> List[float]: ...
def create_benchmark_problem(
    name: str,
    solution_length: int,
    number_of_objectives: int,
    bounds: List[tuple],
    direction: Optional[List[int]] = None,
) -> Problem: ...


# ---------------------------------------------------------------------------
# Front-quality metrics. All assume MINIMIZATION — negate maximized objectives.
# ---------------------------------------------------------------------------

def hypervolume_2d(front: List[List[float]], reference: Tuple[float, float]) -> float:
    """Exact 2-objective hypervolume (dominated area) against a nadir reference point."""
    ...

def igd(front: List[List[float]], reference_set: List[List[float]]) -> float:
    """Inverted generational distance to a reference set. Lower is better."""
    ...

def spacing(front: List[List[float]]) -> float:
    """Schott spacing: std-dev of nearest-neighbour distances. Lower is more uniform."""
    ...


# ---------------------------------------------------------------------------
# Island-model parallelism
# ---------------------------------------------------------------------------

def run_island_model(
    problem: Problem,
    islands: int = 4,
    population_size: int = 50,
    max_nfe_per_island: int = 10_000,
    seed: Optional[int] = None,
) -> List[Solution]:
    """Run independent populations in parallel; return the merged Pareto front.

    Requires a native Rust objective (create_benchmark_problem). A Python callable is
    rejected because the GIL would serialize the islands.
    """
    ...
