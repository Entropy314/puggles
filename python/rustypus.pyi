from typing import Callable, List, Optional, Union

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

class Problem:
    solution_length: int
    number_of_objectives: int
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
    ) -> None: ...

class GpuProblem:
    """
    Optimization problem whose objective function runs on the GPU via a WGSL compute shader.

    The shader must bind:
      @binding(0)  var<storage, read>        solutions  : array<f32>  # flat input
      @binding(1)  var<storage, read_write>  objectives : array<f32>  # flat output
      @binding(2)  var<uniform>              params     : Params       # { solution_length, num_objectives, pop_size }

    Each invocation (global_invocation_id.x) processes one solution.
    """
    solution_length: int
    number_of_objectives: int
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
    def __init__(
        self,
        problem: Union[Problem, GpuProblem],
        population_size: int = 100,
        execution_mode: str = "sequential",
        crossover_config: Optional["CrossoverConfig"] = None,
        mutation_config: Optional["MutationConfig"] = None,
        num_threads: Optional[int] = None,
    ) -> None: ...
    def run(
        self,
        max_nfe: int,
        callback: Optional[
            Callable[[List[Solution], List[Solution], int], Optional[bool]]
        ] = None,
    ) -> None: ...
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
    uniform_probability: float
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
        uniform_probability: float = 1.0,
    ) -> None: ...

class MutationConfig:
    real_mutation: str
    integer_mutation: str
    binary_mutation: str
    probability: float
    polynomial_distribution_index: float
    gaussian_std_dev: float
    def __init__(
        self,
        real_mutation: str = "uniform",
        integer_mutation: str = "uniform",
        binary_mutation: str = "bitflip",
        probability: float = 1.0,
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
