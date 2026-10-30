"""Core wheel tests that hold with or without the optional dashboard add-on."""

import subprocess
import sys

import pytest

import puggles as pg


def zdt1(x):
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    return [x[0], g * (1 - (x[0] / g) ** 0.5)]


@pytest.fixture
def problem():
    return pg.Problem(
        solution_length=6,
        number_of_objectives=2,
        solution_data_types=[pg.Real(0.0, 1.0)] * 6,
        objective_function=zdt1,
        direction=[-1, -1],
    )


def front(ga):
    return [tuple(s.objectives) for s in ga.get_archive()]


@pytest.mark.parametrize("algorithm", ["NSGAII", "NSGAIII"])
def test_seeded_runs_reproduce(problem, algorithm):
    kwargs = {"divisions": 6} if algorithm == "NSGAIII" else {}
    make = lambda seed: getattr(pg, algorithm)(problem, population_size=24, seed=seed, **kwargs)  # noqa: E731
    a, b, c = make(1), make(1), make(2)
    for ga in (a, b, c):
        ga.run(1_500)
    assert front(a) == front(b) and front(a) != front(c)
    assert a.nfe == 1_500


def test_a_plain_callback_sees_each_generation(problem):
    seen = []
    ga = pg.NSGAII(problem, population_size=20, seed=3)
    ga.run(10_000, callback=lambda archive, population, nfe: seen.append(nfe) or len(seen) < 5)
    assert len(seen) == 5 and seen == sorted(seen), "returning False must stop the run"


def test_problem_describes_itself(problem):
    assert problem.direction == [-1, -1]
    assert problem.variable_types[0] == ("real", 0.0, 1.0)


def test_an_objective_that_drops_a_problem_does_not_deadlock(problem):
    # Python objectives used to be called with the callable registry locked; a callable that
    # dropped a Problem (whose destructor takes that lock) then deadlocked the run.
    def objective(x):
        pg.Problem(1, 1, [pg.Real(0.0, 1.0)], objective_function=lambda y: [y[0]])  # dropped at once
        return zdt1(x)

    p = pg.Problem(6, 2, [pg.Real(0.0, 1.0)] * 6, objective_function=objective)
    ga = pg.NSGAII(p, population_size=10, seed=1)
    ga.run(200)
    assert ga.nfe == 200


def test_import_does_not_load_the_dashboard():
    code = "import sys, puggles; print('puggles_dashboard' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def has_dashboard():
    try:
        import puggles_dashboard  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(has_dashboard(), reason="the dashboard add-on is installed")
@pytest.mark.parametrize("name", ["Dashboard", "show", "load", "reference_front"])
def test_without_the_add_on_its_names_explain_how_to_get_it(name):
    with pytest.raises(ImportError, match=r'pip install "puggles\[dashboard\]"'):
        getattr(pg, name)
    with pytest.raises(AttributeError):
        pg.not_a_thing  # noqa: B018


@pytest.mark.skipif(has_dashboard(), reason="the dashboard add-on is installed")
def test_without_the_add_on_the_cli_explains_how_to_get_it():
    out = subprocess.run([sys.executable, "-m", "puggles.dashboard", "run.pgrun.json"], capture_output=True, text=True)
    assert out.returncode != 0 and 'pip install "puggles[dashboard]"' in out.stderr


# -- features from 0.2.0, kept through the 0.3.0 merge ----------------------------------------

def test_permutation_encoding_keeps_permutations():
    pos = [0.0, 9.0, 3.0, 7.0, 1.0, 5.0, 8.0, 2.0]

    def tour(x):
        return [sum(abs(pos[int(a)] - pos[int(b)]) for a, b in zip(x, x[1:]))]

    problem = pg.Problem(8, 1, [pg.Integer(0, 7)] * 8, tour, permutation=True)
    assert problem.permutation
    ga = pg.NSGAII(problem, population_size=40, seed=5)
    ga.run(6000)
    for s in ga.get_population():
        assert sorted(int(v) for v in s.variables) == list(range(8))


def test_checkpoint_resumes_where_it_left_off(problem):
    ga = pg.NSGAII(problem, population_size=20, seed=1)
    ga.run(1_000)
    state = ga.save_state()
    resumed = pg.NSGAII(problem, population_size=20, seed=1)
    resumed.load_state(state)
    resumed.run(2_000)
    assert resumed.nfe == 2_000


def test_metrics_and_hypervolume_stop(problem):
    ga = pg.NSGAII(problem, population_size=30, seed=2)
    generations = ga.run_until_hv_converged(20_000, (11.0, 11.0), patience=10)
    assert generations > 0 and ga.nfe <= 20_000
    front = [s.objectives for s in ga.get_archive()]
    assert ga.archive_hypervolume((11.0, 11.0)) == pytest.approx(pg.hypervolume_2d(front, (11.0, 11.0)))
    assert pg.spacing(front) >= 0 and pg.igd(front, front) == 0


def test_island_model_and_nsga3_batch_objective():
    problem = pg.create_benchmark_problem("dtlz2", 4, 3, [(0.0, 1.0)] * 4)
    front = pg.run_island_model(problem, islands=3, population_size=20, max_nfe_per_island=1_000, seed=0)
    assert front

    def batch(population):
        return [zdt1(x) for x in population]

    p = pg.Problem(6, 2, [pg.Real(0.0, 1.0)] * 6, batch_objective_function=batch)
    ga = pg.NSGAIII(p, divisions=6, seed=3)
    ga.run(600)
    assert ga.nfe == 600 and ga.get_archive()
