"""Dashboard add-on tests: drive the native server over real HTTP. Run with `pytest python-dashboard/tests`."""

import json
import math
import threading
import time
import urllib.request

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


def get_json(dash, path):
    with urllib.request.urlopen(dash.url + path, timeout=5) as r:
        return json.load(r)


def post(dash, path):
    req = urllib.request.Request(dash.url + path, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def test_live_run_records_meta_and_frames(problem):
    ga = pg.NSGAII(problem, population_size=20, seed=1)
    with pg.Dashboard(ga, objective_names=["cost", "risk"], open_browser=False) as dash:
        assert dash.state == "waiting"
        ga.run(400, callback=dash)
        assert dash.state == "done", "passing the dashboard as callback marks the run finished"

        meta = get_json(dash, "api/meta")
        assert meta["n_obj"] == 2 and meta["n_var"] == 6
        assert meta["objective_names"] == ["cost", "risk"]
        assert meta["algorithm"] == "NSGAII"
        assert meta["variable_types"][0] == ["real", 0.0, 1.0]

        hist = get_json(dash, "api/history")
        assert hist["state"] == "done"
        frames = hist["frames"]
        assert len(frames) == dash.frame_count > 1
        last = frames[-1]
        assert last["nfe"] == ga.nfe
        assert len(last["F"]) == len(ga.get_archive())
        assert len(last["X"][0]) == 6 and len(last["PF"]) == 20


def test_native_objective_runs_without_the_gil_and_nsgaiii_records():
    # The built-in DTLZ functions return len(x) - 1 objectives: 7 variables -> 6 objectives.
    problem = pg.create_benchmark_problem("dtlz2", 7, 6, [(0.0, 1.0)] * 7)
    ga = pg.NSGAIII(problem, divisions=2, execution_mode="multithreaded", seed=2)
    with pg.Dashboard(problem, open_browser=False) as dash:
        ga.run(3_000, callback=dash)
        assert dash.state == "done"
        hist = get_json(dash, "api/history")
        assert len(hist["frames"]) > 5
        assert len(hist["frames"][-1]["F"][0]) == 6
        assert get_json(dash, "api/meta")["n_obj"] == 6


def test_stop_button_ends_the_run(problem):
    ga = pg.NSGAII(problem, population_size=20, seed=3)
    dash = pg.Dashboard(problem, open_browser=False)

    def press_stop():
        deadline = time.time() + 10
        while dash.frame_count < 3 and time.time() < deadline:
            time.sleep(0.01)
        post(dash, "api/stop")

    t = threading.Thread(target=press_stop)
    t.start()
    ga.run(10_000_000, callback=dash)
    t.join()
    assert ga.nfe < 10_000_000
    assert dash.state == "stopped"
    dash.close()


def test_sse_streams_frames(problem):
    dash = pg.Dashboard(problem, open_browser=False)
    ga = pg.NSGAII(problem, population_size=20, seed=4)
    ga.run(200, callback=dash)
    with urllib.request.urlopen(dash.url + "api/events", timeout=5) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        seen = b""
        while b"event: state" not in seen or b"event: frame" not in seen:
            seen += r.read1(4096)
    assert b'"state":"done"' in seen
    dash.close()


def test_generic_callback_form_and_python_errors(problem):
    dash = pg.Dashboard(open_browser=False)
    ga = pg.NSGAII(problem, population_size=20, seed=5)
    calls = []

    def mine(archive, population, nfe):
        calls.append(nfe)
        return dash(archive, population, nfe)

    ga.run(300, callback=mine)
    assert dash.frame_count == len(calls) > 0
    assert dash.state == "running", "a wrapped dashboard isn't finished automatically"
    dash.finish()
    assert dash.state == "done"

    def boom(*_):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pg.NSGAIII(problem, divisions=4).run(500, callback=boom)
    dash.close()


@pytest.mark.parametrize("name", ["run.pgrun.json", "run.pgrun.json.gz"])
def test_save_load_and_standalone_html(problem, tmp_path, name):
    ga = pg.NSGAII(problem, population_size=20, seed=6)
    with pg.Dashboard(ga, title="zdt1", open_browser=False) as dash:
        ga.run(300, callback=dash)
        path = dash.save(tmp_path / name)
        html = dash.export_html(tmp_path / "run.html").read_text()
        frames = dash.frame_count
    assert "window.PUGGLES_DATA = " in html and '"kind":"run"' in html

    replay = pg.dashboard.load(path, open_browser=False, block=False)
    assert replay.state == "replay"
    assert replay.frame_count == frames
    assert get_json(replay, "api/meta")["title"] == "zdt1"
    replay.close()

    from puggles.dashboard.__main__ import main
    out = tmp_path / "cli.html"
    assert main([str(path), "--html", str(out)]) == 0
    assert "window.PUGGLES_DATA = " in out.read_text()


def test_show_finished_optimizer(problem):
    ga = pg.NSGAII(problem, population_size=20, seed=7)
    with pytest.raises(ValueError):
        pg.show(ga, open_browser=False, block=False)
    ga.run(300)
    dash = pg.show(ga, open_browser=False, block=False)
    assert dash.state == "done" and dash.frame_count == 1
    assert len(get_json(dash, "api/history")["frames"][0]["F"]) == len(ga.get_archive())
    dash.close()


def test_problem_metadata_getters():
    p = pg.Problem(3, 2, [pg.Real(-1.0, 1.0), pg.Integer(0, 9), pg.BitBinary()],
                   objective_function=lambda x: [x[0], x[1]], direction=[-1, 1])
    assert p.direction == [-1, 1]
    assert p.variable_types == [("real", -1.0, 1.0), ("integer", 0.0, 9.0), ("binary", 0.0, 1.0)]


SPHERE_WGSL = """
@group(0) @binding(0) var<storage, read> solutions: array<f32>;
@group(0) @binding(1) var<storage, read_write> objectives: array<f32>;
struct Params { solution_length: u32, num_objectives: u32, pop_size: u32 };
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if (s >= params.pop_size) { return; }
    let x = solutions[s * 2u];
    objectives[s * 2u] = x * x;
    objectives[s * 2u + 1u] = (x - 1.0) * (x - 1.0) + solutions[s * 2u + 1u];
}
"""


@pytest.mark.skipif(not hasattr(pg, "GpuProblem"), reason="built without the gpu feature")
def test_gpu_problem_is_reported_as_gpu():
    try:
        problem = pg.GpuProblem(2, 2, [pg.Real(0.0, 1.0)] * 2, SPHERE_WGSL)
    except BaseException as e:  # no adapter on this machine (pyo3 surfaces the panic)
        pytest.skip(f"no usable GPU: {e}")
    ga = pg.NSGAII(problem, population_size=20, execution_mode="gpu", seed=8)
    with pg.Dashboard(ga, open_browser=False) as dash:
        ga.run(400, callback=dash)
        meta = get_json(dash, "api/meta")
        last = get_json(dash, "api/history")["frames"][-1]
    assert meta["execution"]["effective"] == "GPU"
    assert meta["execution"]["device"]
    assert last["gpu_evals"] == last["nfe"] == ga.nfe

    with pytest.raises(ValueError, match="GpuProblem"):
        pg.NSGAIII(problem, divisions=4).run(100)


def test_pause_and_resume_over_http(problem):
    ga = pg.NSGAII(problem, population_size=20, seed=9)
    dash = pg.Dashboard(problem, open_browser=False)
    assert post(dash, "api/pause")["paused"] is True
    assert dash.paused

    def resume_later():
        deadline = time.time() + 10
        while dash.state != "paused" and time.time() < deadline:
            time.sleep(0.01)
        frames_while_paused = dash.frame_count
        time.sleep(0.3)
        assert dash.frame_count == frames_while_paused, "a paused run must not advance"
        post(dash, "api/resume")

    t = threading.Thread(target=resume_later)
    t.start()
    ga.run(400, callback=dash)
    t.join()
    assert ga.nfe == 400 and dash.state == "done"
    dash.close()


def test_keep_runs_compare_and_export(problem, tmp_path):
    with pg.Dashboard(problem, open_browser=False, keep_runs=True) as dash:
        pg.NSGAII(problem, population_size=20, seed=1).run(300, callback=dash)
        pg.NSGAIII(problem, divisions=6, seed=1).run(300, callback=dash)
        assert dash.comparisons == ["run 1 · NSGAII"]
        assert get_json(dash, "api/meta")["run_label"] == "run 2 · NSGAIII"
        assert get_json(dash, "api/compare")[0]["name"] == "run 1 · NSGAII"
        saved = dash.save(tmp_path / "b.pgrun.json.gz")
        html = dash.export_html(tmp_path / "both.html").read_text()
        assert '"comparisons":[{"name":"run 1 · NSGAII"' in html

        other = pg.Dashboard(problem, open_browser=False)
        pg.NSGAII(problem, population_size=20, seed=3).run(300, callback=other)
        dash.compare(other, name="seed 3")
        dash.compare(saved)
        assert dash.comparisons == ["run 1 · NSGAII", "seed 3", "b"]
        other.close()

    three = pg.Problem(4, 3, [pg.Real(0.0, 1.0)] * 4, objective_function=lambda x: [x[0], x[1], x[2]])
    with pg.Dashboard(three, open_browser=False) as wrong:
        with pytest.raises(ValueError, match="objectives"):
            wrong.compare(saved)

    from puggles.dashboard.__main__ import main
    out = tmp_path / "cli.html"
    assert main([str(saved), str(saved), "--html", str(out)]) == 0
    assert '"comparisons":[{"name":"b"' in out.read_text()


def test_reference_front_timing_and_notebook(problem):
    front = pg.reference_front("zdt1", 2, 50)
    assert len(front) == 50 and front[0] == [0.0, 1.0]
    assert len(pg.reference_front("dtlz2", 3, 100)[0]) == 3
    with pytest.raises(ValueError):
        pg.reference_front("knapsack")

    with pg.Dashboard(problem, open_browser=False, reference_front=front) as dash:
        pg.NSGAII(problem, population_size=20, seed=4).run(300, callback=dash)
        assert len(get_json(dash, "api/meta")["reference_front"]) == 50
        timing = get_json(dash, "api/history")["frames"][-1]["timing"]
        assert set(timing) == {"variation", "evaluation", "selection", "archive", "record"}
        assert dash._repr_html_().startswith(f'<iframe src="{dash.url}"')


def test_groups_columns_events_and_thinning(problem):
    calls = []

    def bad(x):
        raise ValueError("boom")

    with pg.Dashboard(problem, open_browser=False, keep_runs=True, group="A", max_points=7,
                      columns={"sum": lambda x: (calls.append(1), sum(x))[1], "bad": bad}) as dash:
        pg.NSGAII(problem, population_size=20, seed=1).run(200, callback=dash)
        dash.add_event("tweaked something")
        dash.set_group("B")
        pg.NSGAII(problem, population_size=20, seed=2).run(200, callback=dash)
        frame = get_json(dash, "api/history")["frames"][-1]
        meta = get_json(dash, "api/meta")
        compare = get_json(dash, "api/compare")
    assert calls, "the column function is called"
    assert len(frame["E"]["sum"]) == len(frame["F"])
    assert abs(frame["E"]["sum"][0] - sum(frame["X"][0])) < 1e-9
    assert all(v is None for v in frame["E"]["bad"]), "a raising column yields null (NaN), not a crash"
    assert len(frame["PF"]) == 7 and frame["Pn"] == 20
    assert meta["group"] == "B" and compare[0]["meta"]["group"] == "A"
    assert compare[0]["meta"]["events"][0]["label"] == "tweaked something"


def test_token_protects_a_dashboard(problem):
    import urllib.error
    with pg.Dashboard(problem, open_browser=False, token="letmein") as dash:
        assert dash.token == "letmein" and dash.url.endswith("/?token=letmein")
        base = dash.url.split("?")[0]
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "api/meta", timeout=5)
        assert err.value.code == 403
        with urllib.request.urlopen(dash.url, timeout=5) as r:
            cookie = r.headers["Set-Cookie"].split(";")[0]
        req = urllib.request.Request(base + "api/meta", headers={"Cookie": cookie})
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200
    with pg.Dashboard(problem, open_browser=False, host="0.0.0.0") as open_dash:
        assert open_dash.token and "token=" in open_dash.url, "non-loopback hosts get a token by default"


def test_goals_steer_a_python_nsga3_run():
    def dtlz2(x, m=3):
        g = sum((v - 0.5) ** 2 for v in x[m - 1:])
        out = []
        for i in range(m):
            v = 1 + g
            for xj in x[: m - 1 - i]:
                v *= math.cos(xj * math.pi / 2)
            if i > 0:
                v *= math.sin(x[m - 1 - i] * math.pi / 2)
            out.append(v)
        return out

    problem = pg.Problem(8, 3, [pg.Real(0.0, 1.0)] * 8, objective_function=dtlz2)
    goal = [0.1, 0.1, 0.99]
    dist = lambda F: sum(math.dist(r, goal) for r in F) / len(F)
    free = pg.NSGAIII(problem, divisions=8, seed=4)
    free.run(6000)
    with pg.Dashboard(problem, open_browser=False) as dash:
        body = json.dumps({"goals": goal, "steer": True, "spread": 0.15}).encode()
        req = urllib.request.Request(dash.url + "api/goals", data=body, method="POST", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        steered = pg.NSGAIII(problem, divisions=8, seed=4)
        steered.run(6000, callback=dash)
        meta = get_json(dash, "api/meta")
    assert meta["steerable"] is True and meta["goals"] == goal
    assert meta["events"][0]["label"].startswith("steered to goals")
    front = lambda ga: [s.objectives for s in ga.get_archive()]
    assert dist(front(steered)) < 0.6 * dist(front(free)), "steering concentrates the front near the goal"


# -- the add-on and the core wheel -------------------------------------------------------------

def test_the_core_serves_the_add_on():
    import puggles_dashboard

    assert pg.Dashboard is puggles_dashboard.Dashboard
    assert pg.dashboard.load is puggles_dashboard.load
    assert pg.reference_front is puggles_dashboard.reference_front
    assert puggles_dashboard.__version__ == pg.__version__, "install matching puggles and puggles-dashboard"


class Spy:
    """A callback that forwards the native recorder (or doesn't) and counts Python calls."""

    def __init__(self, dash, recorder):
        self.dash, self.recorder, self.calls = dash, recorder, 0
        if recorder is not None:
            self.__puggles_recorder__ = recorder

    def __call__(self, archive, population, nfe):
        self.calls += 1
        return self.dash(archive, population, nfe)


@pytest.mark.parametrize("algorithm", ["NSGAII", "NSGAIII"])
def test_run_records_natively_without_python_calls(problem, algorithm):
    kwargs = {"divisions": 6} if algorithm == "NSGAIII" else {}
    ga = getattr(pg, algorithm)(problem, population_size=20, seed=4, **kwargs)
    with pg.Dashboard(ga, open_browser=False) as dash:
        spy = Spy(dash, dash.__puggles_recorder__)
        ga.run(400, callback=spy)
        assert spy.calls == 0, "the native recorder must be used, not the Python callback"
        assert dash.state == "done"
        last = get_json(dash, "api/history")["frames"][-1]
        assert last["nfe"] == ga.nfe and len(last["F"]) == len(ga.get_archive())
        assert get_json(dash, "api/meta")["algorithm"] == algorithm


def test_pure_rust_objective_records_with_the_gil_released():
    problem = pg.create_benchmark_problem("dtlz2", 7, 3, [(0.0, 1.0)] * 7)
    ga = pg.NSGAII(problem, population_size=40, seed=2, execution_mode="multithreaded")
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(1)
            time.sleep(0.001)

    with pg.Dashboard(problem, open_browser=False) as dash:
        t = threading.Thread(target=ticker)
        t.start()
        ga.run(60_000, callback=dash)
        stop.set()
        t.join()
        assert dash.frame_count > 10
    assert len(ticks) > 5, "another Python thread must run while a watched pure-Rust run is going"


def test_an_unusable_recorder_falls_back_to_the_callback(problem):
    import ctypes

    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    wrong_name = ctypes.c_char_p(b"someone.else.v9")  # kept alive: the capsule borrows it
    offers = [lambda: 42, lambda: new_capsule(ctypes.addressof(ctypes.c_int(1)), wrong_name, None)]
    for offer in offers:
        ga = pg.NSGAII(problem, population_size=20, seed=5)
        with pg.Dashboard(ga, open_browser=False) as dash:
            spy = Spy(dash, offer)
            with pytest.warns(RuntimeWarning, match="recording through the Python callback"):
                ga.run(300, callback=spy)
            assert spy.calls > 0 and dash.frame_count == spy.calls


def test_generic_callback_thins_the_population(problem):
    ga = pg.NSGAII(problem, population_size=50, seed=6)
    with pg.Dashboard(ga, open_browser=False, max_points=10) as dash:
        ga.run(500, callback=lambda a, p, n: dash(a, p, n))
        last = get_json(dash, "api/history")["frames"][-1]
        assert len(last["PF"]) == 10 and last["Pn"] == 50
