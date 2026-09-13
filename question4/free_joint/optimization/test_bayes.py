"""Fast optimizer checks use an explicit synthetic objective, not UAV scores."""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from . import bayes


# This small evaluator is importable in Windows spawn workers. It is used only
# for proposal, checkpoint, failure handling and process-pool integration tests.
def resolve_configs(specification):
    result = copy.deepcopy(specification)
    for name, spec in result.items():
        spec.setdefault("label", name)
        if spec["policy"] != "test_objective":
            raise ValueError("This test evaluator is not a mission simulator")
    return result


def code_hashes():
    return {__file__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def run_one(job):
    seed, name, spec, directory, record = job
    parameters = spec["parameters"]
    value = 100 + (parameters["x"]-.28)**2 * 30 + (parameters["steps"]-4)**2
    value += seed % 7
    return make_row(seed, value, strategy=name, worker_pid=os.getpid())


def make_row(seed, value, **extra):
    result = dict(seed=seed, completed=True, source_count=10, cleared=10,
                  per_source_s=value, elapsed_per_source_s=value, time_s=value*10,
                  costs=dict(movement=value*10-100, measure=100., switch=0.,
                             clear_success=0., clear_failure=0.), wall_s=.001)
    result.update(extra)
    return result


SPACE = {"x": dict(type="float", low=0., high=1.),
         "steps": dict(type="int", low=2, high=7)}
BASE = dict(policy="test_objective", parameters=dict(x=.65, steps=5, unchanged=True), label="testing")


def test_mixed_space_round_trip_and_bounds():
    space = bayes.SearchSpace(dict(**SPACE, distance=dict(type="float", low=10., high=1000., scale="log")))
    fields = space.decode([.3, .49, .5])
    assert fields == {"x": .3, "steps": 4, "distance": pytest.approx(100.)}
    assert np.allclose(space.encode(fields), [.3, .4, .5])
    assert space.decode([-1., 2., 0.])["steps"] == 7
    with pytest.raises(ValueError, match="outside"):
        space.encode(dict(x=2., steps=3, distance=10.))
    with pytest.raises(ValueError, match="integer"):
        bayes.SearchSpace({"count": dict(type="int", low=.2, high=3)})


def test_failure_costs_and_partial_sources_cannot_win():
    seeds = [10, 11]
    complete = [make_row(10, 400.), make_row(11, 420.)]
    baseline = [make_row(10, 450.), make_row(11, 460.)]
    score = bayes.summarize(complete, seeds, baseline_rows=baseline)
    assert score["mean_per_source_s"] == 410.
    assert score["delta_from_baseline_s"] == -45.
    assert score["paired_mean_variance"] == 25.
    for edit in ({"completed": False, "per_source_s": None}, {"cleared": 9}, {"time_s": 1.}):
        failed = [make_row(10, 1., **edit), make_row(11, 2.)]
        outcome = bayes.summarize(failed, seeds, baseline_rows=baseline)
        assert not outcome["complete"]
        assert outcome["mean_per_source_s"] is None
        assert outcome["objective"] >= 1e6
        assert outcome["failed_seeds"] == [10]
    with pytest.raises(ValueError, match="exactly once"):
        bayes.summarize([complete[0], complete[0]], seeds)


def test_ei_and_matern_gp_recover_a_smooth_objective():
    x = np.linspace(0, 1, 12)[:, None]
    y = (x[:, 0]-.3)**2 * 20
    gp = bayes.GaussianProcess(x, y, np.zeros(len(x)))
    probes = np.linspace(0, 1, 51)[:, None]
    mean, sigma = gp.predict(probes)
    assert np.mean((mean-(probes[:, 0]-.3)**2*20)**2) < .01
    assert np.all(sigma > 0.)
    ei = bayes.expected_improvement(np.array([0., 2., 3.]), np.array([.1, .1, 5.]), 1.)
    assert ei[0] > ei[1]
    assert ei[2] > ei[1]


def test_proposals_have_eight_sobol_before_reproducible_ei():
    space = bayes.SearchSpace(SPACE)
    baseline_fields = {name: BASE["parameters"][name] for name in SPACE}
    trials = [dict(status="complete", proposal=dict(kind="baseline"),
                   searched_parameters=baseline_fields, vector=space.encode(baseline_fields).tolist(),
                   score=dict(delta_from_baseline_s=0., paired_mean_variance=0., complete=True))]
    for iteration in range(10):
        vector, proposal = bayes.propose(space, trials, 8, 42, pool_power=5)
        if iteration < 8:
            assert proposal["kind"] == "sobol"
        else:
            assert proposal["kind"] == "gp_expected_improvement"
            again, details = bayes.propose(space, trials, 8, 42, pool_power=5)
            assert np.array_equal(vector, again)
            assert proposal == details
        fields = space.decode(vector)
        assert fields not in [t["searched_parameters"] for t in trials]
        trials.append(dict(status="complete", proposal=proposal, searched_parameters=fields,
                           vector=vector.tolist(), score=dict(delta_from_baseline_s=float(np.sum(vector**2)),
                                                             paired_mean_variance=.01, complete=True)))


def test_process_pool_checkpoint_resume_and_best_config(tmp_path):
    arguments = dict(evaluator_module="question4.free_joint.optimization.test_bayes",
                     base_config=BASE, space_spec=SPACE, seeds=[71, 72], output=tmp_path,
                     trials=11, initial=8, workers=2, random_seed=82, pool_power=5)
    partial = bayes.run_search(**arguments, max_new_trials=3)
    assert partial["status"] == "checkpointed"
    assert len(partial["trials"]) == 3
    # The baseline is preserved bit-for-bit, including untuned parameters.
    assert partial["trials"][0]["config"] == BASE
    existing_rows = copy.deepcopy(partial["trials"][0]["rows"])
    # Model interruption halfway through a scene batch: a saved successful row
    # must survive, and only the missing seed should be submitted on resume.
    pending = partial["trials"][-1]
    saved_pending_row = copy.deepcopy(pending["rows"][0])
    pending["rows"] = [saved_pending_row]
    pending["status"] = "running"
    pending.pop("score")
    bayes.write_json(tmp_path / "checkpoint.json", partial)
    complete = bayes.run_search(**arguments, resume=True)
    assert complete["status"] == "complete"
    assert complete["trials"][0]["rows"] == existing_rows
    assert complete["trials"][2]["rows"][0] == saved_pending_row
    assert len(complete["trials"]) == 11
    assert sum(t["proposal"]["kind"] == "gp_expected_improvement" for t in complete["trials"]) == 2
    assert complete["best_mean_per_source_s"] <= complete["trials"][0]["score"]["objective"]
    assert all(t["score"]["complete"] for t in complete["trials"])
    assert len({r["worker_pid"] for t in complete["trials"][3:] for r in t["rows"]}) <= 2
    saved = bayes.read_json(tmp_path / "best_config.json")
    assert saved["parameters"]["unchanged"] is True
    assert len(bayes.read_json(tmp_path / "best_trajectory.json")) == 11
    with pytest.raises(ValueError, match="settings differ"):
        bayes.run_search(**dict(arguments, seeds=[73, 74]), resume=True)


def test_code_hash_guard_refuses_resumption(tmp_path, monkeypatch):
    import question4.free_joint.optimization.test_bayes as evaluator
    expected = bayes.source_hashes(evaluator)
    monkeypatch.setattr(evaluator, "code_hashes", lambda: {"deliberately_changed.py": "changed"})
    with pytest.raises(RuntimeError, match="Source hash guard"):
        bayes.check_sources(evaluator, expected, tmp_path)
    assert (tmp_path / "code_changed.json").exists()


def test_seed_file_and_duplicate_rejection(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps({"train_seeds": [10, 12]}), encoding="utf-8")
    assert bayes.parse_seeds(str(path), 0, 2) == [10, 12]
    assert bayes.parse_seeds("10,12", 0, 2) == [10, 12]
    assert bayes.parse_seeds(None, 20, 2) == [20, 21]
    with pytest.raises(ValueError, match="unique"):
        bayes.parse_seeds("10,10", 0, 2)
