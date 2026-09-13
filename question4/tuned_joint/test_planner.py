"""Physical and counterfactual checks for jointly priced unknown scans.

Tiny scenes deliberately isolate accounting; they are not benchmark scores.
"""
from copy import deepcopy

import numpy as np
import pytest

from question4.full_mission.simulator import Simulator, Source
from question4.tuned_joint.planner import Parameters, Planner
from question4.zigzag_study.routes import zigzag
from question4.zigzag_study.study import certificate_details


def isolated(command=lambda *args: None, unknown=(1, 2), **params):
    policy = Planner(command, Parameters(elastic_coverage=False,
                                        refine_candidates=False, **params))
    for c, target in policy.targets.items():
        if c not in unknown:
            target.status = 'absent'
    return policy


def negative(policy, channel, position):
    policy.targets[channel].observe(position, dict(measure_result='no_signal'))
    policy.unknown_map.observe(channel, position, 'no_signal')


def evidence(policy):
    return dict(
        points={c: np.asarray(policy.unknown_map.points(c)).tolist() for c in policy.targets},
        certified={c: policy.unknown_map.channel_certified(c) for c in policy.targets},
        states={c: grid.copy() for c, grid in policy.unknown_map.unseen.items()},
        targets={c: (t.status, deepcopy(t.observations), t.polygon.copy())
                 for c, t in policy.targets.items()},
        commands=deepcopy(policy.commands),
    )


def assert_evidence_equal(left, right):
    assert left['points'] == right['points']
    assert left['certified'] == right['certified']
    for channel in left['states']:
        assert np.array_equal(left['states'][channel], right['states'][channel])
    assert left['commands'] == right['commands']
    for channel in left['targets']:
        old, new = left['targets'][channel], right['targets'][channel]
        assert old[:2] == new[:2]
        assert np.array_equal(old[2], new[2])


def test_future_cost_omits_channels_really_measured_at_each_stop():
    policy = isolated()
    policy.search_plan = [[100., 0.], [200., 0.]]
    negative(policy, 1, [100., 0.])
    negative(policy, 2, [200., 0.])
    before = evidence(policy)
    value = policy._forecast(np.zeros(2), [], [])
    # Travel 200/5 seconds. Only c2 at the first stop and c1 at the
    # second stop remain, each requiring a receiver switch and measurement.
    assert value == pytest.approx(40. + 6. + 6.)
    assert_evidence_equal(before, evidence(policy))


def test_future_same_stop_does_not_charge_the_hypothetical_scan_twice():
    policy = isolated()
    policy.search_plan = [[0., 0.]]
    before = evidence(policy)
    value = policy._forecast(np.zeros(2), [], [1])
    # The stop cannot be deleted for c2. But the candidate action has already
    # scanned c1 there in this conditional continuation; only c2 costs six.
    assert value == pytest.approx(6.)
    assert_evidence_equal(before, evidence(policy))


def test_future_receiver_uses_primary_first_actual_scan_order(monkeypatch):
    policy = isolated(bundle_scans=False)
    policy.search_plan = [[100., 0.]]
    negative(policy, 1, [100., 0.])
    target = policy.targets[3]
    target.status = 'active'
    target.center = np.array([50., 0.])
    target.radius = 50.
    policy.channel = 3
    monkeypatch.setattr(policy, '_known_values', lambda q: [dict(
        channel=3, type='known', required=False, gross_saving_s=20.,
        net_saving_s=15., cost_s=5., expected_center=[50., 0.],
        hit_probability=1.)])
    action = policy._evaluate(dict(position=[0., 0.], kind='measure', owner=3,
                                   required=[1, 2], reason='order_fixture'))
    assert action['predicted_scan_channels'] == [3, 1, 2]
    # The shared stop ends on c2, so future c2 at x=100 needs no switch.
    assert action['forecast_route_s'] == pytest.approx(20. + 5.)


def test_failed_candidate_evaluation_restores_receiver_context(monkeypatch):
    from question4.free_joint.planner import Planner as ParentPlanner

    policy = isolated()
    previous = dict(kind='measure', owner=7, required=[])
    policy._evaluated_row = previous

    def failed(*args, **kwargs):
        raise ArithmeticError('prediction fixture')

    monkeypatch.setattr(ParentPlanner, '_evaluate', failed)
    with pytest.raises(ArithmeticError, match='prediction fixture'):
        policy._evaluate_once(dict(kind='clear', owner=3, required=[]))
    assert policy._evaluated_row is previous


def test_empty_discrete_unknown_map_does_not_authorize_bundle_deletion():
    policy = isolated(unknown_credit_cap_s=0.)
    policy.search_plan = [[100., 0.]]
    for grid in policy.unknown_map.unseen.values():
        grid[:] = False
    before = evidence(policy)
    result = policy._evaluate(dict(position=[110., 0.], owner=None,
                                   kind='measure', required=[], reason='no_proof'))
    assert result is None
    assert not policy.unknown_map.plan_certified([1, 2], [[110., 0.]])
    assert_evidence_equal(before, evidence(policy))


def make_clear_bundle():
    supports = zigzag(990., 1900., 24)
    assert certificate_details(supports)['certified']
    q = supports[1]
    remaining = np.delete(supports, 1, axis=0)
    sim = Simulator(718, [Source(3, *q, 1000., False, 0.)])
    # Weight two makes the bundle preference independent of sub-metre travel
    # differences, while actual commands below always retain physical costs.
    policy = isolated(sim.command, future_scan_weight=2.,
                      unknown_credit_cap_s=0.)
    for p in remaining:
        for channel in (1, 2):
            policy.execute(dict(kind='measure', channel=channel,
                                position=p.tolist(), reason='proof_fixture'), 'search')
    policy.targets[3].status = 'unknown'
    policy.execute(dict(kind='measure', channel=3, position=q.tolist(),
                        reason='clear_fixture'), 'service')
    policy.search_plan = [q.tolist()]
    assert all(not policy.unknown_map.channel_certified(c) for c in (1, 2))
    return policy, sim, q


def test_clear_bundle_is_continuously_certified_but_not_real_evidence_yet():
    policy, _, q = make_clear_bundle()
    before = evidence(policy)
    action = policy._evaluate(dict(position=q.tolist(), kind='clear', owner=3,
                                   required=[], reason='certified_mec_clear'))
    assert action['kind'] == 'clear'
    assert action['channel'] == 3
    assert action['bundle_scan'] is True
    assert action['replaced_future_stops'] == 1
    assert set(action['required_channels']) == {1, 2}
    for channel in (1, 2):
        proof = np.vstack((policy.unknown_map.points(channel), q))
        assert certificate_details(proof)['certified']
    assert_evidence_equal(before, evidence(policy))


def test_clear_bundle_executes_then_reassesses_all_required_real_channels():
    policy, sim, q = make_clear_bundle()
    action = policy._evaluate(dict(position=q.tolist(), kind='clear', owner=3,
                                   required=[], reason='certified_mec_clear'))
    before, time_before = len(policy.commands), policy.time_s
    policy.execute(action, 'service')
    assert policy.targets[3].status == 'cleared'
    assert all(not policy.unknown_map.channel_certified(c) for c in (1, 2))
    policy.assess_stop(action['required_channels'], first_channel=action['channel'])
    added = policy.commands[before:]
    assert [(r['kind'], r['channel']) for r in added] == [
        ('clear', 3), ('measure', 1), ('measure', 2)]
    assert all(r['position'] == q.tolist() and r['move_s'] == 0. for r in added)
    assert all(policy.unknown_map.channel_certified(c) for c in (1, 2))
    assert policy.finished()
    assert policy.time_s - time_before == pytest.approx(5. + 6. + 6.)
    assert policy.time_s == pytest.approx(sim.time_s)
    assert policy.costs == pytest.approx(sim.costs)


def test_run_keeps_public_count_stop_and_charges_all_real_clears():
    sim = Simulator(719, [Source(c, 0., 0., 1000., False, 0.) for c in range(1, 17)])
    policy = Planner(sim.command, Parameters(elastic_coverage=False,
                                            refine_candidates=False))
    result = policy.run()
    assert result['completed'] and result['cleared'] == 16
    assert all(source.cleared for source in sim.sources)
    assert result['command_count'] == 36
    assert result['time_s'] == pytest.approx(199.)
    assert result['time_s'] == pytest.approx(sim.time_s)
    assert result['costs'] == pytest.approx(sim.costs)
