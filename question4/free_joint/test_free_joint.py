import math
import numpy as np
import pytest

from question4.full_mission.service import Target as OriginalTarget, _sample_polygon
from question4.free_joint.belief import Target, integrate
from question4.free_joint.coverage import DirectionalCoverage
from question4.free_joint.planner import Planner, Parameters
from question4.zigzag_study.study import certificate_details
from question4.zigzag_study.routes import zigzag


@pytest.mark.parametrize('position', [[400., 30.], [850., -25.], [1350., 40.]])
def test_angle_radius_integral_against_independent_dense_angles(position):
    obs = [dict(position=[0., 0.], result='direction'),
           dict(position=[100., 300.], result='no_signal'),
           dict(position=[-400., -120.], result='direction'),
           dict(position=[1400., 300.], result='no_signal')]
    _, _, mass = integrate([position], obs)
    angles = (np.arange(65536)+.5)*2*np.pi/65536
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    lower = 1000.
    upper = np.full(len(angles), 1500.)
    valid = np.ones(len(angles), bool)
    omni_upper = 1500.
    for o in obs:
        v = np.asarray(o['position'])-position
        d = np.linalg.norm(v)
        lit = normals@v >= 0
        if o['result'] == 'direction':
            valid &= lit; lower = max(lower, d)
        else:
            upper[lit] = np.minimum(upper[lit], d)
            omni_upper = min(omni_upper, d)
    expected = np.maximum(upper-lower, 0)[valid].sum()/len(angles)+max(0., omni_upper-lower)
    assert abs(mass[0]-expected) < .025


def test_vectorized_mass_matches_original_scalar_integrator():
    t = OriginalTarget(1)
    t.observe([0, 0], dict(measure_result='direction', svd_deg=20.))
    t.observe([600, 300], dict(measure_result='no_signal'))
    p, w, _, _, total = t._quadrature()
    _, _, masses = integrate(p, t.observations)
    assert np.isclose(total, masses.sum(), rtol=1e-10)
    assert np.allclose(w, masses/masses.sum(), atol=1e-12)


def test_adaptive_integration_never_changes_conservative_geometry():
    t = Target(1)
    t.observe([0, 0], dict(measure_result='direction', svd_deg=0.))
    t.observe([600, 80], dict(measure_result='no_signal'))
    before = t.polygon.copy()
    p, w, seg, omni, total = t._quadrature()
    assert np.array_equal(before, t.polygon)
    assert total > 0 and np.isclose(w.sum(), 1.)
    assert len(p) >= 128
    assert t.status == 'active'
    assert 0 <= t.hit_probability([100, 50]) <= 1


def test_elastic_plan_remains_continuously_certified_without_observation():
    cov = DirectionalCoverage(channels=1)
    cov.observe(1, [0., 0.], 'no_signal')
    cov.service_positions = [[700., 300.], [1500., 700.]]
    plan = cov.completion_plan([1], [650., 200.])
    assert cov.plan_certified([1], plan)
    assert not cov.channel_certified(1)
    assert len(cov.points(1)) == 1
    assert cov.last_plan_diagnostics['positions_fixed'] is False


def test_new_candidates_still_obey_common_radio_budget():
    policy = Planner(lambda x: None, Parameters(elastic_coverage=False))
    t = policy.targets[1]
    t.observe([0, 0], dict(measure_result='direction', svd_deg=0.))
    policy.search_plan = []
    t.radio_steps = 12
    rows = policy._raw_candidates()
    assert not any(r['owner'] == 1 and r['kind'] == 'measure' for r in rows)
    assert not policy._radio_eligible(t)


def test_stationary_scan_value_does_not_inflate_moving_credit():
    p = Planner(lambda x: None, Parameters(elastic_coverage=False, stationary_unknown_cap_s=14.))
    p.search_budget_s = 6000.
    q = np.array([800., 0.])
    moving = p._unknown_values(q)
    p._stationary = True
    stopped = p._unknown_values(q)
    assert sum(r['gross_saving_s'] for r in moving) <= 60.+1e-7
    assert sum(r['gross_saving_s'] for r in stopped) > 60.
    assert all(r['gross_saving_s'] <= 14. for r in stopped)


def test_clear_approach_covers_entire_polygon_and_saves_movement():
    p = Planner(lambda x: None, Parameters(elastic_coverage=False,
                approach_clear_boundary=True, near_anchor_probes=False))
    for t in p.targets.values():
        t.status = 'absent'
    t = p.targets[1]; t.status = 'active'
    t.polygon = np.array([[-10., -5.], [10., -5.], [10., 5.], [-10., 5.]])
    t.center = np.zeros(2); t.radius = math.sqrt(125.)
    p.position = np.array([100., 30.])
    rows = [r for r in p._raw_candidates() if r['reason'] == 'near_boundary_certified_clear']
    assert rows
    q = np.asarray(rows[0]['position'])
    assert np.linalg.norm(q-p.position) < np.linalg.norm(t.center-p.position)
    assert np.max(np.linalg.norm(q-t.polygon, axis=1)) < 20.
    rng = np.random.default_rng(731)
    interior = rng.uniform([-10., -5.], [10., 5.], size=(2000, 2))
    assert np.max(np.linalg.norm(interior-q, axis=1)) < 20.
