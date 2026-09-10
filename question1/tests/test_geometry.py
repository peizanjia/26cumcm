import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from question1.geometry import (bearing_halfplanes, intersect_bearings,
                                polygon_area, polygon_diameter)
from question1.simulate import BearingEnvironment, sample_disk


def test_known_square_from_wedges():
    result = intersect_bearings([[-1, 0], [1, 0]], [0, 180], 45)
    assert result.status == "bounded"
    assert len(result.vertices) == 4
    assert polygon_area(result.vertices) == pytest.approx(2)
    assert polygon_diameter(result.vertices)[0] == pytest.approx(2)


def test_forward_wedges_not_bidirectional_lines():
    result = intersect_bearings([[0, 0], [-10, 0]], [0, 180])
    assert result.status == "empty"


def test_unbounded_and_explicit_bounds():
    result = intersect_bearings([[0, 0]], [0])
    assert result.status == "unbounded"
    assert result.vertices.shape == (0, 2)
    bounded = intersect_bearings([[0, 0]], [0], bounds=(-1800, -1800, 1800, 1800))
    assert bounded.status == "bounded" and bounded.search_box_applied
    assert np.max(bounded.vertices[:, 0]) == pytest.approx(1800)


def test_intersection_outside_initial_box():
    target = np.array([1e6, 0])
    points = np.array([[0, -1000], [0, 1000]])
    delta = target - points
    bearings = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 360
    result = intersect_bearings(points, bearings, error_deg=0.001)
    assert result.status == "bounded"
    assert np.min(result.vertices[:, 0]) > 900000


def test_degenerate_point_and_segment():
    point = intersect_bearings([[0, 0], [0, 0]], [0, 180])
    assert point.status == "bounded"
    np.testing.assert_allclose(point.vertices, [[0, 0]], atol=1e-8)
    segment = intersect_bearings([[0, 0], [1, 0]], [45, 225], 45)
    assert segment.status == "bounded"
    assert polygon_area(segment.vertices) == pytest.approx(0, abs=1e-8)
    assert polygon_diameter(segment.vertices)[0] == pytest.approx(1)


def test_wraparound_and_per_station_error():
    result = intersect_bearings([[0, 0], [100, -100]], [360, 90], [1, 0.5])
    same = intersect_bearings([[0, 0], [100, -100]], [0, -270], [1, 0.5])
    np.testing.assert_allclose(result.vertices, same.vertices)
    assert result.status == "bounded"


@pytest.mark.parametrize("bad_points,bad_angles,error", [
    ([], [], 1), ([[0, 0]], [], 1), ([[np.nan, 0]], [0], 1),
    ([[0, 0]], [np.inf], 1), ([[0, 0]], [0], 0), ([[0, 0]], [0], 90),
    ([[0, 0]], [0], [1, 2]),
])
def test_validation(bad_points, bad_angles, error):
    with pytest.raises(ValueError):
        intersect_bearings(bad_points, bad_angles, error)


@pytest.mark.parametrize("seed", range(15))
def test_random_intersection_matches_independent_lp_support(seed):
    env = BearingEnvironment(seed=seed)
    points = env.sample_monitoring_points(5)
    angles = [env.measure(p)["svd_deg"] for p in points]
    result = intersect_bearings(points, angles)
    assert result.status == "bounded"
    a, b = bearing_halfplanes(points, angles)
    assert np.max(a @ env.source - b) <= 1e-7
    assert np.max(a @ result.vertices.T - b[:, None]) <= 1e-7
    for angle in np.linspace(0, 2*np.pi, 9)[:-1]:
        direction = np.array([np.cos(angle), np.sin(angle)])
        lp = linprog(-direction, A_ub=a, b_ub=b, bounds=[(None, None)]*2, method="highs")
        assert lp.success
        assert np.max(result.vertices @ direction) == pytest.approx(-lp.fun, abs=2e-6)


def test_calipers_against_brute_force():
    rng = np.random.default_rng(8)
    for count in (3, 4, 20, 100, 1000):
        for _ in range(10):
            cloud = rng.normal(size=(count, 2))
            hull = cloud[ConvexHull(cloud).vertices]
            brute = np.max(np.linalg.norm(hull[:, None] - hull[None, :], axis=2))
            assert polygon_diameter(hull)[0] == pytest.approx(brute)
            assert polygon_diameter(hull[::-1])[0] == pytest.approx(brute)


@pytest.mark.parametrize("noise", ["uniform", "truncnorm"])
def test_fixed_bounded_error_reproducible_and_query_order_independent(noise):
    env = BearingEnvironment(seed=111, noise=noise)
    other = BearingEnvironment(seed=111, noise=noise)
    points = env.sample_monitoring_points(100)
    np.testing.assert_array_equal(points, other.sample_monitoring_points(100))
    errors = [env.error_at(p) for p in points]
    assert max(abs(e) for e in errors) <= 1
    assert errors == [other.error_at(p) for p in points[::-1]][::-1]
    assert env.measure(points[0]) == env.measure(points[0])
    assert env.error_at([0., -0.]) == env.error_at([-0., 0.])
    assert env.measure(env.source)["measure_result"] == "near"
    assert env.measure(env.source + [1600, 0])["measure_result"] == "no_signal"


def test_uniform_area_sampling():
    points = sample_disk(np.random.default_rng(9), 100000, 1800)
    r2 = np.sum(points**2, axis=1) / 1800**2
    assert np.max(r2) <= 1
    assert np.mean(r2) == pytest.approx(0.5, abs=0.005)
