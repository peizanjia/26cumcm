import importlib
import numpy as np
import pytest
import torch

geometry = importlib.import_module("question2.黑箱思路.geometry")
environment = importlib.import_module("question2.黑箱思路.environment")
objective = importlib.import_module("question2.黑箱思路.objective")
PolicyNet = importlib.import_module("question2.黑箱思路.policy").PolicyNet


def test_cpu_torch_areas_and_truth_containment():
    data = environment.make_dataset(80, 12, 137)
    d = objective.tensor_dataset(data, "cpu")
    rng = np.random.default_rng(75)
    q = rng.uniform([.1, -.6], [1.1, .6], (80, 2))
    result = objective.outcomes(d, torch.tensor(q, dtype=torch.float32))
    for i in range(80):
        poly = data["polygons"][i, :data["counts"][i]]
        for j in range(12):
            g = data["targets"][i, j]
            angle = np.arctan2(*(g-q[i])[::-1]) + data["errors"][i, j]
            clipped = geometry.second_cpu(poly, q[i], angle)
            area = geometry.area_cpu(clipped) * geometry.SCALE**2
            assert float(result["area"][i, j]) == pytest.approx(area, abs=.08, rel=.001)
            # Every edge of the CCW outer polygon must include the true source.
            edges = np.roll(clipped, -1, axis=0) - clipped
            cross = edges[:, 0] * (g-clipped)[:, 1] - edges[:, 1] * (g-clipped)[:, 0]
            assert np.all(cross >= -1e-9)


def test_exact_clipping_gradient_matches_finite_difference():
    data = environment.make_dataset(5, 9, 888)
    d = {key: torch.as_tensor(value, dtype=torch.long if key == "counts" else torch.float64)
         for key, value in data.items()}
    q = torch.tensor([[.45, .28]] * 5, dtype=torch.float64, requires_grad=True)
    loss = objective.outcomes(d, q)["loss"].mean()
    grad, = torch.autograd.grad(loss, q)
    for i in range(5):
        for j in range(2):
            plus, minus = q.detach().clone(), q.detach().clone()
            plus[i, j] += 1e-6
            minus[i, j] -= 1e-6
            fd = (objective.outcomes(d, plus)["loss"].mean()
                  - objective.outcomes(d, minus)["loss"].mean()) / 2e-6
            assert grad[i, j].item() == pytest.approx(fd.item(), rel=.002, abs=.02)


def test_no_signal_and_repeat_cannot_fake_zero_area():
    data = environment.make_dataset(6, 16, 934)
    d = objective.tensor_dataset(data, "cpu")
    for q in (torch.zeros((6, 2)), torch.full((6, 2), 10.)):
        result = objective.outcomes(d, q)
        assert np.allclose(result["loss"].mean(1).numpy(), data["areas"] * geometry.SCALE**2, rtol=1e-5)


def test_posterior_radial_distribution_at_origin():
    g = environment.sample_posterior(np.zeros((1, 2)), 60000, 743)[0]
    r = np.linalg.norm(g, axis=1)
    # Integrate r*w(r) and r^2*w(r) independently of the sampler.
    from scipy.integrate import quad
    lo = geometry.NEAR
    w = lambda r: min(1., (1-r)/(1/3))
    z = quad(lambda r: r*w(r), lo, 1, points=[2/3])[0]
    mean = quad(lambda r: r*r*w(r), lo, 1, points=[2/3])[0] / z
    assert r.mean() == pytest.approx(mean, abs=.0025)
    assert np.all(r > lo) and np.all(r <= 1)
    assert np.max(np.abs(np.arctan2(g[:, 1], g[:, 0]))) <= geometry.DELTA


def test_rotation_equivariance_and_periodic_interface(tmp_path):
    NeuralPolicy = importlib.import_module("question2.黑箱思路.policy").NeuralPolicy
    net = PolicyNet(77)
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": net.state_dict()}, checkpoint)
    pi = NeuralPolicy(checkpoint)
    q = np.array(pi(300, 400, 23))
    assert pi(300, 400, 383) == pytest.approx(q)
    t = .76
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    p2 = rot @ [300, 400]
    q2 = pi(*p2, 23 + np.degrees(t))
    assert np.allclose(q2, rot @ q, atol=.001)


def test_guaranteed_reception_certificate():
    poly = geometry.first_polygon(np.zeros(2))
    q = np.array([.5, .2])
    assert geometry.robust_margin(poly, q) > 0
    rng = np.random.default_rng(471)
    weights = rng.dirichlet(np.ones(len(poly)), 500)
    g = weights @ poly
    assert np.max(np.linalg.norm(g-q, axis=1)) <= 1000/geometry.SCALE


def test_fixed_coordinate_noise_is_reused():
    from question1.simulate import BearingEnvironment
    env = BearingEnvironment(seed=923)
    assert env.error_at([13.1, -76.2]) == env.error_at([13.1, -76.2])


def test_outer_circle_refinement():
    # Nested tangent normals: refinement can only shrink an outer polygon.
    for p in ([1., .1], [.3, 1.1], [-1.1, 0]):
        areas = [geometry.area_cpu(geometry.first_polygon(np.array(p), n))
                 for n in [64, 128, 256, 512]]
        assert np.all(np.diff(areas) <= 1e-12)


def test_certified_policy_covers_all_outer_vertices():
    certify = importlib.import_module("question2.黑箱思路.policy").certify_action
    rng = np.random.default_rng(341)
    for p in [[0., 0.], [1., 0.], [.3, .6]]:
        poly = geometry.first_polygon(np.array(p))
        for q in rng.uniform(-2, 2, (12, 2)):
            safe = certify(p, q)
            assert geometry.robust_margin(poly, safe) >= -1e-12


def test_continuous_circle_reference_matches_sector():
    ref = importlib.import_module("question2.黑箱思路.circle_reference").exact_circle_area
    assert ref([0, 0]) == pytest.approx(geometry.DELTA, rel=1e-10)
    q = np.array([.5, .3])
    theta2 = np.arctan2(-.3, .2)
    poly = geometry.second_cpu(geometry.first_polygon(np.zeros(2)), q, theta2)
    assert geometry.area_cpu(poly) == pytest.approx(ref([0, 0], q, theta2), abs=1e-8)


def test_analytical_baseline_moments_against_independent_quadrature():
    moment = importlib.import_module("question2.黑箱思路.analytic_baseline").moment_action
    from scipy.integrate import quad
    integrals = [quad(lambda r: r**(k+1)*min(1., 3*(1-r)), geometry.NEAR, 1., points=[2/3])[0]
                 for k in (1, 2, 3)]
    a = integrals[1]/integrals[0]
    b = np.sqrt(integrals[2]/integrals[0]-a*a)
    assert moment([[0, 0]])[0] == pytest.approx([a, -b], rel=1e-9)
