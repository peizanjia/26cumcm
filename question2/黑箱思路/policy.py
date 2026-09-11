"""Public interface: pi(x, y, theta_deg) -> (next_x, next_y), in metres."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .geometry import SCALE, ARENA, first_polygon, robust_margin


class PolicyNet(nn.Module):
    def __init__(self, seed=0, side=1):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
        nn.init.normal_(self.layers[-1].weight, std=0.003)
        with torch.no_grad():
            self.layers[-1].bias.copy_(torch.tensor([np.arctanh(.55 / 2),
                                                    np.arctanh(side * .3 / 2)]))

    def forward(self, local_p):
        # Rotation-equivariant feature construction removes arbitrary angular
        # wraparound. Domain: relative movement in [-3000,3000]^2 metres.
        return 2 * torch.tanh(self.layers(local_p / ARENA))


def certify_action(local_p, q):
    """Pull toward a universal feasible anchor until reception is certified.

    This certifies signal OR near, not a new bearing in the <=5 m case.
    It changes the raw network action and is therefore exposed separately.
    """
    poly = first_polygon(np.asarray(local_p, float))
    q = np.asarray(q, float)
    if robust_margin(poly, q) >= 1e-6:
        return q
    anchor = np.array([.5, 0.])
    if robust_margin(poly, anchor) < 0:
        raise ArithmeticError("Universal receiving-wedge anchor failed certification")
    lo, hi = 0., 1.
    for _ in range(45):
        mid = (lo+hi)/2
        if robust_margin(poly, anchor+mid*(q-anchor)) >= 1e-6:
            lo = mid
        else:
            hi = mid
    return anchor+lo*(q-anchor)


class NeuralPolicy:
    def __init__(self, checkpoint=None, device="cpu", guarantee_reception=False):
        if checkpoint is None:
            checkpoint = Path(__file__).parent / "checkpoints" / "best.pt"
        self.device = torch.device(device)
        self.guarantee_reception = guarantee_reception
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.net = PolicyNet().to(self.device)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()

    @torch.inference_mode()
    def __call__(self, x, y, theta_deg):
        values = np.array([x, y, theta_deg], float)
        if not np.isfinite(values).all():
            raise ValueError("x, y, theta must be finite")
        # Scope guard: training states use initial monitors inside the arena.
        if np.hypot(x, y) > 1800 + 1e-8:
            raise ValueError("First monitor outside training domain; use candidate search")
        theta = np.deg2rad(theta_deg % 360)
        c, s = np.cos(theta), np.sin(theta)
        p = np.array([x, y]) / SCALE
        local_p = np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1]])
        q = self.net(torch.tensor(local_p, dtype=torch.float32, device=self.device)).cpu().numpy()
        if self.guarantee_reception:
            q = certify_action(local_p, q)
        delta = SCALE * np.array([c * q[0] - s * q[1], s * q[0] + c * q[1]])
        result = np.array([x, y]) + delta
        return float(result[0]), float(result[1])


_default_policy = None


def pi(x, y, theta_deg):
    global _default_policy
    if _default_policy is None:
        _default_policy = NeuralPolicy()
    return _default_policy(x, y, theta_deg)


_safe_policy = None


def safe_pi(x, y, theta_deg):
    global _safe_policy
    if _safe_policy is None:
        _safe_policy = NeuralPolicy(guarantee_reception=True)
    return _safe_policy(x, y, theta_deg)
