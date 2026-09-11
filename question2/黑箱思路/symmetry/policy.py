"""Share a single branch generator across reflected observations.

Index 0 is right (negative lateral displacement); index 1 is left.
Reflection maps BOTH the state and side: q_s(Fp) = F q_{-s}(p).
It does not assert q_left(p) = F q_right(p) at an asymmetric fixed state.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..geometry import SCALE, ARENA
from ..policy import certify_action


def ray_length(local_p):
    return (-local_p[..., 0] + torch.sqrt((ARENA**2-local_p[..., 1]**2).clamp_min(0))).clamp(0, 1)


class TwoSidedNet(nn.Module):
    def __init__(self, seed=0, boundary_scaled=True):
        super().__init__()
        torch.manual_seed(seed)
        self.boundary_scaled = boundary_scaled
        self.layers = nn.Sequential(nn.Linear(3 if boundary_scaled else 2, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
        nn.init.normal_(self.layers[-1].weight, std=.002)
        with torch.no_grad():
            self.layers[-1].bias.copy_(torch.tensor([np.log(.325/.675), np.log(.18/.82)]))

    def forward(self, local_p):
        """(...,2) state -> (...,2 sides,2 coordinates) local displacements."""
        signs = local_p.new_tensor([-1., 1.])
        p = local_p[..., None, :].expand(*local_p.shape[:-1], 2, 2)
        features = torch.stack([p[..., 0]/ARENA, signs*p[..., 1]/ARENA], -1)
        length = ray_length(local_p)
        if self.boundary_scaled:
            features = torch.cat([features, length[..., None, None].expand(*features.shape[:-1], 1)], -1)
        raw = self.layers(features)
        scale = length[..., None] if self.boundary_scaled else torch.ones_like(raw[..., 0])
        a = 2*torch.sigmoid(raw[..., 0])*scale
        b = torch.sigmoid(raw[..., 1])*scale
        return torch.stack([a, signs*b], -1)


class SidePolicy:
    def __init__(self, checkpoint=None, device="cpu", guarantee_reception=False):
        self.device = torch.device(device)
        if checkpoint is None:
            checkpoint = Path(__file__).parents[1]/"checkpoints"/"symmetry"/"best.pt"
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        self.net = TwoSidedNet(boundary_scaled=payload["boundary_scaled"]).to(device)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        self.guarantee_reception = guarantee_reception

    @torch.inference_mode()
    def options(self, x, y, theta_deg):
        if not np.isfinite([x, y, theta_deg]).all() or np.hypot(x, y)>1800+1e-8:
            raise ValueError("Finite input and first point inside the trained arena required")
        t = np.deg2rad(theta_deg % 360)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        p = np.einsum("ji,j->i", rot, [x/SCALE, y/SCALE])
        q = self.net(torch.tensor(p, device=self.device, dtype=torch.float32)).cpu().numpy()
        if self.guarantee_reception:
            q = np.array([certify_action(p, v) for v in q])
        world = np.einsum("ij,kj->ki", rot, q*SCALE)+[x, y]
        return {False: tuple(map(float, world[0])), True: tuple(map(float, world[1]))}

    def __call__(self, x, y, theta_deg, left):
        if not isinstance(left, (bool, np.bool_)):
            raise TypeError("left must be a Boolean chosen by the global planner")
        return self.options(x, y, theta_deg)[bool(left)]


_policy = None


def pi_side(x, y, theta_deg, left):
    global _policy
    if _policy is None:
        _policy = SidePolicy()
    return _policy(x, y, theta_deg, left)
