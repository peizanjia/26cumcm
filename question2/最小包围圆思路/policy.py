"""Radius policy API; predictions are proposals, not 20 m certificates."""
from pathlib import Path
import numpy as np
import torch
from question2.黑箱思路.symmetry.policy import SidePolicy


def load_policy(guided=True):
    return SidePolicy(Path(__file__).parent/('guided_nn.pt' if guided else 'radius_nn.pt'))


_models={}


def pi_radius(x,y,theta_deg,left: bool,guided=True):
    """Global coordinates; left/right remains an external Boolean decision."""
    if guided not in _models:_models[guided]=load_policy(guided)
    return _models[guided](x,y,theta_deg,left)
