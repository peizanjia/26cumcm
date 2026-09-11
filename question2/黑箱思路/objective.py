"""Expected residual area, analytically marginalizing the hidden radius."""
import torch
from .geometry import NEAR, SCALE, second_area


def tensor_dataset(data, device):
    return {key: torch.as_tensor(value, device=device,
                                dtype=torch.long if key == "counts" else torch.float32)
            for key, value in data.items()}


def outcomes(data, q, radius_mode="uniform"):
    """One action per state, all posterior samples per state.

    direction: outer approximation of D1800 ∩ B(p,1500) ∩ W1 ∩ W2;
    no_signal: conservatively retain the first region;
    near: zero terminal uncertainty AFTER successful optical localization.
    The new 1500 m receiving disc is deliberately not included in the score.
    """
    g = data["targets"]
    batch, samples, _ = g.shape
    diff = g - q[:, None]
    d2 = torch.linalg.vector_norm(diff, dim=-1)
    d1 = torch.linalg.vector_norm(g, dim=-1)
    if radius_mode == "uniform":
        lower = d1.clamp_min(1000 / SCALE)
        detected = ((1 - torch.maximum(d2, lower)) / (1 - lower).clamp_min(1e-7)).clamp(0, 1)
    else:
        detected = (d2 <= float(radius_mode) / SCALE).to(d2.dtype)
    near = (d2 <= NEAR).to(d2.dtype)
    direction = detected * (1 - near)
    angle = torch.atan2(diff[..., 1], diff[..., 0]) + data["errors"]
    poly = data["polygons"][:, None].expand(-1, samples, -1, -1).reshape(batch * samples, -1, 2)
    counts = data["counts"][:, None].expand(-1, samples).reshape(-1)
    qq = q[:, None].expand(-1, samples, -1).reshape(-1, 2)
    area = second_area(poly, counts, qq, angle.reshape(-1)).reshape(batch, samples)
    # A repeated location has the SAME bearing, rather than a new error draw.
    repeated = torch.linalg.vector_norm(q, dim=-1) <= 1e-10
    area = torch.where(repeated[:, None], data["areas"][:, None], area)
    loss = direction * area + (1 - detected) * data["areas"][:, None]
    return dict(loss=loss * SCALE**2, area=area * SCALE**2,
                no_signal=1 - detected, near=near, direction=direction)


def subset(data, indices, sample_indices=None):
    result = {key: value[indices] for key, value in data.items()}
    if sample_indices is not None:
        for key in ("targets", "errors"):
            result[key] = result[key][:, sample_indices]
    return result


@torch.inference_mode()
def evaluate(data, action, batch_size=64, radius_mode="uniform"):
    parts = {}
    for start in range(0, len(data["states"]), batch_size):
        d = subset(data, slice(start, start + batch_size))
        q = action(d) if callable(action) else action[start:start + batch_size]
        result = outcomes(d, q, radius_mode)
        for key, value in result.items():
            parts.setdefault(key, []).append(value.cpu())
        parts.setdefault("q", []).append(q.cpu())
    return {key: torch.cat(value).numpy() for key, value in parts.items()}
