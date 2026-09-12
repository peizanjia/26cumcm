"""In-process, synthetic Q4 physics; no official or network transport.

Only ``command`` is passed to a strategy.  ``truth`` and ``sources`` belong to
the evaluator and visualizer.  Scene sampling matches the earlier Q4 route
study, while actions now include localization measurements and actual clearing.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np

from question4.benchmark import make_scene


@dataclass
class Source:
    channel: int
    x: float
    y: float
    radius: float
    directional: bool
    orientation: float
    cleared: bool = False


class Simulator:
    """Mutable scene with fixed orientation, receive radius and bearing errors."""

    def __init__(self, seed: int = 0, sources: Iterable | None = None):
        self.seed = int(seed)
        supplied = make_scene(self.seed) if sources is None else sources
        # Copy even mutable inputs: strategies evaluated on a shared scene must
        # never inherit another strategy's clears.
        self.sources: list[Source] = []
        for source in supplied:
            data = dict(source) if isinstance(source, Mapping) else vars(source)
            self.sources.append(Source(**{
                key: data[key] for key in Source.__dataclass_fields__ if key in data
            }))
        channels = [source.channel for source in self.sources]
        if len(channels) != len(set(channels)):
            raise ValueError("Each source must occupy a distinct channel")
        for source in self.sources:
            if not 1 <= source.channel <= 20 or int(source.channel) != source.channel:
                raise ValueError("Source channel must be an integer in 1..20")
            if not all(math.isfinite(value) for value in
                       (source.x, source.y, source.radius, source.orientation)):
                raise ValueError("Source parameters must be finite")
            if math.hypot(source.x, source.y) > 1800.0 + 1e-9:
                raise ValueError("Source must lie inside the 1800 m disk")
            if not 1000.0 <= source.radius <= 1500.0:
                raise ValueError("Receive radius must lie in 1000..1500 m")
        self.position = np.zeros(2, dtype=float)
        self.channel = 1
        self.time_s = 0.0
        self.costs = dict(movement=0.0, measure=0.0, switch=0.0,
                          clear_success=0.0, clear_failure=0.0)
        self.history: list[dict] = []

    def error(self, channel: int, position) -> float:
        """Stable local error in [-1, 1] degrees, as in the Q3 simulator.

        Negative zero is normalized, so (0, 0) and (-0, 0) are one physical
        location.  Repeated readings do not provide independent noise samples.
        """
        x, y = (float(v) or 0.0 for v in position)
        key = f"{self.seed}:{int(channel)}:{x.hex()}:{y.hex()}"
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        return 2.0 * (int.from_bytes(digest, "big") / 2**64) - 1.0

    @staticmethod
    def _illuminated(source: Source, position: np.ndarray) -> bool:
        if not source.directional:
            return True
        dx, dy = position - np.array([source.x, source.y])
        projection = dx * math.cos(source.orientation) + dy * math.sin(source.orientation)
        # Closed 180-degree half-plane; tolerance only absorbs trig roundoff at
        # the exact +/-90-degree boundary, in metre units.
        return projection >= -1e-9

    def command(self, kind: str, position, channel: int) -> dict:
        """Move in a straight line, then measure or clear one channel.

        Clearing is optical and independent of emitter orientation.  It does
        not change the receiver channel or incur a receiver switching charge.
        Measurements are made only after arrival, never while moving.
        """
        action = kind.removeprefix("/") if isinstance(kind, str) else ""
        if action not in ("measure", "clear"):
            raise ValueError("Expected measure or clear")
        if isinstance(position, Mapping):
            position = (position["x"], position["y"])
        target = np.asarray(position, dtype=float)
        if target.shape != (2,) or not np.all(np.isfinite(target)) or np.any(np.abs(target) > 2e6):
            raise ValueError("Position must contain two finite coordinates within +/-2e6")
        if isinstance(channel, (bool, np.bool_)) or not isinstance(channel, (int, float, np.integer, np.floating)):
            raise ValueError("Channel must be an integer in 1..20")
        if not math.isfinite(channel) or int(channel) != channel or not 1 <= channel <= 20:
            raise ValueError("Channel must be an integer in 1..20")
        channel = int(channel)
        target = np.where(target == 0.0, 0.0, target)
        previous = self.position.copy()
        before_channel, before_time = self.channel, self.time_s
        movement_s = float(np.linalg.norm(target - previous)) / 5.0
        source = next((s for s in self.sources if s.channel == channel and not s.cleared), None)
        distance = math.hypot(target[0] - source.x, target[1] - source.y) if source else math.inf
        switch_s = 0.0
        if action == "measure":
            action_s = 5.0
            switch_s = float(channel != self.channel)
            self.channel = channel
            self.costs["measure"] += action_s
            self.costs["switch"] += switch_s
            if source is None or distance > source.radius or not self._illuminated(source, target):
                fields = dict(measure_result="no_signal")
            elif distance <= 5.0:
                fields = dict(measure_result="near")
            else:
                bearing = math.degrees(math.atan2(source.y - target[1], source.x - target[0]))
                shown = round((bearing + self.error(channel, target)) % 360.0, 2) % 360.0
                fields = dict(measure_result="direction", svd_deg=shown)
        else:
            success = distance <= 20.0
            action_s = 5.0 if success else 3.0
            self.costs["clear_success" if success else "clear_failure"] += action_s
            if success:
                source.cleared = True
            fields = dict(clear_result="success" if success else "no_target_in_range")
        self.costs["movement"] += movement_s
        self.time_s += movement_s + action_s + switch_s
        self.position = target.copy()
        response = dict(accepted=True, virtual_time_s=self.time_s, **fields)
        self.history.append(dict(
            index=len(self.history), kind=action, path="/" + action,
            **{"from": previous.tolist()}, position=target.tolist(), channel=channel,
            channel_before=before_channel, channel_after=self.channel,
            time_before_s=before_time, end_time_s=self.time_s,
            move_s=movement_s, action_s=action_s, switch_s=switch_s,
            response=response.copy(),
        ))
        return response

    def truth(self) -> list[dict]:
        """Evaluator-only detached source records, including actual clear state."""
        return [asdict(source) for source in self.sources]
