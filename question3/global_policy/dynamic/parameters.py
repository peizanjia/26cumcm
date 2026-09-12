"""Tunable dynamic decisions; inherited physical constants remain fixed."""
from dataclasses import dataclass
import math
from ..model import Parameters


@dataclass
class DynamicParameters(Parameters):
    sector_count: int = 8
    nearby_distance: float = 500.
    frontier_spacing: float = 240.
    frontier_closure_bonus: float = .35
    rollout_samples: int = 12
    probe_lateral: float = 70.
    probe_fraction_near: float = .5
    probe_fraction_far: float = .8
    rollout_probe_limit: int = 3
    action_margin_s: float = 1.

    def validate(self):
        super().validate()
        for name,low,high in (
            ('sector_count',4,16),('nearby_distance',100,900),
            ('frontier_spacing',100,500),('frontier_closure_bonus',0,3),
            ('rollout_samples',4,64),('probe_lateral',10,250),
            ('probe_fraction_near',.1,.95),('probe_fraction_far',.1,.95),
            ('rollout_probe_limit',0,5),('action_margin_s',0,20)):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{name} must be in [{low}, {high}]')
        for name in ('sector_count','rollout_samples','rollout_probe_limit'):
            if not isinstance(getattr(self,name),int):raise ValueError(f'{name} must be an integer')
        if self.probe_fraction_near>=self.probe_fraction_far:raise ValueError('Near probe fraction must be smaller than far fraction')
        return self
