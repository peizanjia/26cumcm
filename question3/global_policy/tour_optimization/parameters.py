from dataclasses import dataclass
import math
from ..adaptive_mpc.sweep_planner import SweepParameters


@dataclass
class Parameters(SweepParameters):
    early_geometry_mapping: bool = True
    early_unknown_scan_gain: float = .08
    refine_coverage: bool = False
    exact_route: bool = False
    defer_scans: bool = False
    coupled_scan_weight: float = 0.
    mapping_radius_ratio: float = .6
    mapping_detection_probability: float = .5
    defer_overlap: float = .9
    defer_distance: float = 900.
    coverage_passes: int = 2

    def validate(self):
        super().validate()
        for key in ('refine_coverage','exact_route','defer_scans'):
            if not isinstance(getattr(self,key),bool):raise ValueError(key)
        for key,lo,hi in [('coupled_scan_weight',0,2),('mapping_radius_ratio',.1,.95),
                          ('mapping_detection_probability',0,1),('defer_overlap',.5,1),
                          ('defer_distance',100,2000),('coverage_passes',1,4)]:
            x=getattr(self,key)
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not lo<=x<=hi:raise ValueError(key)
        if not isinstance(self.coverage_passes,int):raise ValueError('coverage_passes')
        return self
