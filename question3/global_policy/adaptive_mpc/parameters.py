from dataclasses import dataclass
import math
from ..joint_rollout.parameters import JointParameters


@dataclass
class AdaptiveParameters(JointParameters):
    scan_scenarios: int = 24
    scan_candidates: int = 8
    scan_max_known: int = 5
    scan_min_net_s: float = 1.0
    early_scan_min_net_s: float = 0.0
    unknown_scan_gain: float = .035
    early_unknown_scan_gain: float = .015
    scan_risk_weight: float = 0.0
    early_stops: int = 8
    revisit_radius: float = 120.
    revisit_penalty_s: float = 15.
    frontier_candidates: int = 3
    joint_samples: int = 32
    joint_validation_samples: int = 128
    joint_finalists: int = 6
    side_scan_channels: int = 2
    replan_shrink_fraction: float = .3
    target_switch_margin_s: float = 2.
    max_waypoint_step: float = 450.
    stop_scans: bool = True
    early_mapping: bool = True
    anticipate_coverage: bool = False
    angular_bias_s: float = 0.
    early_geometry_mapping: bool = False
    early_geometry_stops: int = 6

    def validate(self):
        super().validate()
        bounds = dict(scan_scenarios=(4,512),scan_candidates=(1,20),scan_max_known=(1,20),
            scan_min_net_s=(0,60),early_scan_min_net_s=(0,60),unknown_scan_gain=(0,1),
            early_unknown_scan_gain=(0,1),scan_risk_weight=(0,5),early_stops=(0,40),
            revisit_radius=(1,500),revisit_penalty_s=(0,120),frontier_candidates=(1,10),
            joint_samples=(8,512),joint_validation_samples=(16,4096),joint_finalists=(1,40),
            side_scan_channels=(0,5),replan_shrink_fraction=(0,1),target_switch_margin_s=(0,30),
            max_waypoint_step=(50,1800),angular_bias_s=(0,500),early_geometry_stops=(0,40))
        integers = {'scan_scenarios','scan_candidates','scan_max_known','early_stops',
                    'frontier_candidates','joint_samples','joint_validation_samples','joint_finalists','side_scan_channels','early_geometry_stops'}
        for name,(lo,hi) in bounds.items():
            x=getattr(self,name)
            if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or not lo<=x<=hi:
                raise ValueError(name)
            if name in integers and not isinstance(x,int): raise ValueError(name)
        for name in ('stop_scans','early_mapping','anticipate_coverage','early_geometry_mapping'):
            if not isinstance(getattr(self,name),bool): raise ValueError(name)
        return self
