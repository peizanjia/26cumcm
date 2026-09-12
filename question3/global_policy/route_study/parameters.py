from dataclasses import dataclass
import math
from ..dynamic.parameters import DynamicParameters


@dataclass
class StudyParameters(DynamicParameters):
    route_method: str = 'insertion_2opt'
    global_mode: str = 'sector'
    scan_mode: str = 'heuristic'
    inner_radius: float = 800.
    inner_nodes: int = 8
    inner_probability: float = .65
    voi_samples: int = 6
    voi_candidates: int = 6
    voi_gain_threshold_s: float = 1.
    voi_uncertainty_weight: float = 0.
    scan_each_probe: bool = False
    route_scope: str = 'local'
    annealing_steps: int = 300
    annealing_temperature: float = 200.
    open_if_no_frontier: bool = False

    def validate(self):
        super().validate()
        for name,allowed in [('route_method',('insertion_2opt','nearest','exact_dp','exact_open','mst_2opt','christofides','annealing')),
                             ('global_mode',('sector','inner_first','adaptive_sector','block_first','global_graph')),
                             ('route_scope',('local','global_guided')),
                             ('scan_mode',('heuristic','voi'))]:
            if getattr(self,name) not in allowed:raise ValueError(f'Invalid {name}')
        for name,lo,hi in [('inner_radius',300,1300),('inner_nodes',4,16),('inner_probability',.1,1),
                           ('voi_samples',4,64),('voi_candidates',1,20),('voi_gain_threshold_s',0,60),
                           ('voi_uncertainty_weight',0,3),('annealing_steps',10,5000),('annealing_temperature',1,2000)]:
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError(name)
        for name in ('inner_nodes','voi_samples','voi_candidates','annealing_steps'):
            if not isinstance(getattr(self,name),int):raise ValueError(name)
        if not isinstance(self.scan_each_probe,bool):raise ValueError('scan_each_probe')
        if not isinstance(self.open_if_no_frontier,bool):raise ValueError('open_if_no_frontier')
        return self
