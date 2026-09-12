from dataclasses import dataclass
from ..route_study.parameters import StudyParameters


@dataclass
class JointParameters(StudyParameters):
    linked_terminal: bool = True
    continuous_service: bool = True
    samples: int = 1024
    search_scenarios: int = 384
    validation_scenarios: int = 2048
    refine_iterations: int = 24
    refine_finalists: int = 6
    next_entry_limit: int = 6

    def validate(self):
        super().validate()
        for name in ('linked_terminal','continuous_service'):
            if not isinstance(getattr(self,name),bool):raise ValueError(name)
        for name,lo,hi in [('search_scenarios',16,2048),('validation_scenarios',32,8192),
                          ('refine_iterations',0,200),('refine_finalists',2,24),('next_entry_limit',1,16)]:
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or not lo<=v<=hi:raise ValueError(name)
        return self
