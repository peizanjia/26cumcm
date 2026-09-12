"""Joint terminal routing and receding-horizon single-source service."""
import numpy as np
from pathlib import Path
from ..route_study.planner import Planner as StudyPlanner
from ..route_study.routes import held_karp
from .parameters import JointParameters
from .service import choose_action


def open_cost(start,points,end=None):
    points=np.asarray(points).reshape(-1,2)
    if not len(points):return (float(np.linalg.norm(start-end)) if end is not None else 0.)
    first=np.linalg.norm(points-start,axis=1)
    edges=np.linalg.norm(points[:,None]-points[None,:],axis=2)
    last=np.zeros(len(points)) if end is None else np.linalg.norm(points-end,axis=1)
    return float(held_karp(first,edges,last)[0])


class Planner(StudyPlanner):
    def __init__(self,client,parameters=None):super().__init__(client,parameters or JointParameters())

    def terminal_options(self,targets):
        w=self.world;original=self.sweep.index;excluded={t.channel for t in targets}
        options=[]
        try:
            # First unfinished region with useful known/unknown public work.
            for index in range(original,w.params.sector_count):
                if index in self.sweep.closed:continue
                self.sweep.index=index
                tail=[t for t in self.sweep.outstanding() if t.channel not in excluded]
                frontier=self.sweep.next_point()
                if not tail and frontier is None:continue
                points=np.array([t.center for t in tail]).reshape(-1,2)
                entries=[(t.center,'known',t.channel) for t in tail]
                if frontier is not None:entries.append((frontier[1],'coverage',None))
                # Keep both frontier and spatially distinct service entry candidates.
                entries=entries[:w.params.next_entry_limit-1]+([entries[-1]] if entries else [])
                unique=[]
                for entry,kind,ch in entries:
                    if any(np.linalg.norm(entry-e[0])<1e-6 for e in unique):continue
                    unique.append((entry,kind,ch))
                for entry,kind,ch in unique:
                    end=frontier[1] if frontier is not None and kind!='coverage' else None
                    tail_cost=open_cost(entry,points,end)/5+5*len(tail)
                    if frontier is not None:tail_cost+=6*len(frontier[2])
                    options.append(dict(position=np.asarray(entry).copy(),tail_s=tail_cost,
                                        sector=index,kind=kind,channel=ch))
                break
        finally:self.sweep.index=original
        return options

    def route(self,targets,destination,open_end=False):
        if not self.world.params.linked_terminal:return super().route(targets,destination,open_end)
        if not targets:return []
        w=self.world;points=np.array([t.center for t in targets]);options=self.terminal_options(targets)
        if options:
            costs=np.array([np.linalg.norm(points-o['position'],axis=1)/5+o['tail_s'] for o in options])
            choices=np.argmin(costs,axis=0);end_cost=np.min(costs,axis=0)
        else:choices=np.zeros(len(points),int);end_cost=np.zeros(len(points))
        first=np.linalg.norm(points-w.position,axis=1)/5
        edges=np.linalg.norm(points[:,None]-points[None,:],axis=2)/5
        cost,ids=held_karp(first,edges,end_cost);order=[targets[int(i)].channel for i in ids]
        selected=options[int(choices[ids[-1]])] if options else None
        self.route_sizes.append(len(targets))
        self.event('linked_terminal',sector=self.sweep.index,channels=order,
                   centers=points.tolist(),start=w.position.tolist(),exit=points[ids[-1]].tolist(),
                   entry=selected['position'].tolist() if selected else None,
                   next_sector=selected['sector'] if selected else None,
                   surrogate_time_s=float(cost),terminal_options=[{**o,'position':o['position'].tolist()} for o in options])
        self.event('route_problem',channels=[t.channel for t in targets],centers=points.tolist(),
                   destination=selected['position'].tolist() if selected else w.position.tolist(),
                   selected_order=order,method='linked_terminal_dp',open_end=not options)
        return order

    def service_target(self,channel):
        if not self.world.params.continuous_service:return super().service_target(channel)
        w=self.world;t=w.targets[channel];self.event('service_start',channel=channel,radius=t.radius,sector=self.sweep.index)
        for step in range(24):
            if t.status=='cleared':return
            path,q,reason,values=choose_action(w,t,step)
            self.event('local_decision',channel=channel,action=path,destination=q.tolist(),radius=t.radius,
                       hit_probability=t.hit_probability(q),reason=reason,candidates=values,sector=self.sweep.index)
            self.execute(path,q,channel,reason)
        raise RuntimeError(f'Channel {channel}: joint service guard reached')

    def run(self):
        row=super().run();row['policy']='joint_rollout'
        row['linked_routes']=sum(e['phase']=='linked_terminal' for e in self.events)
        row['cross_region_terminals']=sum(e['phase']=='linked_terminal' and e['next_sector'] is not None
                                            and e['next_sector']>e['sector'] for e in self.events)
        row['joint_decisions']=sum(e['phase']=='local_decision' and e['reason'].startswith('joint_') for e in self.events)
        return row


def run_local(seed,params=None,output=None):
    from question3.local_sim.simulator import Simulator,Client
    sim=Simulator(seed);planner=Planner(Client(simulator=sim),params);row=planner.run()
    row.update(seed=seed,evaluation='local_synthetic',true_total=len(sim.sources),true_cleared=sum(s.cleared for s in sim.sources))
    if row['complete'] and not all(s.cleared for s in sim.sources):row.update(complete=False,failure='Evaluator found uncleared sources')
    if output:planner.save(output,row);sim.dump(Path(output)/'evaluator')
    return row,planner
