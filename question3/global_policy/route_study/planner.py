"""Alternative local and global routes through the unchanged public interface."""
import math
from pathlib import Path
import numpy as np
from ..dynamic.runner import Planner as DynamicPlanner
from ..dynamic.frontier import Sweep,nearby_targets
from ..dynamic.local_rollout import choose_action
from ..decisions import select_known_channels,select_unknown_channels
from .parameters import StudyParameters
from .routes import plan_route
from .information import rank_known_scans


class Planner(DynamicPlanner):
    def __init__(self,client,parameters=None):
        super().__init__(client,parameters or StudyParameters());self.inner_phase=False;self.route_sizes=[]

    def route(self,targets,destination,open_end=False):
        self.route_sizes.append(len(targets))
        w=self.world
        method='exact_open' if open_end and w.params.open_if_no_frontier else w.params.route_method
        if w.params.route_scope=='global_guided':
            all_targets=w.active();global_order=plan_route(w,all_targets,destination,method)
            eligible={t.channel for t in targets};order=[c for c in global_order if c in eligible]
            self.event('global_route_guidance',all_channels=global_order,eligible=sorted(eligible),selected_order=order)
        else:order=plan_route(w,targets,destination,method)
        self.event('route_problem',channels=[t.channel for t in targets],centers=[t.center.tolist() for t in targets],
                   destination=np.asarray(destination).tolist(),selected_order=order,method=method,open_end=open_end)
        return order

    def scan_here(self,mandatory=(),reason='opportunistic_stop'):
        if self.world.params.scan_mode=='heuristic':return super().scan_here(mandatory,reason)
        w=self.world
        # Coverage-required measurements are not dropped by optional-information rules.
        for channel in dict.fromkeys(list(mandatory)+select_unknown_channels(w,w.position)):
            t=w.targets[channel]
            if t.status!='cleared' and not t.measured_at(w.position):self.execute('/measure',w.position,channel,reason)
        channels,values=rank_known_scans(w)
        self.event('stationary_information_value',selected=channels,values=values)
        for channel in channels:
            if not w.targets[channel].measured_at(w.position):self.execute('/measure',w.position,channel,'positive_scan_value')

    def service_target(self,channel):
        if not self.world.params.scan_each_probe:return super().service_target(channel)
        w=self.world;t=w.targets[channel];probes=0
        self.event('service_start',channel=channel,radius=t.radius,sector=self.sweep.index)
        for _ in range(16):
            if t.status=='cleared':return
            path,q,reason,values=choose_action(w,t,probes)
            if path=='/measure':probes+=1
            self.event('local_decision',channel=channel,action=path,destination=q.tolist(),radius=t.radius,
                       hit_probability=t.hit_probability(q),reason=reason,candidates=values,sector=self.sweep.index)
            self.execute(path,q,channel,reason)
            if path=='/measure':self.scan_here(reason='probe_stop')
        raise RuntimeError('Local service guard reached')

    def inner_probability(self,target):
        if target.particles is None:return float(np.linalg.norm(target.center)+target.radius<=self.world.params.inner_radius)
        return float(target.weights[np.linalg.norm(target.particles,axis=1)<=self.world.params.inner_radius].sum())

    def clean_neighborhood(self):
        w=self.world
        for _ in range(20):
            nearby=nearby_targets(w)
            if self.inner_phase:nearby=[t for t in nearby if self.inner_probability(t)>=w.params.inner_probability]
            if not nearby:return
            frontier=self.sweep.next_point()
            destination=frontier[1] if frontier else w.position
            route=self.route(nearby,destination,open_end=frontier is None)
            self.event('nearby_cleanup_route',channels=route,sector=self.sweep.index,inner_phase=self.inner_phase)
            self.service_target(route[0]);self.scan_here()
        raise RuntimeError('Neighborhood cleanup guard reached')

    def inner_pass(self):
        w=self.world;self.inner_phase=True
        angles=self.sweep.heading+self.sweep.sign*np.arange(w.params.inner_nodes)*2*math.pi/w.params.inner_nodes
        for angle in angles:
            q=w.params.inner_radius*np.array([math.cos(angle),math.sin(angle)])
            heading=math.atan2(*(q-w.position)[::-1])
            channels=list(dict.fromkeys(select_known_channels(w,q,heading)+select_unknown_channels(w,q)))
            if not channels:continue
            self.event('inner_search',destination=q.tolist(),channels=channels)
            self.execute('/measure',q,channels[0],'inner_search');self.scan_here(channels[1:],'inner_search')
            self.clean_neighborhood()
        self.inner_phase=False
        # Inner pass never falsely closes an outer region. Re-anchor the outer sweep
        # near the current angle while retaining its direction and coverage history.
        width=self.sweep.width;angle=math.atan2(*w.position[::-1])
        shift=int(round((self.sweep.sign*(angle-self.sweep.heading))%(2*math.pi)/width))%w.params.sector_count
        self.sweep.heading+=self.sweep.sign*shift*width
        self.sweep.regions=(self.sweep.regions-shift)%w.params.sector_count
        self.sweep.region_masks=self.sweep.region_masks[shift:]+self.sweep.region_masks[:shift]
        self.sweep.region_planes=self.sweep.region_planes[shift:]+self.sweep.region_planes[:shift]
        self.sweep.index=0
        self.event('outer_sweep_start',heading=self.sweep.heading,sign=self.sweep.sign,sectors=w.params.sector_count)

    def next_sector(self,remaining):
        if self.world.params.global_mode!='adaptive_sector':return min(remaining)
        w=self.world;choices=[]
        for index in remaining:
            self.sweep.index=index
            targets=self.sweep.outstanding();frontier=self.sweep.next_point()
            distances=[float(np.linalg.norm(t.center-w.position)) for t in targets]
            if frontier is not None:distances.append(float(np.linalg.norm(frontier[1]-w.position)))
            choices.append((min(distances) if distances else 0.,index))
        return min(choices)[1]

    def plan(self):
        w=self.world;self.event('origin_scan')
        for channel in range(1,21):self.execute('/measure',np.zeros(2),channel,'origin_all_channels')
        self.sweep=Sweep(w)
        self.event('sweep_start',heading=self.sweep.heading,sign=self.sweep.sign,sectors=w.params.sector_count)
        if w.params.global_mode=='inner_first':self.inner_pass()
        if w.params.global_mode=='global_graph':return self.global_graph_pass()
        remaining=set(range(w.params.sector_count))
        while remaining:
            index=self.next_sector(remaining);self.sweep.index=index
            if w.params.global_mode=='block_first':self.map_block_first()
            for _ in range(100):
                self.clean_neighborhood()
                if self.sweep.can_close():
                    self.sweep.closed.append(index);self.event('sector_closed',sector=index);remaining.remove(index);break
                outstanding=self.sweep.outstanding();frontier=self.sweep.next_point()
                if outstanding:
                    route=self.route(outstanding,frontier[1] if frontier else w.position,open_end=frontier is None)
                    self.event('region_cleanup_route',channels=route,sector=index)
                    self.service_target(route[0]);self.scan_here();continue
                if frontier is None:raise RuntimeError('No frontier available for incomplete region')
                score,q,channels=frontier
                if nearby_targets(w):raise AssertionError('Leaving known nearby source')
                self.event('coverage_frontier',sector=index,destination=q.tolist(),gain_per_second=score,channels=channels,nearby_left=[])
                self.execute('/measure',q,channels[0],'coverage_frontier');self.scan_here(channels[1:],'coverage_frontier')
            else:raise RuntimeError('Region progress guard reached')
        if not w.finished():raise RuntimeError('Invalid completion certificate')
        self.event('coverage_closed')

    def map_block_first(self):
        """Experimental scan-then-freeze architecture, deliberately defers nearby clearing."""
        w=self.world
        for _ in range(50):
            frontier=self.sweep.next_point()
            if frontier is None:break
            _,q,channels=frontier
            self.event('block_mapping',sector=self.sweep.index,destination=q.tolist())
            self.execute('/measure',q,channels[0],'block_mapping');self.scan_here(channels[1:],'block_mapping')
        else:raise RuntimeError('Block mapping guard reached')
        for _ in range(20):
            targets=self.sweep.outstanding()
            if not targets:return
            order=self.route(targets,w.position)
            self.event('frozen_block_order',channels=order,centers=[t.center.tolist() for t in targets],sector=self.sweep.index)
            for channel in order:
                if w.targets[channel].status!='cleared':self.service_target(channel);self.scan_here()
        raise RuntimeError('Frozen block service guard reached')

    def global_graph_pass(self):
        """Global known-source graph, with globally scored unknown-frontier actions."""
        w=self.world
        for _ in range(100):
            for index in range(w.params.sector_count):
                if index in self.sweep.closed:continue
                self.sweep.index=index
                if self.sweep.can_close():self.sweep.closed.append(index);self.event('sector_closed',sector=index)
            if w.finished():self.event('coverage_closed');return
            if w.active():
                order=self.route(w.active(),w.position)
                self.event('global_known_route',channels=order)
                self.service_target(order[0]);self.scan_here();continue
            options=[]
            for index in range(w.params.sector_count):
                if index in self.sweep.closed:continue
                self.sweep.index=index;frontier=self.sweep.next_point()
                if frontier is not None:options.append((frontier[0],index,frontier[1],frontier[2]))
            if not options:raise RuntimeError('No global frontier')
            score,index,q,channels=max(options,key=lambda v:v[0]);self.sweep.index=index
            self.event('coverage_frontier',sector=index,destination=q.tolist(),gain_per_second=score,channels=channels,nearby_left=[])
            self.execute('/measure',q,channels[0],'coverage_frontier');self.scan_here(channels[1:],'coverage_frontier')
        raise RuntimeError('Global graph guard reached')

    def run(self):
        summary=super().run();summary['policy']='route_study'
        summary['route_sizes']=self.route_sizes
        summary['voi_scans']=sum(c['reason']=='positive_scan_value' for c in self.world.commands)
        return summary


def run_local(seed,params=None,output=None):
    from question3.local_sim.simulator import Simulator,Client
    sim=Simulator(seed);planner=Planner(Client(simulator=sim),params);summary=planner.run()
    summary.update(seed=seed,evaluation='local_synthetic',true_total=len(sim.sources),true_cleared=sum(s.cleared for s in sim.sources))
    if summary['complete'] and not all(s.cleared for s in sim.sources):summary.update(complete=False,failure='Evaluator found missed sources')
    if output:planner.save(output,summary);sim.dump(Path(output)/'evaluator')
    return summary,planner
