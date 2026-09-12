"""Experimental tour improvements, consuming only public command feedback."""
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ..adaptive_mpc.tour_planner import Planner as PreviousPlanner
from ..adaptive_mpc.planning import coverage_tail
from ..adaptive_mpc.anticipated_tail import anticipated_tail
from ..adaptive_mpc.scanning import assess_stop, known_scan_value, _known_geometry
from ..decisions import choose_initial_probe, localization_value
from ..joint_rollout.service import choose_action
from .parameters import Parameters
from .routing import refine_route


class Planner(PreviousPlanner):
    def __init__(self,client,parameters=None):
        super().__init__(client,parameters or Parameters())
        self.forecast_route=[]
        self.deferred_scans=0

    def scan_here(self,mandatory=(),reason='opportunistic_stop'):
        w=self.world;p=w.params
        channels,rows=assess_stop(w,early=self.early(),mandatory=mandatory)
        geometric=[];nonorigin=len(self.memory.visits)-1
        by_channel={r['channel']:r for r in rows}
        if p.early_geometry_mapping and p.stop_scans and 1<=nonorigin<=p.early_geometry_stops:
            for target in w.active():
                if target.channel in channels or target.radius<=20 or target.measured_at(w.position):continue
                row=by_channel[target.channel]
                if not row.get('valid'):
                    row=known_scan_value(w,target,w.position);by_channel[target.channel]=row
                if (row.get('valid') and row.get('expected_radius_ratio',math.inf)<p.mapping_radius_ratio
                        and row.get('detection_probability',0)>p.mapping_detection_probability):
                    row.update(selected=True,reason='early_geometry_mapping_spend',geometry_mapping=True,
                               value_kind='geometry_map_investment_not_positive_time_voi')
                    geometric.append(target.channel);channels.append(target.channel)
        if p.defer_scans and not self.early() and self.forecast_route:
            upcoming=[]
            for point in self.forecast_route:
                for target in w.active():
                    if (target.channel!=self.last_target and target.radius<=100 and
                        np.linalg.norm(target.center-point)<1e-5 and
                        np.linalg.norm(point-w.position)<=p.defer_distance):
                        upcoming.append(w.coverage.mask_at(point))
                if len(upcoming)>=2:break
            if upcoming:
                predicted=np.logical_or.reduce(upcoming)
                here=w.coverage.mask_at(w.position)
                for channel in channels.copy():
                    if channel in mandatory or w.targets[channel].status!='unknown':continue
                    newly=here & ~w.coverage.covered[channel-1]
                    overlap=float(np.count_nonzero(newly & predicted)/max(1,np.count_nonzero(newly)))
                    if overlap>=p.defer_overlap:
                        channels.remove(channel);self.deferred_scans+=1
                        by_channel[channel].update(selected=False,reason='deferred_to_nearby_known_stop',
                                                   predicted_overlap=overlap)
        if not p.stop_scans:channels=[c for c in channels if c in mandatory]
        self.event('stop_scan',selected=channels,values=[by_channel[c] for c in sorted(by_channel)],
                   early=self.early(),geometric_mapping_channels=geometric,nonorigin_stop=nonorigin)
        for channel in channels:
            if not w.targets[channel].measured_at(w.position):
                self.execute('/measure',w.position,channel,
                             'early_geometry_mapping_spend' if channel in geometric else reason)

    def service_target(self,channel):
        if self.world.params.coupled_scan_weight<=0:return super().service_target(channel)
        w=self.world;t=w.targets[channel]
        steps=self.service_steps.get(channel,0)
        if steps>=80:raise RuntimeError('Coupled service progress guard')
        path,q,reason,values=choose_action(w,t,steps)
        for row in values:
            y=np.asarray(row['destination']);credits=[]
            for other in w.active():
                if (other.channel==channel or other.radius<=20 or other.measured_at(y)
                        or other.particles is None):continue
                credits.append((_known_geometry(other,y)['geometry_screen_score'],other))
            credit=0.;scans=[]
            for _,other in sorted(credits,key=lambda z:-z[0])[:2]:
                estimate=known_scan_value(w,other,y)
                if estimate['valid'] and estimate['net_saved_s']>0:
                    credit+=min(40.,estimate['net_saved_s']);scans.append(other.channel)
            useful=(row['action']=='/clear' and t.radius<=20) or localization_value(t,y,w)>.15
            penalty=self.memory.penalty(y,channel=channel,useful=useful)
            row.update(side_scan_credit_s=credit,side_scan_channels=scans,revisit_penalty_s=penalty,
                       selection_score_s=row['expected_remaining_s']+penalty-w.params.coupled_scan_weight*credit)
            row.pop('selected',None)
        if values:
            best=min(values,key=lambda r:r['selection_score_s']);best['selected']=True
            path,q,reason=best['action'],np.asarray(best['destination']),'coupled_'+best['name']
        self.replans+=1
        if self.last_target is not None and channel!=self.last_target:self.target_switches+=1
        self.last_target=channel
        self.event('local_decision',channel=channel,action=path,destination=q.tolist(),radius=t.radius,
                   candidates=values,interruptible=True,reason=reason)
        self.execute(path,q,channel,reason);self.service_steps[channel]=steps+1

    def plan(self):
        w=self.world
        self.sweep=SimpleNamespace(index=-1,closed=[],heading=0.,sign=1.,width=2*np.pi)
        self.event('origin_scan')
        for channel in range(1,21):self.execute('/measure',np.zeros(2),channel,'origin_all_channels')
        if w.params.initial_map_probe and w.active():
            q=choose_initial_probe(w);t=max(w.active(),key=lambda t:localization_value(t,q,w))
            self.event('initial_probe_uncommitted',channel=t.channel,destination=q.tolist())
            self.execute('/measure',q,t.channel,'initial_map_probe');self.scan_here()
        for cycle in range(240):
            if w.finished():self.event('coverage_closed');return
            self.tour_replans+=1
            stations=coverage_tail(w)
            cost,route=anticipated_tail(w,{t.channel:0. for t in w.active()},w.position,None,
                                       w.position,[],stations,early=self.early())
            route,refinement=refine_route(w,route)
            self.forecast_route=route
            action=self._route_action(route)
            self.event('global_tour_replan',cycle=cycle,route=[q.tolist() for q in route],
                       forecast_route_seconds=cost+(refinement['after_m']-refinement['before_m'])/5,
                       channel=action['channel'],kind=action['kind'],destination=action['position'].tolist(),
                       refinement=refinement)
            if action['kind']=='known':self.service_target(action['channel']);self.scan_here()
            else:
                self.execute('/measure',action['position'],action['channel'],'tour_coverage_frontier')
                self.scan_here(action['mandatory'],'tour_coverage_frontier')
        raise RuntimeError('Tour optimization progress guard')

    def run(self):
        row=super().run();row.update(policy='optimized_tour',deferred_scans=self.deferred_scans)
        return row


def run_local(seed,params=None,output=None):
    from question3.local_sim.simulator import Simulator,Client
    simulator=Simulator(seed);planner=Planner(Client(simulator=simulator),params)
    row=planner.run()
    row.update(seed=seed,evaluation='local_synthetic',true_total=len(simulator.sources),
               true_cleared=sum(s.cleared for s in simulator.sources))
    if row['complete'] and not all(s.cleared for s in simulator.sources):
        row.update(complete=False,failure='Evaluator found an uncleared source')
    if output:
        planner.save(output,row);simulator.dump(Path(output)/'evaluator')
    return row,planner
