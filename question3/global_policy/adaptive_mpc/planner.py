"""One real action per planning cycle, followed by a public stop scan."""
from pathlib import Path
import json
import numpy as np
from ..dynamic.runner import Planner as DynamicPlanner
from ..decisions import choose_initial_probe, localization_value
from .parameters import AdaptiveParameters
from .exploration import ExplorationMemory
from .scanning import assess_stop, known_scan_value
from .planning import choose_next


class Planner(DynamicPlanner):
    def __init__(self,client,parameters=None):
        super().__init__(client,parameters or AdaptiveParameters())
        self.memory=ExplorationMemory(self.world.params)
        self.changes=[];self.cycles=0;self.previous_channel=None;self.target_switches=0

    def execute(self,path,q=None,channel=None,reason=''):
        w=self.world
        old=w.targets[channel] if channel else None
        status=old.status if old else None;radius=old.radius if old else np.inf
        result=super().execute(path,q,channel,reason)
        if q is not None:
            self.memory.observe(path,q,channel,result,len(w.commands)-1)
            t=w.targets[channel]
            if status=='unknown' and t.status=='active':self.changes.append(dict(kind='new_source',channel=channel))
            if status=='active' and t.status=='cleared':self.changes.append(dict(kind='cleared',channel=channel))
            if status=='active' and t.radius<radius*(1-w.params.replan_shrink_fraction):
                self.changes.append(dict(kind='radius_shrank',channel=channel,before=radius,after=t.radius))
        return result

    def early(self):
        return self.world.params.early_mapping and len(self.memory.visits)<=self.world.params.early_stops

    def scan_here(self,mandatory=(),reason='stop_scan'):
        w=self.world
        # Selection is repeated after actual observations so new sources can enter
        # the public map. A stop never measures a channel twice at that location.
        channels,values=assess_stop(w,early=self.early(),mandatory=mandatory)
        if w.params.early_geometry_mapping and 0<len(self.memory.visits)-1<=w.params.early_geometry_stops:
            for t in w.active():
                if t.channel in channels or t.radius<=20 or t.measured_at(w.position):continue
                row=known_scan_value(w,t,w.position)
                if row.get('valid') and row.get('expected_radius_ratio',1)<.6 and row.get('detection_probability',0)>.5:
                    channels.append(t.channel)
                    row.update(selected=True,reason='early_geometry_mapping_spend')
                    values=[v for v in values if v['channel']!=t.channel]+[row]
        if not w.params.stop_scans:channels=[c for c in channels if c in mandatory]
        self.event('stop_scan',early=self.early(),selected=channels,values=values)
        chosen=[]
        for c in channels:
            if w.targets[c].measured_at(w.position):raise AssertionError('Duplicate scan proposal')
            self.execute('/measure',w.position,c,reason)
            chosen.append(c)
        self.event('stop_scan_finished',channels=chosen,early=self.early())

    def waypoint_action(self,action):
        """Interrupt a long leg only if an intermediate measurement can be useful."""
        w=self.world;a=action.copy();q=np.asarray(a['position']);d=np.linalg.norm(q-w.position)
        if d<=w.params.max_waypoint_step:return a
        y=w.position+(q-w.position)*(w.params.max_waypoint_step/d)
        threshold=w.params.early_unknown_scan_gain if self.early() else w.params.unknown_scan_gain
        known=[t for t in w.active() if not t.measured_at(y) and t.radius>20 and localization_value(t,y,w)>.05]
        unknown=[t for t in w.unknown() if not t.measured_at(y) and w.coverage.gain(t.channel,y)>=threshold]
        if known:
            channel=max(known,key=lambda t:localization_value(t,y,w)).channel
        elif unknown:channel=max(unknown,key=lambda t:w.coverage.gain(t.channel,y)).channel
        else:return a
        return dict(path='/measure',position=y,channel=channel,name='enroute_probe',mandatory=[],
                    intended_destination=q.tolist(),intended_channel=a['channel'])

    def plan(self):
        w=self.world;self.event('origin_scan')
        for c in range(1,21):self.execute('/measure',np.zeros(2),c,'origin_all_channels')
        from ..dynamic.frontier import Sweep
        orientation=Sweep(w)
        self.memory.sweep_sign=orientation.sign
        self.event('soft_sweep_orientation',heading=orientation.heading,sign=orientation.sign)
        if w.active():
            q=choose_initial_probe(w)
            t=max(w.active(),key=lambda t:localization_value(t,q,w))
            self.event('initial_probe_uncommitted',channel=t.channel,destination=q.tolist())
            self.execute('/measure',q,t.channel,'initial_map_probe')
            self.scan_here()
        for cycle in range(w.params.max_commands):
            if w.finished():self.event('coverage_closed');return
            self.cycles+=1
            triggers=self.changes or [dict(kind='new_stop_or_feedback')];self.changes=[]
            self.event('global_replan',triggers=triggers,active=[t.channel for t in w.active()])
            action,values=choose_next(w,self.memory,self.previous_channel,self.early())
            executed=action
            if self.previous_channel is not None and executed['channel']!=self.previous_channel:
                self.target_switches+=1
            self.previous_channel=executed['channel']
            self.event('global_action_decision',channel=executed['channel'],action=executed['path'],
                       destination=np.asarray(executed['position']).tolist(),candidates=values,
                       waypoint=executed['name']=='enroute_probe',reason=executed['name'])
            self.execute(executed['path'],executed['position'],executed['channel'],executed['name'])
            # Actual state decides scans, including after miss/success, not only
            # after a full source service. Full frontier masks retain mandatory scans.
            mandatory=executed.get('mandatory',[])
            self.scan_here(mandatory)
        raise RuntimeError('Global planning guard reached before completion')

    def run(self):
        row=super().run()
        row.update(policy='adaptive_mpc',replans=self.cycles,target_switches=self.target_switches,
                   stop_count=len(self.memory.visits))
        w=self.world;switches=0;current=1;scan=0;clear=0
        for cmd in w.commands:
            if cmd['path']=='/measure':
                scan+=5;switches+=int(current!=cmd['channel']);current=cmd['channel']
            if cmd['path']=='/clear':clear+=5 if cmd['response']['clear_result']=='success' else 3
        row['time_components_s']=dict(movement=w.distance/5,measure=scan,switch=switches,clear=clear)
        return row

    def save(self,directory,summary):
        super().save(directory,summary)
        (Path(directory)/'exploration_memory.json').write_text(json.dumps(self.memory.snapshot(),ensure_ascii=False),encoding='utf8')


def run_local(seed,params=None,output=None):
    from question3.local_sim.simulator import Simulator,Client
    sim=Simulator(seed);planner=Planner(Client(simulator=sim),params);row=planner.run()
    row.update(seed=seed,evaluation='local_synthetic',true_total=len(sim.sources),true_cleared=sum(s.cleared for s in sim.sources))
    if row['complete'] and not all(s.cleared for s in sim.sources):row.update(complete=False,failure='Evaluator found uncleared sources')
    if output:planner.save(output,row);sim.dump(Path(output)/'evaluator')
    return row,planner
