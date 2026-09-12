"""Dynamic coverage sweep. Default execution is entirely local and synthetic."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
from ..runner import Planner as BasePlanner, HTTPClient
from ..decisions import select_known_channels, select_unknown_channels
from .parameters import DynamicParameters
from .frontier import Sweep, nearby_targets, short_service_route
from .local_rollout import choose_action


class Planner(BasePlanner):
    def __init__(self,client,parameters=None):
        super().__init__(client,parameters or DynamicParameters())
        self.sweep=None

    def service_target(self,channel):
        w=self.world;target=w.targets[channel];probes=0
        self.event('service_start',channel=channel,radius=target.radius,sector=self.sweep.index)
        for _ in range(16):
            if target.status=='cleared':return
            path,q,reason,values=choose_action(w,target,probes)
            if path=='/measure':probes+=1
            self.event('local_decision',channel=channel,action=path,destination=q.tolist(),
                       radius=target.radius,hit_probability=target.hit_probability(q),reason=reason,
                       candidates=values,sector=self.sweep.index)
            self.execute(path,q,channel,reason)
        raise RuntimeError(f'Channel {channel}: service guard reached')

    def scan_here(self,mandatory=(),reason='opportunistic_stop'):
        w=self.world;heading=self.sweep.heading+self.sweep.sign*self.sweep.index*self.sweep.width
        channels=list(dict.fromkeys(list(mandatory)+select_known_channels(w,w.position,heading)+select_unknown_channels(w,w.position)))
        for channel in channels:
            target=w.targets[channel]
            if target.status=='cleared' or target.measured_at(w.position):continue
            self.execute('/measure',w.position,channel,reason)

    def clean_neighborhood(self):
        for _ in range(20):
            nearby=nearby_targets(self.world)
            if not nearby:return
            frontier=self.sweep.next_point()
            destination=frontier[1] if frontier else self.world.position
            route=short_service_route(self.world,nearby,destination)
            self.event('nearby_cleanup_route',channels=route,sector=self.sweep.index)
            self.service_target(route[0]);self.scan_here()
        raise RuntimeError('Neighborhood cleanup did not terminate')

    def plan(self):
        w=self.world
        self.event('origin_scan')
        for channel in range(1,21):self.execute('/measure',np.zeros(2),channel,'origin_all_channels')
        self.sweep=Sweep(w)
        self.event('sweep_start',heading=self.sweep.heading,sign=self.sweep.sign,sectors=w.params.sector_count)
        for index in range(w.params.sector_count):
            self.sweep.index=index
            for _ in range(100):
                self.clean_neighborhood()
                if self.sweep.can_close():
                    self.sweep.closed.append(index)
                    self.event('sector_closed',sector=index)
                    break
                outstanding=self.sweep.outstanding();frontier=self.sweep.next_point()
                # All known sources that could occupy this region are serviced before
                # relocation to another unknown patch. No radius-based eligibility gate.
                if outstanding:
                    destination=frontier[1] if frontier else w.position
                    route=short_service_route(w,outstanding,destination)
                    self.event('region_cleanup_route',channels=route,sector=index)
                    self.service_target(route[0]);self.scan_here()
                    continue
                if frontier is None:raise RuntimeError('Unclosed region has no frontier or service target')
                score,q,channels=frontier
                if nearby_targets(w):raise AssertionError('Leaving a known nearby target behind')
                self.event('coverage_frontier',sector=index,destination=q.tolist(),
                           gain_per_second=score,channels=channels,nearby_left=[])
                # Movement is attached to a useful measurement, preserving the protocol.
                first=channels[0];self.execute('/measure',q,first,'coverage_frontier')
                self.scan_here(channels[1:],'coverage_frontier')
            else:raise RuntimeError('Region progress guard reached')
        if not w.finished():raise RuntimeError('Sweep closed but public completion check failed')
        self.event('coverage_closed')

    def run(self):
        started=time.perf_counter()
        try:self.execute('/enter');self.plan()
        except Exception as exc:self.failure=f'{type(exc).__name__}: {exc}'
        finally:
            if self.entered:
                try:self.execute('/exit',reason='complete' if self.world.finished() and not self.failure else 'incomplete')
                except Exception as exc:self.failure=self.failure or f'Exit failed: {exc}'
        w=self.world;removed=sum(t.status=='cleared' for t in w.targets.values())
        frontiers=[e for e in self.events if e['phase']=='coverage_frontier']
        return dict(policy='dynamic_frontier_rollout',complete=w.finished() and self.failure is None,
                    cleared=removed,virtual_time_s=w.virtual_time,average_time_s=w.virtual_time/removed if removed else None,
                    wall_time_s=time.perf_counter()-started,distance_m=w.distance,
                    measures=sum(x['path']=='/measure' for x in w.commands),
                    clears=sum(x['path']=='/clear' for x in w.commands),misses=sum(t.misses for t in w.targets.values()),
                    failure=self.failure,coverage=w.coverage.summary(),parameters=asdict(w.params),
                    frontier_radii_m=[float(np.linalg.norm(e['destination'])) for e in frontiers],
                    closed_sectors=self.sweep.closed if self.sweep else [],
                    frontier_departures_with_nearby=sum(bool(e['nearby_left']) for e in frontiers))


def run_local(seed,params=None,output=None):
    from question3.local_sim.simulator import Simulator,Client
    sim=Simulator(seed);planner=Planner(Client(simulator=sim),params)
    summary=planner.run()
    summary.update(evaluation='local_synthetic',seed=seed,true_total=len(sim.sources),true_cleared=sum(s.cleared for s in sim.sources))
    if summary['complete'] and not all(s.cleared for s in sim.sources):
        summary.update(complete=False,failure='Evaluator found missed sources despite completion claim')
    if output:
        planner.save(output,summary);sim.dump(Path(output)/'evaluator')
    return summary,planner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20260911)
    parser.add_argument('--params')
    parser.add_argument('--output',default='question3/global_policy/dynamic/outputs/demo')
    parser.add_argument('--mode',choices=['local','http'],default='local')
    parser.add_argument('--base-url',default='http://127.0.0.1:2027')
    parser.add_argument('--robot-id',default='local')
    parser.add_argument('--http-context',choices=['local','practice'],default='local')
    parser.add_argument('--practice-confirmed',action='store_true')
    args=parser.parse_args()
    params=DynamicParameters(**json.loads(Path(args.params).read_text(encoding='utf8'))) if args.params else DynamicParameters()
    params.validate()
    if args.mode=='local':summary,_=run_local(args.seed,params,args.output)
    else:
        from urllib.parse import urlsplit
        if args.http_context=='practice' and not args.practice_confirmed:
            parser.error('Practice requires explicit UI verification and --practice-confirmed; formal tests are forbidden')
        if args.http_context=='local' and urlsplit(args.base_url).port==2026:
            parser.error('Port 2026 requires explicit practice context; local simulator uses 2027')
        planner=Planner(HTTPClient(args.robot_id,args.base_url),params)
        summary=planner.run();summary['evaluation']='official_practice' if args.http_context=='practice' else 'local_http'
        planner.save(args.output,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not summary['complete']:raise SystemExit(1)


if __name__=='__main__':main()
