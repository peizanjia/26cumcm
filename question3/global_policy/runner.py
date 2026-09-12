"""Execute decisions through the same four documented JSON commands."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np
from .model import Parameters,World
from .decisions import (choose_initial_probe,choose_ring_route,select_known_channels,
    select_unknown_channels,plan_cleanup_route,choose_clear_or_probe,choose_coverage_patch)


class HTTPClient:
    def __init__(self,robot_id,base_url,retries=3):
        self.robot_id,self.base_url,self.retries=robot_id,base_url.rstrip('/'),retries
        self.prefix=uuid.uuid4().hex;self.counter=0

    def command(self,path,position=None,channel=None):
        from urllib.request import Request,urlopen
        from urllib.error import HTTPError,URLError
        self.counter+=1
        payload=dict(arena_id='default',robot_id=self.robot_id,request_id=f'{self.prefix}-{self.counter}')
        if position is not None:
            payload.update(position=dict(x=float(position[0]),y=float(position[1])),channel=int(channel))
        body=json.dumps(payload).encode('utf8')
        for attempt in range(self.retries):
            try:
                request=Request(self.base_url+path,body,{'Content-Type':'application/json'},method='POST')
                with urlopen(request,timeout=10) as response:result=json.load(response)
                if not result.get('accepted'):raise RuntimeError(f'Action rejected: {result}')
                return result
            except HTTPError as exc:
                raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode("utf8",errors="replace")}') from exc
            except (URLError,TimeoutError,ConnectionError,OSError):
                if attempt+1==self.retries:raise
                time.sleep(.2)
        raise RuntimeError('Unreachable retry state')


class Planner:
    def __init__(self,client,parameters=None):
        self.client=client;self.world=World(parameters or Parameters())
        self.deadline=math.inf;self.events=[];self.failure=None;self.entered=False

    def execute(self,path,q=None,channel=None,reason=''):
        w=self.world
        if path not in ('/enter','/exit'):
            if len(w.commands)>=w.params.max_commands:raise RuntimeError('Command budget exhausted before coverage closure')
            if time.monotonic()>=self.deadline-2:raise RuntimeError('Real-time budget exhausted before coverage closure')
        result=self.client.command(path,q,channel)
        w.commands.append(dict(path=path,position=np.asarray(q).tolist() if q is not None else None,
                               channel=channel,reason=reason,response=result))
        if not result.get('accepted'):raise RuntimeError('Rejected action cannot update world state')
        w.virtual_time=result['virtual_time_s']
        if path=='/enter':
            self.entered=True;self.deadline=time.monotonic()+result['remaining_real_duration_s']
        if q is not None:
            q=np.asarray(q,dtype=float)
            w.distance+=float(np.linalg.norm(q-w.position));w.position=q.copy()
            target=w.targets[channel]
            if path=='/measure':
                w.channel=channel;w.coverage.mark(channel,q)
                target.update_measurement(q,result,w.params)
            elif path=='/clear':target.update_clear(q,result,w.params)
        return result

    def event(self,phase,**extra):
        self.events.append(dict(phase=phase,command_index=len(self.world.commands),
                                position=self.world.position.tolist(),**extra))

    def scan_stop(self,q,heading,force_unknown=False,reason='scan'):
        w=self.world;q=np.asarray(q)
        channels=select_known_channels(w,q,heading)
        channels+=select_unknown_channels(w,q,force_unknown)
        channels=list(dict.fromkeys(channels))
        if not channels and np.linalg.norm(w.position-q)>1e-7 and not w.finished():
            # Moving has no independent command. Attach one useful measurement.
            eligible=[t for t in w.active()+w.unknown() if not t.measured_at(q)]
            if eligible:channels=[max(eligible,key=lambda t:t.radius if math.isfinite(t.radius) else 2000).channel]
        self.event('scan_selection',destination=q.tolist(),channels=channels,force_unknown=force_unknown)
        for channel in channels:
            target=w.targets[channel]
            if target.status=='cleared' or target.measured_at(q):continue
            self.execute('/measure',q,channel,reason)

    def service_target(self,channel):
        w=self.world;target=w.targets[channel]
        self.event('service_start',channel=channel,radius=target.radius)
        for _ in range(16):
            if target.status=='cleared':return
            path,q,reason=choose_clear_or_probe(w,target)
            if path=='/measure' and target.measured_at(q):
                # Should not occur for normal center contraction; do not loop on fixed error.
                raise ArithmeticError(f'Channel {channel}: repeated center measurement without geometric progress')
            self.event('local_decision',channel=channel,action=path,destination=q.tolist(),
                       radius=target.radius,hit_probability=target.hit_probability(q),reason=reason)
            self.execute(path,q,channel,reason)
        raise RuntimeError(f'Channel {channel}: service guard reached; not declared cleared')

    def cleanup_along_route(self,destination):
        w=self.world
        # Each service removes a distinct target; there are at most 16.
        for _ in range(20):
            route=plan_cleanup_route(w,destination)
            if not route:return
            self.event('cleanup_route',channels=route,destination=np.asarray(destination).tolist())
            self.service_target(route[0])
            delta=np.asarray(destination)-w.position
            heading=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>0 else w.dense_heading
            self.scan_stop(w.position,heading,False,'opportunistic_stop')
        raise RuntimeError('Too many target services; check source uniqueness')

    def advance(self,destination,phase,force_unknown=False):
        w=self.world
        self.cleanup_along_route(destination)
        if w.finished():return
        delta=destination-w.position
        heading=math.atan2(delta[1],delta[0])
        self.event(phase,destination=destination.tolist())
        self.scan_stop(destination,heading,force_unknown,phase)

    def run(self):
        started=time.perf_counter()
        try:
            self.execute('/enter')
            self.event('origin_scan')
            for channel in range(1,21):self.execute('/measure',np.zeros(2),channel,'origin_all_channels')
            if self.world.active():
                first=choose_initial_probe(self.world)
                self.advance(first,'dense_direction_probe')
            ring=choose_ring_route(self.world)
            for i,point in enumerate(ring):
                if self.world.finished():break
                # If every unknown channel is already certified, skip unused outer stops.
                if not self.world.unknown():break
                self.event('ring_schedule',index=i)
                self.advance(point,'outer_ring',True)
            while not self.world.finished():
                patch=choose_coverage_patch(self.world)
                if patch is not None:
                    self.advance(patch,'coverage_patch',True)
                elif self.world.active():
                    # Search is certified complete. Finish known targets in short-route batches.
                    route=plan_cleanup_route(self.world,self.world.position,force=True)
                    if not route:raise RuntimeError('Active target missing from final route')
                    self.event('final_cleanup_route',channels=route)
                    self.service_target(route[0])
                else:raise RuntimeError('Inconsistent completion state')
            self.event('coverage_closed')
        except Exception as exc:
            self.failure=f'{type(exc).__name__}: {exc}'
        finally:
            if self.entered:
                try:self.execute('/exit',reason='complete' if self.world.finished() and not self.failure else 'incomplete')
                except Exception as exc:
                    self.failure=self.failure or f'Exit failed: {exc}'
        w=self.world
        removed=sum(t.status=='cleared' for t in w.targets.values())
        return dict(complete=w.finished() and self.failure is None,cleared=removed,
                    virtual_time_s=w.virtual_time,average_time_s=w.virtual_time/removed if removed else None,
                    wall_time_s=time.perf_counter()-started,distance_m=w.distance,
                    measures=sum(x['path']=='/measure' for x in w.commands),
                    clears=sum(x['path']=='/clear' for x in w.commands),misses=sum(t.misses for t in w.targets.values()),
                    failure=self.failure,coverage=w.coverage.summary(),parameters=asdict(w.params))

    def save(self,directory,summary):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for name,value in [('summary.json',summary),('map.json',self.world.snapshot()),('decisions.json',self.events)]:
            (directory/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
        (directory/'commands.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in self.world.commands),encoding='utf8')


def load_parameters(path=None):
    return Parameters(**json.loads(Path(path).read_text(encoding='utf8'))).validate() if path else Parameters().validate()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['local','http'],default='local')
    parser.add_argument('--seed',type=int,default=20260911)
    parser.add_argument('--robot-id',default='local')
    parser.add_argument('--base-url',default='http://127.0.0.1:2027')
    parser.add_argument('--http-context',choices=['local','practice'],default='local',
                        help='No formal-test context is supported')
    parser.add_argument('--practice-confirmed',action='store_true',
                        help='Only after user starts Q3 PRACTICE in the official UI')
    parser.add_argument('--params')
    parser.add_argument('--output',default='question3/global_policy/outputs/demo')
    args=parser.parse_args();params=load_parameters(args.params)
    if args.mode=='http':
        from urllib.parse import urlsplit
        if args.http_context=='practice' and not args.practice_confirmed:
            parser.error('Practice requires explicit --practice-confirmed after checking the simulator UI; formal tests are forbidden')
        if args.http_context=='local' and urlsplit(args.base_url).port==2026:
            parser.error('Port 2026 is reserved here for explicit practice context; use the local simulator on 2027')
    simulator=None
    if args.mode=='local':
        from question3.local_sim.simulator import Simulator,Client
        simulator=Simulator(args.seed,robot_id=args.robot_id)
        client=Client(args.robot_id,simulator=simulator)
    else:client=HTTPClient(args.robot_id,args.base_url)
    planner=Planner(client,params);summary=planner.run()
    if simulator is not None:
        summary.update(evaluation='local_synthetic',seed=args.seed,true_total=len(simulator.sources),
                       true_cleared=sum(s.cleared for s in simulator.sources))
        if summary['complete'] and not all(s.cleared for s in simulator.sources):
            summary.update(complete=False,failure='Evaluator found missed sources despite completion claim')
        simulator.dump(Path(args.output)/'evaluator')
    planner.save(args.output,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not summary['complete']:raise SystemExit(1)


if __name__=='__main__':main()
