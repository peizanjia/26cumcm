"""Dynamic Q4 sensing stops jointly evaluated against all observable targets.

Only real stationary commands update the beliefs. The route, radius potential
and coverage credits are labelled approximations used to select actual stops;
continuous per-channel proofs alone certify an unknown channel absent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from question4.full_mission.planner import Parameters as BaseParameters, Planner as BasePlanner, open_order
from question4.full_mission.service import _optical_candidate
from .coverage import DirectionalCoverage
from .information import measurement_value, receiver_hit_weights, service_potential, known_scan_candidates, coupled_points


@dataclass
class Parameters(BaseParameters):
    strategy: str = 'dynamic'
    cross_source_weight: float = 1.0
    unknown_credit_weight: float = .65
    unknown_credit_cap_s: float = 150.0
    forecast_weight: float = 1.0
    scan_min_net_s: float = 2.0
    unknown_scan_min_net_s: float = 3.0
    revisit_penalty_s: float = 20.0
    revisit_radius_m: float = 110.0
    candidate_limit: int = 20
    service_candidates_per_source: int = 3
    information_samples: int = 3
    frontier_candidates: int = 14
    shared_candidates: int = 8
    waypoint_candidates: int = 8
    refine_candidates: bool = True
    refine_step_m: float = 70.0
    forecast_passes: int = 1
    coverage_spacing: float = 225.0
    coverage_directions: int = 24
    max_dynamic_stops: int = 500


class Planner(BasePlanner):
    def __init__(self, command, params=None, record=False):
        super().__init__(command, params or Parameters(), record)
        # The inherited object supplies only response accounting and Target
        # updates. Its fixed search station list is never used by this policy.
        self.stations = []
        self.unknown_map = DirectionalCoverage(spacing=self.params.coverage_spacing,
                                                directions=self.params.coverage_directions)
        self.visits = [np.zeros(2)]
        self.search_plan = []
        self.search_budget_s = 6000.
        self.dynamic_stops = 0
        self.counters.update(dynamic_search_moves=0, dynamic_service_moves=0,
            shared_point_moves=0, intermediate_stops=0, refined_stops=0,
            stop_reassessments=0, unknown_skipped=0, known_skipped=0,
            predicted_cross_source_choices=0, actual_cross_source_measurements=0,
            coverage_required_scans=0, position_candidates_evaluated=0)

    def search_needed(self):
        found = sum(t.status in ('active', 'cleared') for t in self.targets.values())
        return found < 16 and bool(self.unknown())

    def finished(self):
        return sum(t.status == 'cleared' for t in self.targets.values()) >= 16 or (
            not self.active() and not self.unknown())

    def _refresh_absence(self):
        for target in self.unknown():
            if self.unknown_map.channel_certified(target.channel):
                target.status = 'absent'
        # Used by the common replay/accounting schema, not by the proof itself.
        self.coverage.certified = not bool(self.unknown())

    def snapshots(self):
        rows = super().snapshots()
        for row in rows:
            if row['status'] == 'absent':
                row.update(polygon=None, center=None, radius=None, radius_m=None)
            if row['status'] == 'unknown':
                row.update(radius_m=None)
        return rows

    def execute(self, action, phase, decision=None):
        before = self.position.copy()
        if action.get('reason') == 'optical_finite_cover':
            committed = _optical_candidate(self.targets[action['channel']], self.position)
            if committed is None or np.linalg.norm(np.asarray(committed['position'])-action['position']) > .03:
                raise ArithmeticError('Speculative optical enumeration changed before execution')
        super().execute(action, phase, decision)
        if action['kind'] == 'measure':
            self.unknown_map.observe(action['channel'], self.position,
                                     self.commands[-1]['response']['measure_result'])
        self._refresh_absence()
        if np.linalg.norm(self.position-before) > 1.:
            self.visits.append(self.position.copy())
        channels = [t.channel for t in self.unknown()]
        mass = float(np.mean([self.unknown_map.residual_mass(c) for c in channels])) if channels else 0.
        update = dict(coverage_certified=self.coverage.certified, coverage_state_mass=mass)
        self.commands[-1].update(update)
        if self.record:
            self.frames[-1].update(update, targets=self.snapshots())
            if decision is not None:
                self.frames[-1]['unknown_domains'] = [self.unknown_map.snapshot(c) for c in channels]

    def _covered_at(self, target, q):
        return any(o['result'] in ('direction', 'near', 'no_signal') and
                   np.linalg.norm(np.asarray(o['position'])-q) < .03 for o in target.observations)

    def _radio_eligible(self, target):
        """One recovery gate shared by every global and stationary scan path.

        Exhausted radio service becomes optical-only even before its first
        optical action wins the global ranking. Otherwise shared/refined
        candidates can bypass the finite single-target recovery policy.
        """
        if target.status!='active' or target.optical_started or target.radius<=19.8:
            return False
        limit=int(self.params.max_radio_steps)
        failures=sum(o['result'] in ('no_target_in_range','failed','failure') for o in target.observations)
        return target.stagnant_steps<limit and target.radio_steps<limit+4 and failures<8

    def _unknown_values(self, q, required=()):
        if not self.search_needed():
            return []
        channels = [t.channel for t in self.unknown()]
        mass = sum(self.unknown_map.residual_mass(c) for c in channels)
        values = []
        for c in channels:
            if self._covered_at(self.targets[c], q):
                continue
            gain = self.unknown_map.gain(c, q)
            # A coverage-effort allocation, not a posterior existence probability.
            gross = self.params.unknown_credit_weight*self.search_budget_s*gain/max(mass, 1e-9)
            fee = 5.+float(c != self.channel)
            values.append(dict(channel=c, type='unknown', gain=gain, gross_saving_s=gross,
                cost_s=fee, net_saving_s=gross-fee, required=c in required,
                model='residual_direction_mass_share_of_forecast_search_effort'))
        total=sum(r['gross_saving_s'] for r in values)
        scale=min(1.,self.params.unknown_credit_cap_s/max(total,1e-9))
        for row in values:
            row['gross_saving_s']*=scale
            row['net_saving_s']=row['gross_saving_s']-row['cost_s']
            row['coverage_credit_scale']=scale
        return values

    def _known_values(self, q):
        values = []
        for target in self.active():
            if not self._radio_eligible(target):
                continue
            row = measurement_value(target, q, current_channel=self.channel,
                                    samples=self.params.information_samples)
            values.append(dict(row, type='known', required=False))
        return values

    def _reference_plan(self):
        channels = [t.channel for t in self.unknown()] if self.search_needed() else []
        self.search_plan = self.unknown_map.completion_plan(channels, self.position) if channels else []
        if self.search_plan:
            order = open_order(self.position, self.search_plan, passes=2)
            self.search_plan = [self.search_plan[i] for i in order]
            points = np.vstack((self.position, self.search_plan))
            self.search_budget_s = float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()/5
                                        + 6*len(channels)*len(self.search_plan))
        else:
            self.search_budget_s = 0.
        self.route_hint = [list(q) for q in self.search_plan]

    def _raw_candidates(self):
        raw = []
        active = self.active()
        channels = [t.channel for t in self.unknown()] if self.search_needed() else []
        if channels:
            for row in self.unknown_map.candidate_points(self.position, channels,
                                                        limit=self.params.frontier_candidates):
                raw.append(dict(position=row['position'], reason='dynamic_direction_frontier',
                                owner=None, kind='measure', required=[], generation=row))
            # These coordinates are generated from present coverage geometry;
            # they are future candidates, never an executed fixed ring order.
            for q in self.search_plan[:6]:
                raw.append(dict(position=list(q), reason='dynamic_coverage_completion', owner=None,
                    kind='measure', required=self.unknown_map.required_channels(q, channels)))
        hint = np.asarray(self.search_plan[0]) if self.search_plan else None
        for target in active:
            rows = known_scan_candidates(target, self.position, asdict(self.params), hint)
            selected = rows[:self.params.service_candidates_per_source]
            # Keep certified and finite optical clearance even if its immediate
            # movement cost is unattractive to the localization proxy.
            for row in selected:
                raw.append(dict(position=row['position'], reason=row['reason'], owner=target.channel,
                                kind=row['kind'], required=[], service_action=row))
        regular = [t for t in active if self._radio_eligible(t)]
        for q in coupled_points(regular, self.position, [r['position'] for r in raw],
                                limit=self.params.shared_candidates):
            raw.append(dict(position=np.asarray(q).tolist(), reason='multiple_source_shared_point',
                            owner=None, kind='measure', required=[]))
        # A segment is never treated as a measurement. Its intermediate points
        # compete as real stop-and-measure actions, with their operation costs.
        destinations = sorted(raw, key=lambda r:np.linalg.norm(np.asarray(r['position'])-self.position))
        added=0
        for row in destinations:
            delta=np.asarray(row['position'])-self.position
            length=np.linalg.norm(delta)
            if length < 450.:
                continue
            for fraction in (.45,.7):
                q=self.position+fraction*delta
                raw.append(dict(position=q.tolist(), reason='valuable_intermediate_stop',
                                owner=None, kind='measure', required=[]))
                added+=1
                if added>=self.params.waypoint_candidates: break
            if added>=self.params.waypoint_candidates: break
        raw.append(dict(position=self.position.tolist(),reason='current_stop_reassessment',owner=None,
                        kind='measure',required=[]))
        unique=[];seen=set()
        for r in raw:
            # Future certificate supports must keep their exact coordinates.
            # Moving a support by rounding can invalidate its geometric proof.
            r['position']=(np.asarray(r['position'],dtype=float) if r['required']
                           else np.round(r['position'],2)).tolist()
            key=(*r['position'],r['kind'],r['owner'],bool(r['required']))
            if key in seen: continue
            if not np.isfinite(r['position']).all(): continue
            seen.add(key);unique.append(r)
        return unique

    def _quick_score(self, row):
        q=np.asarray(row['position']); move=np.linalg.norm(q-self.position)/5
        benefit=sum(max(0.,r['net_saving_s']) for r in self._unknown_values(q,row['required']))
        known_benefits=[]
        for target in self.active():
            if not self._radio_eligible(target) or self._covered_at(target,q): continue
            if target.radius<=19.8: continue
            chance=target.hit_probability(q)
            positives=[o for o in target.observations if o['result'] in ('direction','near')]
            a=np.asarray(positives[-1]['position'])-target.center
            b=q-target.center
            cross=abs(a[0]*b[1]-a[1]*b[0])/max(1.,np.linalg.norm(a)*np.linalg.norm(b))
            credit=chance*service_potential(target)*(.2+.8*cross)
            known_benefits.append((target.channel,credit))
        reference=row['owner'] if row['owner'] is not None else max(known_benefits,key=lambda x:x[1],default=(None,0))[0]
        benefit+=sum(v*(1. if c==reference else self.params.cross_source_weight) for c,v in known_benefits)
        if row['kind']=='clear':
            target=self.targets[row['owner']]
            benefit+=target.clear_probability(q)*(service_potential(target)+100.)
        return move-benefit

    def _revisit_penalty(self,q,novelty):
        previous=self.visits[:-1]
        if not previous or np.linalg.norm(q-self.position)<1.:
            return 0.
        closest=min(np.linalg.norm(q-p) for p in previous)
        return self.params.revisit_penalty_s*math.exp(-.5*(closest/self.params.revisit_radius_m)**2)*max(0.,1.-min(1.,novelty*10.))

    def _forecast(self,q,known_values,selected_channels,clearing=None):
        predicted={r['channel']:r['expected_center'] for r in known_values if r['channel'] in selected_channels}
        known=[np.asarray(predicted.get(t.channel,t.center)) for t in self.active() if t.channel != clearing]
        plan=[np.asarray(p) for p in self.search_plan]
        channels=[t.channel for t in self.unknown()] if self.search_needed() else []
        if plan and channels and selected_channels:
            plan=self.unknown_map.forecast_plan(channels,plan,q,selected_channels)
        nodes=known+plan
        if not nodes: return 0.
        order=open_order(q,nodes,passes=self.params.forecast_passes)
        points=np.vstack((q,[nodes[i] for i in order]))
        return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()/5)

    def _evaluate(self,row):
        q=np.asarray(row['position'])
        if row['kind']=='measure' and row['owner'] is not None and (
            not self._radio_eligible(self.targets[row['owner']]) or self._covered_at(self.targets[row['owner']],q)):
            return None
        unknown=self._unknown_values(q,row['required'])
        known=self._known_values(q)
        owner=row['owner']
        reference=owner if owner is not None else max(known,key=lambda r:r['gross_saving_s'],default={'channel':None})['channel']
        chosen=[]
        for r in unknown+known:
            weight=1. if r['type']=='unknown' or reference==r['channel'] else self.params.cross_source_weight
            threshold = (self.params.unknown_scan_min_net_s
                         if r['type'] == 'unknown' else self.params.scan_min_net_s)
            if r.get('required') or weight*r['gross_saving_s']-r['cost_s']>=threshold:
                chosen.append(dict(r,credit_weight=weight))
        if row['kind']=='measure' and owner is not None and all(r['channel']!=owner for r in chosen):
            r=next((r for r in known if r['channel']==owner),None)
            if r is not None:chosen.append(dict(r,credit_weight=1.,forced_service=True))
        if row['kind']=='measure' and not chosen:
            return None
        if row['kind']=='clear':
            chosen=[r for r in chosen if r['channel']!=owner]
            primary=owner
        else:
            primary=owner if owner is not None else max(chosen,key=lambda r:(r.get('required',False),r['net_saving_s']))['channel']
        channels=[r['channel'] for r in chosen]
        # Put the moving radio action first, then keep the receiver if possible.
        scan_order=([primary] if row['kind']=='measure' else [])+[c for c in channels if c!=primary]
        receiver=self.channel; scan_cost=0.
        for c in scan_order:
            scan_cost+=5.+float(c!=receiver); receiver=c
        move=float(np.linalg.norm(q-self.position))/5
        base_potential=sum(service_potential(t) for t in self.active())
        known_credit=sum(r['credit_weight']*r['gross_saving_s'] for r in chosen if r['type']=='known')
        unknown_credit=sum(r['gross_saving_s'] for r in chosen if r['type']=='unknown')
        probability=None; clear_cost=0.; clearing=None
        if row['kind']=='clear':
            target=self.targets[owner]
            probability=1. if target.radius<=19.8 else target.clear_probability(q)
            clear_cost=3.+2.*probability
            clearing=owner if probability>=.999 else None
            known_credit+=probability*service_potential(target)
        forecast=self._forecast(q,known,channels,clearing)
        if row['kind']=='clear' and 0.<probability<.999:
            forecast=(1.-probability)*forecast+probability*self._forecast(q,known,channels,owner)
        novelty=sum(r['gain'] for r in unknown)
        revisit=self._revisit_penalty(q,novelty)
        remaining_clears=5.*(len(self.active())-(probability or 0.))
        score=move+scan_cost+clear_cost+self.params.forecast_weight*forecast+remaining_clears+base_potential-known_credit-unknown_credit+revisit
        if row['reason']=='optical_finite_cover':
            score-=8.  # One distinct finite-cover query makes persistent progress.
        cross=[r for r in chosen if r['type']=='known' and r['channel']!=primary]
        return dict(kind=row['kind'],channel=primary,position=q.tolist(),reason=row['reason'],
            score_s=float(score),selection_s=float(score),move_s=move,scan_cost_s=scan_cost,
            expected_clear_cost_s=clear_cost,forecast_route_s=forecast,
            localization_potential_s=base_potential,known_credit_s=known_credit,
            unknown_credit_s=unknown_credit,revisit_penalty_s=revisit,
            hit_probability=probability if probability is not None else next((r.get('hit_probability') for r in chosen if r['channel']==primary),None),
            predicted_scans=chosen,predicted_scan_channels=scan_order,required_channels=list(row['required']),
            cross_source_channels=[r['channel'] for r in cross],owner=owner,
            cross_source_credit_s=sum(r['credit_weight']*r['gross_saving_s'] for r in cross),
            generation_reason=row['reason'],score_model='dynamic_route_plus_localization_potential_minus_coverage_effort_credit')

    def choose(self):
        self._reference_plan()
        raw=self._raw_candidates()
        raw.sort(key=self._quick_score)
        selected=raw[:self.params.candidate_limit]
        for target in self.active():
            row=next((r for r in raw if r['owner']==target.channel),None)
            if row is not None and row not in selected:selected.append(row)
        required=next((r for r in raw if r['required']),None)
        if required is not None and required not in selected:selected.append(required)
        values=[v for row in selected if (v:=self._evaluate(row)) is not None]
        if not values:raise RuntimeError('No observable dynamic action makes progress')
        winner=min(values,key=lambda r:r['score_s'])
        if self.params.refine_candidates and winner['kind']=='measure' and not winner['required_channels']:
            # A small continuous-coordinate pattern search evaluates the SAME
            # multi-target objective, so other sources can move the stop itself.
            origin=np.asarray(winner['position']); step=self.params.refine_step_m
            for delta in ([step,0],[-step,0],[0,step],[0,-step]):
                row=dict(position=np.round(origin+delta,2).tolist(),reason='multisource_refined_point',
                         owner=winner['owner'],kind='measure',required=[])
                if (v:=self._evaluate(row)) is not None:values.append(v)
            winner=min(values,key=lambda r:r['score_s'])
        self.counters['position_candidates_evaluated']+=len(values)
        self.counters['predicted_cross_source_choices']+=bool(winner['cross_source_channels'])
        reason=winner['reason']
        self.counters['shared_point_moves']+=int(reason=='multiple_source_shared_point')
        self.counters['intermediate_stops']+=int(reason=='valuable_intermediate_stop')
        self.counters['refined_stops']+=int(reason=='multisource_refined_point')
        phase='service' if self.targets[winner['channel']].status=='active' or winner['kind']=='clear' else 'search'
        self.counters['dynamic_service_moves' if phase=='service' else 'dynamic_search_moves']+=1
        decision=dict(candidates=values,selected=winner,score_model=winner['score_model'],
                      forecast_search_stations=self.search_plan,search_budget_s=self.search_budget_s,
                      coverage_plan_diagnostics=dict(self.unknown_map.last_plan_diagnostics),
                      unknown_channels=[t.channel for t in self.unknown()])
        return winner,phase,decision

    def assess_stop(self,required=(),first_channel=None):
        mandatory=set(required)
        for _ in range(40):
            self.counters['stop_reassessments']+=1
            # Other channels can be optically cleared here with a conservative
            # circle proof, even if the stop was selected for a different task.
            clearable=[t for t in self.active() if np.linalg.norm(t.center-self.position)+t.radius<=19.8]
            if clearable:
                target=min(clearable,key=lambda t:t.channel)
                self.execute(dict(kind='clear',channel=target.channel,position=self.position.tolist(),
                                  reason='shared_stop_certified_clear'),'service')
                continue
            rows=self._unknown_values(self.position,mandatory)+self._known_values(self.position)
            accepted=[]
            for row in rows:
                threshold = (self.params.unknown_scan_min_net_s
                             if row['type'] == 'unknown' else self.params.scan_min_net_s)
                if row.get('required') or row['net_saving_s']>=threshold:
                    accepted.append(row)
            if not accepted:
                self.counters['unknown_skipped']+=sum(r['type']=='unknown' for r in rows)
                self.counters['known_skipped']+=sum(r['type']=='known' for r in rows)
                break
            row=max(accepted,key=lambda r:(r.get('required',False),r['net_saving_s']))
            phase='side_known' if row['type']=='known' else ('search' if row.get('required') else 'side_search')
            reason='reassessed_known_value' if row['type']=='known' else ('required_continuous_coverage_scan' if row.get('required') else 'reassessed_unknown_value')
            self.execute(dict(kind='measure',channel=row['channel'],position=self.position.tolist(),reason=reason),phase,
                         dict(stop_values=rows,selected_channels=[row['channel']],reassessed_after_feedback=True))
            mandatory.discard(row['channel'])
            self.counters['coverage_required_scans']+=int(row.get('required',False))
            self.counters['side_known_measures']+=int(row['type']=='known')
            self.counters['actual_cross_source_measurements']+=int(row['type']=='known' and row['channel']!=first_channel)
        else:
            raise RuntimeError('Stationary assessment failed to terminate')

    def run(self):
        started=time.perf_counter()
        for c in range(1,21):
            self.execute(dict(kind='measure',channel=c,position=[0.,0.],reason='origin_all_channels'),'origin')
        while not self.finished():
            if self.dynamic_stops>=self.params.max_dynamic_stops:
                raise RuntimeError('Dynamic movement budget exceeded; no shortened success reported')
            self.counters['replans']+=1
            action,phase,decision=self.choose()
            self.execute(action,phase,decision)
            self.dynamic_stops+=1
            self.assess_stop(action['required_channels'],first_channel=action['channel'])
        return dict(time_s=self.time_s,costs=self.costs,counters=self.counters,completed=True,
            cleared=sum(t.status=='cleared' for t in self.targets.values()),source_times=self.target_times,
            coverage_certified=self.coverage.certified,planning_wall_s=time.perf_counter()-started,
            command_count=len(self.commands),parameters=asdict(self.params),dynamic_stop_count=self.dynamic_stops)
