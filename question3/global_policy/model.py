"""Public observations, conservative geometry, and channel-wise coverage."""
from dataclasses import asdict, dataclass, field
import math
import numpy as np
from question1.geometry import _clip, bearing_halfplanes, polygon_area
from question1.enclosing_circle import minimum_enclosing_circle

ARENA_R, RECEIVE_MIN, RECEIVE_MAX = 1800., 1000., 1500.
CLEAR_R, NEAR_R, SPEED, ERROR_DEG = 20., 5., 5., 1.005


@dataclass
class Parameters:
    # These are decisions; physical constants above are never tuned.
    initial_step: float = 550.
    initial_lateral: float = 70.
    density_sigma_deg: float = 30.
    heading_candidates: int = 72
    ring_radius: float = 1400.
    ring_nodes: int = 6
    max_known_scans: int = 5
    known_score_min: float = .025
    rear_scan_weight: float = .35
    unknown_gain_min: float = .08
    service_radius: float = 60.
    probe_service_radius: float = 300.
    probe_detour_max: float = 100.
    trial_probability_min: float = .4
    max_speculative_clears: int = 2
    max_detour: float = 220.
    route_detour_budget: float = 420.
    route_batch: int = 4
    information_weight: float = 1.
    proximity_weight: float = 1.
    cleanup_weight: float = 1.
    coverage_weight: float = 1.
    count_bound_stop: bool = False
    samples: int = 256
    circle_sides: int = 32
    coverage_cell: float = 60.
    max_commands: int = 4000
    model_seed: int = 12345

    def validate(self):
        bounds = {
            'initial_step':(50,1000),'initial_lateral':(0,250),'density_sigma_deg':(5,120),
            'heading_candidates':(12,360),'ring_radius':(1000,1700),'ring_nodes':(4,12),
            'max_known_scans':(1,20),'known_score_min':(0,2),'rear_scan_weight':(0,1),
            'unknown_gain_min':(0,1),'service_radius':(20,200),'probe_service_radius':(60,800),
            'probe_detour_max':(0,500),'trial_probability_min':(0,1),
            'max_speculative_clears':(0,5),'max_detour':(0,1000),'route_detour_budget':(0,3000),
            'route_batch':(1,16),'information_weight':(0,10),'proximity_weight':(0,10),
            'cleanup_weight':(.01,10),'coverage_weight':(.01,10),'samples':(32,4096),
            'circle_sides':(16,64),'coverage_cell':(20,150),'max_commands':(200,20000),
            'model_seed':(0,2147483647)}
        for name,(low,high) in bounds.items():
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{name} must be in [{low}, {high}]')
        for name in ('heading_candidates','ring_nodes','max_known_scans','max_speculative_clears','route_batch','samples','circle_sides','max_commands','model_seed'):
            if not isinstance(getattr(self,name),int): raise ValueError(f'{name} must be an integer')
        if not isinstance(self.count_bound_stop,bool): raise ValueError('count_bound_stop must be bool')
        return self


def clip_circle_outer(poly, center, radius, sides):
    """Tangent halfplanes circumscribe the true disk; never cut away a true target."""
    for angle in np.linspace(0,2*math.pi,sides,endpoint=False):
        normal=np.array([math.cos(angle),math.sin(angle)])
        poly=_clip(poly,normal,float(normal@center+radius),1e-8)
    return poly


def initial_region(sides=32):
    box=np.array([[-ARENA_R,-ARENA_R],[ARENA_R,-ARENA_R],[ARENA_R,ARENA_R],[-ARENA_R,ARENA_R]])
    return clip_circle_outer(box,np.zeros(2),ARENA_R,sides)


def sample_polygon(poly, count, rng):
    """Uniform area samples of a convex polygon, without thin-wedge rejection."""
    if len(poly)<3 or polygon_area(poly)<1e-9:
        return np.repeat(poly.mean(axis=0,keepdims=True),count,axis=0)
    u=poly[1:-1]-poly[0];v=poly[2:]-poly[0]
    areas=np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])
    if areas.sum()<=1e-12:return np.repeat(poly.mean(axis=0,keepdims=True),count,axis=0)
    ids=rng.choice(len(areas),size=count,p=areas/areas.sum())
    a=np.sqrt(rng.random(count));b=rng.random(count)
    return poly[0]+a[:,None]*((1-b[:,None])*u[ids]+b[:,None]*v[ids])


@dataclass
class Observation:
    position: np.ndarray
    result: str
    bearing: float | None = None


@dataclass
class Target:
    channel: int
    status: str = 'unknown'
    polygon: np.ndarray | None = None
    observations: list = field(default_factory=list)
    exclusions: list = field(default_factory=list)
    center: np.ndarray | None = None
    radius: float = math.inf
    particles: np.ndarray | None = None
    weights: np.ndarray | None = None
    misses: int = 0
    particle_resets: int = 0

    def measured_at(self, q):
        return any(np.linalg.norm(o.position-q)<1e-7 for o in self.observations)

    def update_measurement(self,q,response,params):
        kind=response['measure_result'];q=np.asarray(q,dtype=float)
        if self.status=='cleared': return
        if self.measured_at(q): return  # Fixed same-location error, no independent evidence.
        obs=Observation(q.copy(),kind,response.get('svd_deg'))
        self.observations.append(obs)
        if kind=='no_signal':
            self.exclusions.append((q.copy(),RECEIVE_MIN,'no_signal'))
            if self.polygon is not None: self.refresh_particles(params)
            return
        self.status='active'
        poly=initial_region(params.circle_sides) if self.polygon is None else self.polygon.copy()
        if kind=='near':
            poly=clip_circle_outer(poly,q,NEAR_R,params.circle_sides)
        else:
            a,b=bearing_halfplanes([q],[obs.bearing],ERROR_DEG)
            for normal,offset in zip(a,b):poly=_clip(poly,normal,offset,1e-8)
            poly=clip_circle_outer(poly,q,RECEIVE_MAX,params.circle_sides)
        if not len(poly): raise ArithmeticError(f'Channel {self.channel}: empty conservative geometry')
        self.polygon=poly
        circle=minimum_enclosing_circle(poly)
        self.center,self.radius=circle.center,float(circle.radius)
        self.refresh_particles(params)

    def update_clear(self,q,response,params):
        if response['clear_result']=='success': self.status='cleared'
        else:
            self.misses+=1
            self.exclusions.append((np.asarray(q).copy(),CLEAR_R,'clear_miss'))
            # Keep convex outer polygon; removing disks would be nonconvex.
            self.refresh_particles(params)

    def refresh_particles(self,params):
        if self.polygon is None:return
        rng=np.random.default_rng(params.model_seed+1009*self.channel+31*len(self.observations)+self.misses)
        points=sample_polygon(self.polygon,params.samples*4,rng)
        valid=np.linalg.norm(points,axis=1)<=ARENA_R+1e-8
        lower=np.full(len(points),RECEIVE_MIN);upper=np.full(len(points),RECEIVE_MAX)
        for o in self.observations:
            d=np.linalg.norm(points-o.position,axis=1)
            if o.result=='no_signal': upper=np.minimum(upper,d)
            else:
                lower=np.maximum(lower,d)
                valid &= d<=NEAR_R+1e-8 if o.result=='near' else d>NEAR_R
        for q,r,kind in self.exclusions:
            if kind=='clear_miss':valid &= np.linalg.norm(points-q,axis=1)>r
        weights=np.maximum(upper-lower,0)*valid
        good=weights>0
        if not good.any():
            # Probability approximation failed; do not allow speculative clears.
            self.particles,self.weights=None,None
            self.particle_resets+=1
            return
        ids=np.flatnonzero(good)
        if len(ids)>params.samples:ids=rng.choice(ids,params.samples,replace=False)
        self.particles=points[ids];self.weights=weights[ids]/weights[ids].sum()

    def hit_probability(self,q):
        if self.polygon is not None and np.max(np.linalg.norm(self.polygon-q,axis=1))<=CLEAR_R:
            return 1.
        if self.particles is None:return 0.
        return float(self.weights[np.linalg.norm(self.particles-q,axis=1)<=CLEAR_R].sum())

    def robust_clear_point(self,current):
        if self.radius>CLEAR_R:raise ValueError('No certified 20m circle')
        delta=np.asarray(current)-self.center;d=np.linalg.norm(delta)
        slack=max(0,CLEAR_R-self.radius-1e-6)
        return self.center+delta*(min(1,slack/d) if d else 0.)

    def snapshot(self):
        return dict(channel=self.channel,status=self.status,
                    polygon=self.polygon.tolist() if self.polygon is not None else None,
                    center=self.center.tolist() if self.center is not None else None,
                    radius=self.radius if math.isfinite(self.radius) else None,misses=self.misses,
                    observations=[dict(position=o.position.tolist(),result=o.result,bearing=o.bearing) for o in self.observations],
                    exclusions=[dict(center=q.tolist(),radius=r,reason=k) for q,r,k in self.exclusions],
                    particle_resets=self.particle_resets)


class Coverage:
    """Each bit certifies a WHOLE square cell, not just its sample center."""
    def __init__(self,cell=60.):
        self.cell=cell;self.half=cell/2
        # Cover the disk even if the user selects a non-divisor of its diameter.
        axis=np.arange(-ARENA_R,ARENA_R+cell,cell)+cell/2
        x,y=np.meshgrid(axis,axis)
        points=np.column_stack((x.ravel(),y.ravel()))
        closest=np.maximum(np.abs(points)-self.half,0)
        self.centers=points[np.linalg.norm(closest,axis=1)<=ARENA_R+1e-9]
        self.covered=np.zeros((20,len(self.centers)),dtype=bool)

    def mask_at(self,q):
        # Exact farthest-corner distance of an axis-aligned square to q.
        far=np.abs(self.centers-np.asarray(q))+self.half
        return np.sum(far*far,axis=1)<=RECEIVE_MIN**2-1e-5

    def mark(self,channel,q):
        self.covered[channel-1] |= self.mask_at(q)

    def complete(self,channel):return bool(self.covered[channel-1].all())

    def gain(self,channel,q):
        return float(np.mean(self.mask_at(q)&~self.covered[channel-1]))

    def summary(self):
        return dict(cell_size=self.cell,cell_count=len(self.centers),
                    fraction_by_channel={str(c):float(self.covered[c-1].mean()) for c in range(1,21)})


class World:
    def __init__(self,params):
        self.params=params.validate();self.position=np.zeros(2);self.channel=1
        self.targets={c:Target(c) for c in range(1,21)}
        self.coverage=Coverage(params.coverage_cell)
        self.virtual_time=0.;self.commands=[];self.decisions=[]
        self.distance=0.;self.dense_heading=0.

    def active(self):return [t for t in self.targets.values() if t.status=='active']
    def unknown(self):return [t for t in self.targets.values() if t.status=='unknown' and not self.coverage.complete(t.channel)]

    def finished(self):
        if self.params.count_bound_stop and sum(t.status=='cleared' for t in self.targets.values())==16:return True
        return not self.active() and not self.unknown()

    def snapshot(self):
        targets=[]
        for t in self.targets.values():
            item=t.snapshot()
            item['coverage_complete']=self.coverage.complete(t.channel)
            item['absence_certified']=t.status=='unknown' and item['coverage_complete']
            if item['absence_certified']:item['status']='absent'
            targets.append(item)
        return dict(parameters=asdict(self.params),position=self.position.tolist(),virtual_time_s=self.virtual_time,
                    complete=self.finished(),targets=targets,coverage=self.coverage.summary())
