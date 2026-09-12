"""Channel coverage frontiers with monotonically closed angular regions."""
import math
import numpy as np
from ..decisions import choose_dense_heading
from question1.geometry import _clip,bearing_halfplanes


def possible_cells(target,coverage,polygon=None):
    """Conservative convex-polygon/square overlap (SAT), minus certified exclusions."""
    poly=target.polygon if polygon is None else polygon;centers=coverage.centers;half=coverage.half
    if poly is None:return np.ones(len(centers),dtype=bool)
    if not len(poly):return np.zeros(len(centers),dtype=bool)
    mask=np.all(centers+half>=poly.min(axis=0)-1e-7,axis=1)&np.all(centers-half<=poly.max(axis=0)+1e-7,axis=1)
    edges=np.roll(poly,-1,axis=0)-poly
    for edge in edges:
        normal=np.array([-edge[1],edge[0]])
        projections=poly@normal;mid=centers@normal;spread=half*np.abs(normal).sum()
        mask &= (mid+spread>=projections.min()-1e-7)&(mid-spread<=projections.max()+1e-7)
    for q,r,kind in target.exclusions:
        far=np.abs(centers-q)+half
        mask &= np.sum(far*far,axis=1)>r*r-1e-5
    return mask


class Sweep:
    def __init__(self,world):
        self.world=world;self.heading=choose_dense_heading(world)
        world.dense_heading=self.heading
        self.width=2*math.pi/world.params.sector_count
        angles=np.arctan2(world.coverage.centers[:,1],world.coverage.centers[:,0])
        def region(angle,sign):return np.floor(((sign*(angle-self.heading)+self.width/2)%(2*math.pi))/self.width).astype(int)
        targets=world.active()
        scores=[]
        for sign in (1,-1):
            score=sum(float(region(math.atan2(t.center[1],t.center[0]),sign)) for t in targets)
            scores.append(score)
        self.sign=(1,-1)[int(np.argmin(scores))]
        self.regions=region(angles,self.sign);self.index=0;self.closed=[]
        self.region_masks=[];self.region_planes=[]
        for i in range(world.params.sector_count):
            bearing=math.degrees(self.heading+self.sign*i*self.width)
            a,b=bearing_halfplanes([[0.,0.]],[bearing],math.degrees(self.width/2))
            self.region_planes.append((a,b))
            # Include every square that may intersect the wedge, including BOTH
            # neighboring regions at an angular boundary. Never center-only coverage.
            minimum=world.coverage.centers@a.T-world.coverage.half*np.abs(a).sum(axis=1)
            self.region_masks.append(np.all(minimum<=b+1e-7,axis=1))
        axis=np.arange(-1800,1800+1,world.params.frontier_spacing)
        xx,yy=np.meshgrid(axis,axis)
        grid=np.column_stack((xx.ravel(),yy.ravel()))
        self.grid=grid[np.linalg.norm(grid,axis=1)<=1800]

    def outstanding(self):
        return [t for t in self.world.active() if self.target_in_region(t,self.index)]

    def target_in_region(self,target,index):
        poly=target.polygon.copy();a,b=self.region_planes[index]
        for normal,offset in zip(a,b):poly=_clip(poly,normal,offset,1e-8)
        if not len(poly):return False
        exclusions=list(target.exclusions)
        exclusions += [(o.position,5.,'not_near') for o in target.observations if o.result=='direction']
        # Convex disk contains a convex polygon iff it contains every vertex.
        # This removes shared-origin degeneracies without deleting a real source.
        for q,r,kind in exclusions:
            if np.max(np.linalg.norm(poly-q,axis=1))<=r-1e-7:return False
        return bool(np.any(possible_cells(target,self.world.coverage,poly)&self.region_masks[index]))

    def holes(self):
        w=self.world;channels=[t.channel for t in w.unknown()]
        if not channels:return channels,np.zeros((0,len(self.regions)),dtype=bool)
        return channels,(~w.coverage.covered[np.array(channels)-1])&self.region_masks[self.index]

    def can_close(self):
        return not self.holes()[1].any() and not self.outstanding()

    def next_point(self):
        w=self.world;channels,holes=self.holes()
        if not holes.any():return None
        needed=holes.sum(axis=0);missing=w.coverage.centers[needed>0]
        # A selected cell center always makes progress, including thin residual gaps.
        samples=missing[np.linspace(0,len(missing)-1,min(80,len(missing)),dtype=int)]
        mean=np.average(w.coverage.centers,axis=0,weights=needed)
        candidates=np.vstack((self.grid,samples,mean*.65,mean*.85,mean,w.position))
        angle=np.arctan2(candidates[:,1],candidates[:,0])
        offset=(self.sign*(angle-self.heading)-self.index*self.width+math.pi)%(2*math.pi)-math.pi
        candidates=candidates[(np.abs(offset)<=self.width*.75)&(np.linalg.norm(candidates,axis=1)<=1800)]
        # Boundary cell centers may lie just outside the arena. Projection keeps commands valid.
        fallback=samples*np.minimum(1,1799.99/np.maximum(1,np.linalg.norm(samples,axis=1)))[:,None]
        candidates=np.vstack((candidates,fallback))
        best=None
        for q in candidates:
            mask=w.coverage.mask_at(q)
            gains=np.sum(holes&mask,axis=1);count=np.count_nonzero(gains)
            if not count:continue
            gain=float(gains.sum()/holes.sum())
            closes=int(np.all(holes[:,~mask].sum(axis=1)==0))
            travel=np.linalg.norm(q-w.position)/5
            score=(gain+w.params.frontier_closure_bonus*closes)/(travel+6*count+1)
            if best is None or score>best[0]:best=(score,q.copy(),[c for c,g in zip(channels,gains) if g>0])
        return best


def nearby_targets(world):
    # Radius of the uncertainty polygon is deliberately NOT an eligibility threshold.
    return [t for t in world.active() if np.linalg.norm(t.center-world.position)<=world.params.nearby_distance]


def short_service_route(world,targets,destination):
    """Cheapest insertion then 2-opt for a local open route; execute just its first visit."""
    if not targets:return []
    points={t.channel:t.center for t in targets};route=[];remaining=list(points)
    while remaining:
        choices=[]
        for channel in remaining:
            for i in range(len(route)+1):
                a=world.position if i==0 else points[route[i-1]]
                b=destination if i==len(route) else points[route[i]]
                extra=np.linalg.norm(a-points[channel])+np.linalg.norm(points[channel]-b)-np.linalg.norm(a-b)
                choices.append((float(extra),channel,i))
        _,channel,i=min(choices);route.insert(i,channel);remaining.remove(channel)
    def length(order):
        p=np.vstack((world.position,[points[c] for c in order],destination))
        return np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
    changed=True
    while changed:
        changed=False
        for i in range(len(route)):
            for j in range(i+1,len(route)):
                trial=route[:i]+route[i:j+1][::-1]+route[j+1:]
                if length(trial)<length(route)-1e-7:route=trial;changed=True
    return route
