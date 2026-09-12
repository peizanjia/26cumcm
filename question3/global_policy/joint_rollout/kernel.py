"""Compiled conservative geometry and sampled full base-policy continuations.

Latent source coordinates generate outcomes only; decisions use the updated
polygon. Negative disks are convexified conservatively, never truth-clipped.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def clip(poly,a,b):
    n=len(poly)
    out=np.empty((2*n+2,2));k=0
    if n==0:return out[:0]
    prev=poly[-1];sp=prev@a-b
    for i in range(n):
        cur=poly[i];sc=cur@a-b
        if (sp<=1e-8)!=(sc<=1e-8):
            t=sp/(sp-sc);out[k]=prev+t*(cur-prev);k+=1
        if sc<=1e-8:out[k]=cur;k+=1
        prev=cur;sp=sc
    return out[:k]


@njit(cache=True)
def circle_outer(poly,q,r,sides=32):
    # Convex hull already inside the true disk: every tangent constraint is redundant.
    inside=True
    for point in poly:
        d=point-q
        if d@d>r*r:inside=False;break
    if inside:return poly.copy()
    for i in range(sides):
        a=np.array([math.cos(2*math.pi*i/sides),math.sin(2*math.pi*i/sides)])
        poly=clip(poly,a,a@q+r)
    return poly


@njit(cache=True)
def hull(points):
    n=len(points)
    if n<=1:return points.copy()
    ids=np.arange(n)
    for i in range(1,n):
        j=i
        while j>0:
            a=points[ids[j]];b=points[ids[j-1]]
            if a[0]>b[0] or (a[0]==b[0] and a[1]>=b[1]):break
            ids[j],ids[j-1]=ids[j-1],ids[j];j-=1
    out=np.empty((2*n,2));k=0
    for half in range(2):
        lower=k
        for z in range(n):
            p=points[ids[z if half==0 else n-1-z]]
            while k>=lower+2:
                u=out[k-1]-out[k-2];v=p-out[k-1]
                if u[0]*v[1]-u[1]*v[0]>1e-10:break
                k-=1
            out[k]=p;k+=1
        k-=1
    return out[:max(1,k)]


@njit(cache=True)
def exclude_disk(poly,q,r):
    """Outer convex hull of polygon minus the open disk (retains boundary)."""
    # If every original vertex survives, its convex hull is still the whole polygon.
    outside=True
    for point in poly:
        d=point-q
        if d@d<r*r-1e-7:outside=False;break
    if outside:return poly.copy()
    n=len(poly);points=np.empty((3*n+1,2));k=0
    for i in range(n):
        p=poly[i];d=poly[(i+1)%n]-p;z=p-q
        if z@z>=r*r-1e-7:points[k]=p;k+=1
        a=d@d
        if a<1e-16:continue
        b=2*(z@d);c=z@z-r*r;disc=b*b-4*a*c
        if disc>=0:
            for sign in (-1.,1.):
                t=(-b+sign*math.sqrt(disc))/(2*a)
                if 0<t<1:points[k]=p+t*d;k+=1
    return hull(points[:k])


@njit(cache=True)
def mec(poly):
    n=len(poly)
    if n==0:return np.zeros(2),1e8
    best=-1.;ii=0;jj=0
    for i in range(n):
        for j in range(i):
            d=poly[i]-poly[j];r=d@d
            if r>best:best=r;ii=i;jj=j
    c=(poly[ii]+poly[jj])/2;radius2=0.
    for p in poly:
        d=p-c;radius2=max(radius2,d@d)
    if radius2<=max(0.,best)/4+1e-7:return c,math.sqrt(radius2)
    best=math.inf;center=c.copy()
    for i in range(n):
        for j in range(i):
            for k in range(j):
                u=poly[j]-poly[i];v=poly[k]-poly[i]
                cross=u[0]*v[1]-u[1]*v[0]
                if abs(cross)<1e-10:continue
                uu=u@u;vv=v@v
                c=poly[i]+np.array([uu*v[1]-vv*u[1],u[0]*vv-v[0]*uu])/(2*cross)
                delta=c-poly[i];r=delta@delta
                if r>=best:continue
                ok=True;mx=r
                for p in poly:
                    d=p-c;rr=d@d;mx=max(mx,rr)
                    if rr>r+1e-5:ok=False;break
                if ok:best=mx;center=c.copy()
    return center,math.sqrt(best)


@njit(cache=True)
def update(poly,q,x,receive_radius,error,sides):
    d=np.linalg.norm(q-x)
    if d>receive_radius:return exclude_disk(poly,q,1000.)
    if d<=5:return circle_outer(poly,q,5.,sides)
    theta=round((math.degrees(math.atan2(x[1]-q[1],x[0]-q[0]))+error)%360,2)
    lo=math.radians(theta-1.005);hi=math.radians(theta+1.005)
    a=np.array([math.sin(lo),-math.cos(lo)])
    poly=clip(poly,a,a@q)
    a=np.array([-math.sin(hi),math.cos(hi)])
    poly=clip(poly,a,a@q)
    poly=circle_outer(poly,q,1500.,sides)
    return exclude_disk(poly,q,5.)


@njit(cache=True)
def evaluate(poly,current,candidate,is_clear,points,radii,errors,history,switch,sides):
    costs=np.empty(len(points));endpoints=np.empty_like(points)
    for i in range(len(points)):
        x=points[i];q=candidate.copy();p=poly.copy()
        cost=np.linalg.norm(q-current)/5
        if is_clear and np.linalg.norm(x-q)<=20:
            costs[i]=cost+5;endpoints[i]=q;continue
        seen=np.empty((len(history)+20,2));used=len(history)
        for j in range(used):seen[j]=history[j]
        if is_clear:
            cost+=3;p=exclude_disk(p,q,20.)
            pending_switch=switch
        else:
            cost+=5+switch;p=update(p,q,x,radii[i],errors[i,0],sides)
            seen[used]=q;used+=1;pending_switch=0
        finished=False
        for step in range(14):
            if len(p)==0:break
            center,r=mec(p)
            if not math.isfinite(r):break
            if r<=20:
                delta=q-center;d=np.linalg.norm(delta)
                dest=center+delta*min(1.,max(0.,20-r-1e-6)/d) if d>0 else center.copy()
                cost+=np.linalg.norm(dest-q)/5+5
                endpoints[i]=dest;finished=True;break
            dest=center.copy()
            for attempt in range(20):
                repeated=False
                for j in range(used):
                    if np.linalg.norm(dest-seen[j])<1e-6:repeated=True;break
                if not repeated:break
                dest[0]+=.05
            cost+=np.linalg.norm(dest-q)/5+5+pending_switch;pending_switch=0
            p=update(p,dest,x,radii[i],errors[i,step+1],sides)
            seen[used]=dest;used+=1;q=dest
        costs[i]=cost if finished else 1e6
        if not finished:endpoints[i]=q
    return costs,endpoints
