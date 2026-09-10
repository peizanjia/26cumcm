"""Independently replay all saved counterexamples and audit saved statistics."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .geometry import bearing_halfplanes, intersect_bearings
from .enclosing_circle import diameter_circle_metrics


def verify(path):
    examples=json.loads((path/'examples.json').read_text(encoding='utf-8'))
    for name,case in examples.items():
        points=np.array(case['points']);source=np.array(case['source'])
        angles=np.array([m['svd_deg'] for m in case['measurements']])
        delta=source-points
        true=np.degrees(np.arctan2(delta[:,1],delta[:,0]))%360
        errors=(angles-true+180)%360-180
        assert np.max(np.abs(errors))<=1+1e-10
        distance=np.linalg.norm(points-source,axis=1)
        assert np.min(distance)>5 and np.max(distance)<=case['receive_radius_m']+1e-8
        assert np.max(np.linalg.norm(points,axis=1))<=1800+1e-8
        result=intersect_bearings(points,angles)
        assert result.status=='bounded'
        np.testing.assert_allclose(result.vertices,case['vertices'],atol=1e-7)
        p=result.vertices
        a,b=bearing_halfplanes(points,angles)
        assert np.max(a@source-b)<=1e-7
        assert np.max(a@p.T-b[:,None])<=1e-6
        diameter=np.max(np.linalg.norm(p[:,None]-p[None,:],axis=2))
        np.testing.assert_allclose(diameter,case['metrics']['diameter_m'],rtol=1e-9)
        # Independent convex optimization in normalized coordinates; initial
        # center is the vertex mean, not the enumerated enclosing-circle center.
        q=(p-p[0])/diameter
        center=q.mean(axis=0)
        initial=np.r_[center,np.max(np.sum((q-center)**2,axis=1))+1]
        ref=minimize(lambda z:z[2],initial,jac=lambda z:np.array([0.,0.,1.]),
                     constraints={'type':'ineq',
                        'fun':lambda z:z[2]-np.sum((q-z[:2])**2,axis=1),
                        'jac':lambda z:np.column_stack((2*(q-z[:2]),np.ones(len(q))))},
                     method='SLSQP',options={'ftol':1e-10,'maxiter':1000})
        assert ref.success, (name,ref.message)
        np.testing.assert_allclose(np.sqrt(ref.fun)*diameter,
                                   case['metrics']['mec_radius_m'],rtol=2e-7,atol=1e-7)
        assert diameter_circle_metrics(p)['failure']

    with gzip.open(path/'trials.csv.gz','rt',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    metadata=json.loads((path/'metadata.json').read_text())
    assert len(rows)==metadata['intersection_evaluations']
    assert len({(r['layout'],r['noise'],r['seed'],r['n']) for r in rows})==len(rows)
    assert sum(r['status']=='bounded' for r in rows)==metadata['bounded']
    assert sum(r['failure']=='True' for r in rows)==metadata['failures']
    for s in csv.DictReader((path/'summary_by_n.csv').open()):
        group=[r for r in rows if all(r[k]==s[k] for k in ('layout','noise','n'))]
        assert len(group)==int(s['total'])
        assert sum(r['status']=='bounded' for r in group)==int(s['bounded'])
        assert sum(r['failure']=='True' for r in group)==int(s['failures'])
    print(f'Verified {len(examples)} saved counterexamples with independent optimization; '
          f'audited {len(rows)} raw rows and all per-count summaries.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path,nargs='?',default=Path(__file__).parent/'output'/'circle_study')
    verify(parser.parse_args().path)
