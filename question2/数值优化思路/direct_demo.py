"""One observed state, no NN: verify action gradients and optimize two coordinates.

Uses the existing polygon/Monte Carlo score, NOT certified exact geometry or
global optimization. Common random samples make finite differences comparable.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from question2.黑箱思路.environment import sample_posterior
from question2.黑箱思路.geometry import first_polygons, SCALE, DELTA
from question2.黑箱思路.objective import outcomes


def main():
    torch.set_num_threads(4)
    p = np.zeros((1, 2))
    poly, counts, areas = first_polygons(p, 256, 64)
    def dataset(seed, n):
        return {
            'targets': torch.tensor(sample_posterior(p, n, seed), dtype=torch.float64),
            'errors': torch.tensor(np.random.default_rng(seed+1).uniform(-DELTA, DELTA, (1,n))),
            'polygons': torch.tensor(poly[:, :counts.max()], dtype=torch.float64),
            'counts': torch.tensor(counts, dtype=torch.long),
            'areas': torch.tensor(areas, dtype=torch.float64),
        }
    data = dataset(25260911, 2048)
    def fg(x):
        q_m = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        loss = outcomes(data, q_m[None]/SCALE)['loss'].mean()
        loss.backward()
        return float(loss.detach()), q_m.grad.numpy().copy()
    x0 = np.array([750., -375.])
    f0, grad = fg(x0)
    h = .05
    fd = np.array([(fg(x0+np.eye(2)[i]*h)[0]-fg(x0-np.eye(2)[i]*h)[0])/(2*h) for i in range(2)])
    x1 = x0 - 20*grad
    result = minimize(fg, x0, jac=True, method='L-BFGS-B',
                      bounds=[(50,1500),(-750,-20)],
                      options={'maxiter':80, 'ftol':1e-12, 'gtol':1e-7})
    test = dataset(26260911, 16384)
    def test_score(x):
        with torch.no_grad():
            return float(outcomes(test, torch.tensor(x)[None]/SCALE)['loss'].mean())
    report = dict(state=dict(p_m=[0,0],theta_rad=0),
        training_seed=25260911, training_samples=2048,
        test_seed=26260911,test_samples=16384,
        initial_q_m=x0.tolist(), initial_score_m2=f0,
        gradient_m2_per_m=grad.tolist(),finite_difference_m2_per_m=fd.tolist(),
        demonstration_step_q_m=x1.tolist(),demonstration_step_score_m2=fg(x1)[0],
        optimized_q_m=result.x.tolist(),optimized_training_score_m2=float(result.fun),
        initial_test_score_m2=test_score(x0),optimized_test_score_m2=test_score(result.x),
        optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),
        limits='Single-state, single-start, restricted right-side action box, polygon score and finite posterior samples; no global certificate.')
    (Path(__file__).parent/'direct_demo_result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
