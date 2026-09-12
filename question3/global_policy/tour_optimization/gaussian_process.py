"""Small Matérn-5/2 GP with observation variance and expected improvement."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve,solve_triangular
from scipy.optimize import minimize
from scipy.special import ndtr


def kernel(a,b,length):
    d=np.sqrt(np.maximum(0.,np.sum(((np.asarray(a)[:,None]-np.asarray(b)[None,:])/length)**2,axis=2)))
    z=np.sqrt(5.)*d
    return (1.+z+z*z/3.)*np.exp(-z)


class GaussianProcess:
    def __init__(self,x,y,variance):
        self.x=np.asarray(x,float)
        y=np.asarray(y,float);self.offset=float(y.mean());self.scale=max(1.,float(y.std()))
        self.y=(y-self.offset)/self.scale
        self.variance=np.maximum(1e-5,np.asarray(variance)/self.scale**2)
        def likelihood(log_length):
            length=np.exp(log_length[0])
            K=kernel(self.x,self.x,length)+np.diag(self.variance+1e-7)
            factor=cho_factor(K,lower=True)
            return float(.5*self.y@cho_solve(factor,self.y)+np.log(np.diag(factor[0])).sum())
        choices=[minimize(likelihood,[np.log(l)],bounds=[(np.log(.08),np.log(3.))],method='L-BFGS-B')
                 for l in (.2,.7,1.5)]
        optimum=min(choices,key=lambda r:r.fun)
        self.length=float(np.exp(optimum.x[0]))
        K=kernel(self.x,self.x,self.length)+np.diag(self.variance+1e-7)
        self.factor=cho_factor(K,lower=True)
        self.alpha=cho_solve(self.factor,self.y)

    def predict(self,points):
        cross=kernel(self.x,np.asarray(points),self.length)
        mean=self.offset+self.scale*(cross.T@self.alpha)
        v=solve_triangular(self.factor[0],cross,lower=True)
        variance=np.maximum(1e-12,1.-np.sum(v*v,axis=0))*self.scale**2
        return mean,np.sqrt(variance)


def expected_improvement(mean,std,incumbent,margin=.05):
    delta=incumbent-np.asarray(mean)-margin
    sigma=np.maximum(np.asarray(std),1e-12)
    z=delta/sigma
    return np.maximum(0.,delta*ndtr(z)+sigma*np.exp(-.5*z*z)/np.sqrt(2*np.pi))
