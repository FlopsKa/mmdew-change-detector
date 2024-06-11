import numpy as np
import numexpr as ne
from scipy.linalg.blas import sgemm

# code from https://stackoverflow.com/a/55834552/2510367
def sum_k(X,Y, gamma=1, var=1):
    X_norm = -gamma*np.einsum('ij,ij->i',X,X)
    Y_norm = -gamma*np.einsum('ij,ij->i',Y,Y)
    return np.sum(ne.evaluate('v * exp(A + B + C)', {\
        'A' : X_norm[:,None],\
        'B' : Y_norm[None,:],\
        'C' : sgemm(alpha=2.0*gamma, a=X, b=Y, trans_b=True),\
        'g' : gamma,\
        'v' : var\
    }))


# implementation of the median heuristic. See https://arxiv.org/abs/1707.07269
def _est_nu(X_, max_len=1000):
    n = min(len(X_), max_len)
    dists = []
    X = np.random.default_rng(1234).choice(X_,n)
    for i in range(n):
        for j in range(i,n):
            dists += [np.linalg.norm(X[i]-X[j],ord=2)**2]
    nu = np.median(dists)
    return np.sqrt(nu*0.5)

def _nu2gamma(nu):
        return 1/(2*nu**2)

def est_gamma(X_, max_len=500):
        return _nu2gamma(_est_nu(X_=X_, max_len=max_len))