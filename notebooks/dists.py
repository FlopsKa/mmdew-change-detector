import numpy as np
from enum import Enum


import matplotlib.pyplot as plt
from time import time
import sys
import json


class DistributionType(Enum):
    MIXED = 'mixed'
    LAPLACE = 'laplace'
    UNIFORM = 'uniform'




def _get_distribution(distribution_type, size, dim):
    rng = np.random.default_rng()
    sigma = 2
    if distribution_type == DistributionType.MIXED.value:
        print("MIXED")
        mu = 0
        s2 = sigma ** 2

        p1 = rng.normal(size=(size, dim))
        p2 = rng.multivariate_normal(mu * np.ones(dim), s2 * np.identity(dim), size=size)
        alpha = rng.uniform(size=size)

        p = p2
        p[alpha < 0.3] = p1[alpha < 0.3]

        return p
    
    elif distribution_type == DistributionType.LAPLACE.value:
        print("LAPLACE")
        return rng.laplace(scale=sigma, size=(size, dim))
    
    elif distribution_type == DistributionType.UNIFORM.value:
        print("UNIFORM")
        return rng.uniform(low=-sigma, high=sigma, size=(size, dim))

    else:
        return None
