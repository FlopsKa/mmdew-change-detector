import psutil
from time import time, strftime, gmtime
import pathlib
from sklearn import preprocessing
import pandas as pd
from copy import deepcopy, copy
import uuid
from experiments.competitors import WATCH, IBDD, D3, AdwinK
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import numpy as np

from experiments.abstract import DriftDetector
from experiments.data import *
from experiments.mmdew_adapter import MMDEWAdapter
from run_experiments_f1_score import *
from mmdew import mmdew
import onlinecp.algos as algos
import onlinecp.utils.feature_functions as feat
from mmdew.fast_rbf_kernel import est_gamma



class MMDEWAdapter(MMDEWAdapter):
    def pre_train(self, data):
        self.gamma = est_gamma(data)
        self.detector = mmdew.MMDEW(gamma=self.gamma,alpha=self.alpha,min_elements_per_window=256,max_windows=0,cooldown=0)
    


if __name__ == "__main__":
    parameter_choices = {
        MMDEWAdapter: {"gamma": [1], "alpha": np.geomspace(0.001,1,10)},
      
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg]))
        for alg in parameter_choices
    }

    max_len = None
    n_reps = 1

    datasets = [
        Uniform(10000, preprocess=preprocess),
        Laplace(10000, preprocess=preprocess),
        Mixed(10000, preprocess=preprocess)
    ]

    ex = Experiment(algorithms, datasets, reps=n_reps)

    tasks = ex.generate_tasks()
    for t in tasks:
        t.output = "../results/arl"
    start_time = time()
    print(f"Total tasks: {len(tasks)}")
    Parallel(n_jobs=-4)(delayed(Task.run)(t) for t in tasks)
    end_time = time()
    print(f"Total runtime: {end_time-start_time}")
