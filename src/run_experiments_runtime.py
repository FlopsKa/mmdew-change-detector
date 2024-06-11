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
from experiments.scanb_adapter import ScanB
from experiments.newma_adapter import NewMA
from experiments.data import *
from experiments.mmdew_adapter import MMDEWAdapter
from run_experiments_f1_score import *
from mmdew import mmdew
import onlinecp.algos as algos
import onlinecp.utils.feature_functions as feat


class MMDEWAdapter(MMDEWAdapter):
    def pre_train(self, data):
        self.gamma = 1
        self.detector = mmdew.MMDEW(gamma=self.gamma,alpha=self.alpha,min_elements_per_window=1,max_windows=0,cooldown=500)
    

class NewMA(NewMA):
    def pre_train(self, data):
        big_Lambda, small_lambda = (0.1, 0.01)
        m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
        d = data.shape[1]
        W, sigmasq = feat.generate_frequencies(m, d, data=data, choice_sigma=None)

        def feat_func(x):
            return feat.fourier_feat(x, W)
        self.detector = algos.NEWMA(data[0], forget_factor=self.forget_factor, feat_func=feat_func, adapt_forget_factor=self.forget_factor, thresholding_quantile=self.thresholding_quantile)
    
class ScanB(ScanB):
    def pre_train(self, data):
        big_Lambda, small_lambda = (0.1, 0.01)
        m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
        d = data.shape[1]
        W, sigmasq = feat.generate_frequencies(m, d, data=data, choice_sigma=None)
        self.detector = algos.ScanB(data[0], kernel_func=lambda x, y: feat.gauss_kernel(x, y, np.sqrt(sigmasq)), window_size=self.window_size, nbr_windows=self.num_windows, adapt_forget_factor=self.forget_factor,thresholding_quantile=self.thresholding_quantile)
    
    


if __name__ == "__main__":
    parameter_choices = {
        MMDEWAdapter: {"gamma": [1], "alpha": [0.1]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg]))
        for alg in parameter_choices
    }

    max_len = None
    n_reps = 1

    datasets = [Constant(n=n,preprocess=preprocess,max_len=max_len) for n in np.geomspace(5000,1000000,10,dtype=int)]

    ex = Experiment(algorithms, datasets, reps=n_reps)

    tasks = ex.generate_tasks()
    for t in tasks:
        t.output = "../results/runtime"
    start_time = time()
    print(f"Total tasks: {len(tasks)}")
    Parallel(n_jobs=-10)(delayed(Task.run)(t) for t in tasks)
    end_time = time()
    print(f"Total runtime: {end_time-start_time}")
