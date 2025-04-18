import numpy as np
from sklearn.model_selection import ParameterGrid


if __name__ == "__main__":
    parameters = {
        "algorithm" : ["mmdew", "mmd"],
        "dataset" : ["normalUnif", "normalLaplace", "normalMixed"],
        "n" : [1024],
        "repetitions" : [20],
        "alpha" : np.geomspace(0.001, 1, 10)
    }
    for ex in list(ParameterGrid(parameters)):
        print(f"{ex["algorithm"]} {ex["dataset"]} {ex["n"]} --repetitions {ex["repetitions"]} -c alpha={ex["alpha"]}")
