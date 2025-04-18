import argparse
import os
import numpy as np
from pathlib import Path
import base
from experiments.data import Laplace, Mixed, NormalToLaplace, NormalToMixed, NormalToUniform, Uniform
from mmdew.detectors import MMDAdapter, MMDEWAdapter, OnlineKernelCUSUM
import pandas as pd
from mmdew import fast_rbf_kernel, mmdew

class ParseKwargs(argparse.Action):
    """Source: https://sumit-ghosh.com/posts/parsing-dictionary-key-value-pairs-kwargs-argparse-python/"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

def run(args):
    if args.datagenerator == "uniform":
        stream = Uniform(args.n, preprocess=base.preprocess)
    elif args.datagenerator == "laplace":
        stream = Laplace(args.n, preprocess=base.preprocess)
    elif args.datagenerator == "mixed":
        stream = Mixed(args.n, preprocess=base.preprocess)
    elif args.datagenerator == "normalUnif":
        stream = NormalToUniform(n1=args.n,n2=args.n, preprocess=base.preprocess)
    elif args.datagenerator == "normalLaplace":
        stream = NormalToLaplace(n1=args.n,n2=args.n, preprocess=base.preprocess)
    elif args.datagenerator == "normalMixed":
        stream = NormalToMixed(n1=args.n,n2=args.n, preprocess=base.preprocess)

    if args.algorithm == "mmdew":
        alpha = float(args.config["alpha"]) if "alpha" in args.config else 0.01
        config = args.config | {"alpha" : alpha} # rhs overwrites lhs
        print(f"Initializing MMDEW with conf: {config}") ## selbst in pre_train berechnen.
        detector = MMDEWAdapter(**config)
    elif args.algorithm == "mmd":
        alpha = float(args.config["alpha"]) if "alpha" in args.config else 0.01
        config = args.config | {"alpha" : alpha} # rhs overwrites lhs
        print(f"Initializing MMD with conf: {config}") ## selbst in pre_train berechnen.
        detector = MMDAdapter(**config)
    elif args.algorithm == "okcusum":
        alpha = float(args.config["alpha"]) if "alpha" in args.config else 0.01
        B_max = int(args.config["B_max"]) if "B_max" in args.config else 20
        N = int(args.config["N"]) if "N" in args.config else 5
        null_size = int(args.config["null_size"]) if "null_size" in args.config else 100
        config = args.config | {"alpha" : alpha, "B_max" : B_max, "N" : N, "null_size" : null_size} # rhs overwrites lhs
        print(f"Initializing Online Kernel CUSUM with conf: {config}") ## selbst in pre_train berechnen.
        detector = OnlineKernelCUSUM(**config)



    # run the algorithm
    if args.pretrain > 0:
        pre_train_dat = np.array(
            [stream.next_sample()[0] for _ in range(args.pretrain)]
        ).squeeze(1)
        detector.pre_train(pre_train_dat)
        stream.restart()


    actual_cps = []
    detected_cps = []
    detected_cps_at = []
    i = 0
    with base.ContextTimer() as timer:
        while stream.has_more_samples():
            next_sample, _, is_change = stream.next_sample()
            if is_change:
                actual_cps += [i]
            detector.add_element(next_sample)
            if detector.detected_change():
                detected_cps_at += [i]
                if detector.delay:
                    detected_cps += [i - detector.delay]
            i += 1


    # collect results
    res = {
        "algorithm" : [detector.name()],
        "dataset" : [stream.id()],
        "actual_cps" : [actual_cps],
        "detected_cps" : [detected_cps],
        "detected_cps_at" : [detected_cps_at],
        "runtime" : [timer.secs],
        "config" : [detector.parameter_str()],
    }
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", type=str, help="the algorithm to run")
    parser.add_argument("datagenerator", type=str, help="generator to use")
    parser.add_argument("n", type=int, help="number of samples to generate")
    parser.add_argument('-c', '--config', nargs='*', action=ParseKwargs, help="key=value pairs of the algorithm config. Keys must match the respective __init__.")
    parser.add_argument("--pretrain", type=int, default=100, help="Number of samples to use for pre training")
    parser.add_argument("--repetitions", type=int, default=1, help="Repetitions of this configuration")
    args = parser.parse_args()

    df = pd.DataFrame()
    for rep in range(args.repetitions):
        res = run(args) | {"rep" : rep}
        df = pd.concat((df,pd.DataFrame(res)))

    output_dir = Path("../results_rebuttal") / args.datagenerator / f"n={args.n}" / args.algorithm
    os.makedirs(output_dir, exist_ok=True)
    print("Done. Writing...")
    df.to_csv(output_dir / (df.iloc[0]["config"] + ".csv")  )
