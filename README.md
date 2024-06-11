# Maximum Mean Discrepancy on Exponential Windows for Online Change Detection

This repository hosts the code for reproducing the experiments in the article:

__Maximum Mean Discrepancy on Exponential Windows for Online Change Detection__.


## Installation

To install the package and its dependencies, clone the repository, and run the following commands.

    conda create -n mmdew python=3.11
    conda activate mmdew
    pip install psutil  scikit-learn pandas tensorflow numexpr
    pip install git+https://github.com/lightonai/newma.git
    pip install -e .


## Obtaining the data sets

We use publicly available data sets. The sources are stated in the article. We also collect them here for easy reference.

CIFAR10, FashionMNIST, and MNIST are automatically downloaded through keras.
[Gas](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) and [HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) are available through UCI.

The code assumes the latter two to be present in the  `./data` folder.


## Reproducing the results

The scripts for reproducing the results are collected in `./src`. E.g., for reproducing the F1-scores, run

    python src/run_experiments_f1_score.py

which will save the results in the `./results/<current-date>` folder. Each configuration results in one file. These can be aggregated by

    python experiments/unify_results.py <beta> ../<date>/

In the article, we choose $\beta \in \{1,2,4\}$, that is, we ran the script three times.


## Figures

The figures were produced with the notebooks in `./notebooks`. These read the output of the `unify_results.py` script.

