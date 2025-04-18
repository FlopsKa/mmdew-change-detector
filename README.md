# Maximum Mean Discrepancy on Exponential Windows for Online Change Detection

This repository hosts the code for reproducing the experiments in the article:

F. Kalinke, M. Heyden, E. Fouché Georg Gntuni and K. Böhm. Maximum Mean Discrepancy on Exponential Windows for Online Change Detection. Transactions on Machine Learning Research, 2025. [OpenReview](https://openreview.net/forum?id=OGaTF9iOxi).


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

### Figure 3: Synthetic ARL+MTD

For producing the ARL and MTD results, we use the `src/run_experiment.py` script
with the output of either `src/arl_experiments.py` or `src/mtd_experiments.py`.
E.g., to run the scripts parallely on 8 cores, one may use:

  python src/arl_experiments.py | xargs -P 8 -L 1 python src/run_experiment.py
  python src/mtd_experiments.py | xargs -P 8 -L 1 python src/run_experiment.py

Or, to run both:

    { python src/mtd_experiments.py ; python src/arl_experiments.py ; } | xargs -P 8 -L 1 python src/run_experiment.py

For post-processing, use `03_prepare_arl.ipynb` and `04_prepare_edd.ipynb`. To plot the data, see `notebooks/07_plot_edd+arl.ipynb`.

### Figure 4: Synthetic runtime

Create the results with `src/run_experiments_runtime.py` and plot with `notebooks/02_plot_runtime.ipynb`.

### Figure 5: Real-world F1-score

For reproducing the F1-scores, run

    python src/run_experiments_f1_score.py

which will save the results in the `./results/<current-date>` folder. Each configuration results in one file. These can be aggregated by

    python experiments/unify_results.py <beta> ../<date>/

In the article, we choose $\beta \in \{1,2,4\}$, that is, we ran the script three times.

### Figure 6: Real-world MTD/PCD

The data is the same as for __Figure 5__ but plotted with `notebooks/01_plot_pcd_mtd.ipynb`.

### Figure 7: Synthetic kernel-based EDD/MTD

We must first generate the Monte Carlo thresholds. These are in

- `08_kernel-cusum_thresholds.ipynb`,
- `08_scanb_thresholds.ipynb`,
- `09_mmdew-threholds_d=20.ipynb`, and
- `13_newma_threholds.ipynb`.

After creating the threholds, we post-process with

- `09_mmdew-arl-vs-edd.ipynb`,
- `10_okcusum-arl-vs-edd.ipynb`,
- `11_newma-arl-vs-edd.ipynb`, and
- `11_scanb-arl-vs-edd.ipynb`.

### Figure 8: Synthetic univariate test-statistics

CPM is implemented in R. We run the code in `src/R_experiments.R` and post-process with `notebooks/post_process_R.ipynb`.

The remaining algorithms are in 

- `15_statistics_univariate-mmd.ipynb`,
- `15_statistics_univariate-mmdew.ipynb`, and
- `16_statistics_univariate-focus.ipynb`.

Plotting is done in `17_plot_univariate.ipynb`.

## Test of online kernel CUSUM

We test our implementation in `demo-kcusum.ipynb`.





## Figures

The figures were produced with the notebooks in `./notebooks`. These read the output of the `unify_results.py` script.
