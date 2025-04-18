from abc import ABC, abstractmethod

import numpy as np

from mmdew.fast_rbf_kernel import est_gamma
from mmdew.mmdew import MMDEW
from scipy.special import comb as nchoosek
from sklearn.metrics.pairwise import rbf_kernel




class ChangeDetector(ABC):
    """Common interface for all change detectors to simplify experiments."""
    @abstractmethod
    def pre_train(self, data):
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def parameter_str(self) -> str:
        raise NotImplementedError


class MMDEWAdapter(ChangeDetector):
    def __init__(self, gamma=1, alpha=.1):
        """
        :param gamma: The scale of the data
        :param alpha: alpha value for the hypothesis test
        """

        self.gamma=gamma
        self.alpha = alpha
        self.detector = MMDEW(gamma=gamma,alpha=alpha,min_elements_per_window=32,max_windows=0,cooldown=0)
        self.element_count = 0
        super(MMDEWAdapter, self).__init__()

    def name(self) -> str:
        return "MMDEW"

    def parameter_str(self) -> str:
        return r"$\alpha = {}$".format(self.alpha)

    def pre_train(self, data):
        self.gamma = est_gamma(data)
        self.detector = MMDEW(gamma=self.gamma,alpha=self.alpha,min_elements_per_window=32,max_windows=0,cooldown=0)

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """

        self.element_count+=1
        self.detected_cp = False
        prev_cps = len(self.detector.changes_detected_at)
        self.detector.insert(input_value[0])
        if len(self.detector.changes_detected_at) > prev_cps:
            self.delay = self.element_count - self.detector.changes_detected_at[-1]
            self.detected_cp = True

    def detected_change(self):
        return self.detected_cp

class MMDAdapter(ChangeDetector):
    """Compute the quadratic time MMD on a datastream, i.e., use MMDEW without subsampling."""
    def __init__(self, gamma=1, alpha=.1):
        """
        :param gamma: The scale of the data
        :param alpha: alpha value for the hypothesis test
        """
        self.gamma=gamma
        self.alpha = alpha
        self.detector = MMDEW(gamma=gamma,alpha=alpha,min_elements_per_window=np.inf,max_windows=0,cooldown=0)
        self.element_count = 0
        super(MMDAdapter, self).__init__()

    def name(self) -> str:
        return "MMD"

    def parameter_str(self) -> str:
        return r"$\alpha = {}$".format(self.alpha)

    def pre_train(self, data):
        self.gamma = est_gamma(data)
        self.detector = MMDEW(gamma=self.gamma,alpha=self.alpha,min_elements_per_window=np.inf,max_windows=0,cooldown=0)

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """

        self.element_count+=1
        self.detected_cp = False
        prev_cps = len(self.detector.changes_detected_at)
        self.detector.insert(input_value[0])
        if len(self.detector.changes_detected_at) > prev_cps:
            self.delay = self.element_count - self.detector.changes_detected_at[-1]
            self.detected_cp = True

    def detected_change(self):
        return self.detected_cp

class ScanBStatistic:
    def __init__(self, reference_sample, B0, N):
        self.rng = np.random.default_rng()
        self.B0 = B0 # block size
        self.N = N # number of blocks
        self.gamma = est_gamma(reference_sample)

        self.ref_blocks = self.rng.choice(reference_sample, size=N*B0, replace=False) # store the blocks that make the references
        self.ref_grams_XX = []
        for i in range(self.N):
            x = self.ref_blocks[i*self.B0:(i+1)*self.B0]
            self.ref_grams_XX += [rbf_kernel(x, gamma=self.gamma)]

        self.post_block = reference_sample[-B0:] # store the block with the most recent data
        self.var = 1/nchoosek(B0,2)*(1/N*self.expectation_h_squared(reference_sample) + (N-1)/N*self.covariance_h(reference_sample))
        self.stats = []

    def add_element(self, input_value):
        self.post_block = np.concatenate((self.post_block[1:], input_value))
        acc = 0
        gram_YY = rbf_kernel(self.post_block, gamma=self.gamma)
        for i in range(self.N):
            acc += self.mmd_u(self.ref_blocks[i*self.B0:(i+1)*self.B0], self.ref_grams_XX[i], self.post_block, gram_YY)
        self.stats += [acc / (self.N * np.sqrt(self.var))]


    def mmd_u(self, sample_x, gram_XX, sample_y, gram_YY):
        n_x = len(sample_x)
        n_y = len(sample_y)
        XX = gram_XX - np.eye(n_x)
        YY = gram_YY - np.eye(n_y)
        XY = rbf_kernel(sample_x, sample_y, gamma=self.gamma)

        return np.sum(XX)/(n_x*(n_x-1)) + np.sum(YY)/(n_y*(n_y-1)) - 2*np.mean(XY)

    def expectation_h_squared(self, sample): # corresponds to (7) in their paper
        # we assume a shuffled sample
        n = int(len(sample)/4)
        x   = sample[0*n:1*n]
        x_p = sample[1*n:2*n]
        y   = sample[2*n:3*n]
        y_p = sample[3*n:4*n]

        K_xx = rbf_kernel(x,x_p, gamma=self.gamma)
        K_yy = rbf_kernel(y,y_p, gamma=self.gamma)
        K_xy1 = rbf_kernel(x,y_p, gamma=self.gamma)
        K_xy2 = rbf_kernel(x_p,y, gamma=self.gamma)

        return np.mean((K_xx + K_yy - K_xy1 - K_xy2)**2)

    def covariance_h(self, sample): # corresponds to (7) in their paper
        # we assume a shuffled sample
        n     = int(len(sample)/6)
        x     = sample[0*n:1*n]
        x_p   = sample[1*n:2*n]
        x_pp  = sample[2*n:3*n]
        x_ppp = sample[3*n:4*n]
        y     = sample[4*n:5*n]
        y_p   = sample[5*n:6*n]

        K1 = rbf_kernel(x,x_p, gamma=self.gamma)
        K2 = rbf_kernel(y,y_p, gamma=self.gamma)
        K3 = rbf_kernel(x,y_p, gamma=self.gamma)
        K4 = rbf_kernel(x_p,y, gamma=self.gamma)
        K5 = rbf_kernel(x_pp,x_ppp, gamma=self.gamma)
        K6 = rbf_kernel(x_pp,y_p, gamma=self.gamma)
        K7 = rbf_kernel(x_ppp,y, gamma=self.gamma)

        h1 = K1 + K2 - K3 - K4
        h2 = K5 + K2 - K6 - K7

        return np.mean(h1*h2) - np.mean(h1)*np.mean(h2)


class NaiveOnlineKernelCUSUM(ChangeDetector):
    def __init__(self, reference_sample, B_max, N, alpha=.05):
        self.B_stats = []
        for B0 in range(2,B_max+1, 2):
            self.B_stats += [ScanBStatistic(reference_sample, B0=B0, N=N)]

        self.alpha = alpha
        self.B_max = B_max
        self.N = N

        self.delay = False
        self.element_count = 0

    def add_element(self, input_value):
        self.element_count += 1
        self.detected_cp = False

        for scan_b_stat in self.B_stats:
            scan_b_stat.add_element(input_value)

        # we estimate the null on the first few observations
        #if self.element_count == self.null_size:
        #    self.threshold = np.quantile(self.stats(), 1-self.alpha)

        #if self.threshold and self.stats()[-1] >= self.threshold:
        #    self.detected_cp = True

    def detected_change(self):
        return self.detected_cp

    def name(self) -> str:
        return "OnlineKernelCUSUM"

    def stats(self):
        stats = np.zeros((len(self.B_stats),len(self.B_stats[0].stats)))
        for i, _ in enumerate(self.B_stats):
            stats[i] = np.array((self.B_stats[i].stats))
        return np.max(stats,axis=0)

    def pre_train(self, data):
        self.threshold = 4.6
        self.B_stats = []
        for B0 in range(2,self.B_max+1, 2):
            self.B_stats += [ScanBStatistic(data, B0=B0, N=self.N)]
        self.element_count = 0

    def parameter_str(self) -> str:
        return r"$\alpha = {}$, $B = {}$, $N = {}$".format(self.alpha, self.B_max, self.N)

class FastOKCUSUM(ChangeDetector):
    def __init__(self, reference_sample, B_max, N, B_min=2,alpha=0.05):
        self.rng = np.random.default_rng()
        self.gamma = est_gamma(reference_sample)
        self.B_max = B_max
        self.B_min = B_min
        self.N = N

        self.reference_sample = self.rng.choice(reference_sample, size=N*B_max, replace=False) # store the blocks that make the references
        self.XX = rbf_kernel(self.reference_sample, gamma=self.gamma)

        self.post_block = reference_sample[-B_max:] # store the block with the most recent data

        self.vars = []
        e_h_sq = self.expectation_h_squared(self.reference_sample)
        cov_h = self.covariance_h(self.reference_sample)
        for B0 in range(B_min,B_max+1,2):
            self.vars += [1/nchoosek(B0,2)*(1/N*e_h_sq + (N-1)/N*cov_h)] # store the variances corresponding to each B0
        self.stats = []
        self.alpha = alpha

    def pre_train(self, data):
        pass

    def parameter_str(self) -> str:
        return r"$\alpha = {}$, $B = {}$, $N = {}$".format(self.alpha, self.B_max, self.N)

    def expectation_h_squared(self, sample): # corresponds to (7) in their paper
        # we assume a shuffled sample
        n = int(len(sample)/4)
        x   = sample[0*n:1*n]
        x_p = sample[1*n:2*n]
        y   = sample[2*n:3*n]
        y_p = sample[3*n:4*n]

        K_xx = rbf_kernel(x,x_p, gamma=self.gamma)
        K_yy = rbf_kernel(y,y_p, gamma=self.gamma)
        K_xy1 = rbf_kernel(x,y_p, gamma=self.gamma)
        K_xy2 = rbf_kernel(x_p,y, gamma=self.gamma)

        return np.mean((K_xx + K_yy - K_xy1 - K_xy2)**2)

    def covariance_h(self, sample): # corresponds to (7) in their paper
        # we assume a shuffled sample
        n     = int(len(sample)/6)
        x     = sample[0*n:1*n]
        x_p   = sample[1*n:2*n]
        x_pp  = sample[2*n:3*n]
        x_ppp = sample[3*n:4*n]
        y     = sample[4*n:5*n]
        y_p   = sample[5*n:6*n]

        K1 = rbf_kernel(x,x_p, gamma=self.gamma)
        K2 = rbf_kernel(y,y_p, gamma=self.gamma)
        K3 = rbf_kernel(x,y_p, gamma=self.gamma)
        K4 = rbf_kernel(x_p,y, gamma=self.gamma)
        K5 = rbf_kernel(x_pp,x_ppp, gamma=self.gamma)
        K6 = rbf_kernel(x_pp,y_p, gamma=self.gamma)
        K7 = rbf_kernel(x_ppp,y, gamma=self.gamma)

        h1 = K1 + K2 - K3 - K4
        h2 = K5 + K2 - K6 - K7

        return np.mean(h1*h2) - np.mean(h1)*np.mean(h2)

    def add_element(self, input_value):
        self.post_block = np.concatenate((self.post_block[1:], input_value))
        YY = rbf_kernel(self.post_block, gamma=self.gamma)
        XY = rbf_kernel(self.reference_sample, self.post_block, gamma=self.gamma)

        stats = -np.inf

        for i, B0 in enumerate(range(self.B_min,self.B_max+1,2)): # different window sizes
            acc = 0
            yy = YY[-B0:,-B0:]
            for n in range(self.N):
                xx = self.XX[n*self.B_max:n*self.B_max+B0,n*self.B_max:n*self.B_max+B0]
                xy = XY[n*self.B_max:n*self.B_max+B0,-B0:]
                n_x = len(xx)
                n_y = len(yy)
                acc += np.sum(xx-np.eye(n_x))/(n_x*(n_x-1)) + np.sum(yy-np.eye(n_y))/(n_y*(n_y-1)) - 2*np.mean(xy)
                #import pdb; pdb.set_trace()
            tmp = acc / (self.N * np.sqrt(self.vars[i]))

            stats = max(tmp, stats)

        self.stats += [stats]
