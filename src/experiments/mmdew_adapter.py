import numpy as np

from experiments.abstract import DriftDetector
from mmdew import mmdew
from mmdew.fast_rbf_kernel import est_gamma

class MMDEWAdapter(DriftDetector):
    def __init__(self, gamma, alpha=.1):
        """
        :param gamma: The scale of the data
        :param alpha: alpha value for the hypothesis test
      
        """
        self.gamma=gamma
        self.alpha = alpha
        self.logger = None
        self.detector = mmdew.MMDEW(gamma=gamma,alpha=alpha,min_elements_per_window=32,max_windows=0,cooldown=500)
        self.element_count = 0
        super(MMDEWAdapter, self).__init__()

    def name(self) -> str:
        return "MMDEW"

    def parameter_str(self) -> str:
        return r"$\alpha = {}$".format(self.alpha)

    def pre_train(self, data):
        self.gamma = est_gamma(data)
        self.detector = mmdew.MMDEW(gamma=self.gamma,alpha=self.alpha,min_elements_per_window=32,max_windows=0,cooldown=500)
    

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
    
    def metric(self):
        return 0

 
