import numpy as np
import mmdew.fast_rbf_kernel as fast_rbf

class MMDEW:
    def __init__(self, gamma, alpha=0.01, min_elements_per_window=1, max_windows=0, cooldown=500, seed=1234):   
        self.gamma = gamma
        self.alpha = alpha
        self.min_elements_per_window = min_elements_per_window
        self.max_windows = max_windows
        self.rng = np.random.default_rng(seed)
        self.windows = []
        self.changes_detected_at = []
        self.total_insertions = 0
        self.cooldown = cooldown
        self.cooling_down = 0

    def insert(self, element):
        self.total_insertions += 1
        x0 = np.array(element).reshape(1,-1)
        XY = [np.sum(self.k(x0,w.elements)) for w in reversed(self.windows)]
        X0 = Window(
            elements = x0,
            length = 1,
            XX = self.k(x0,x0),
            XY = XY,
            n_XX = 1,
            n_XY = [len(w.elements) for w in reversed(self.windows)]
        )
        self.windows += [X0]

        if self.max_windows > 0 and len(self.windows) > self.max_windows:
            self.windows = self.windows[1:]
            for w in self.windows:
                w.XY = w.XY[:-1]    
                w.n_XY = w.n_XY[:-1]
        self.cooling_down -= 1
        while split := self.has_change() > 0 :
            if self.cooling_down <= 0:
                self.changes_detected_at += [self.total_insertions]
                self.cooling_down = self.cooldown
                
            self.windows = self.windows[split:]
            for w in self.windows:
                w.XY = w.XY[:-split]
                w.n_XY = w.n_XY[:-split]
        self.merge()

    def mmd_at_position(self, split):
        XX = np.sum([w.XX for w in self.windows[:split]])
        XX += np.sum([2*xy for w in self.windows[:split] for xy in w.XY[:split]])
        n_XX = np.sum([w.n_XX for w in self.windows[:split]])
        n_XX += np.sum([2*n_xy for w in self.windows[:split] for n_xy in w.n_XY[:split]])
    
        YY = np.sum([w.XX for w in self.windows[split:]])
        YY += np.sum([2*xy for w in self.windows[split:] for xy in w.XY[:-split]])
        n_YY = np.sum([w.n_XX for w in self.windows[split:]])
        n_YY += np.sum([2*n_xy for w in self.windows[split:] for n_xy in w.n_XY[:-split]])

        XY = np.sum([xy for w in self.windows[split:] for xy in w.XY[-split:]])
        n_XY = np.sum([n_xy for w in self.windows[split:] for n_xy in w.n_XY[-split:]])

        mmd = 1/n_XX * XX + 1/n_YY * YY - 2/n_XY * XY
        return mmd, int(np.sqrt(n_XX)), int(np.sqrt(n_YY)), int(np.sqrt(n_XY))
    

    def has_change(self):
        for split in range(1,len(self.windows)):
            stat, n_XX, n_YY, n_XY = self.mmd_at_position(split)
            threshold = self.threshold(max(n_XX,1),max(n_YY,1),self.alpha/(len(self.windows)-1))
            if stat >= threshold:
                return split
        return 0

    def merge(self):
        if len(self.windows) < 2:
            return ## nothing to do
        X0 = self.windows[-1]
        X1 = self.windows[-2]
        if X1.length == X0.length:
            X1.length *= 2
            sample_size = int(np.log2(X1.length))
            if X1.length < self.min_elements_per_window:
                X1.elements = np.concatenate((X1.elements,X0.elements))
            else:
                X1.elements = self.rng.choice(np.concatenate((X1.elements,X0.elements)), size=sample_size, replace=False)
            
            X1.XX += X0.XX + 2* X0.XY[0]
            X1.n_XX += X0.n_XX + 2* X0.n_XY[0]
            X1.XY = [xy + X0.XY[i+1] for i, xy in enumerate(X1.XY)]
            X1.n_XY = [n_xy + X0.n_XY[i+1] for i, n_xy in enumerate(X1.n_XY)]
            self.windows = self.windows[:-1]
            self.merge()


    def k(self, x, y):
        return fast_rbf.sum_k(x,y,gamma=self.gamma)
    
    def threshold(self,m,n,alpha):
        K = 1
        return (
                np.sqrt(K / m + K / n)
                + np.sqrt((2 * K * (m + n) * np.log(1 / alpha)) / (m * n))
            ) ** 2

class Window:
    def __init__(self, elements, length, XX, XY, n_XX, n_XY):
        self.elements = elements
        self.length = length
        self.XX = XX
        self.XY = XY
        self.n_XX = n_XX
        self.n_XY = n_XY