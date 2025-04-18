import numpy as np

class RandomData:
    def draw(self):
        pass

class MixedNormal(RandomData):
    def __init__(self, n, d, prob):
        self.n = n
        self.d = d
        self.prob = prob
        self.rng = np.random.default_rng()
    def draw(self):
        return np.where((self.rng.uniform(size=self.n)>self.prob).reshape(-1,1),
                        self.rng.normal(scale=2,size=(self.n,self.d)),
                        self.rng.normal(size=(self.n,self.d)))

class Laplace(RandomData):
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.rng = np.random.default_rng()
    def draw(self):
        return self.rng.laplace(scale=2,size=(self.n,self.d))

class Uniform(RandomData):
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.rng = np.random.default_rng()
    def draw(self):
        return self.rng.uniform(-1,1,size=(self.n,self.d))*4
