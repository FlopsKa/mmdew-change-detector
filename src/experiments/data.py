from itertools import permutations
import os

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

from abc import abstractmethod, ABCMeta, ABC
from typing import Protocol, runtime_checkable

from skmultiflow.data_stream import DataStream

class ChangeStream(DataStream, metaclass=ABCMeta):
    def next_sample(self, batch_size=1):
        change = self._is_change()
        x, y = super(ChangeStream, self).next_sample(batch_size)
        return x, y, change

    @abstractmethod
    def change_points(self):
        raise NotImplementedError

    @abstractmethod
    def _is_change(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def id(self) -> str:
        raise NotImplementedError

    def type(self) -> str:
        raise NotImplementedError

def get_perm_for_cd(df):
    rng = np.random.default_rng()
    classes = sorted(df["Class"].unique())
    perms = list(permutations(classes, len(classes)))
    use_perm = rng.integers(len(perms))
    mapping = dict(zip(classes, list(perms)[use_perm]))
    df = df.sort_values("Class", key=lambda series : series.apply(lambda x: mapping[x])).reset_index(drop=True)
    return df

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

class Constant(ChangeStream):
    def __init__(self, n, preprocess=None, max_len=None):
        self.n = n
        data = np.ones(n).reshape(-1,1)
        y = np.zeros(n)

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(Constant, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "Constant" + str(self.n)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class TrafficUnif(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        df = pd.read_csv("../data/traffic/traffic.csv",sep=";",decimal=",")
        est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        est.fit(df["Slowness in traffic (%)"].values.reshape(-1,1))
        df["Class"] = est.transform(df["Slowness in traffic (%)"].values.reshape(-1,1))

        df = get_perm_for_cd(df)

        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(TrafficUnif, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "TrafficUnif"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class GasSensors(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        df = pd.read_csv("../data/gas-drift_csv.csv")
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(GasSensors, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "GasSensors"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class MNIST(ChangeStream):
    def __init__(self, preprocess=None, max_len = None):
        from tensorflow import keras

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        df = pd.DataFrame(x)
        df["Class"] = y
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(MNIST, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "MNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class FashionMNIST(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        from tensorflow import keras

        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])


        df = pd.DataFrame(x)
        df["Class"] = y
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(FashionMNIST, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "FMNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class HAR(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        har_data_dir = "../data/har"
        test = pd.read_csv(os.path.join(har_data_dir, "test.csv"))
        train = pd.read_csv(os.path.join(har_data_dir, "train.csv"))
        x = pd.concat([test, train])
        x = x.sort_values(by="Activity")
        y = LabelEncoder().fit_transform(x["Activity"])
        x = x.drop(["Activity", "subject"], axis=1)

        df = pd.DataFrame(x)
        df["Class"] = y

        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)

        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(HAR, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "HAR"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class CIFAR10(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])

        df = pd.DataFrame(x)
        df["Class"] = y
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(CIFAR10, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "CIFAR10"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class ImageNet(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        d1 = pd.read_feather("../data/Imagenet64_train_part1/df1.feather")
        d2 = pd.read_feather("../data/Imagenet64_train_part1/df2.feather")
        d3 = pd.read_feather("../data/Imagenet64_train_part1/df3.feather")
        d4 = pd.read_feather("../data/Imagenet64_train_part1/df4.feather")
        d5 = pd.read_feather("../data/Imagenet64_train_part1/df5.feather")

        df = pd.concat((d1,d2,d3,d4,d5))

        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(ImageNet, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "ImageNet64"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class Uniform(ChangeStream):
    def __init__(self, n, preprocess=None, max_len=None):
        self.n = n
        data = np.random.default_rng().uniform(size=(n,5))
        y = np.zeros(n)

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(Uniform, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "Uniform" + str(self.n)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class Laplace(ChangeStream):
    def __init__(self, n, scale=3, preprocess=None, max_len=None):
        self.n = n
        data = np.random.default_rng().laplace(scale=scale**2,size=(n,5))
        y = np.zeros(n)

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(Laplace, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "Laplace" + str(self.n)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class Mixed(ChangeStream):
    def __init__(self, n, scale=3, proba=0.3, preprocess=None, max_len=None):
        self.n = n
        alpha = np.random.default_rng().uniform(size=n)
        p1 = np.random.default_rng().normal(size=(n,5))
        data = np.random.default_rng().normal(scale=scale**2,size=(n,5))
        data[alpha < proba] = p1[alpha<proba]

        y = np.zeros(n)

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(Mixed, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "Mixed" + str(self.n)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class NormalToUniform(ChangeStream):
    def __init__(self, n1, n2, preprocess=None, max_len=None):
        self.n = n1+n2
        unif = np.random.default_rng().uniform(size=(n2,5))
        normal = np.random.default_rng().normal(size=(n1,5))
        y = np.hstack((np.zeros(n1),np.ones(n2)))
        data = np.vstack((normal,unif))

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(NormalToUniform, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "NormalToUnif" + str(self.n)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class NormalToLaplace(ChangeStream):
    def __init__(self, n1, n2, scale=3, preprocess=None, max_len=None):
        self.n = n1+n2
        laplace = np.random.default_rng().laplace(scale=scale**2,size=(n2,5))
        normal = np.random.default_rng().normal(size=(n1,5))
        y = np.hstack((np.zeros(n1),np.ones(n2)))
        data = np.vstack((normal,laplace))

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(NormalToLaplace, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "NormalToLaplace" + str(self.n)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

class NormalToMixed(ChangeStream):
    def __init__(self, n1, n2, scale=3, proba=0.3, preprocess=None, max_len=None):
        self.n = n1+n2
        normal = np.random.default_rng().normal(size=(n1,5))

        alpha = np.random.default_rng().uniform(size=n2)
        p1 = np.random.default_rng().normal(size=(n2,5))
        mixed = np.random.default_rng().normal(scale=scale**2,size=(n2,5))
        mixed[alpha < proba] = p1[alpha<proba]

        y = np.hstack((np.zeros(n1),np.ones(n2)))
        data = np.vstack((normal,mixed))

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(NormalToMixed, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "NormalToMixed" + str(self.n)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]
