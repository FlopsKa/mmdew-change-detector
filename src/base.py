import time
from sklearn import preprocessing
import seaborn as sns

def sns_palette():
    palette_sns = sns.color_palette("Set1", n_colors=n)
    palette_sns = [palette_sns[i] for i in range(n)]
    palette = { k:v for k,v in zip(hues, palette_sns) }

def sns_markers(hues):
    """Use with `style=` keyword to make use of markers"""
    possible_markers = ["*", "d", "X", "P", "o", "v", "^", "p", ">", "<"]
    n = min(len(hues),len(possible_markers))
    markers = { k:v for k,v in  zip(hues[:n], possible_markers)}
    return {"markers" : markers}

    #return { "palette" : palette, "markers" : markers, "hue_order" : hues}

def preprocess(x):
    return preprocessing.minmax_scale(x)


class ContextTimer(object):
    """
    A class used to time an executation of a code snippet.
    Use it with with .... as ...
    For example,
        with ContextTimer() as t:
            # do something
        time_spent = t.secs
    From https://www.huyng.com/posts/python-performance-analysis
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        if self.verbose:
            print('elapsed time: %f ms' % (self.secs*1000))
