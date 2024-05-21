from scipy import stats
import numpy as np
import torch


class LibNotMR(object):
    """
        Instead of using LibMR (https://github.com/abhijitbendale/OSDN/tree/master/libMR) we implemented
        the simple operations with Scipy. The output is checked against the original library for verification.
    """
    def __init__(self, tailsize=20):
        self.tailsize = tailsize
        self.min_val = None
        self.translation = 10000  # this constant comes from the library.
        # it only makes small numerical differences.
        # we keep it for authenticity.
        self.a = 1
        self.loc = 0
        self.c = None
        self.scale = None

    def fit_high(self, inputs):
        inputs = inputs.numpy()
        tailtofit = sorted(inputs)[-self.tailsize:]
        self.min_val = np.min(tailtofit)
        new_inputs = [i + self.translation - self.min_val for i in tailtofit]
        params = stats.exponweib.fit(new_inputs, floc=0, f0=1)
        self.c = params[1]
        self.scale = params[3]

    def w_score(self, inputs):
        new_inputs = inputs + self.translation - self.min_val
        new_score = stats.exponweib.cdf(new_inputs, a=self.a, c=self.c, loc=self.loc, scale=self.scale)
        return new_score

    def serialize(self):
        return torch.FloatTensor([self.min_val, self.c, self.scale])

    def deserialize(self, params):
        self.min_val = params[0].item()
        self.c = params[1].item()
        self.scale = params[2].item()

    def __str__(self):
        return 'Weib: C=%.2f scale=%.2f min_val=%.2f' % (self.c, self.scale, self.min_val)
