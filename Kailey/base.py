import torch
import numpy as np

class RangeException(Exception):
    pass

class DimensionException(Exception):
    pass

class BenchmarkProblem():

    """
    Base class for Bayesian Optimization benchmark problems.
    """

    def __init__(self, dim = 1, num_obj = 1, num_cons = 0, bounds = None, optimizers = [0], opt_value = 0, ref_point = None, to_verify = True, out_type = torch, tags = []):
        self.dim = dim
        self.num_obj = num_obj
        self.num_cons = num_cons
        self.bounds = bounds
        self.optimizers = optimizers
        self.opt_value = opt_value
        self.ref_point = ref_point
        self.to_verify = to_verify
        self.out_type = out_type
        self.tags = tags


    def scale(self, X, to_verify):
        """
        (Optionally) verifies that X is in the correct range [0, 1] and has the correct dimensions.
        Converts X to a torch.Tensor if necessary and scales X to the problem's bounds.

        Parameters:
            X (array, np.array, or torch.Tensor): data in range of [0, 1]

        Returns:
            X (Torch.tensor): data scaled to bounds

        """

        if not torch.is_tensor(X):
            X = torch.tensor(X)

        if to_verify:
            if X.size(1) != self.dim:
                raise DimensionException("Incorrect X dimensions.")
            if torch.max(X) >= 1 or torch.min(X) <= 0:
                raise RangeException("Incorrect X range: must be [0, 1].")

        if not torch.is_tensor(self.bounds):
            self.bounds = torch.tensor(self.bounds)
        X_scaled = torch.add(torch.mul(X, (self.bounds[:, 1] - self.bounds[:, 0])), self.bounds[:, 0])
        return X_scaled
