import torch
import numpy as np
from base import BenchmarkProblem

class MODAct(BenchmarkProblem):

    r'''
    C. Picard and J. Schiffmann, “Realistic Constrained Multi-Objective
    Optimization Benchmark Problems from Design,” IEEE Transactions on
    Evolutionary Computation, pp. 1–1, 2020, doi: 10.1109/TEVC.2020.3020046.
    '''

    tags = {"single_objective", "multi_objective", "constrained", "continuous", "extra_imports", "20D"}

    def __init__(self, problem_name, single_objective = False):
        import modact.modact.problems as pb
        self.prob = pb.get_problem(problem_name)
        xl, xu = self.prob.bounds()
        num_obj = None
        if not single_objective:
            num_obj = len(self.prob.weights)
        else:
            num_obj = 1
        super().__init__(dim = len(xl), num_obj = num_obj, num_cons = len(self.prob.c_weights), bounds = [[xl[i], xu[i]] for i in range(len(xl))])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        FX = torch.zeros((X.shape[0],self.num_obj))
        GX = torch.zeros((X.shape[0],self.num_cons))

        for ii in range(X.shape[0]):
            f, g = self.prob(X[ii,:].numpy())

            # Pymoo: f = np.array(f)*-1*cs1.weights
            f = np.array(f)*self.prob.weights # BO is maximizing
            g = np.array(g)*self.prob.c_weights
            if self.num_obj > 1:
                FX[ii, :] = torch.tensor(f)
            else:
                FX[ii,0] = torch.tensor(f[0])
            GX[ii,:] = torch.from_numpy(g)

        return GX, FX
