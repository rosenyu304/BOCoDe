from ..base import BenchmarkProblem, DataType


class EulerBernoulliBeamBending(BenchmarkProblem):
    """
    Cuesta Ramirez, J., Le Riche, R., Roustant, O. et al.
    (2022) A comparison of mixed-variables Bayesian optimization
    approaches. Adv. Model. and Simul. in Eng. Sci. 9, 6 .
    """

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    # 3D objective, 0 constraints, X = n-by-3

    tags = {"single_objective", "unconstrained", "mixed", "3D"}

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            optimum=[-1.287385e3],
            x_opt=[[0.0, 0.43, 0.380]],
            bounds=[(0, 1)] * 3,
            #  MIXED = MixIntConfig(is_mixed: True,
            #                       discrete_dict = {2: torch.tensor([0.083, 0.139, 0.380,
            #                                                         0.080, 0.133, 0.363,
            #                                                         0.086, 0.136, 0.360,
            #                                                         0.092, 0.138, 0.369])},
            #                       continuous_dict = {0: torch.tensor([0,1]),
            #                                          1: torch.tensor([0,1]),
            #                                         }
            #                      ),
        )

    def _evaluate_implementation(self, X):
        # X = super().scale(X, to_verify)

        # # x0: [0, 1]
        # # x1: [0, 1]
        # # x2: {0.083, 0.139, 0.380, 0.080, 0.133, 0.363, 0.086, 0.136, 0.360, 0.092, 0.138, 0.369}

        # if self.to_print_Xscaled:
        #     print(f'X: {X}')

        # BO comparison paper: https://amses-journal.springeropen.com/articles/10.1186/s40323-022-00218-8
        E = 600
        P = 600
        alpha = 60

        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

        L = 10 + 10 * x1
        S = 1 + x2
        I = x3

        D = P * L**3 / (3 * E * S**2 * I)
        y = D + alpha * L * S
        return None, -y.reshape(-1, 1)
