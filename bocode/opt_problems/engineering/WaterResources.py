import torch

from ...base import BenchmarkProblem


class WaterResources(BenchmarkProblem):
    """
    K. Musselman and J. Talavage. A trade-off cut approach to multiple objective optimization.
    Operations Research 28(6):1424–1435, 1980

    The objectives AND the seven constraints are numerically identical to ``CRE51``,
    the constrained water-resource planning problem of Tanabe & Ishibuchi (their
    CRE5-3-1 / RE6-3-1, eqs. S.143-S.154 of the supplementary file), negated into
    BoCoDe's maximization frame (verified: ``WaterResources(x) == -CRE51(x)``).
    Three constraints previously deviated from that published definition (g2 was
    scaled by 0.1, g3 had ``- 4051.02`` instead of ``+ 4051.02``, and g6 used
    ``0.417 / (x1 * x2) ... - 136.52`` instead of ``0.417 * x1 * x2 ... - 136.54``);
    they now follow the published equations.

    Reference point (DERIVED from published RE data, not published for this exact
    formulation): CRE51's objectives are the first five of ``RE61``, whose
    approximated ideal/nadir points are published at
    https://github.com/ryojitanabe/reproblems/tree/master/ideal_nadir_points.
    Following the paper's convention ``r = z_ideal + 1.1 * (z_nadir - z_ideal)``
    (minimization frame), then negated for maximization.
    """

    available_dimensions = 3
    num_objectives = 5
    num_constraints = 7

    # 3D objective, 7 constraints, X = 7-by-dim

    tags = {"multi_objective", "constrained", "continuous", "3D"}

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=5,
            num_constraints=7,
            bounds=[(0.01, 0.45), (0.01, 0.10), (0.01, 0.10)],
            ref_point=[
                -82602.57638,
                -1482.0,
                -3110281.172,
                -7766172.841,
                -96522.77513,
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        fx = torch.zeros((n, self.num_objectives))
        # negate for maximization
        fx[:, 0] = -(106780.37 * x2 + 106780.37 * x3 + 61704.67)
        fx[:, 1] = -(3000 * x1)
        fx[:, 2] = -((305700 / (0.06 * 2289) ** 0.65) * 2289 * x2)
        fx[:, 3] = -(250 * 2289 * torch.e ** (-39.75 * x2 + 9.9 * x3 + 2.74))
        fx[:, 4] = -(25 * (1.39 / (x1 * x2) + 4940 * x3 - 80))

        gx = torch.zeros((n, self.num_constraints))
        gx[:, 0] = 0.00139 / (x1 * x2) + 4.94 * x3 - 0.08 - 1
        gx[:, 1] = 0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986 - 1
        gx[:, 2] = 12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02 - 50000
        gx[:, 3] = 2.098 / (x1 * x2) + 8046.33 * x3 - 696.71 - 16000
        gx[:, 4] = 2.138 / (x1 * x2) + 7883.39 * x3 - 705.04 - 10000
        gx[:, 5] = 0.417 * x1 * x2 + 1721.26 * x3 - 136.54 - 2000
        gx[:, 6] = 0.164 / (x1 * x2) + 631.13 * x3 - 54.48 - 550

        return gx, fx
