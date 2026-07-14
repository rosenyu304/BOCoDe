import torch

from ...base import BenchmarkProblem


class TwoBarTruss(BenchmarkProblem):
    """
    S. S. Rao. Game theory approach for multiobjective structural optimization.
    Computers and Structures 26(1):119–127, 1987

    Reference point (DERIVED, not published): no reference point or approximated
    ideal/nadir point is published for this formulation. Derived with the standard
    convention (Tanabe & Ishibuchi, Sec. 3.1; Picard & Schiffmann, Sec. V-A)
    ``r = z_ideal + 1.1 * (z_nadir - z_ideal)`` in the minimization frame, where
    ``z_ideal`` / ``z_nadir`` are the min / max over the non-dominated set of a
    fixed Latin-hypercube sample (``problem.sample(2048, seed=0)``), then negated
    for BoCoDe's maximization frame. Derived values (minimization frame):
    ``z_ideal = (0.00974, 0.04605)``, ``z_nadir = (76.20809, 346.58270)``.

    KNOWN PROBLEM BUG (pre-existing, not fixed here): ``bounds`` are ``[(0, 1),
    (0, 1)]`` while constraints 3 and 4 require ``x1 >= 0.1`` and ``x2 >= 1.0``.
    ``x2 >= 1.0`` is only attainable exactly at the upper bound, so essentially no
    point in the box is feasible (0/512 feasible in a Latin-hypercube sample) and
    the feasible hypervolume is 0 for every algorithm. The bounds look like they
    should be ``[(0.1, ...), (1.0, ...)]``. Constraint 5 (``gx[:, 4]``) is also
    never assigned and stays 0.
    """

    available_dimensions = 2
    num_objectives = 2
    num_constraints = 5

    # 2D objective, 5 constraints, X = 2-by-dim

    tags = {"multi_objective", "constrained", "continuous", "2D"}

    def __init__(self):
        super().__init__(
            dim=2,
            num_objectives=2,
            num_constraints=5,
            bounds=[(0, 1)] * 2,
            ref_point=[-83.82792735, -381.23636827],
        )

    def _evaluate_implementation(self, X):
        X = super().scale(X)

        n = X.size(0)

        rho = 0.283
        h = 100
        P = 10000
        sigma_0 = 20000
        E = 30 * 10**6

        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_lower_bound = 0.1
        x2_lower_bound = 1.0

        fx = torch.zeros((n, self.num_objectives))
        # negate for maximization
        fx[:, 0] = -(2 * rho * h * x2 * (1 + x1**2) ** 0.5)
        fx[:, 1] = -(P * h * (1 + x1**2) ** 1.5 * (1 + x1**4) ** 0.5) / (
            2 * 2**0.5 * E * x1**2 * x2
        )

        gx = torch.zeros((n, self.num_constraints))
        gx[:, 0] = (P * (1 + x1) * (1 + x1**2) ** 0.5) / (
            2 * 2**0.5 * x1 * x2
        ) - sigma_0
        gx[:, 1] = (P * (x1 - 1) * (1 + x1**2) ** 0.5) / (
            2 * 2**0.5 * x1 * x2
        ) - sigma_0
        gx[:, 2] = x1_lower_bound - x1
        gx[:, 3] = x2_lower_bound - x2

        return gx, fx
