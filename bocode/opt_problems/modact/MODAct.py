"""MODAct: Multi-Objective Design of electro-mechanical ACTuators.

Sign convention
---------------
MODAct returns each objective in its own natural direction, flagged by
``problem.weights`` (``-1`` minimize / ``+1`` maximize; see the ``modact`` README
and ``modact.interfaces.pymoo``). BoCoDe maximizes, so the *minimization*
objectives are negated here. (This is the mirror image of MODAct's own pymoo
adapter, which negates the maximization objectives to get an all-minimization
problem.)

Reference points (DERIVED, not published)
-----------------------------------------
Picard & Schiffmann compute the hypervolume the same way Tanabe & Ishibuchi do:
the front is normalized with the estimated ideal ``z*`` and nadir ``z_nad`` of the
best-known Pareto front and the HV is then taken against ``r = (1.1, ..., 1.1)``
(Section V-A of the paper). But neither the paper nor
https://github.com/epfl-lamd/modact publishes the ``z*`` / ``z_nad`` vectors or the
best-known fronts, so the un-normalized reference point cannot be reconstructed
from the source.

The reference points below are therefore **derived, not published**. The same
recipe is applied to an approximated front obtained from a fixed, seeded
Latin-hypercube sample instead of from EMOA runs:

    A            = non-dominated set of ``problem.sample(2048, seed=0)``
                   (minimization frame, all sampled points, feasible or not)
    z*_i         = min_{x in A} f_i(x)
    z_nad_i      = max_{x in A} f_i(x)
    r_i          = z*_i + 1.1 * (z_nad_i - z*_i)

The 20 MODAct problems share only five distinct objective functions (CS, CT, CTS,
CTSE, CTSEI); the numbered variants differ only in their constraint set. The
reference point is therefore keyed on the objective family, so e.g. CS1..CS4 are
scored against the same reference point.

Sources:
C. Picard and J. Schiffmann. Realistic Constrained Multi-Objective Optimization
Benchmark Problems from Design. IEEE Transactions on Evolutionary Computation
25(2):234-246, 2021. https://ieeexplore.ieee.org/document/9179777
https://github.com/epfl-lamd/modact
"""

import torch

try:
    import modact.problems as pb
except ImportError as _exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "MODAct problems require the optional 'modact' dependency. "
        "Install it with: pip install 'bocode[modact]'"
    ) from _exc

from ...base import BenchmarkProblem

#: Objective-family -> derived HV reference point, in the ORIGINAL MINIMIZATION
#: frame. See the module docstring for the derivation. DERIVED, NOT PUBLISHED.
_REF_POINT_MIN = {
    "CS": [0.41829938, -144.54084625],
    "CT": [1.11193959, 1.20777957],
    "CTS": [1.17214278, 1.25492908, 219.06426423],
    "CTSE": [1.17214278, 1.25492908, 219.06426423, 0.01632634],
    "CTSEI": [1.17214278, 1.25925277, 219.06426423, 0.01707531, 227.20479530],
}


class BaseModactProblem(BenchmarkProblem):
    def __init_subclass__(subcls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        subcls.problem_name = subcls.__name__.lower()
        subcls.problem = pb.get_problem(subcls.problem_name)
        subcls.available_dimensions = len(subcls.problem.bounds()[0])
        subcls.num_objectives = len(subcls.problem.weights)
        subcls.num_constraints = len(subcls.problem.c_weights)

    def __init__(self, optimum=None, x_opt=None):
        bounds = list(zip(*self.problem.bounds(), strict=False))
        dim = len(self.problem.bounds()[0])
        num_obj = len(self.problem.weights)
        num_cons = len(self.problem.c_weights)

        # "CTSE3" -> family "CTSE"; the trailing digit only selects the constraints.
        family = type(self).__name__.rstrip("0123456789")

        super().__init__(
            dim=dim,
            num_objectives=num_obj,
            num_constraints=num_cons,
            bounds=bounds,
            x_opt=x_opt,
            optimum=optimum,
            # Negated: BoCoDe maximizes, the reference point above is minimization.
            ref_point=[-r for r in _REF_POINT_MIN[family]],
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        X = X.numpy()

        fx = torch.zeros((X.shape[0], self.num_objectives))
        gx = torch.zeros((X.shape[0], self.num_constraints))

        for i in range(X.shape[0]):
            f, g = self.problem(X[i, :])
            fx[i, :], gx[i, :] = torch.tensor(f), torch.tensor(g)

        for i, w in enumerate(self.problem.weights):
            # Objective weights: -1 --> minimization / 1 --> maximization
            # BoCoDe maximizes: negate the minimization objectives.
            if w == -1:
                fx[:, i] = -fx[:, i]

        for i, w in enumerate(self.problem.c_weights):
            # Constraints weights: -1 --> g(x) >= 0 / 1 --> g(x) <= 0
            # Convert everything to g(x) <= 0
            if w == -1:
                gx[:, i] = -gx[:, i]

        return gx, fx


class CS1(BaseModactProblem):
    pass


class CT1(BaseModactProblem):
    pass


class CTS1(BaseModactProblem):
    pass


class CTSE1(BaseModactProblem):
    pass


class CTSEI1(BaseModactProblem):
    pass


class CS2(BaseModactProblem):
    pass


class CT2(BaseModactProblem):
    pass


class CTS2(BaseModactProblem):
    pass


class CTSE2(BaseModactProblem):
    pass


class CTSEI2(BaseModactProblem):
    pass


class CS3(BaseModactProblem):
    pass


class CT3(BaseModactProblem):
    pass


class CTS3(BaseModactProblem):
    pass


class CTSE3(BaseModactProblem):
    pass


class CTSEI3(BaseModactProblem):
    pass


class CS4(BaseModactProblem):
    pass


class CT4(BaseModactProblem):
    pass


class CTS4(BaseModactProblem):
    pass


class CTSE4(BaseModactProblem):
    pass


class CTSEI4(BaseModactProblem):
    pass
