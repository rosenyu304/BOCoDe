"""MODAct: Multi-Objective Design of electro-mechanical ACTuators.

Sign convention
---------------
MODAct returns each objective in its own natural direction, flagged by
``problem.weights`` (``-1`` minimize / ``+1`` maximize; see the ``modact`` README
and ``modact.interfaces.pymoo``). BoCoDe maximizes, so the *minimization*
objectives are negated here. (This is the mirror image of MODAct's own pymoo
adapter, which negates the maximization objectives to get an all-minimization
problem.)

Reference points (derived from the OFFICIAL best-known Pareto fronts)
--------------------------------------------------------------------
Picard & Schiffmann compute the hypervolume as follows (Section V-A): the front
is transformed to a minimization-only problem, normalized with the estimated
ideal ``z*`` and nadir ``z_nad`` collected "to provide per-problem front
normalization", and the HV is then taken against ``r = (1.1, ..., 1.1)``.

The paper does not print ``z*`` / ``z_nad``, but it does publish the best-known
Pareto fronts they are computed from (reference [42] of the paper):

    C. Picard and J. Schiffmann. Multi-Objective Design of Actuators: Pareto
    Fronts. Zenodo, version 1.1.0, May 2020. https://doi.org/10.5281/zenodo.3824302

The reference points below are therefore reconstructed from the official fronts,
using the paper's own recipe:

    A            = official ``<name>_PF.dat`` (natural frame, as ``problem(x)``
                   returns it), mapped to the minimization frame via ``-1 * w``
    z*_i         = min_{f in A} f_i
    z_nad_i      = max_{f in A} f_i
    r_i          = z*_i + 1.1 * (z_nad_i - z*_i)

``r`` is the un-normalized image of ``(1.1, ..., 1.1)``, so scoring against it
reproduces the paper's normalized HV up to the constant factor prod(z_nad - z*).

Keyed PER PROBLEM, not per objective family: the best-known front depends on the
constraint level as well as the objectives (the paper says the normalization is
"per-problem"), so e.g. CS1 and CS4 have different reference points.

Sources:
C. Picard and J. Schiffmann. Realistic Constrained Multi-Objective Optimization
Benchmark Problems from Design. IEEE Transactions on Evolutionary Computation
25(2):234-246, 2021. https://ieeexplore.ieee.org/document/9179777
https://github.com/epfl-lamd/modact

Search space (continuous, but piecewise-continuous response)
-----------------------------------------------------------
All 20 variables are continuous doubles, exactly as in the official pymoo adapter
(``vtype=np.double``). Do NOT declare any of them integer: MODAct packs a discrete
and a continuous quantity into a single variable via ``math.modf`` (see
``modact.util.create_actuator_from_x``). ``x0`` carries the motor index in its
integer part and the coil fill factor in its fractional part; each stage's first
two variables carry a gear tooth count (integer part) and a profile-shift
coefficient (fractional part). Rounding them would destroy the profile-shift
dimension and silently change the benchmark. The consequence is that objectives
and constraints are discontinuous across integer boundaries of 7 of the 20
variables -- this is inherent to the source, not an artefact of this wrapper.

Likewise, constraint index 7 (the 3-D collision constraint, present at constraint
levels 2/3/4, i.e. problems ``*2``/``*3``/``*4``) is quasi-discrete by
construction: ``modact.actuator.Actuator.internal_collisions`` returns
``(number of colliding mesh pairs) / (number of faces)``, a ratio of integer
counts that is 0 exactly when the design has no internal collision. It is a
step-like function in the source, not a rounding introduced here.
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

#: Problem -> HV reference point, in the ORIGINAL MINIMIZATION frame, computed
#: from the official best-known Pareto fronts (Zenodo 10.5281/zenodo.3824302) as
#: ``z* + 1.1 * (z_nad - z*)``. See the module docstring for the derivation.
_REF_POINT_MIN = {
    "CS1": [1.05958393, -8.47810402],
    "CS2": [1.40440601, -5.99616879],
    "CS3": [0.45454181, -12.59592776],
    "CS4": [0.90131334, -18.67311177],
    "CT1": [1.05089426, 0.85545936],
    "CT2": [1.06567993, 0.85709077],
    "CT3": [0.48475323, 0.76631995],
    "CT4": [1.02587117, 0.85740007],
    "CTS1": [1.88265842, 0.85810407, 42.23955453],
    "CTS2": [1.91544973, 0.85808384, 42.34471998],
    "CTS3": [1.06353849, 0.82486893, 10.31520340],
    "CTS4": [1.70041478, 0.85799142, 38.15696404],
    "CTSE1": [1.93059612, 0.85790543, 40.24526584, 0.01987067],
    "CTSE2": [2.00514792, 0.85796001, 40.19966872, 0.01775356],
    "CTSE3": [1.01650380, 0.79007123, 10.98092948, 0.01288903],
    "CTSE4": [1.67414955, 0.85628818, 26.46193406, 0.01428249],
    "CTSEI1": [2.10500194, 0.85792421, 45.00040889, 0.02435908, 487.58523267],
    "CTSEI2": [2.13799550, 0.85783628, 44.22073412, 0.02354771, 507.96320712],
    "CTSEI3": [1.08149921, 0.78571464, 11.15206796, 0.01398496, 289.53359505],
    "CTSEI4": [1.54361994, 0.85712121, 36.51275660, 0.01536087, 273.31055290],
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

        super().__init__(
            dim=dim,
            num_objectives=num_obj,
            num_constraints=num_cons,
            bounds=bounds,
            x_opt=x_opt,
            optimum=optimum,
            # Negated: BoCoDe maximizes, the reference point above is minimization.
            ref_point=[-r for r in _REF_POINT_MIN[type(self).__name__]],
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
