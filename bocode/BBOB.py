"""
https://coco-platform.org/testsuites/bbob/overview.html

"We present here 24 noise-free real-parameter single-objective benchmark functions (see (Hansen et al. 2009, 2016)
for our experimental setup and (Hansen et al. 2022) for our performance assessment methodology)."

Supported problem dimensions: 2,3,5,10,20,40

bbob-constrained: 54 functions

The bbob-mixint suite contains 24 mixed-integer functions. When instantiated with six dimensions (n=5,10,20,40,80,160)
and 15 instances per function, this results in 2160 problem instances in total.

"""

import cocoex
import numpy as np
import torch

from .base import BenchmarkProblem, DataType


class BaseBBOB(BenchmarkProblem):
    """
    N. Hansen, A. Auger, R. Ros, O. Mersmann, T. Tušar, D. Brockhoff.
    COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting,
    Optimization Methods and Software, 36(1), pp. 114-144, 2021.
    """

    input_type = DataType.CONTINUOUS

    def __init__(self, dim, suite, function_number, instance_number):
        problem = cocoex.Suite(
            suite, "", ""
        ).get_problem_by_function_dimension_instance(
            function_number, dim, instance_number
        )
        num_objectives = problem.number_of_objectives
        num_constraints = problem.number_of_constraints

        lower_bounds = torch.tensor(problem.lower_bounds)
        upper_bounds = torch.tensor(problem.upper_bounds)
        bounds = torch.cat(
            (lower_bounds.unsqueeze(-1), upper_bounds.unsqueeze(-1)), dim=1
        )

        x_opt = (
            torch.tensor(problem._best_parameter())
            if problem._best_parameter() is not None
            else None
        )
        optimum = (
            torch.tensor(problem.best_observed_fvalue1)
            if problem.best_observed_fvalue1 is not None
            else None
        )

        super().__init__(
            dim=dim,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            bounds=bounds,
            x_opt=x_opt,
            optimum=optimum,
        )

        self.problem = problem
        self.suite = suite
        self.function_number = function_number
        self.instance_number = instance_number

    def _evaluate_implementation(self, X: torch.Tensor, scaling=False):
        if scaling:
            X = super().scale(X)

        X_np = X.detach().cpu().numpy()
        fx = []
        gx = []

        fx = torch.from_numpy(np.array([self.problem(x) for x in X_np]))

        if self.problem.number_of_constraints > 0:
            gx = torch.from_numpy(np.array([self.problem.constraint(x) for x in X_np]))
        else:
            gx = None

        if self.num_objectives == 1:
            fx = fx.unsqueeze(-1)

        return gx, fx


class BBOB(BaseBBOB):
    available_dimensions = {2, 3, 5, 10, 20, 40}
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        dim=2,
        function_number=1,
        instance_number=1,
    ):
        """
        Available Function Numbers:

        Separable Functions
            f1 - Sphere Function
            f2 - Separable Ellipsoidal Function
            f3 - Rastrigin Function
            f4 - Büche-Rastrigin Function
            f5 - Linear Slope

        Functions with Low or Moderate Conditioning
            f6 - Attractive Sector Function
            f7 - Step Ellipsoidal Function
            f8 - Rosenbrock Function (original)
            f9 - Rosenbrock Function (rotated)

        Functions with High Conditioning and Unimodal
            f10 - Ellipsoidal Function
            f11 - Discus Function
            f12 - Bent Cigar Function
            f13 - Sharp Ridge Function
            f14 - Different Powers Function

        Multi-modal Functions with Adequate Global Structure
            f15 - Rastrigin Function
            f16 - Weierstrass Function
            f17 - Schaffer's F7 Function
            f18 - Schaffer's F7 Function (moderately ill-conditioned)
            f19 - Composite Griewank-Rosenbrock Function F8F2

        Multi-modal Functions with Weak Global Structure
            f20 - Schwefel Function
            f21 - Gallagher's Gaussian 101-me Peaks Function
            f22 - Gallagher's Gaussian 21-hi Peaks Function
            f23 - Katsuura Function
            f24 - Lunacek bi-Rastrigin Function
        """
        suite = "bbob"
        super().__init__(dim, suite, function_number, instance_number)


class BBOB_Biobj(BaseBBOB):
    available_dimensions = {2, 3, 5, 10, 20, 40}
    num_objectives = 2
    num_constraints = 0

    def __init__(
        self,
        dim=2,
        function_number=1,
        instance_number=1,
    ):
        """
        Available Function Numbers:
            f1 - Sphere/Sphere
            f2 - Sphere/Ellipsoid separable
            f3 - Sphere/Attractive sector
            f4 - Sphere/Rosenbrock original
            f5 - Sphere/Sharp ridge
            f6 - Sphere/Sum of different powers
            f7 - Sphere/Rastrigin
            f8 - Sphere/Schaffer F7, condition 10
            f9 - Sphere/Schwefel
            f10 - Sphere/Gallagher 101 peaks
            f11 - Ellipsoid separable/Ellipsoid separable
            f12 - Ellipsoid separable/Attractive sector
            f13 - Ellipsoid separable/Rosenbrock original
            f14 - Ellipsoid separable/Sharp ridge
            f15 - Ellipsoid separable/Sum of different powers
            f16 - Ellipsoid separable/Rastrigin
            f17 - Ellipsoid separable/Schaffer F7, condition 10
            f18 - Ellipsoid separable/Schwefel
            f19 - Ellipsoid separable/Gallagher 101 peaks
            f20 - Attractive sector/Attractive sector
            f21 - Attractive sector/Rosenbrock original
            f22 - Attractive sector/Sharp ridge
            f23 - Attractive sector/Sum of different powers
            f24 - Attractive sector/Rastrigin
            f25 - Attractive sector/Schaffer F7, condition 10
            f26 - Attractive sector/Schwefel
            f27 - Attractive sector/Gallagher 101 peaks
            f28 - Rosenbrock original/Rosenbrock original
            f29 - Rosenbrock original/Sharp ridge
            f30 - Rosenbrock original/Sum of different powers
            f31 - Rosenbrock original/Rastrigin
            f32 - Rosenbrock original/Schaffer F7, condition 10
            f33 - Rosenbrock original/Schwefel
            f34 - Rosenbrock original/Gallagher 101 peaks
            f35 - Sharp ridge/Sharp ridge
            f36 - Sharp ridge/Sum of different powers
            f37 - Sharp ridge/Rastrigin
            f38 - Sharp ridge/Schaffer F7, condition 10
            f39 - Sharp ridge/Schwefel
            f40 - Sharp ridge/Gallagher 101 peaks
            f41 - Sum of different powers/Sum of different powers
            f42 - Sum of different powers/Rastrigin
            f43 - Sum of different powers/Schaffer F7, condition 10
            f44 - Sum of different powers/Schwefel
            f45 - Sum of different powers/Gallagher 101 peaks
            f46 - Rastrigin/Rastrigin
            f47 - Rastrigin/Schaffer F7, condition 10
            f48 - Rastrigin/Schwefel
            f49 - Rastrigin/Gallagher 101 peaks
            f50 - Schaffer F7, condition 10/Schaffer F7, condition 10
            f51 - Schaffer F7, condition 10/Schwefel
            f52 - Schaffer F7, condition 10/Gallagher 101 peaks
            f53 - Schwefel/Schwefel
            f54 - Schwefel/Gallagher 101 peaks
            f55 - Gallagher 101 peaks/Gallagher 101 peaks
            f56 - Sphere/Rastrigin separable
            f57 - Sphere/Rastrigin-Büche
            f58 - Sphere/Linear slope
            f59 - Separable Ellipsoid/Separable Rastrigin
            f60 - Separable Ellipsoid/Büche-Rastrigin
            f61 - Separable Ellipsoid/Linear Slope
            f62 - Separable Rastrigin/Büche-Rastrigin
            f63 - Separable Rastrigin/Linear Slope
            f64 - Büche-Rastrigin/Linear slope
            f65 - Attractive Sector/Step-ellipsoid
            f66 - Attractive Sector/rotated Rosenbrock
            f67 - Step-ellipsoid/separable Rosenbrock
            f68 - Step-ellipsoid/rotated Rosenbrock
            f69 - separable Rosenbrock/rotated Rosenbrock
            f70 - Ellipsoid/Discus
            f71 - Ellipsoid/Bent Cigar
            f72 - Ellipsoid/Sharp Ridge
            f73 - Ellipsoid/Sum of different powers
            f74 - Discus/Bent Cigar
            f75 - Discus/Sharp Ridge
            f76 - Discus/Sum of different powers
            f77 - Bent Cigar/Sharp Ridge
            f78 - Bent Cigar/Sum of different powers
            f79 - Rastrigin/Schaffer F7 with conditioning of 1000
            f80 - Rastrigin/Griewank-Rosenbrock
            f81 - Schaffer F7/Schaffer F7 with conditioning 1000
            f82 - Schaffer F7/Griewank-Rosenbrock
            f83 - Schaffer F7 with conditioning 1000/Griewank-Rosenbrock
            f84 - Schwefel/Gallagher 21
        """

        suite = "bbob-biobj"
        super().__init__(dim, suite, function_number, instance_number)


class BBOB_BiobjMixInt(BaseBBOB):
    available_dimensions = {2, 3, 5, 10, 20, 40}
    num_objectives = 2
    num_constraints = 0

    def __init__(
        self,
        dim=5,
        function_number=1,
        instance_number=1,
    ):
        suite = "bbob-biobj-mixint"
        super().__init__(dim, suite, function_number, instance_number)


class BBOB_Boxed(BaseBBOB):
    available_dimensions = {2, 3, 5, 10, 20, 40}
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        dim=2,
        function_number=1,
        instance_number=1,
    ):
        suite = "bbob-boxed"
        super().__init__(dim, suite, function_number, instance_number)


class BBOB_Constrained(BaseBBOB):
    available_dimensions = {2, 3, 5, 10, 20, 40}
    num_objectives = 1
    num_constraints = (1, None)

    def __init__(
        self,
        dim=2,
        function_number=1,
        instance_number=1,
    ):
        """
        Each function is a constrained optimization problem built by combining an objective function
        with varying numbers of nonlinear constraints and transformations.

        Available Function Numbers:

        f1 - Sphere
            Objective: Sphere
            Constraints: 1 constraint
            Transformation: id
            Scaling Factor: 1

        f2 - Sphere
            Objective: Sphere
            Constraints: 3 constraints
            Transformation: id
            Scaling Factor: 1

        f3 - Sphere
            Objective: Sphere
            Constraints: 9 constraints
            Transformation: id
            Scaling Factor: 1

        f4 - Sphere
            Objective: Sphere
            Constraints: 9 + ⌊3n/4⌋ constraints
            Transformation: id
            Scaling Factor: 1

        f5 - Sphere
            Objective: Sphere
            Constraints: 9 + ⌊3n/2⌋ constraints
            Transformation: id
            Scaling Factor: 1

        f6 - Sphere
            Objective: Sphere
            Constraints: 9 + ⌊9n/2⌋ constraints
            Transformation: id
            Scaling Factor: 1

        f7 - Separable Ellipsoid
            Objective: Separable Ellipsoid
            Constraints: 1 constraint
            Transformation: Tosz
            Scaling Factor: 1e-4

        f8 - Separable Ellipsoid
            Objective: Separable Ellipsoid
            Constraints: 3 constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f9 - Separable Ellipsoid
            Objective: Separable Ellipsoid
            Constraints: 9 constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f10 - Separable Ellipsoid
            Objective: Separable Ellipsoid
            Constraints: 9 + ⌊3n/4⌋ constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f11 - Separable Ellipsoid
            Objective: Separable Ellipsoid
            Constraints: 9 + ⌊3n/2⌋ constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f12 - Separable Ellipsoid
            Objective: Separable Ellipsoid
            Constraints: 9 + ⌊9n/2⌋ constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f13 - Linear Slope
            Objective: Linear Slope
            Constraints: 1 constraint
            Transformation: id
            Scaling Factor: 1

        f14 - Linear Slope
            Objective: Linear Slope
            Constraints: 3 constraints
            Transformation: id
            Scaling Factor: 1

        f15 - Linear Slope
            Objective: Linear Slope
            Constraints: 9 constraints
            Transformation: id
            Scaling Factor: 1

        f16 - Linear Slope
            Objective: Linear Slope
            Constraints: 9 + ⌊3n/4⌋ constraints
            Transformation: id
            Scaling Factor: 1

        f17 - Linear Slope
            Objective: Linear Slope
            Constraints: 9 + ⌊3n/2⌋ constraints
            Transformation: id
            Scaling Factor: 1

        f18 - Linear Slope
            Objective: Linear Slope
            Constraints: 9 + ⌊9n/2⌋ constraints
            Transformation: id
            Scaling Factor: 1

        f19 - Rotated Ellipsoid
            Objective: Rotated Ellipsoid
            Constraints: 1 constraint
            Transformation: Tosz
            Scaling Factor: 1e-4

        f20 - Rotated Ellipsoid
            Objective: Rotated Ellipsoid
            Constraints: 3 constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f21 - Rotated Ellipsoid
            Objective: Rotated Ellipsoid
            Constraints: 9 constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f22 - Rotated Ellipsoid
            Objective: Rotated Ellipsoid
            Constraints: 9 + ⌊3n/4⌋ constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f23 - Rotated Ellipsoid
            Objective: Rotated Ellipsoid
            Constraints: 9 + ⌊3n/2⌋ constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f24 - Rotated Ellipsoid
            Objective: Rotated Ellipsoid
            Constraints: 9 + ⌊9n/2⌋ constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f25 - Discus
            Objective: Discus
            Constraints: 1 constraint
            Transformation: Tosz
            Scaling Factor: 1e-4

        f26 - Discus
            Objective: Discus
            Constraints: 3 constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f27 - Discus
            Objective: Discus
            Constraints: 9 constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f28 - Discus
            Objective: Discus
            Constraints: 9 + ⌊3n/4⌋ constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f29 - Discus
            Objective: Discus
            Constraints: 9 + ⌊3n/2⌋ constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f30 - Discus
            Objective: Discus
            Constraints: 9 + ⌊9n/2⌋ constraints
            Transformation: Tosz
            Scaling Factor: 1e-4

        f31 - Bent Cigar
            Objective: Bent Cigar
            Constraints: 1 constraint
            Transformation: Tβ_asy (β=0.5)
            Scaling Factor: 1e-4

        f32 - Bent Cigar
            Objective: Bent Cigar
            Constraints: 3 constraints
            Transformation: Tβ_asy (β=0.5)
            Scaling Factor: 1e-4

        f33 - Bent Cigar
            Objective: Bent Cigar
            Constraints: 9 constraints
            Transformation: Tβ_asy (β=0.5)
            Scaling Factor: 1e-4

        f34 - Bent Cigar
            Objective: Bent Cigar
            Constraints: 9 + ⌊3n/4⌋ constraints
            Transformation: Tβ_asy (β=0.5)
            Scaling Factor: 1e-4

        f35 - Bent Cigar
            Objective: Bent Cigar
            Constraints: 9 + ⌊3n/2⌋ constraints
            Transformation: Tβ_asy (β=0.5)
            Scaling Factor: 1e-4

        f36 - Bent Cigar
            Objective: Bent Cigar
            Constraints: 9 + ⌊9n/2⌋ constraints
            Transformation: Tβ_asy (β=0.5)
            Scaling Factor: 1e-4

        f37 - Different Powers
            Objective: Different Powers
            Constraints: 1 constraint
            Transformation: id
            Scaling Factor: 1e-2

        f38 - Different Powers
            Objective: Different Powers
            Constraints: 3 constraints
            Transformation: id
            Scaling Factor: 1e-2

        f39 - Different Powers
            Objective: Different Powers
            Constraints: 9 constraints
            Transformation: id
            Scaling Factor: 1e-2

        f40 - Different Powers
            Objective: Different Powers
            Constraints: 9 + ⌊3n/4⌋ constraints
            Transformation: id
            Scaling Factor: 1e-2

        f41 - Different Powers
            Objective: Different Powers
            Constraints: 9 + ⌊3n/2⌋ constraints
            Transformation: id
            Scaling Factor: 1e-2

        f42 - Different Powers
            Objective: Different Powers
            Constraints: 9 + ⌊9n/2⌋ constraints
            Transformation: id
            Scaling Factor: 1e-2

        f43 - Separable Rastrigin
            Objective: Separable Rastrigin
            Constraints: 1 constraint
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f44 - Separable Rastrigin
            Objective: Separable Rastrigin
            Constraints: 3 constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f45 - Separable Rastrigin
            Objective: Separable Rastrigin
            Constraints: 9 constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f46 - Separable Rastrigin
            Objective: Separable Rastrigin
            Constraints: 9 + ⌊3n/4⌋ constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f47 - Separable Rastrigin
            Objective: Separable Rastrigin
            Constraints: 9 + ⌊3n/2⌋ constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f48 - Separable Rastrigin
            Objective: Separable Rastrigin
            Constraints: 9 + ⌊9n/2⌋ constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f49 - Rotated Rastrigin
            Objective: Rotated Rastrigin
            Constraints: 1 constraint
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f50 - Rotated Rastrigin
            Objective: Rotated Rastrigin
            Constraints: 3 constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f51 - Rotated Rastrigin
            Objective: Rotated Rastrigin
            Constraints: 9 constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f52 - Rotated Rastrigin
            Objective: Rotated Rastrigin
            Constraints: 9 + ⌊3n/4⌋ constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f53 - Rotated Rastrigin
            Objective: Rotated Rastrigin
            Constraints: 9 + ⌊3n/2⌋ constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        f54 - Rotated Rastrigin
            Objective: Rotated Rastrigin
            Constraints: 9 + ⌊9n/2⌋ constraints
            Transformation: Tβ_asy ∘ Tosz
            Scaling Factor: 10

        Notes:
        - "id" means identity transformation.
        - "T" is a generic non-linear transformation applied after initial scaling.
        - "Tβ_asy" and "Tosz" are defined as coordinate-wise asymmetry and oscillation transforms.
        """

        suite = "bbob-constrained"
        super().__init__(dim, suite, function_number, instance_number)


class BBOB_LargeScale(BaseBBOB):
    available_dimensions = {20, 40, 80, 160, 320, 640}
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        dim=20,
        function_number=1,
        instance_number=1,
    ):
        """
        Available Function Numbers:

        Separable Functions
        --------------------------------
        f1 - Sphere Function
            f(x) = γ(n) * Σ(z_i^2) + fopt
            z = x - xopt

        f2 - Ellipsoidal Function
            f(x) = γ(n) * Σ(10^(6*(i-1)/(n-1)) * z_i^2) + fopt
            z = Tosz(x - xopt)

        f3 - Rastrigin Function
            f(x) = γ(n) * [10n - 10Σcos(2πz_i) + ||z||²] + fopt
            z = Λ₁₀ T_asy^0.2(Tosz(x - xopt))

        f4 - Bueche-Rastrigin Function
            f(x) = γ(n) * [10n - 10Σcos(2πz_i) + ||z||²] + 100*fpen(x) + fopt
            z_i = s_i * Tosz(x_i - xopt_i), s_i depends on sign/oddness

        f5 - Linear Slope
            f(x) = γ(n) * Σ(5|s_i| - s_i * z_i) + fopt
            z_i, s_i depend on xopt

        Low or Moderate Conditioning
        -------------------------------------
        f6 - Attractive Sector Function
            f(x) = Tosz(γ(n) * Σ(s_i*z_i)^2)^0.9 + fopt
            z = QΛ₁₀R(x - xopt)

        f7 - Step Ellipsoidal Function
            f(x) = γ(n) * 0.1 * max(|ẑ1|/1e4, Σ10^(2*(i-1)/(n-1))*z_i^2) + fpen(x) + fopt
            ẑ = Λ₁₀R(x - xopt), z̃_i = step(ẑ_i), z = Qz̃

        f8 - Rosenbrock, Original
            f(x) = γ(n) * Σ[100*(z_i² - z_{i+1})² + (z_i - 1)²] + fopt
            z = max(1, sqrt(s)/8) * (x - xopt) + 1

        f9 - Rosenbrock, Rotated
            f(x) = γ(n) * Σ[100*(z_i² - z_{i+1})² + (z_i - 1)²] + fopt
            z = max(1, sqrt(s)/8) * R(x - xopt) + 1

        High Conditioning, Unimodal
        ------------------------------------
        f10 - Ellipsoidal (Rotated)
            f(x) = γ(n) * Σ(10^(6*(i-1)/(n-1)) * z_i^2) + fopt
            z = Tosz(R(x - xopt))

        f11 - Discus
            f(x) = γ(n) * [1e6 * Σ(z_i^2 for i in 1..k) + Σ(z_i^2 for i in k+1..n)] + fopt
            z = Tosz(R(x - xopt)), k = ceil(n/40)

        f12 - Bent Cigar
            f(x) = γ(n) * [Σ(z_i^2 for i in 1..k) + 1e6 * Σ(z_i^2 for i in k+1..n)] + fopt
            z = R T_asy^0.5(R(x - xopt))

        f13 - Sharp Ridge
            f(x) = γ(n) * [Σ(z_i^2 for i in 1..k) + 100 * sqrt(Σ(z_i^2 for i in k+1..n))] + fopt
            z = QΛ₁₀R(x - xopt)

        f14 - Different Powers
            f(x) = γ(n) * Σ(|z_i|^(2 + 4*(i-1)/(n-1))) + fopt
            z = R(x - xopt)

        Multi-modal with Adequate Global Structure
        ---------------------------------------------------
        f15 - Rastrigin (Rotated)
            f(x) = γ(n) * [10n - 10Σcos(2πz_i) + ||z||²] + fopt
            z = RΛ₁₀Q T_asy^0.2(Tosz(R(x - xopt)))

        f16 - Weierstrass
            f(x) = 10 * [mean Σ(1/2^k * cos(2π3^k(z_i+0.5))) - f0]^3 + 10/n*fpen(x) + fopt
            z = RΛ₁/₁₀₀Q Tosz(R(x - xopt))

        f17 - Schaffers F7
            f(x) = [1/(n−1) * Σ(sqrt(s_i) + sqrt(s_i) * sin²(50*s_i^1/5))]^2 + 10*fpen(x) + fopt
            z = Λ₁₀Q T_asy^0.5(R(x - xopt)), s_i = sqrt(z_i² + z_{i+1}²)

        f18 - Schaffers F7 (mod. ill-conditioned)
            Same as f17 but uses Λ₁₀₀₀

        f19 - Composite Griewank-Rosenbrock F8F2
            f(x) = 10/(n−1) * Σ(s_i/4000 - cos(s_i)) + 10 + fopt
            z = max(1, sqrt(s)/8) * R(x) + 0.5, s_i = 100*(z_i² - z_{i+1})² + (z_i - 1)²

        Multi-modal with Weak Global Structure
        -----------------------------------------------
        f20 - Schwefel
            f(x) = -1/(100n) * Σ(z_i * sin(√|z_i|)) + 4.189... + 100*fpen(z/100) + fopt
            z derived from x and xopt through several transformations

        f21 - Gallagher 101 Peaks
            f(x) = Tosz(10 - max_i w_i * exp(-0.5/n * (z - y_i)^T B^T C_i B (z - y_i)))^2 + fpen(x) + fopt
            101 Gaussian peaks, B is block-diagonal

        f22 - Gallagher 21 Peaks
            Similar to f21 but with 21 Gaussian peaks

        f23 - Katsuura
            f(x) = [10 / n^2 * Π(1 + i * Σ|2^j z_i - [2^j z_i]| / 2^j)^10/n^1.2 - 10/n^2] + fpen(x) + fopt
            z = QΛ₁₀₀R(x - xopt)

        f24 - Lunacek bi-Rastrigin
            f(x) = γ(n) * [min(||x̂ - μ0||², n + s * ||x̂ - μ1||²) + 10(n - Σcos(2πz_i))] + 1e4*fpen(x) + fopt
            x̂ = 2sign(xopt) ⊗ x, z = QΛ₁₀₀R(x̂ - μ0)
        """

        suite = "bbob-largescale"
        super().__init__(dim, suite, function_number, instance_number)


class BBOB_MixInt(BaseBBOB):
    available_dimensions = {5, 10, 20, 40, 80, 160}
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        dim=5,
        function_number=1,
        instance_number=1,
    ):
        suite = "bbob-mixint"
        super().__init__(dim, suite, function_number, instance_number)


class BBOB_Noisy(BaseBBOB):
    available_dimensions = {2, 3, 5, 10, 20, 40}
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        dim=2,
        function_number=101,
        instance_number=1,
    ):
        suite = "bbob-noisy"
        super().__init__(dim, suite, function_number, instance_number)


__all__ = [
    "BBOB",
    "BBOB_Biobj",
    "BBOB_BiobjMixInt",
    "BBOB_Boxed",
    "BBOB_Constrained",
    "BBOB_LargeScale",
    "BBOB_MixInt",
    "BBOB_Noisy",
]
