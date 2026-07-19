"""High-dimensional variants of the Rover trajectory-planning problem used by
GIT-BO (Figure 17), at dims {100, 200, 300, 400, 500}.

Each class fixes ``Rover``'s dimension; the objective and helpers are inherited
unchanged from :mod:`~bocode.opt_problems.engineering.Rover`.

Sources:
https://github.com/zi-w/Ensemble-Bayesian-Optimization/tree/4e6f9ed04833cc2e21b5906b1181bc067298f914
"""

from .Rover import Rover


class Rover_100D(Rover):
    """Rover trajectory planning fixed to 100 dimensions."""

    available_dimensions = 100

    def __init__(self):
        super().__init__(dim=100)


class Rover_200D(Rover):
    """Rover trajectory planning fixed to 200 dimensions."""

    available_dimensions = 200

    def __init__(self):
        super().__init__(dim=200)


class Rover_300D(Rover):
    """Rover trajectory planning fixed to 300 dimensions."""

    available_dimensions = 300

    def __init__(self):
        super().__init__(dim=300)


class Rover_400D(Rover):
    """Rover trajectory planning fixed to 400 dimensions."""

    available_dimensions = 400

    def __init__(self):
        super().__init__(dim=400)


class Rover_500D(Rover):
    """Rover trajectory planning fixed to 500 dimensions."""

    available_dimensions = 500

    def __init__(self):
        super().__init__(dim=500)
