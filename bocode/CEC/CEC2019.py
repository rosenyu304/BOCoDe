"""
CEC 2019 Benchmark Functions

K. V. Price, N. H. Awad, M. Z. Ali, P. N. Suganthan, "Problem Definitions and Evaluation Criteria for the 100-Digit Challenge Special Session and Competition on Single Objective Numerical Optimization," Technical Report, Nanyang Technological University, Singapore, November 2018.

+-----+--------------------------------------------------------------+----+------------------+
| No. | Functions                                                    | D  | Bounds           |
+-----+--------------------------------------------------------------+----+------------------+
|  1  | Storn's Chebyshev Polynomial Fitting Problem                 |  9 | [-8192, 8192]    |
|  2  | Inverse Hilbert Matrix Problem                               | 16 | [-16384, 16384]  |
|  3  | Lennard-Jones Minimum Energy Cluster                         | 18 | [-4, 4]          |
|  4  | Rastrigin’s Function                                         | 10 | [-100, 100]      |
|  5  | Griewangk’s Function                                         | 10 | [-100, 100]      |
|  6  | Weierstrass Function                                         | 10 | [-100, 100]      |
|  7  | Modified Schwefel’s Function                                 | 10 | [-100, 100]      |
|  8  | Expanded Schaffer’s F6 Function                              | 10 | [-100, 100]      |
|  9  | Happy Cat Function                                           | 10 | [-100, 100]      |
| 10  | Ackley Function                                              | 10 | [-100, 100]      |
+-----+--------------------------------------------------------------+----+------------------+
"""

import opfunu

from .BaseOpfunuCEC import BaseOpfunuCEC


class CEC2019_p1(BaseOpfunuCEC):
    """
    Storn's Chebyshev Polynomial Fitting Problem
    """

    problem = opfunu.cec_based.cec2019.F12019
    available_dimensions = 9
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2019_p2(BaseOpfunuCEC):
    """
    Inverse Hilbert Matrix Problem
    """

    problem = opfunu.cec_based.cec2019.F22019
    available_dimensions = 16
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2019_p3(BaseOpfunuCEC):
    """
    Lennard-Jones Minimum Energy Cluster
    """

    problem = opfunu.cec_based.cec2019.F32019
    available_dimensions = 18
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2019_p4(BaseOpfunuCEC):
    """
    Rastrigin’s Function
    """

    problem = opfunu.cec_based.cec2019.F42019
    available_dimensions = 10
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2019_p5(BaseOpfunuCEC):
    """
    Griewangk’s Function
    """

    problem = opfunu.cec_based.cec2019.F52019
    available_dimensions = 10
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2019_p6(BaseOpfunuCEC):
    """
    Weierstrass Function
    """

    problem = opfunu.cec_based.cec2019.F62019
    available_dimensions = 10
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2019_p7(BaseOpfunuCEC):
    """
    Modified Schwefel’s Function
    """

    problem = opfunu.cec_based.cec2019.F72019
    available_dimensions = 10
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2019_p8(BaseOpfunuCEC):
    """
    Expanded Schaffer’s F6 Function
    """

    problem = opfunu.cec_based.cec2019.F82019
    available_dimensions = 10
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2019_p9(BaseOpfunuCEC):
    """
    Happy Cat Function
    """

    problem = opfunu.cec_based.cec2019.F92019
    available_dimensions = 10
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2019_p10(BaseOpfunuCEC):
    """
    Ackley Function
    """

    problem = opfunu.cec_based.cec2019.F102019
    available_dimensions = 10
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)
