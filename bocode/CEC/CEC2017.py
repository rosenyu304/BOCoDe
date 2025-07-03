"""
CEC 2017 Benchmark Functions

G. Wu, R. Mallipeddi, and P. N. Suganthan. Problem definitions and evaluation criteria for the CEC 2017 competition on constrained real-parameter optimization. National University of Defense Technology, China, 2016.

+-----+--------------------------------------------------------------+-----+------------------+
| No. | Functions                                                    | D   | Bounds           |
+-----+--------------------------------------------------------------+-----+------------------+
|  1  | Shifted and Rotated Bent Cigar Function                      | 100 | [-100, 100]      |
|  2  | Shifted and Rotated Zakharov Function                        | 200 | [-100, 100]      |
|  3  | Shifted and Rotated Rosenbrock’s Function                    | 300 | [-100, 100]      |
|  4  | Shifted and Rotated Rastrigin’s Function                     | 400 | [-100, 100]      |
|  5  | Shifted and Rotated Expanded Scaffer’s F6 Function           | 500 | [-100, 100]      |
|  6  | Shifted and Rotated Lunacek Bi_Rastrigin Function            | 600 | [-100, 100]      |
|  7  | Shifted and Rotated Non-Continuous Rastrigin’s Function      | 700 | [-100, 100]      |
|  8  | Shifted and Rotated Levy Function                            | 800 | [-100, 100]      |
|  9  | Shifted and Rotated Schwefel’s Function                      | 900 | [-100, 100]      |
| 10  | Hybrid Function 1 (N=3)                                      |1000 | [-100, 100]      |
| 11  | Hybrid Function 2 (N=3)                                      |1100 | [-100, 100]      |
| 12  | Hybrid Function 3 (N=3)                                      |1200 | [-100, 100]      |
| 13  | Hybrid Function 4 (N=4)                                      |1300 | [-100, 100]      |
| 14  | Hybrid Function 5 (N=4)                                      |1400 | [-100, 100]      |
| 15  | Hybrid Function 6 (N=4)                                      |1500 | [-100, 100]      |
| 16  | Hybrid Function 6 (N=5)                                      |1600 | [-100, 100]      |
| 17  | Hybrid Function 6 (N=5)                                      |1700 | [-100, 100]      |
| 18  | Hybrid Function 6 (N=5)                                      |1800 | [-100, 100]      |
| 19  | Hybrid Function 6 (N=6)                                      |1900 | [-100, 100]      |
| 20  | Composition Function 1 (N=3)                                 |2000 | [-100, 100]      |
| 21  | Composition Function 2 (N=3)                                 |2100 | [-100, 100]      |
| 22  | Composition Function 3 (N=4)                                 |2200 | [-100, 100]      |
| 23  | Composition Function 4 (N=4)                                 |2300 | [-100, 100]      |
| 24  | Composition Function 5 (N=5)                                 |2400 | [-100, 100]      |
| 25  | Composition Function 6 (N=5)                                 |2500 | [-100, 100]      |
| 26  | Composition Function 7 (N=6)                                 |2600 | [-100, 100]      |
| 27  | Composition Function 8 (N=6)                                 |2700 | [-100, 100]      |
| 28  | Composition Function 9 (N=3)                                 |2800 | [-100, 100]      |
| 29  | Composition Function 10 (N=3)                                |2900 | [-100, 100]      |
+-----+--------------------------------------------------------------+-----+------------------+

"""

import opfunu

from .BaseOpfunuCEC import BaseOpfunuCEC


class CEC2017_p1(BaseOpfunuCEC):
    """
    Shifted and Rotated Bent Cigar Function
    """

    problem = opfunu.cec_based.cec2017.F12017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p2(BaseOpfunuCEC):
    """
    Shifted and Rotated Zakharov Function
    """

    problem = opfunu.cec_based.cec2017.F22017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p3(BaseOpfunuCEC):
    """
    Shifted and Rotated Rosenbrock’s Function
    """

    problem = opfunu.cec_based.cec2017.F32017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p4(BaseOpfunuCEC):
    """
    Shifted and Rotated Rastrigin’s Function
    """

    problem = opfunu.cec_based.cec2017.F42017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p5(BaseOpfunuCEC):
    """
    Shifted and Rotated Expanded Scaffer’s F6 Function
    """

    problem = opfunu.cec_based.cec2017.F52017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p6(BaseOpfunuCEC):
    """
    Shifted and Rotated Lunacek Bi_Rastrigin Function
    """

    problem = opfunu.cec_based.cec2017.F62017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p7(BaseOpfunuCEC):
    """
    Shifted and Rotated Non-Continuous Rastrigin’s Function
    """

    problem = opfunu.cec_based.cec2017.F72017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p8(BaseOpfunuCEC):
    """
    Shifted and Rotated Levy Function
    """

    problem = opfunu.cec_based.cec2017.F82017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p9(BaseOpfunuCEC):
    """
    Shifted and Rotated Schwefel’s Function
    """

    problem = opfunu.cec_based.cec2017.F92017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p10(BaseOpfunuCEC):
    """
    Hybrid Function 1 (N=3)
    """

    problem = opfunu.cec_based.cec2017.F102017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p11(BaseOpfunuCEC):
    """
    Hybrid Function 2 (N=3)
    """

    problem = opfunu.cec_based.cec2017.F112017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p12(BaseOpfunuCEC):
    """
    Hybrid Function 3 (N=3)
    """

    problem = opfunu.cec_based.cec2017.F122017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p13(BaseOpfunuCEC):
    """
    Hybrid Function 4 (N=4)
    """

    problem = opfunu.cec_based.cec2017.F132017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p14(BaseOpfunuCEC):
    """
    Hybrid Function 5 (N=4)
    """

    problem = opfunu.cec_based.cec2017.F142017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p15(BaseOpfunuCEC):
    """
    Hybrid Function 6 (N=4)
    """

    problem = opfunu.cec_based.cec2017.F152017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p16(BaseOpfunuCEC):
    """
    Hybrid Function 6 (N=5)
    """

    problem = opfunu.cec_based.cec2017.F162017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p17(BaseOpfunuCEC):
    """
    Hybrid Function 6 (N=5)
    """

    problem = opfunu.cec_based.cec2017.F172017
    available_dimensions = [30, 50, 100]
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p18(BaseOpfunuCEC):
    """
    Hybrid Function 6 (N=5)
    """

    problem = opfunu.cec_based.cec2017.F182017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p19(BaseOpfunuCEC):
    """
    Hybrid Function 6 (N=6)
    """

    problem = opfunu.cec_based.cec2017.F192017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p20(BaseOpfunuCEC):
    """
    Composition Function 1 (N=3)
    """

    problem = opfunu.cec_based.cec2017.F202017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p21(BaseOpfunuCEC):
    """
    Composition Function 2 (N=3)
    """

    problem = opfunu.cec_based.cec2017.F212017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p22(BaseOpfunuCEC):
    """
    Composition Function 3 (N=4)
    """

    problem = opfunu.cec_based.cec2017.F222017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p23(BaseOpfunuCEC):
    """
    Composition Function 4 (N=4)
    """

    problem = opfunu.cec_based.cec2017.F232017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p24(BaseOpfunuCEC):
    """
    Composition Function 5 (N=5)
    """

    problem = opfunu.cec_based.cec2017.F242017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p25(BaseOpfunuCEC):
    """
    Composition Function 6 (N=5)
    """

    problem = opfunu.cec_based.cec2017.F252017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p26(BaseOpfunuCEC):
    """
    Composition Function 7 (N=6)
    """

    problem = opfunu.cec_based.cec2017.F262017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p27(BaseOpfunuCEC):
    """
    Composition Function 8 (N=6)
    """

    problem = opfunu.cec_based.cec2017.F272017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p28(BaseOpfunuCEC):
    """
    Composition Function 9 (N=3)
    """

    problem = opfunu.cec_based.cec2017.F282017
    available_dimensions = problem().dim_supported
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)


class CEC2017_p29(BaseOpfunuCEC):
    """
    Composition Function 10 (N=3)
    """

    problem = opfunu.cec_based.cec2017.F292017
    available_dimensions = [30, 50, 100]
    num_constraints = 0

    def __init__(self, dim=None):
        super().__init__(dim=dim, num_objectives=1, num_constraints=0)
