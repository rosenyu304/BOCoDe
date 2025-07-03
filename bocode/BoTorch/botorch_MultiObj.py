import botorch
import botorch.test_functions.multi_objective

from .BaseBotorch import MultiObjBotorchProblem


class BraninCurrin(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.BraninCurrin().dim
    num_objectives = (
        botorch.test_functions.multi_objective.BraninCurrin().num_objectives
    )
    num_constraints = (
        botorch.test_functions.multi_objective.BraninCurrin().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.BraninCurrin(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.BraninCurrin,
        )


class DH1(MultiObjBotorchProblem):
    available_dimensions = (2, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DH1(dim=3).num_constraints
        if hasattr(botorch.test_functions.multi_objective.DH1(dim=3), "num_constraints")
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DH1, dim=dim
        )


class DH2(MultiObjBotorchProblem):
    available_dimensions = (2, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DH2(dim=2).num_constraints
        if hasattr(botorch.test_functions.multi_objective.DH2(dim=2), "num_constraints")
        else 0
    )

    def __init__(self, dim=2):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DH2, dim=dim
        )


class DH3(MultiObjBotorchProblem):
    available_dimensions = (3, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DH3(dim=3).num_constraints
        if hasattr(botorch.test_functions.multi_objective.DH3(dim=3), "num_constraints")
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DH3, dim=dim
        )


class DH4(MultiObjBotorchProblem):
    available_dimensions = (3, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DH4(dim=3).num_constraints
        if hasattr(botorch.test_functions.multi_objective.DH4(dim=3), "num_constraints")
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DH4, dim=dim
        )


class DTLZ1(MultiObjBotorchProblem):
    available_dimensions = (3, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DTLZ1(dim=3).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.DTLZ1(dim=3), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DTLZ1, dim=dim
        )


class DTLZ2(MultiObjBotorchProblem):
    available_dimensions = (3, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DTLZ2(dim=3).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.DTLZ2(dim=3), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DTLZ2, dim=dim
        )


class DTLZ3(MultiObjBotorchProblem):
    available_dimensions = (3, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DTLZ3(dim=3).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.DTLZ3(dim=3), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DTLZ3, dim=dim
        )


class DTLZ4(MultiObjBotorchProblem):
    available_dimensions = (3, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DTLZ4(dim=3).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.DTLZ4(dim=3), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DTLZ4, dim=dim
        )


class DTLZ5(MultiObjBotorchProblem):
    available_dimensions = (3, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DTLZ5(dim=3).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.DTLZ5(dim=3), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DTLZ5, dim=dim
        )


class DTLZ7(MultiObjBotorchProblem):
    available_dimensions = (3, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.DTLZ7(dim=3).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.DTLZ7(dim=3), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DTLZ7, dim=dim
        )


class GMM(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.GMM().dim
    num_objectives = botorch.test_functions.multi_objective.GMM().num_objectives
    num_constraints = (
        botorch.test_functions.multi_objective.GMM().num_constraints
        if hasattr(botorch.test_functions.multi_objective.GMM(), "num_constraints")
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.GMM,
        )


class Penicillin(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.Penicillin().dim
    num_objectives = botorch.test_functions.multi_objective.Penicillin().num_objectives
    num_constraints = (
        botorch.test_functions.multi_objective.Penicillin().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.Penicillin(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.Penicillin,
        )


class ToyRobust(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.ToyRobust().dim
    num_objectives = botorch.test_functions.multi_objective.ToyRobust().num_objectives
    num_constraints = (
        botorch.test_functions.multi_objective.ToyRobust().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.ToyRobust(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.ToyRobust,
        )


class VehicleSafety(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.VehicleSafety().dim
    num_objectives = (
        botorch.test_functions.multi_objective.VehicleSafety().num_objectives
    )
    num_constraints = (
        botorch.test_functions.multi_objective.VehicleSafety().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.VehicleSafety(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.VehicleSafety,
        )


class ZDT1(MultiObjBotorchProblem):
    available_dimensions = (2, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.ZDT1(dim=2).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.ZDT1(dim=2), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=2):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.ZDT1, dim=dim
        )


class ZDT2(MultiObjBotorchProblem):
    available_dimensions = (2, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.ZDT2(dim=2).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.ZDT2(dim=2), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=2):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.ZDT2, dim=dim
        )


class ZDT3(MultiObjBotorchProblem):
    available_dimensions = (2, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.ZDT3(dim=2).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.ZDT3(dim=2), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=2):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.ZDT3, dim=dim
        )


class CarSideImpact(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.CarSideImpact().dim
    num_objectives = (
        botorch.test_functions.multi_objective.CarSideImpact().num_objectives
    )
    num_constraints = (
        botorch.test_functions.multi_objective.CarSideImpact().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.CarSideImpact(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.CarSideImpact,
        )


class BNH(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.BNH().dim
    num_objectives = botorch.test_functions.multi_objective.BNH().num_objectives
    num_constraints = (
        botorch.test_functions.multi_objective.BNH().num_constraints
        if hasattr(botorch.test_functions.multi_objective.BNH(), "num_constraints")
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.BNH,
        )


class CONSTR(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.CONSTR().dim
    num_objectives = botorch.test_functions.multi_objective.CONSTR().num_objectives
    num_constraints = (
        botorch.test_functions.multi_objective.CONSTR().num_constraints
        if hasattr(botorch.test_functions.multi_objective.CONSTR(), "num_constraints")
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.CONSTR,
        )


class ConstrainedBraninCurrin(MultiObjBotorchProblem):
    available_dimensions = (
        botorch.test_functions.multi_objective.ConstrainedBraninCurrin().dim
    )
    num_objectives = (
        botorch.test_functions.multi_objective.ConstrainedBraninCurrin().num_objectives
    )
    num_constraints = (
        botorch.test_functions.multi_objective.ConstrainedBraninCurrin().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.ConstrainedBraninCurrin(),
            "num_constraints",
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.ConstrainedBraninCurrin,
        )


class C2DTLZ2(MultiObjBotorchProblem):
    available_dimensions = (3, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.C2DTLZ2(dim=3).num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.C2DTLZ2(dim=3), "num_constraints"
        )
        else 0
    )

    def __init__(self, dim=3):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.C2DTLZ2, dim=dim
        )


class DiscBrake(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.DiscBrake().dim
    num_objectives = botorch.test_functions.multi_objective.DiscBrake().num_objectives
    num_constraints = (
        botorch.test_functions.multi_objective.DiscBrake().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.DiscBrake(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.DiscBrake,
        )


class MW7(MultiObjBotorchProblem):
    available_dimensions = (2, None)
    num_objectives = 2
    num_constraints = (
        botorch.test_functions.multi_objective.MW7(dim=2).num_constraints
        if hasattr(botorch.test_functions.multi_objective.MW7(dim=2), "num_constraints")
        else 0
    )

    def __init__(self, dim=2):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.MW7, dim=dim
        )


class OSY(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.OSY().dim
    num_objectives = botorch.test_functions.multi_objective.OSY().num_objectives
    num_constraints = (
        botorch.test_functions.multi_objective.OSY().num_constraints
        if hasattr(botorch.test_functions.multi_objective.OSY(), "num_constraints")
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.OSY,
        )


class SRN(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.SRN().dim
    num_objectives = botorch.test_functions.multi_objective.SRN().num_objectives
    num_constraints = (
        botorch.test_functions.multi_objective.SRN().num_constraints
        if hasattr(botorch.test_functions.multi_objective.SRN(), "num_constraints")
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.SRN,
        )


class WeldedBeam(MultiObjBotorchProblem):
    available_dimensions = botorch.test_functions.multi_objective.WeldedBeam().dim
    num_objectives = botorch.test_functions.multi_objective.WeldedBeam().num_objectives
    num_constraints = (
        botorch.test_functions.multi_objective.WeldedBeam().num_constraints
        if hasattr(
            botorch.test_functions.multi_objective.WeldedBeam(), "num_constraints"
        )
        else 0
    )

    def __init__(self):
        super().__init__(
            botorch_problem=botorch.test_functions.multi_objective.WeldedBeam,
        )
