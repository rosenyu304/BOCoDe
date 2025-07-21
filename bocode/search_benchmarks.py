import math
from collections import defaultdict
from typing import Callable, Dict, List, Union

import bocode
from .base import BenchmarkProblem, DataType

ValType = Union[int, tuple, set, list]

SyntheticsFuncs = [
    bocode.Synthetics.Ackley,
    bocode.Synthetics.Bukin,
    bocode.Synthetics.DixonPrice,
    bocode.Synthetics.Goldstein,
    bocode.Synthetics.Goldstein_Discrete,
    bocode.Synthetics.Griewank,
    bocode.Synthetics.Levy,
    bocode.Synthetics.Michalewicz,
    bocode.Synthetics.Powell,
    bocode.Synthetics.Rastrigin,
    bocode.Synthetics.Rosenbrock,
    bocode.Synthetics.StyblinskiTang,
    bocode.Synthetics.Beale,
    bocode.Synthetics.Cosine8,
    bocode.Synthetics.DropWave,
    bocode.Synthetics.EggHolder,
    bocode.Synthetics.Hartmann3D,
    bocode.Synthetics.Hartmann6D,
    bocode.Synthetics.HolderTable,
    bocode.Synthetics.Shekelm5,
    bocode.Synthetics.Shekelm7,
    bocode.Synthetics.Shekelm10,
    bocode.Synthetics.Shekel,
    bocode.Synthetics.SixHumpCamel,
    bocode.Synthetics.ThreeHumpCamel,
    bocode.Synthetics.ConstrainedGramacy,
    bocode.Synthetics.ConstrainedHartmann,
    bocode.Synthetics.ConstrainedHartmannSmooth,
    bocode.Synthetics.PressureVessel,
    bocode.Synthetics.WeldedBeamSO,
    bocode.Synthetics.TensionCompressionString,
    bocode.Synthetics.SpeedReducer,
    bocode.Synthetics.SVM,
]
LassoBenchFuncs = [
    bocode.LassoBench.LassoBreastCancer,
    bocode.LassoBench.LassoDiabetes,
    bocode.LassoBench.LassoDNA,
    bocode.LassoBench.LassoLeukemia,
    bocode.LassoBench.LassoRCV1,
    bocode.LassoBench.LassoSyntHard,
    bocode.LassoBench.LassoSyntHigh,
    bocode.LassoBench.LassoSyntMedium,
    bocode.LassoBench.LassoSyntSimple,
]
EngineeringFuncs = [
    bocode.Engineering.CarSideImpact,
    bocode.Engineering.CantileverBeam,
    bocode.Engineering.Car,
    bocode.Engineering.CompressionSpring,
    bocode.Engineering.EulerBernoulliBeamBending,
    bocode.Engineering.GearTrain,
    bocode.Engineering.HeatExchanger,
    bocode.Engineering.KeaneBump,
    bocode.Engineering.Mazda,
    bocode.Engineering.Mazda_SCA,
    bocode.Engineering.MOPTA08Car,
    bocode.Engineering.PD4CartPole,
    bocode.Engineering.PID4Acrobot,
    bocode.Engineering.PressureVessel,
    bocode.Engineering.ReinforcedConcreteBeam,
    bocode.Engineering.RobotPush,
    bocode.Engineering.Rover,
    bocode.Engineering.SpeedReducer,
    bocode.Engineering.Truss10D,
    bocode.Engineering.Truss25D,
    bocode.Engineering.Truss72D_FourForces,
    bocode.Engineering.Truss72D_SingleForce,
    bocode.Engineering.Truss120D,
    bocode.Engineering.Truss200D,
    bocode.Engineering.TwoBarTruss,
    bocode.Engineering.ThreeTruss,
    bocode.Engineering.WaterProblem,
    bocode.Engineering.WaterResources,
]
MujocoFuncs = [
    bocode.Engineering.AntProblem,
    bocode.Engineering.HalfCheetahProblem,
    bocode.Engineering.HopperProblem,
    bocode.Engineering.HumanoidProblem,
    bocode.Engineering.HumanoidStandupProblem,
    bocode.Engineering.InvertedDoublePendulumProblem,
    bocode.Engineering.InvertedPendulumProblem,
    bocode.Engineering.PusherProblem,
    bocode.Engineering.ReacherProblem,
    bocode.Engineering.SwimmerProblem,
    bocode.Engineering.Walker2DProblem,
    bocode.Engineering.SwimmerPolicySearchProblem,
    bocode.Engineering.AntPolicySearchProblem,
    bocode.Engineering.HalfCheetahPolicySearchProblem,
    bocode.Engineering.HopperPolicySearchProblem,
    bocode.Engineering.Walker2DPolicySearchProblem,
]
BayesianCHTFuncs = [
    bocode.Engineering.NonLinearConstraintProblemA3,
    bocode.Engineering.NonLinearConstraintProblemA4,
    bocode.Engineering.NonLinearConstraintProblemA7,
    bocode.Engineering.NonLinearConstraintProblemA8,
    bocode.Engineering.NonLinearConstraintProblemB3,
    bocode.Engineering.NonLinearConstraintProblemB4,
    bocode.Engineering.NonLinearConstraintProblemB7,
    bocode.Engineering.NonLinearConstraintProblemB8,
]
CEC2020Funcs = [
    bocode.CEC.CEC2020_p1,
    bocode.CEC.CEC2020_p2,
    bocode.CEC.CEC2020_p3,
    bocode.CEC.CEC2020_p4,
    bocode.CEC.CEC2020_p5,
    bocode.CEC.CEC2020_p6,
    bocode.CEC.CEC2020_p7,
    bocode.CEC.CEC2020_p8,
    bocode.CEC.CEC2020_p9,
    bocode.CEC.CEC2020_p10,
    bocode.CEC.CEC2020_p11,
    bocode.CEC.CEC2020_p12,
    bocode.CEC.CEC2020_p13,
    bocode.CEC.CEC2020_p14,
    bocode.CEC.CEC2020_p15,
    bocode.CEC.CEC2020_p16,
    bocode.CEC.CEC2020_p17,
    bocode.CEC.CEC2020_p18,
    bocode.CEC.CEC2020_p19,
    bocode.CEC.CEC2020_p20,
    bocode.CEC.CEC2020_p21,
    bocode.CEC.CEC2020_p22,
    bocode.CEC.CEC2020_p23,
    bocode.CEC.CEC2020_p24,
    bocode.CEC.CEC2020_p25,
    bocode.CEC.CEC2020_p26,
    bocode.CEC.CEC2020_p27,
    bocode.CEC.CEC2020_p28,
    bocode.CEC.CEC2020_p29,
    bocode.CEC.CEC2020_p30,
    bocode.CEC.CEC2020_p31,
    bocode.CEC.CEC2020_p32,
    bocode.CEC.CEC2020_p33,
    bocode.CEC.CEC2020_p34,
    bocode.CEC.CEC2020_p35,
    bocode.CEC.CEC2020_p36,
    bocode.CEC.CEC2020_p37,
    bocode.CEC.CEC2020_p38,
    bocode.CEC.CEC2020_p39,
    bocode.CEC.CEC2020_p40,
    bocode.CEC.CEC2020_p41,
    bocode.CEC.CEC2020_p42,
    bocode.CEC.CEC2020_p43,
    bocode.CEC.CEC2020_p44,
    bocode.CEC.CEC2020_p45,
    bocode.CEC.CEC2020_p46,
    bocode.CEC.CEC2020_p47,
    bocode.CEC.CEC2020_p48,
    bocode.CEC.CEC2020_p49,
    bocode.CEC.CEC2020_p50,
    bocode.CEC.CEC2020_p51,
    bocode.CEC.CEC2020_p52,
    bocode.CEC.CEC2020_p53,
    bocode.CEC.CEC2020_p54,
    bocode.CEC.CEC2020_p55,
    bocode.CEC.CEC2020_p56,
    bocode.CEC.CEC2020_p57,
]
BBOBFuncs = [
    bocode.BBOB.BBOB,
    bocode.BBOB.BBOB_Biobj,
    bocode.BBOB.BBOB_BiobjMixInt,
    bocode.BBOB.BBOB_Boxed,
    bocode.BBOB.BBOB_Constrained,
    bocode.BBOB.BBOB_LargeScale,
    bocode.BBOB.BBOB_MixInt,
    bocode.BBOB.BBOB_Noisy,
]
BotorchFuncs = [
    bocode.BoTorch.AugmentedBranin,
    bocode.BoTorch.AugmentedHartmann,
    bocode.BoTorch.AugmentedRosenbrock,
    bocode.BoTorch.BraninCurrin,
    bocode.BoTorch.DH1,
    bocode.BoTorch.DH2,
    bocode.BoTorch.DH3,
    bocode.BoTorch.DH4,
    bocode.BoTorch.DTLZ1,
    bocode.BoTorch.DTLZ2,
    bocode.BoTorch.DTLZ3,
    bocode.BoTorch.DTLZ4,
    bocode.BoTorch.DTLZ5,
    bocode.BoTorch.DTLZ7,
    bocode.BoTorch.GMM,
    bocode.BoTorch.Penicillin,
    bocode.BoTorch.ToyRobust,
    bocode.BoTorch.VehicleSafety,
    bocode.BoTorch.ZDT1,
    bocode.BoTorch.ZDT2,
    bocode.BoTorch.ZDT3,
    bocode.BoTorch.CarSideImpact,
    bocode.BoTorch.BNH,
    bocode.BoTorch.CONSTR,
    bocode.BoTorch.ConstrainedBraninCurrin,
    bocode.BoTorch.C2DTLZ2,
    bocode.BoTorch.DiscBrake,
    bocode.BoTorch.MW7,
    bocode.BoTorch.OSY,
    bocode.BoTorch.SRN,
    bocode.BoTorch.WeldedBeam,
    bocode.BoTorch.MOMFBraninCurrin,
    bocode.BoTorch.MOMFPark1,
    bocode.BoTorch.Ishigami,
    bocode.BoTorch.Gsobol,
    bocode.BoTorch.Morris,
]
MODActFuncs = [
    bocode.MODAct.CS1,
    bocode.MODAct.CT1,
    bocode.MODAct.CTS1,
    bocode.MODAct.CTSE1,
    bocode.MODAct.CTSEI1,
    bocode.MODAct.CS2,
    bocode.MODAct.CT2,
    bocode.MODAct.CTS2,
    bocode.MODAct.CTSE2,
    bocode.MODAct.CTSEI2,
    bocode.MODAct.CS3,
    bocode.MODAct.CT3,
    bocode.MODAct.CTS3,
    bocode.MODAct.CTSE3,
    bocode.MODAct.CTSEI3,
    bocode.MODAct.CS4,
    bocode.MODAct.CT4,
    bocode.MODAct.CTS4,
    bocode.MODAct.CTSE4,
    bocode.MODAct.CTSEI4,
]
CEC2017Funcs = [
    bocode.CEC.CEC2017_p1,
    bocode.CEC.CEC2017_p2,
    bocode.CEC.CEC2017_p3,
    bocode.CEC.CEC2017_p4,
    bocode.CEC.CEC2017_p5,
    bocode.CEC.CEC2017_p6,
    bocode.CEC.CEC2017_p7,
    bocode.CEC.CEC2017_p8,
    bocode.CEC.CEC2017_p9,
    bocode.CEC.CEC2017_p10,
    bocode.CEC.CEC2017_p11,
    bocode.CEC.CEC2017_p12,
    bocode.CEC.CEC2017_p13,
    bocode.CEC.CEC2017_p14,
    bocode.CEC.CEC2017_p15,
    bocode.CEC.CEC2017_p16,
    bocode.CEC.CEC2017_p17,
    bocode.CEC.CEC2017_p18,
    bocode.CEC.CEC2017_p19,
    bocode.CEC.CEC2017_p20,
    bocode.CEC.CEC2017_p21,
    bocode.CEC.CEC2017_p22,
    bocode.CEC.CEC2017_p23,
    bocode.CEC.CEC2017_p24,
    bocode.CEC.CEC2017_p25,
    bocode.CEC.CEC2017_p26,
    bocode.CEC.CEC2017_p27,
    bocode.CEC.CEC2017_p28,
    bocode.CEC.CEC2017_p29,
]
WFGFuncs = [
    bocode.WFG.WFG1,
    bocode.WFG.WFG2,
    bocode.WFG.WFG3,
    bocode.WFG.WFG4,
    bocode.WFG.WFG5,
    bocode.WFG.WFG6,
    bocode.WFG.WFG7,
    bocode.WFG.WFG8,
    bocode.WFG.WFG9,
]
ZDTFuncs = [
    bocode.ZDT.ZDT1,
    bocode.ZDT.ZDT2,
    bocode.ZDT.ZDT3,
    bocode.ZDT.ZDT4,
    bocode.ZDT.ZDT5,
    bocode.ZDT.ZDT6,
]
DTLZFuncs = [
    bocode.DTLZ.DTLZ1,
    bocode.DTLZ.DTLZ2,
    bocode.DTLZ.DTLZ3,
    bocode.DTLZ.DTLZ4,
    bocode.DTLZ.DTLZ5,
    bocode.DTLZ.DTLZ6,
    bocode.DTLZ.DTLZ7,
]
CEC2007Funcs = [
    bocode.CEC.CEC2007_OKA2,
    bocode.CEC.CEC2007_S_ZDT1,
    bocode.CEC.CEC2007_S_ZDT2,
    bocode.CEC.CEC2007_S_ZDT4,
    bocode.CEC.CEC2007_S_ZDT6,
    bocode.CEC.CEC2007_SYMPART,
    bocode.CEC.CEC2007_R_ZDT4,
    bocode.CEC.CEC2007_S_DTLZ2,
    bocode.CEC.CEC2007_S_DTLZ3,
    bocode.CEC.CEC2007_R_DTLZ2,
    bocode.CEC.CEC2007_WFG1,
    bocode.CEC.CEC2007_WFG8,
    bocode.CEC.CEC2007_WFG9,
]
CEC2019Funcs = [
    bocode.CEC.CEC2019_p1,
    bocode.CEC.CEC2019_p2,
    bocode.CEC.CEC2019_p3,
    bocode.CEC.CEC2019_p4,
    bocode.CEC.CEC2019_p5,
    bocode.CEC.CEC2019_p6,
    bocode.CEC.CEC2019_p7,
    bocode.CEC.CEC2019_p8,
    bocode.CEC.CEC2019_p9,
    bocode.CEC.CEC2019_p10,
]
NEORLFuncs = [
    bocode.NEORL.TSP_51Cities,
    bocode.NEORL.TSP_100Cities,
    bocode.NEORL.ReactivityModel,
    bocode.NEORL.QPowerModel,
]

categorized_classes = {
    "Synthetics": SyntheticsFuncs,
    "LassoBench": LassoBenchFuncs,
    "Engineering": EngineeringFuncs,
    "Engineering.Gym": MujocoFuncs,
    "Engineering.BayesianCHT": BayesianCHTFuncs,
    "BBOB": BBOBFuncs,
    "BoTorch": BotorchFuncs,
    "MODAct": MODActFuncs,
    "WFG": WFGFuncs,
    "ZDT": ZDTFuncs,
    "DTLZ": DTLZFuncs,
    "CEC.CEC2007": CEC2007Funcs,
    "CEC.CEC2017": CEC2017Funcs,
    "CEC.CEC2019": CEC2019Funcs,
    "CEC.CEC2020_RW_Constrained": CEC2020Funcs,
    "NEORL": NEORLFuncs,
}


def _has_valid_val(val: ValType, constraint=Callable[[int], bool]) -> bool:
    if isinstance(val, int):
        return constraint(val)
    elif isinstance(val, (set, list)):
        return any(constraint(d) for d in val)
    elif isinstance(val, tuple) and len(val) == 2:
        start, end = val
        # For open interval
        if end is None or end == math.inf:
            sample_limit = 100  # sample the first 100 values
            for d in range(start, start + sample_limit):
                if constraint(d):
                    return True
            return False
        else:
            # For closed interval
            return any(constraint(d) for d in range(start, end + 1))
    else:
        raise ValueError(f"Unsupported val type: {val}")


def filter_functions(
    dimension_filter: Callable[[int], bool] = lambda x: x > 0,
    input_type_filter: Callable[[DataType], bool] = lambda x: True,
    objectives_filter: Callable[[int], bool] = lambda x: x > 0,
    constraints_filter: Callable[[int], bool] = lambda x: x >= 0,
    category_filter: Callable[[str], bool] = lambda x: True,
) -> Dict[str, List[BenchmarkProblem]]:
    """
    Filter functions based on the given constraints.

    Available Categories: ['Synthetics', 'LassoBench', 'Engineering', 'Engineering.Gym', 'Engineering.BayesianCHT', 'CEC.CEC2020_RW_Constrained', 'BBOB', 'BoTorch', 'MODAct', 'CEC.CEC2017', 'WFG', 'ZDT', 'DTLZ', 'CEC.CEC2007', 'CEC.CEC2019', 'NEORL']

    Parameters
    ----------
    dimension_filter : Callable[[int], bool], optional
        A function that takes a dimension number and returns a boolean, by default unfiltered

    input_type_filter : Callable[[DataType], bool], optional
        A function that takes an input type and returns a boolean, by default unfiltered.

        Available DataTypes:
         - DataType.CONTINUOUS
         - DataType.DISCRETE
         - DataType.CATEGORICAL

    objectives_filter : Callable[[int], bool], optional
        A function that takes an objective number and returns a boolean, by default unfiltered

    constraints_filter : Callable[[int], bool], optional
        A function that takes a constraint number and returns a boolean, by default unfiltered

    category_filter : Callable[[str], bool], optional
        A function that takes a category string and returns a boolean, by default unfiltered

    Returns
    -------
    Dict[str, List[str]]
        A dictionary where the keys are the categories and the values are the names of the functions that satisfy the constraints.
    """

    filtered_funcs = defaultdict(list)

    for category, functions in categorized_classes.items():
        if not category_filter(category):
            continue

        for func in functions:
            input_type = getattr(func, "input_type", None)
            dimensions = getattr(func, "available_dimensions", None)
            objectives = getattr(func, "num_objectives", None)
            constraints = getattr(func, "num_constraints", None)

            if input_type is not None and not input_type_filter(input_type):
                continue

            if dimensions is not None and not _has_valid_val(
                dimensions, dimension_filter
            ):
                continue

            if objectives is not None and not _has_valid_val(
                objectives, objectives_filter
            ):
                continue

            if constraints is not None and not _has_valid_val(
                constraints, constraints_filter
            ):
                continue

            filtered_funcs[category].append(func)

    return dict(filtered_funcs)
