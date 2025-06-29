import math
from typing import Callable, Iterable, List, Union, Dict
from collections import defaultdict
from bocode import *
import importlib

ValType = Union[int, tuple, set, list]

def qualify_classes(class_list, module_name):
    """
    Given a list of class objects and a module name (string),
    return a list of fully qualified class objects from the specified module.
    """
    full_module_path = "bocode." + module_name.replace('.', '/').replace('/', '.')
    module = importlib.import_module(full_module_path)
    return [getattr(module, cls.__name__) for cls in class_list]

SyntheticsFuncs = [Ackley, Bukin, DixonPrice, Goldstein, Goldstein_Discrete, Griewank, Levy, Michalewicz, 
                   Powell, Rastrigin, Rosenbrock, StyblinskiTang, Beale, Cosine8, DropWave, EggHolder, 
                   Hartmann3D, Hartmann6D, HolderTable, Shekelm5, Shekelm7, Shekelm10, Shekel, SixHumpCamel, 
                   ThreeHumpCamel, ConstrainedGramacy, ConstrainedHartmann, ConstrainedHartmannSmooth, 
                   PressureVessel, WeldedBeamSO, TensionCompressionString, SpeedReducer, SVM]
LassoBenchFuncs = [LassoBreastCancer, LassoDiabetes, LassoDNA, LassoLeukemia, LassoRCV1, LassoSyntHard, 
                   LassoSyntHigh, LassoSyntMedium, LassoSyntSimple]
EngineeringFuncs = [CarSideImpact, CantileverBeam, Car, CompressionSpring, EulerBernoulliBeamBending, GearTrain, 
                    KeaneBump, Mazda, Mazda_SCA, MOPTA08Car, ReinforcedConcreteBeam, RobotPush, Rover, Truss10D, 
                    Truss25D, Truss72D_FourForces, Truss72D_SingleForce, Truss120D, Truss200D, TwoBarTruss, 
                    ThreeTruss, WaterProblem, WaterResources]
MujocoFuncs = [AntProblem, HalfCheetahProblem, HopperProblem, HumanoidProblem, HumanoidStandupProblem, InvertedDoublePendulumProblem, 
               InvertedPendulumProblem, PusherProblem, ReacherProblem, SwimmerProblem, Walker2DProblem]
BayesianCHTFuncs = [NonLinearConstraintProblemA3, NonLinearConstraintProblemA4, NonLinearConstraintProblemA7, NonLinearConstraintProblemA8, 
                    NonLinearConstraintProblemB3, NonLinearConstraintProblemB4, NonLinearConstraintProblemB7, NonLinearConstraintProblemB8]
CEC2020Funcs = [CEC2020_p1, CEC2020_p2, CEC2020_p3, CEC2020_p4, CEC2020_p5, CEC2020_p6, CEC2020_p7, CEC2020_p8, CEC2020_p9, 
                CEC2020_p10, CEC2020_p11, CEC2020_p12, CEC2020_p13, CEC2020_p14, CEC2020_p15, CEC2020_p16, CEC2020_p17, 
                CEC2020_p18, CEC2020_p19, CEC2020_p20, CEC2020_p21, CEC2020_p22, CEC2020_p23, CEC2020_p24, CEC2020_p25, 
                CEC2020_p26, CEC2020_p27, CEC2020_p28, CEC2020_p29, CEC2020_p30, CEC2020_p31, CEC2020_p32, CEC2020_p33, 
                CEC2020_p34, CEC2020_p35, CEC2020_p36, CEC2020_p37, CEC2020_p38, CEC2020_p39, CEC2020_p40, CEC2020_p41, 
                CEC2020_p42, CEC2020_p43, CEC2020_p44, CEC2020_p45, CEC2020_p46, CEC2020_p47, CEC2020_p48, CEC2020_p49, 
                CEC2020_p50, CEC2020_p51, CEC2020_p52, CEC2020_p53, CEC2020_p54, CEC2020_p55, CEC2020_p56, CEC2020_p57]
BBOBFuncs = [BBOB, BBOB_Biobj, BBOB_BiobjMixInt, BBOB_Boxed, BBOB_Constrained, BBOB_LargeScale, BBOB_MixInt, BBOB_Noisy]
BotorchFuncs = [AugmentedBranin, AugmentedHartmann, AugmentedRosenbrock, BraninCurrin, DH1, DH2, DH3, DH4, DTLZ1, DTLZ2, 
                DTLZ3, DTLZ4, DTLZ5, DTLZ7, GMM, Penicillin, ToyRobust, VehicleSafety, ZDT1, ZDT2, ZDT3, CarSideImpact, 
                BNH, CONSTR, ConstrainedBraninCurrin, C2DTLZ2, DiscBrake, MW7, OSY, SRN, WeldedBeam, MOMFBraninCurrin, 
                MOMFPark1, Ishigami, Gsobol, Morris]
MODActFuncs = [CS1, CT1, CTS1, CTSE1, CTSEI1, CS2, CT2, CTS2, CTSE2, CTSEI2, CS3, CT3, CTS3, CTSE3, CTSEI3, CS4, CT4, 
               CTS4, CTSE4, CTSEI4]
CEC2017Funcs = [CEC2017_p1, CEC2017_p2, CEC2017_p3, CEC2017_p4, CEC2017_p5, CEC2017_p6, CEC2017_p7, CEC2017_p8, CEC2017_p9, CEC2017_p10, 
                 CEC2017_p11, CEC2017_p12, CEC2017_p13, CEC2017_p14, CEC2017_p15, CEC2017_p16, CEC2017_p17, CEC2017_p18, CEC2017_p19, 
                 CEC2017_p20, CEC2017_p21, CEC2017_p22, CEC2017_p23, CEC2017_p24, CEC2017_p25, CEC2017_p26, CEC2017_p27, CEC2017_p28, 
                 CEC2017_p29]
WFGFuncs = [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9]
ZDTFuncs = [ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6]
DTLZFuncs = [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7]
CEC2007Funcs = [CEC2007_OKA2, CEC2007_S_ZDT1, CEC2007_S_ZDT2, CEC2007_S_ZDT4, CEC2007_S_ZDT6, CEC2007_SYMPART, CEC2007_R_ZDT4, 
                CEC2007_S_DTLZ2, CEC2007_S_DTLZ3, CEC2007_R_DTLZ2, CEC2007_WFG1, CEC2007_WFG8, CEC2007_WFG9]
CEC2019Funcs = [CEC2019_p1, CEC2019_p2, CEC2019_p3, CEC2019_p4, CEC2019_p5, CEC2019_p6, CEC2019_p7, CEC2019_p8, CEC2019_p9, CEC2019_p10]
NEORLFuncs = [TSP_51Cities, TSP_100Cities, ReactivityModel, QPowerModel]

categorized_classes = {
    "Synthetics": SyntheticsFuncs,
    "LassoBench": LassoBenchFuncs,
    "Engineering": EngineeringFuncs,
    "Engineering.Gym": MujocoFuncs,
    "Engineering.BayesianCHT": BayesianCHTFuncs,
    "CEC.CEC2020_RW_Constrained": CEC2020Funcs,
    "BBOB": BBOBFuncs,
    "BoTorch": BotorchFuncs,
    "MODAct": MODActFuncs,
    "CEC.CEC2017": CEC2017Funcs,
    "WFG": WFGFuncs,
    "ZDT": ZDTFuncs,
    "DTLZ": DTLZFuncs,
    "CEC.CEC2007": CEC2007Funcs,
    "CEC.CEC2019": CEC2019Funcs,
    "NEORL": NEORLFuncs,
}

for category, functions in categorized_classes.items():
    categorized_classes[category] = qualify_classes(functions, category)

def _has_valid_val(val: ValType, constraint = Callable[[int], bool]) -> bool:
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

def filter_functions(dimension_filter: Callable[[int], bool] = lambda x: x > 0,
                     input_type_filter: Callable[[DataType], bool] = lambda x: True,
                     objectives_filter: Callable[[int], bool] = lambda x: x > 0,
                     constraints_filter: Callable[[int], bool] = lambda x: x >= 0,
                     category_filter: Callable[[str], bool] = lambda x: True,
                     ) -> Dict[str, List[str]]:
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

            if dimensions is not None and not _has_valid_val(dimensions, dimension_filter):
                continue

            if objectives is not None and not _has_valid_val(objectives, objectives_filter):
                continue

            if constraints is not None and not _has_valid_val(constraints, constraints_filter):
                continue

            # Compose the path as 'bocode.<category>.<ClassName>'
            class_name = func.__name__ if hasattr(func, "__name__") else func.__class__.__name__
            filtered_funcs[category].append(f"bocode.{category}.{class_name}")

    return dict(filtered_funcs)