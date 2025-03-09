import math
from typing import Callable, Iterable, List, Union, Dict
from collections import defaultdict
from OptBenchmarksLibrary import *

ValType = Union[int, tuple, set, list]

SyntheticsFuncs = [Ackley, Bukin, DixonPrice, Goldstein, Goldstein_Discrete, Griewank, Levy, Michalewicz, Powell, Rastrigin, Rosenbrock, StyblinskiTang]
LassoBenchFuncs = [LassoBreastCancer, LassoDiabetes, LassoDNA, LassoLeukemia, LassoRCV1, LassoSyntHard, LassoSyntHigh, LassoSyntMedium, LassoSyntSimple]
EngineeringFuncs = [CarSideImpact, EulerBernoulliBeamBending, GearTrain, Mazda, Mazda_SCA, MOPTA08Car, RobotPush, Rover, Truss10D, Truss25D, TwoBarTruss, WaterProblem, WaterResources]
CEC2020Funcs = [CEC2020_p1, CEC2020_p2, CEC2020_p3, CEC2020_p4, CEC2020_p5, CEC2020_p6, CEC2020_p7, CEC2020_p8, CEC2020_p9, CEC2020_p10, CEC2020_p11, CEC2020_p12, CEC2020_p13, CEC2020_p14, CEC2020_p15, CEC2020_p16, CEC2020_p17, CEC2020_p18, CEC2020_p19, CEC2020_p20, CEC2020_p21, CEC2020_p22, CEC2020_p23, CEC2020_p24, CEC2020_p25, CEC2020_p26, CEC2020_p27, CEC2020_p28, CEC2020_p29, CEC2020_p30, CEC2020_p31, CEC2020_p32, CEC2020_p33, CEC2020_p34, CEC2020_p35, CEC2020_p36, CEC2020_p37, CEC2020_p38, CEC2020_p39, CEC2020_p40, CEC2020_p41, CEC2020_p42, CEC2020_p43, CEC2020_p44, CEC2020_p45, CEC2020_p46, CEC2020_p47, CEC2020_p48, CEC2020_p49, CEC2020_p50, CEC2020_p51, CEC2020_p52, CEC2020_p53, CEC2020_p54, CEC2020_p55, CEC2020_p56, CEC2020_p57]
BBOBFuncs = [BBOB, BBOB_Biobj, BBOB_BiobjMixInt, BBOB_Boxed, BBOB_Constrained, BBOB_LargeScale, BBOB_MixInt, BBOB_Noisy]

categorized_classes = {
    "Synthetics": SyntheticsFuncs,
    "LassoBench": LassoBenchFuncs,
    "Engineering": EngineeringFuncs,
    "CEC2020_RW_Constrained": CEC2020Funcs,
    "BBOB": BBOBFuncs,
}

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
                     objectives_filter: Callable[[int], bool] = lambda x: x > 0,
                     category_filter: Callable[[str], bool] = lambda x: True,
                     ) -> Dict[str, List[str]]:
    """
    Filter functions based on the given constraints.

    Available Categories: ["Synthetics", "LassoBench", "Engineering", "CEC2020_RW_Constrained", "BBOB"]

    Parameters
    ----------
    dimension_filter : Callable[[int], bool], optional
        A function that takes a dimension number and returns a boolean, by default unfiltered
    objectives_filter : Callable[[int], bool], optional
        A function that takes an objective number and returns a boolean, by default unfiltered
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
            func_instance = func()
            dimensions = getattr(func_instance, "available_dimensions", None)
            objectives = getattr(func_instance, "num_objectives", None)

            if dimensions is not None and not _has_valid_val(dimensions, dimension_filter):
                continue

            if objectives is not None and not _has_valid_val(objectives, objectives_filter):
                continue

            filtered_funcs[category].append(func.__name__)

    return dict(filtered_funcs)