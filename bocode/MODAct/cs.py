import modact.problems as pb
from ..base import *
from .baseModact import BaseModactProblem
from typing import Tuple
import numpy as np
    
class CS1(BaseModactProblem):

    problem_name = "cs1"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("cs1")

class CT1(BaseModactProblem):

    problem_name = "ct1"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ct1")

class CTS1(BaseModactProblem):
    
    problem_name = "cts1"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("cts1")

class CTSE1(BaseModactProblem):
    
    problem_name = "ctse1"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ctse1")

class CTSEI1(BaseModactProblem):
    
    problem_name = "ctsei1"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ctsei1")

class CS2(BaseModactProblem):
    
    problem_name = "cs2"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("cs2")

class CT2(BaseModactProblem):
    
    problem_name = "ct2"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ct2")

class CTS2(BaseModactProblem):
    
    problem_name = "cts2"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("cts2")

class CTSE2(BaseModactProblem):
    
    problem_name = "ctse2"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ctse2")

class CTSEI2(BaseModactProblem):
    
    problem_name = "ctsei2"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ctsei2")

class CS3(BaseModactProblem):
    
    problem_name = "cs3"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("cs3")

class CT3(BaseModactProblem):
    
    problem_name = "ct3"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ct3")

class CTS3(BaseModactProblem):
    
    problem_name = "cts3"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("cts3")

class CTSE3(BaseModactProblem):
    
    problem_name = "ctse3"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ctse3")

class CTSEI3(BaseModactProblem):
    
    problem_name = "ctsei3"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ctsei3")

class CS4(BaseModactProblem):
    
    problem_name = "cs4"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("cs4")
    
class CT4(BaseModactProblem):
    
    problem_name = "ct4"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)

    def __init__(self):
        super().__init__("ct4")

class CTS4(BaseModactProblem):
    
    problem_name = "cts4"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("cts4")

class CTSE4(BaseModactProblem):
    
    problem_name = "ctse4"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ctse4")

class CTSEI4(BaseModactProblem):
    
    problem_name = "ctsei4"
    p = pb.get_problem(problem_name)
    available_dimensions = len(p.bounds()[0])
    num_objectives = len(p.weights)
    num_constraints = len(p.c_weights)
    
    def __init__(self):
        super().__init__("ctsei4")