import torch
from ..base import *



class BraninCurrin(BenchmarkProblem):

    def __init__(self, 
                 dim=2, 
                 num_objs = 2, 
                 MULTIOBJ = MultiObjConfig(type='MULTI_OBJ')):
        
        tags = ["BraninCurrin",
                "-----------------------------",
                "OBJECTIVES: Multi Objective (2)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: 2-Dim", 
                "IMPORTS: BoTorch",
               ]
        
        super().__init__(dim, 
                         num_obj = 2, 
                         num_cons = 0,  
                         bounds = [[0, 1]],
                         ref_point = [18.0, 6.0],
                         MULTIOBJ = MULTIOBJ,
                         tags = tags,
                        )
        

    def _evaluate_implementation(self, X):
        from botorch.test_functions.multi_objective import BraninCurrin as BraninCurrin_imported
        
        fun = BraninCurrin_imported(negate=True)
        fx = fun(X)

        return None, fx






















