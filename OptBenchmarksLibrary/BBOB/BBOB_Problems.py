import torch
from ..base import *


@dataclass
class BBOBConfig:
    suite: Optional[str] = 'bbob'
    function_number: Optional[int] = None
    dimension: Optional[int] = None
    instance_number: Optional[int] = None
    problem = None
    _cocoex = None  # Class variable to store the module

    @classmethod
    def _get_cocoex(cls):
        """Lazy import of cocoex module"""
        if cls._cocoex is None:
            import cocoex
            cls._cocoex = cocoex
        return cls._cocoex

    def __post_init__(self):
        cocoex = self._get_cocoex() 
        suite = cocoex.Suite(self.suite, "", "")
        self.problem = suite.get_problem_by_function_dimension_instance(self.function_number,self.dimension, self.instance_number)
            
    # Get BBOB Problem
    # def get_problem(self):
    #     import cocoex
    #     suite = cocoex.Suite(self.suite, "", "")
    #     self.problem = suite.get_problem_by_function_dimension_instance(self.function_number,self.dimension, self.instance_number)
    #     return self.problem
    
    
    


class BBOB_Problems(BenchmarkProblem):

    r'''
    Sources:
    (1) https://numbbo.github.io/coco/testsuites/bbob
    (2) https://coco-platform.org/
    '''
    

    

    def __init__(self, 
                 CONSTRAINTS = ConstraintConfig(type='UNCONSTRAINED'),
                 flag = BBOBConfig(suite = 'bbob',
                                   function_number = 1,
                                   dimension = 2,
                                   instance_number = 1,
                                  ),
                 debug = False,
                ):

        
        import cocoex
        tags = ["BBOB_Problems",
                "-----------------------------",
                "NOTE: ",
                "Please set the flag = BBOBConfig()",
                "to select the problem you want to work with",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                f"ALL SUITES: {cocoex.known_suite_names}",
                "IMPORT: cocoex",
               ]

        NUM_OBJ, NUM_CONS, bounds, MIXED, CONSTRAINTS = self.BBOB_preprocess(flag, CONSTRAINTS)
        
        super().__init__(dim = flag.dimension, 
                         num_obj = NUM_OBJ, 
                         num_cons = NUM_CONS, 
                         optimum = [[]], 
                         bounds = bounds,
                         MIXED = MIXED,
                         CONSTRAINTS = CONSTRAINTS,
                         tags = tags,
                         flag = flag,
                         debug = debug,
                        )

    
    def BBOB_preprocess(self,flag,CONSTRAINTS):
        
        # Interfacing with BBOB
        import cocoex

        # Get BBOB Problem
        # problem = flag.get_problem()
        problem = flag.problem

        # There is a constrained suite in BBOB
        NUM_CONS = 0
        if (flag.suite == 'bbob-constrained') & (CONSTRAINTS.type == 'UNCONSTRAINED'):
            raise TypeException(f'Error: Using bbob-constrained but have CONSTRAINTS=unconstrained(),'
                                f'Please define the constraint type by setting up CONSTRAINTS.'
                               )
        if (flag.suite == 'bbob-constrained'):
            NUM_CONS = problem.number_of_constraints

        
        # There are two mixed integer suites in BBOB
        if (flag.suite == "bbob-mixint") | (flag.suite == "bbob-biobj-mixint"):
            D_dict = {}
            for ind in range(problem.number_of_integer_variables):
                # Use dictionary assignment instead of append
                D_dict[ind] = torch.arange(problem.lower_bounds[ind], problem.upper_bounds[ind]+1)
            
            C_dict = {}
            for ind in range(problem.number_of_integer_variables, problem.dimension):
                # Use dictionary assignment instead of append
                C_dict[ind] = torch.tensor([problem.lower_bounds[ind], problem.upper_bounds[ind]])
                
            MIXED = MixIntConfig(is_mixed=True,
                                 discrete_dict=D_dict,
                                 continuous_dict=C_dict
                                )
        else:
            MIXED = MixIntConfig(is_mixed=False)

        # There is one bi-objective suite in BBOB
        if (flag.suite == "bbob-biobj-mixint"):
            NUM_OBJ = 2
        else:
            NUM_OBJ = 1

        # Now we setup the bound
        lower_bounds = torch.tensor(problem.lower_bounds)
        upper_bounds = torch.tensor(problem.upper_bounds)
        bounds = torch.cat( (lower_bounds.unsqueeze(-1), 
                             upper_bounds.unsqueeze(-1)) , dim=1)

        return NUM_OBJ, NUM_CONS, bounds, MIXED, CONSTRAINTS


    
    def _evaluate_implementation(self, X):
        
        import cocoex
        X_numpy = X.cpu().numpy()
        problem = self.flag.problem
        

        FX = torch.zeros((X.shape[0], 1))

        if self.flag.suite == "bbob-constrained":
            test_gx = problem.constraint(np.ones((X.shape[1],)))
            GX = torch.zeros((X.shape[0], test_gx.shape[0]))
        else:
            GX = None

        for i in range(X.shape[0]):
            fx = problem(X[i,:])
            FX[i,:] = fx
            
            if self.flag.suite == "bbob-constrained":
                gx = problem.constraint(X[i,:])
                GX[i,:] = torch.from_numpy(gx)
        

        return GX, FX













