import torch
from typing import Union, Tuple, Set

class BenchmarkProblem:
    def __init__(
        self,
        dim: int = 1,
        bound: Union[Tuple, Set],
        ref_point: torch.Tensor
    ) -> None:
        """Initialize the BenchmarkProblem class.
        
        Args:
            dim (int, optional): Dimension of the problem. Defaults to 1.
            bound (Union[Tuple, Set]): Bounds of the problem space. Tuple (lower_bound, upper_bound) for continuous values. Set {discrete values} for discrete values.
            ref_point (torch.Tensor): Reference point for the multi-objective problem.
        """
        self.dim = dim
        self.bound = bound
        self.ref_point = ref_point



# import torch
# import numpy as np
# import jax
# import jax.numpy as jnp
# from .configs import *

# class BenchmarkProblem():

#     """
#     Base class for Bayesian Optimization benchmark problems.
#     """

#     def __init__(self, 
#                  dim = 1, 
#                  num_obj = 1, 
#                  num_cons = 0, 
#                  bounds = None, 
#                  x_optimum = [[]],
#                  optimum = [[]], 
#                  ref_point = None, 
#                  out_type = 'torch', 
#                  device = 'cpu',
#                  tags = [], 
#                  MIXED: MixIntConfig = MixIntConfig(), 
#                  MULTIOBJ: MultiObjConfig = MultiObjConfig(),
#                  CONSTRAINTS: ConstraintConfig = ConstraintConfig(),
#                  MAXIMIZATION: bool = True,
#                  debug: bool = False, 
#                  flag = '',
#                  **kwargs
#                 ):
#         '''
#         Parameters:
#             dim:
#                 The search space's dimension
                
#             num_obj:
#                 The number of the objectives
                
#             num_cons:
#                 The number of the constraints
                
#             bounds:
#                 The bound of the search space. 
#                 The input X is to this library is strictly defined in [0,1].
#                 The bound will be used to scale the X to the correct search space domain. 

#             x_optimum:
#                 The "x" design variable value of the theoretical "optimum".
            
#             optimum:
#                 The theoretical or experimental value of the optimum values of the problem.
                
#             ref_point:
#                 The reference point for calculating the hyper volume for Multi Objective problems
                
#             out_type:
#                 "torch", "np", or "jnp"
                
#             device:
#                 "cpu" or "cuda"s
                
#             tags:
#                 A list that contains the information for each problem.
                
#             MIXED:
#                 - MixIntConfig. See "configs.py"
                
#             MULTIOBJ:
#                 - MultiObjConfig. See "configs.py"
                    
#             CONSTRAINTS:
#                 - ConstraintConfig. See "configs.py"
                
#             MAXIMIZATION: 
#                 True: The problem is formulated for optimizers that does maximization.
#                 False: ... minimization.
                
#             debug:
#                 True: For debugging and printing out some info.
#                 False: Do not output anything.
                
#             flag:
#                 Take in any string if a user want to add flags.
#         '''
        
#         self.dim = dim
#         self.num_obj = num_obj
#         self.num_cons = num_cons
#         self.tags = tags

#         # True optimum if have one
#         self.optimum = optimum

#         # Real bounds for X
#         self.bounds = bounds
#         if bounds != None:
#             if isinstance(bounds, list):
#                 self.bounds = bounds
#             elif isinstance(bounds, torch.Tensor):
#                 self.bounds = bounds
#             elif isinstance(bounds, np.ndarray):
#                 self.bounds = torch.from_numpy(bounds)
#             elif isinstance(bounds, jnp.ndarray):
#                 self.bounds = torch.from_numpy(np.array(bounds))
#             else:
#                 raise TypeException("Error: The type of the bound is not supported")

#         # For Multi-Objective Optimization
#         self.ref_point = ref_point
#         if ref_point != None:
#             if isinstance(ref_point, list):
#                 self.ref_point = ref_point
#             elif isinstance(ref_point, torch.Tensor):
#                 self.ref_point = ref_point
#             elif isinstance(ref_point, np.ndarray):
#                 self.ref_point = torch.from_numpy(ref_point)
#             elif isinstance(ref_point, jnp.ndarray):
#                 self.ref_point = torch.from_numpy(np.array(ref_point))
#             else:
#                 raise TypeException("Error: The type of the ref_point is not supported")
        
        
#         # Problem type
#         self.MIXED = MIXED
#         self.CONSTRAINTS = CONSTRAINTS
#         self.MULTIOBJ = MULTIOBJ

#         # Debugging
#         self.debug = debug

#         self.MAXIMIZATION = MAXIMIZATION
#         self.flag = flag

#         for key, value in kwargs.items():
#             setattr(self, key, value)


#     def INPUT_TYPE_CONVERT(self, X):
#         '''
#         Datatype (numerical computing libraries) conversion

#         Input Args:
#             X: an array-like object
#         Returns: 
#             X_converted: an array-like object with self.out_type
#         '''
        
#         # Detect input type
#         if isinstance(X, torch.Tensor):
#             self.out_type = 'torch'
#             self.device = X.device
#             X_converted = X.cpu()
#         elif isinstance(X, np.ndarray):
#             self.out_type = 'np'
#             X_converted = torch.from_numpy(X).cpu()
#         elif isinstance(X, jnp.ndarray):
#             self.out_type = 'jnp'
#             X_converted = torch.from_numpy(np.array(X)).cpu()
#             self.device = list(X.sharding.device_set)[0]
#         else:
#             raise TypeException("Error: The type of the X data is not supported")

#         return X_converted



#     def OUTPUT_TYPE_CONVERT(self, Y):
#         '''
#         Datatype (numerical computing libraries) conversion

#         Input Args:
#             Y: an array-like object
#         Returns: 
#             Y_converted: an array-like object with self.out_type
#         '''
#         if not torch.is_tensor(Y):
#             raise TypeException("Error: Y in OUTPUT_TYPE_CONVERT() is not torch tensor")
        
#         # Detect input type
#         if self.out_type == 'torch':
#             Y_converted = Y.detach().to(self.device)
#         elif self.out_type == 'np':
#             Y_converted = Y.detach().cpu().numpy()
#         elif self.out_type == 'jnp':
#             Y_converted = jnp.array(Y.detach().cpu().numpy())
#             Y_converted = jax.device_put(Y_converted, self.device)
#         else:
#             raise TypeException("Error: The type of the X data is not supported")

#         return Y_converted


#     def _evaluate_implementation(self, X):
#         """
#         Implementation of the actual evaluation logic.
#         Must be overridden by child classes for each problem.
#         """
#         raise NotImplementedError("Subclasses must implement _evaluate_implementation")

    
#     def evaluate(self, X):
#         """
#         Wrapper method that handles input/output processing for all benchmark problems.

#         Input Args:
#             X: A design variable of N-by-dim
#         Returns: 
#             gx: Constraints in a shape of N-by-#_of_constraints. None for unconstrained problems or where constraints are not needed.
#             fx: Objectives in a shape of N-by-#_of_objectives.
#         """
#         # Process input
#         X = self.INPUT_TYPE_CONVERT(X)
        
#         if self.MIXED.is_mixed:
#             X = self.mixed_int_scale(X)
#         else:
#             X = self.scale(X)
        
#         # Call the actual evaluation implementation
#         gx, fx = self._evaluate_implementation(X)

#         # Process Constraints
#         gx, fx = self.constraint_processing(gx, fx)

#         # Process if MultiObjective
#         fx = self.multiobj_processing(fx)
        
#         # Process output type
#         fx = self.OUTPUT_TYPE_CONVERT(fx)
#         if gx != None:
#             gx = self.OUTPUT_TYPE_CONVERT(gx)

#         # Negate for minimization setting
#         if not self.MAXIMIZATION:
#             fx = -fx
        
#         return gx, fx


    

    
#     def scale(self, X):
#         """
#         Scales a fully continuous X to the problem's bounds.

#         Input Args:
#             X (torch.Tensor): continuous data in range of [0, 1]

#         Returns:
#             X (torch.Tensor): continuous data scaled to bounds
#         """

#         if not torch.is_tensor(X):
#             raise TypeException("Error: X in scale() is not torch tensor")

#         if X.size(1) != self.dim:
#             raise DimensionException("Error: Incorrect X dimensions.")
#         if torch.max(X) > 1 or torch.min(X) < 0:
#             raise RangeException("Error: Incorrect X range: must be [0, 1].")

#         if not torch.is_tensor(self.bounds):
#             self.bounds = torch.tensor(self.bounds).cpu()

            
#         X_scaled = torch.add(torch.mul(X, (self.bounds[:, 1] - self.bounds[:, 0])), self.bounds[:, 0])

#         if self.debug:
#             print(f'self.bounds[:, 0]: {self.bounds[:, 0]}')
#             print(f'self.bounds[:, 1]: {self.bounds[:, 1]}')

            
#         return X_scaled

    
#     def mixed_int_scale(self, X):
#         """
#         Scales a mixed_int X to the problem's bounds.

#         Args:
#             X (torch.Tensor): continuous data in range of [0, 1]

#         Returns:
#             X (torch.Tensor): mixed-int data scaled to bounds
#         """

#         mix_configs = self.MIXED
#         if (len(mix_configs.continuous_dict) + len(mix_configs.discrete_dict)) != X.shape[1]:
#             raise DimensionException(f'Error: Incorrect mixed_bounds ({(len(mix_configs.continuous_dict) + len(mix_configs.discrete_dict))}) to X shape[1] {X.shape[1]} .')

#         X_scaled = X.clone().to(torch.float32)
#         for i in mix_configs.discrete_dict.keys():
#             X_scaled[:,i] = self.continuous_to_discrete(X[:,i], mix_configs.discrete_dict[i]).to(torch.float32)


#         continuous_index = list(mix_configs.continuous_dict.keys())
        
#         self.bounds = torch.vstack(list(mix_configs.continuous_dict.values()))
        
#         self.dim = len(continuous_index)
#         X_scaled[:,continuous_index] = self.scale(X[:,continuous_index]).to(torch.float32)

#         if self.debug:
#             print(f'X_scaled: {X_scaled}')
#             # print(f'self.dim: {self.dim}')
#             # print(f'continuous_index: {continuous_index}')
#             # print(f'self.bounds.shape: {self.bounds.shape}')
#             # print(f'X.shape: {X.shape}')
#             # print(f'X_scaled[:,continuous_index].shape: {X[:,continuous_index].shape}')
                
#         return X_scaled
            
            
        


#     def continuous_to_discrete(self, x, disc_values):
#         '''
#         (For Mixed Variable Problems)
#         Convert continuous value to discrete value
#         Input Args:
#           x: continuous value in [0, 1]
#           disc_values: discrete values
#         Returns: discrete values
#         '''
#         idx = torch.floor(x * len(disc_values)).long()
#         return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]


    
#     def constraint_processing(self, GX, FX):
#         '''
#         Function for processing the constraint output.
        
#         '''
        
#         if self.CONSTRAINTS.type == 'UNCONSTRAINED':
#             return None, FX

#         if self.CONSTRAINTS.type == 'CONSTRAINTS':
#             return GX, FX
        
#         if self.CONSTRAINTS.type == 'PENALTY':
            
#             # Equaly weighted each constraint and objective to a single objective problem
#             # In the Maximization Setting: 
#             if self.CONSTRAINTS.weight == None: 
                
#                 # relu(x) = max(0, x)
#                 violation = torch.relu(GX)
                
#                 # Penalize objective: fx - penalty_weight(=1) * sum(max(0, gx)) = sum( fx, -(max(0, gx)) )
#                 Pen_FX = torch.cat( (FX, -violation), dim=1 ).sum(dim=-1, keepdim=True)
                
#                 return None, Pen_FX
    
#             # User Define Weighted Penalty:
#             if isinstance(self.CONSTRAINTS.weight, list): 
#                 if len(self.CONSTRAINTS.weight) != (GX.shape[1] + 1):
#                     raise DimensionException("Error: The number of weights does not match constraint + 1")
#                 else:
#                     # relu(x) = max(0, x)
#                     violation = torch.relu(GX)
                    
#                     # if penalty_weight != 1
#                     Weighted_Pen_FX = ( torch.cat( (FX, -violation), dim=1 ) * torch.tensor(self.CONSTRAINTS.weight) ).sum(dim=-1, keepdim=True)
#                     return None, Weighted_Pen_FX


    
#     def multiobj_processing(self, FX):
#         '''
#         Function for processing multi-objective values.
#         '''
#         if self.MULTIOBJ.type == 'MULTI_OBJ':
#             return FX
#         elif self.MULTIOBJ.type == 'SINGLE_OBJ':
#             return FX

        
#         if self.MULTIOBJ.type == 'MERGE_TO_SINGLE':
            
#             if self.MULTIOBJ.weight == None: 
#                 Single_FX = torch.mean( FX , dim=1 , keepdim=True)
#                 return Single_FX
                
#             elif isinstance(self.MULTIOBJ.weight, list): 
#                 if len(self.MULTIOBJ.weight) != (FX.shape[1]):
#                     raise DimensionException("Error: The number of weights does not match the number of objectives")
#                 else:
#                     Weighted_FX = ( FX * torch.tensor(self.MULTIOBJ.weight) ).sum(dim=-1, keepdim=True)
#                     return Weighted_FX
                

#         if self.MULTIOBJ.type == 'SELECT_OBJ':
#             return FX[:,self.MULTIOBJ.idx]
        
        



    

#     def info(self):
#         '''
#         Print out the info of each problem.
#         '''
#         if len(self.tags) == 0:
#             print(f'This problem\'s info is not defined')
#         else:
#             for item in self.tags:
#                 print(item)
            
#             print(f'MIXED: {self.MIXED}, \n'
#                   f'MULTIOBJ: {self.MULTIOBJ}, \n'
#                   f'CONSTRAINTS: {self.CONSTRAINTS}, \n'
#                   f'MAXIMIZATION: {self.MAXIMIZATION} \n')




