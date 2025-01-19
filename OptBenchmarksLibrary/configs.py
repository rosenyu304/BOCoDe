from dataclasses import dataclass
from typing import Optional, List, Union, Any

# Exceptions
class RangeException(Exception):
    pass

class DimensionException(Exception):
    pass

class TypeException(Exception):
    pass


@dataclass
class ConstraintConfig:
    '''Configuration class for specifying optimization constraints.
    
    Attributes:
        type (str): The type of constraint handling method. Must be one of:
            - 'UNCONSTRAINED': 
                Default. Unconstrained optimization problem. No constraints returned. 
                Constraint is None for evaluation.
            - 'CONSTRAINTS': 
                Hard constraints enforced. 
                Return the constraint variable for evaluation.
            - 'PENALTY': 
                Soft constraints with penalty function.
                Constraint is None for evaluation.
                
            (Defaults to 'UNCONSTRAINED')
        
        weight (Optional[List[float]]): 
            Weights for penalty terms when using penalty-based constraints. 
            Only relevant when type='PENALTY'. Defaults to None.
    
    Raises:
        ValueError: If type is not one of the valid constraint handling methods defined
            in VALID_TYPES.
    
    Example:
        >>> config = ConstraintConfig(type='PENALTY', weight=[1.0, 0.5])
        >>> config.type
        'PENALTY'
    '''
    type: str = 'UNCONSTRAINED'
    weight: Optional[List[float]] = None
    
    VALID_TYPES = {'UNCONSTRAINED', 'CONSTRAINTS', 'PENALTY'}
    
    def __post_init__(self):
        if self.type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}, got {self.type}")


@dataclass
class MultiObjConfig:
    '''Configuration class for multi-objective problems.
    
    Attributes:
        type (str): The type of objectives handling method. Must be one of:
            - 'SINGLE_OBJ': 
                Default. Single objective function. 
                The returned f(x) objective will be N-by-1.
            - 'MULTI_OBJ': 
                Multi-objective function with M objectives.
                The returned f(x) objective will be N-by-M.
            - 'MERGE_TO_SINGLE': 
                Weighted multi-objective function transformed to single objective.
                If weight is not provided, we weight each objective equally.
                    > "base.py": torch.mean( FX , dim=1 , keepdim=True)
                The returned f(x) objective will be N-by-1.
            - 'SELECT_OBJ':
                Select one of the functions of the multi-objective problems.
                Must provide the idx (index) of the selected function.
                The returned f(x) objective will be N-by-1.
                
            (Defaults to 'SINGLE_OBJ')
        
        weight (Optional[List[float]]): 
            Weights for weighting multi-objective function transformed to single objective. 
            Only relevant when type='MERGE_TO_SINGLE'. Defaults to None (weighted equally).

        idx (Optional[List[int]]):
            Index for the objective you selected for evaluation. Defaults to None.
    
    Raises:
        ValueError: If type is not one of the valid objective handling methods defined
            in VALID_TYPES.
    
    Example:
        >>> config = MixIntConfig(type='MERGE_TO_SINGLE', weight=[0.4, 0.6])
        >>> config.type
        'MERGE_TO_SINGLE'
    '''
    type: str = 'SINGLE_OBJ'
    idx: Optional[List[int]] = None
    weight: Optional[List[float]] = None
    
    VALID_TYPES = {'SINGLE_OBJ', 'MULTI_OBJ', 'MERGE_TO_SINGLE', 'SELECT_OBJ'}
    
    def __post_init__(self):
        if self.type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}, got {self.type}")
        
        if (self.type =='SELECT_OBJ') & (self.idx==None):
            raise NotImplementedError(f"Missing idx: For {self.VALID_TYPES} problem, the self.idx have to be defined.")

        if (self.idx != None) & (not isinstance(self.idx, list)):
            raise TypeException(f"Variable type for MultiObjConfig.idx have to be a list. For example: MixIntConfig(type='MERGE_TO_SINGLE', idx=[0]).")
            
        if (self.weight != None) & (not isinstance(self.weight, list)):
            raise TypeException(f"Variable type for MultiObjConfig.weight have to be a list. For example: MixIntConfig(type='MERGE_TO_SINGLE', weight=[0.4, 0.6]).")


@dataclass
class MixIntConfig:
    '''Configuration class for mixed-integer problem.
    
    Attributes:
        is_mixed (bool): 
            False for continuous problem. True for mix-int problems.

        discrete_dict (Optional[dict]): 
            A dictionary that stores the indicies and the discrete values for the design variable x. 
            The key is the indices of the X columns where they are discrete. 
            The values should be a torch tensor that has the discrete values we can select from.
            For example:
            >>> discrete_dict =  {0: tensor([1.1, 2.2, 3.3]), 
                                  1: tensor([0., 1., 2., 3.]), 
                                  2: tensor([0.3, 0.8, 0.65])
        
        continuous_dict (Optional[dict]):
            A dictionary that stores the indicies and the continuous values for the design variable x. 
            The key is the indices of the X columns where they are continuous. 
            The values should be a torch tensor bouds for continuous values.
            For example:
            >>> continuous_dict ={0: tensor([0., 3.]), 
                                  1: tensor([-5., 5.]))
        
    '''
    is_mixed: bool = False
    discrete_dict: Optional[dict] = None
    continuous_dict: Optional[dict] = None











