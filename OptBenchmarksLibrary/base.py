import torch
from .configs import *
from typing import Union, Tuple, Set, Optional

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL

import numpy as np

class BenchmarkProblem:

    available_dimensions = None
    num_objectives = None

    def __init__(
        self,

        dim: int = 1,
        bounds: Union[List[Union[Tuple, Set]], None] = None,
        num_objectives: int = 1,
        num_constraints: int = 0,
        x_opt: Optional[torch.Tensor] = None,
        optimum: Optional[torch.Tensor] = None,
        ref_point: Optional[torch.Tensor] = None,
        CONSTRAINTS: Optional[ConstraintConfig] = None,
        tags: Optional[List[str]] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the BenchmarkProblem class.
        
        Args:
            dim (int, optional): Dimension of the decision space. Defaults to 1.
            bound (Union[Tuple, Set]): Bounds of the decision variables.
            num_objectives (int, optional): Number of objective functions. Defaults to 1.
            num_constraints (int, optional): Number of constraint functions. Defaults to 0.
            x_opt (torch.Tensor, optional): The decision variables that maximize the objective function(s). Defaults to None.
            optimum (torch.Tensor, optional): The optimal objective values corresponding to the x_opt. Defaults to None.
            ref_point (torch.Tensor, optional): Reference point for calculating hypervolume. Defaults to None.
            constraints (ConstraintConfig, optional): Additional Constraints. Defaults to None.
            tags (List[str], optional): More information for the benchmark problem. Defaults to None.
            debug (bool, optional): Debugging flag. Defaults to False.
        """

        if self.__class__.available_dimensions is None or self.__class__.num_objectives is None:
            raise NotImplementedError("This benchmark problem is not fully implemented yet.")
        
        if len(bounds) != dim:
            raise ValueError(f"Error: The number of bounds ({len(bounds)}) does not match the dimension ({dim}).")

        self.dim = dim
        self.bounds = bounds
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.is_constrained = num_constraints > 0
        self.x_opt = x_opt
        self.optimum = optimum
        self.ref_point = ref_point
        self.CONSTRAINTS = CONSTRAINTS
        self.tags = tags

    def scale(self, X):
        """
        Scales a fully continuous X to the problem's bounds.

        Input Args:
            X (torch.Tensor): continuous data in range of [0, 1]

        Returns:
            X (torch.Tensor): continuous data scaled to bounds
        """

        if not torch.is_tensor(X):
            raise TypeException("Error: X in scale() is not torch tensor")

        if X.size(1) != self.dim:
            raise DimensionException("Error: Incorrect X dimensions.")
        if torch.max(X) > 1 or torch.min(X) < 0:
            raise RangeException("Error: Incorrect X range: must be [0, 1].")

        if not torch.is_tensor(self.bounds):
            self.bounds = torch.tensor(self.bounds).cpu()

        X_scaled = torch.add(torch.mul(X, (self.bounds[:, 1] - self.bounds[:, 0])), self.bounds[:, 0])

        return X_scaled
    
    def visualize_function(self, sampling_density=50):
        """
        Decrease sampling_density for faster rendering. Default is 50. Increase for better resolution.
        -----
        sampling_density: sampling density per axis. Number of evaluated points = sampling_density^2
        """

        bounds = self.bounds

        if any(isinstance(elem, list) for elem in bounds):
            print("Visualization is not supported for discrete functions.")
            return

        D = self.dim
        M = self.num_objectives

        # sample random points
        # move the following inside if statement later
        num_samples = 500
        lbs = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        ubs = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        X_rand = lbs + (ubs - lbs) * torch.rand(num_samples, D)
        Y_rand = self._evaluate_implementation(X_rand)[1]

        Xn = X_rand.detach().cpu().numpy()
        Yn = Y_rand.detach().cpu().numpy()

        # 1D input, M outputs --> 2D lines
        if D == 1 and M>=1:
            for i in range(M):
                xs = np.linspace(bounds[0][0], bounds[0][1], sampling_density)
                ys = (
                    self._evaluate_implementation(torch.from_numpy(xs.reshape(-1, 1).astype(np.float32)))[1]
                    .detach()
                    .cpu()
                    .numpy()[:, i]
                    .ravel()
                )
                plt.figure()
                plt.plot(xs, ys, "-")
                plt.scatter(Xn.ravel(), Yn[:, i].ravel(), alpha=0.3)
                plt.xlabel("x₀")
                plt.ylabel("f(x)")
                plt.title(self.__class__.__name__+f" Objective {i+1} Function Visualization")
                plt.show()
            return

        # 2D input, M objectives --> 3D surfaces
        if D == 2 and M>=1:
            # create grid
            for i in range(M):
                xs = np.linspace(bounds[0][0], bounds[0][1], sampling_density)
                ys = np.linspace(bounds[1][0], bounds[1][1], sampling_density)
                Xg, Yg = np.meshgrid(xs, ys)
                pts = torch.from_numpy(
                    np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)
                )
                Zg = (
                    self._evaluate_implementation(pts)[1]
                    .detach()
                    .cpu()
                    .numpy()[:, i]
                    .reshape(sampling_density, sampling_density)
                )

                surf = go.Surface(x=Xg, y=Yg, z=Zg, colorscale="Viridis", opacity=0.7, name="surface")
                fig = go.Figure(data=[surf])
                fig.update_layout(
                    scene=dict(
                        xaxis_title="x₀",
                        yaxis_title="x₁",
                        zaxis_title=f"f{i}(x)",
                    ),
                    title=self.__class__.__name__+f" Objective {i+1} Function Visualization",
                    width=800,
                    height=700,
                )
                fig.show()
            return
        
        lbs = np.array([b[0] for b in bounds], dtype=float)
        ubs = np.array([b[1] for b in bounds], dtype=float)
        mids = (lbs + ubs) / 2.0

        # populate dropdown options
        pair_options = [
            {"label": f"x{i} vs x{j}", "value": f"{i},{j}"}
            for i in range(D) for j in range(i+1, D)
        ]
        obj_options = [{"label": f"f_{k}", "value": str(k)} for k in range(M)]

        app = dash.Dash(__name__)

        # OLD LAYOUT: VERITCAL STACKING
        # app.layout = html.Div([
        #     html.Div([
        #         html.Label("Dimension pair:"),
        #         dcc.Dropdown(id="dimension-pair", options=pair_options, value=pair_options[0]["value"])
        #     ], style={"width":"30%", "display":"inline-block"}),
        #     html.Div([
        #         html.Label("Objective:"),
        #         dcc.Dropdown(id="objective", options=obj_options, value="0")
        #     ], style={"width":"20%", "display":"inline-block", "marginLeft":"2%"}),
        #     html.Div(id="sliders", style={"marginTop":"20px"}),
        #     dcc.Graph(id="graph", style={"height":"80vh"})
        # ])

        app.layout = html.Div([
            html.Div([
                html.Div([
                    html.Label("Dimension pair:"),
                    dcc.Dropdown(
                        id="dimension-pair",
                        options=pair_options,
                        value=pair_options[0]["value"]
                    )
                ], style={"width": "30%", "display": "inline-block"}),
                html.Div([
                    html.Label("Objective:"),
                    dcc.Dropdown(
                        id="objective",
                        options=obj_options,
                        value="0"
                    )
                ], style={"width": "20%", "display": "inline-block", "marginLeft": "2%"})
            ], style={"marginBottom": "20px"}),

            # graph + sliders next to each other horizontally
            html.Div([
                dcc.Loading(
                    id="loading-graph",
                    type="circle",
                    fullscreen=True,
                    style={"backgroundColor": "rgba(255,255,255,0.5)"}, # change opacity of loader
                    children=dcc.Graph(
                        id="graph",
                        style={"width": "100%", "height": "100%", "marginRight": "0px"}
                    )
                ),

                html.Div(
                    id="sliders",
                    style={
                        "width": "100%",
                        "paddingLeft": "0px",
                        "boxSizing": "border-box",
                        "marginLeft": "0px"
                    }
                )
            ], style={
                "display": "flex",
                "alignItems": "flex-start"
            })
        ])

        # regenerate sliders
        @app.callback(
            Output("sliders", "children"),
            Input("dimension-pair", "value")
        )
        def update_sliders(pair_value):
            i, j = map(int, pair_value.split(","))
            sliders = []
            for dim in range(D):
                if dim in (i, j):
                    # print(f"Skipping slider for dimension {dim}")
                    continue
                low, high = bounds[dim]
                step = (high - low) / 100.0
                sliders.append(html.Div([
                    html.Label(f"x{dim} ="),
                    dcc.Slider(
                        id={"type":"slider","index": dim},
                        min=low, max=high, step=step, value=mids[dim],
                        marks={low:str(low), high:str(high)}
                    )
                ], style={"margin":"10px 0"}))
            return sliders

        # updates 3d visualization when anything updates
        @app.callback(
            Output("graph", "figure"),
            Input("dimension-pair", "value"),
            Input("objective", "value"),
            Input({"type":"slider","index": ALL}, "value"),
            State({"type":"slider","index": ALL}, "id")
        )
        def update_graph(pair_value, obj_value, slider_vals, slider_ids):
            i, j = map(int, pair_value.split(","))
            obj = int(obj_value)

            # start from mids, overwrite with slider settings
            x_fixed = mids.copy()
            for val, id_dict in zip(slider_vals, slider_ids):
                x_fixed[id_dict["index"]] = val

            # do grid
            xi = np.linspace(bounds[i][0], bounds[i][1], sampling_density)
            xj = np.linspace(bounds[j][0], bounds[j][1], sampling_density)
            Xi, Xj = np.meshgrid(xi, xj)
            pts = np.tile(x_fixed, (sampling_density*sampling_density, 1))
            pts[:, i] = Xi.ravel()
            pts[:, j] = Xj.ravel()

            with torch.no_grad():
                Y = self._evaluate_implementation(torch.from_numpy(pts.astype(np.float32)))[1]
            Z = Y.detach().cpu().numpy()[:, obj].reshape(sampling_density, sampling_density)

            surf = go.Surface(x=Xi, y=Xj, z=Z, colorscale="Viridis", opacity=0.8)
            fig = go.Figure(data=[surf])
            fig.update_layout(
                title=f"{self.__class__.__name__} cross‐section (x{i}, x{j}) & f_{obj}",
                scene=dict(
                    xaxis_title=f"x{i}",
                    yaxis_title=f"x{j}",
                    zaxis_title=f"f_{obj}(x)"
                ),
                width=800, height=700,
            )
            return fig

        print("Go to http://127.0.0.1:8050/ to view the visualization.")
        app.run(debug=False)

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
                 # x_optimum = [[]],
                 # optimum = [[]], 
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




