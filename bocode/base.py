import torch
from typing import Union, Tuple, Set, Optional, List

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL

import numpy as np
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

class DataType:
    """
    Data types for the decision variables.
    Available DataTypes:
        - DataType.CONTINUOUS
        - DataType.DISCRETE
        - DataType.CATEGORICAL
    """
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    MIXED = "mixed"

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
        tags: Optional[List[str]] = None,
        debug: bool = False
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

        if any(getattr(self.__class__, attr, None) is None for attr in ["available_dimensions", "num_objectives", "num_constraints", "input_type"]):
            raise NotImplementedError("This benchmark problem is not fully implemented yet.")

        self.dim = dim
        self.bounds = bounds
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.is_constrained = num_constraints > 0
        self.x_opt = x_opt
        self.optimum = optimum
        self.ref_point = ref_point
        self.tags = tags
        self.debug = debug
        
    def evaluate(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the objective and constraint functions.
        """
        constraints, values = self._evaluate_implementation(X)
        return values, constraints


    def _evaluate_implementation(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the objective and constraint functions.
        """
        raise NotImplementedError("This benchmark problem is not fully implemented yet.")

    def scale(self, X):
        """
        Scales a fully continuous X to the problem's bounds.

        Input Args:
            X (torch.Tensor): continuous data in range of (0, 1)

        Returns:
            X (torch.Tensor): continuous data scaled to bounds
        """

        if not torch.is_tensor(X):
            raise TypeException("Error: X in scale() is not torch tensor")

        if X.size(1) != self.dim:
            raise DimensionException("Error: Incorrect X dimensions.")
        if torch.max(X) > 1 or torch.min(X) < 0:
            raise RangeException("Error: Incorrect X range: must be (0, 1).")

        if not torch.is_tensor(self.bounds):
            self.bounds = torch.tensor(self.bounds).cpu()

        X_scaled = torch.add(torch.mul(X, (self.bounds[:, 1] - self.bounds[:, 0])), self.bounds[:, 0])

        return X_scaled
    
    def show_info(self):
        """
        Prints the information about the benchmark problem.
        """
        print(f"Function info:\n",
              f"Number of objectives: {self.num_objectives}\n",
              f"Number of constraints: {self.num_constraints}\n",
              f"Number of dimensions: {self.dim}\n",
              f"Optimum Value: {self.optimum}\n",
              f"Optimal Decision Variables: {self.x_opt}\n",
              f"Bounds: {self.bounds}\n",)
    
    def visualize_function(self, sampling_density=50):
        """
        Decrease sampling_density for faster rendering. Default is 50. Increase for better resolution.
        -----
        sampling_density: sampling density per axis. Number of evaluated points = sampling_density^2
        """

        if self.bounds is None or len(self.bounds) != self.dim:
            raise ValueError(f"Bounds are not set or do not match the dimension of the problem.")

        bounds = self.bounds

        if self.__class__.input_type == DataType.DISCRETE or self.__class__.input_type == DataType.MIXED:
            print("Visualization is not supported for discrete functions.")
            return

        if self.dim > 15:
            print("Visualization may take a while to render for functions with high dimensionality.")

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
                    continue
                low, high = bounds[dim]
                step = (high - low) / 100.0
                sliders.append(html.Div([
                    html.Label(f"x{dim} ="),
                    dcc.Slider(
                        id={"type": "slider", "index": int(dim)},
                        min=float(low),
                        max=float(high),
                        step=float(step),
                        value=float(mids[dim]),
                        marks={str(low): str(low), str(high): str(high)}
                    )
                ], style={"margin": "10px 0"}))
            return html.Div(sliders)

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

