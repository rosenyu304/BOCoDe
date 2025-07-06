import dash
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State

from bocode.base import BenchmarkProblem, DataType


def visualize_function(prob: BenchmarkProblem, sampling_density: int = 50) -> None:
    """
    Decrease sampling_density for faster rendering. Default is 50. Increase for better resolution.
    -----
    sampling_density: sampling density per axis. Number of evaluated points = sampling_density^2 for any problem with 2 or more decision variables
    """

    if prob.bounds is None or len(prob.bounds) != prob.dim:
        raise ValueError(
            "Bounds are not set or do not match the dimension of the problem."
        )

    bounds = prob.bounds

    if (
        prob.__class__.input_type == DataType.DISCRETE
        or prob.__class__.input_type == DataType.MIXED
    ):
        print("Visualization is not supported for discrete functions.")
        return

    if prob.dim > 15:
        print(
            "Visualization may take a while to render for functions with high dimensionality."
        )

    D = prob.dim
    M = prob.num_objectives

    # sample random points
    # move the following inside if statement later
    num_samples = 500
    lbs = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
    ubs = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
    X_rand = lbs + (ubs - lbs) * torch.rand(num_samples, D)
    Y_rand = prob.evaluate(X_rand)[0]

    Xn = X_rand.detach().cpu().numpy()
    Yn = Y_rand.detach().cpu().numpy()

    # 1D input, M outputs --> 2D lines
    if D == 1 and M >= 1:
        for i in range(M):
            xs = np.linspace(bounds[0][0], bounds[0][1], sampling_density)
            ys = (
                prob.evaluate(torch.from_numpy(xs.reshape(-1, 1).astype(np.float32)))[0]
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
            plt.title(
                prob.__class__.__name__ + f" Objective {i + 1} Function Visualization"
            )
            plt.show()
        return

    # 2D input, M objectives --> 3D surfaces
    if D == 2 and M >= 1:
        # create grid
        for i in range(M):
            xs = np.linspace(bounds[0][0], bounds[0][1], sampling_density)
            ys = np.linspace(bounds[1][0], bounds[1][1], sampling_density)
            Xg, Yg = np.meshgrid(xs, ys)
            pts = torch.from_numpy(
                np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)
            )
            Zg = (
                prob.evaluate(pts)[0]
                .detach()
                .cpu()
                .numpy()[:, i]
                .reshape(sampling_density, sampling_density)
            )

            surf = go.Surface(
                x=Xg, y=Yg, z=Zg, colorscale="Viridis", opacity=0.7, name="surface"
            )
            fig = go.Figure(data=[surf])
            fig.update_layout(
                scene=dict(
                    xaxis_title="x₀",
                    yaxis_title="x₁",
                    zaxis_title=f"f{i}(x)",
                ),
                title=prob.__class__.__name__
                + f" Objective {i + 1} Function Visualization",
                width=800,
                height=700,
            )
            fig.show()
        return

    # 3+ dimensional input, M objectives --> 3D cross section surfaces with sliders
    lbs = np.array([b[0] for b in bounds], dtype=float)
    ubs = np.array([b[1] for b in bounds], dtype=float)
    mids = (lbs + ubs) / 2.0

    # populate dropdown options
    pair_options = [
        {"label": f"x{i} vs x{j}", "value": f"{i},{j}"}
        for i in range(D)
        for j in range(i + 1, D)
    ]
    obj_options = [{"label": f"f_{k}", "value": str(k)} for k in range(M)]

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Dimension pair:"),
                            dcc.Dropdown(
                                id="dimension-pair",
                                options=pair_options,
                                value=pair_options[0]["value"],
                            ),
                        ],
                        style={"width": "30%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            html.Label("Objective:"),
                            dcc.Dropdown(
                                id="objective", options=obj_options, value="0"
                            ),
                        ],
                        style={
                            "width": "20%",
                            "display": "inline-block",
                            "marginLeft": "2%",
                        },
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            # graph + sliders next to each other horizontally
            html.Div(
                [
                    dcc.Loading(
                        id="loading-graph",
                        type="circle",
                        fullscreen=True,
                        style={
                            "backgroundColor": "rgba(255,255,255,0.5)"
                        },  # change opacity of loader
                        children=dcc.Graph(
                            id="graph",
                            style={
                                "width": "100%",
                                "height": "100%",
                                "marginRight": "0px",
                            },
                        ),
                    ),
                    html.Div(
                        id="sliders",
                        style={
                            "width": "100%",
                            "paddingLeft": "0px",
                            "boxSizing": "border-box",
                            "marginLeft": "0px",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "flex-start"},
            ),
        ]
    )

    # regenerate sliders
    @app.callback(Output("sliders", "children"), Input("dimension-pair", "value"))
    def update_sliders(pair_value):
        i, j = map(int, pair_value.split(","))
        sliders = []
        for dim in range(D):
            if dim in (i, j):
                continue
            low, high = bounds[dim]
            step = (high - low) / 100.0
            sliders.append(
                html.Div(
                    [
                        html.Label(f"x{dim} ="),
                        dcc.Slider(
                            id={"type": "slider", "index": int(dim)},
                            min=float(low),
                            max=float(high),
                            step=float(step),
                            value=float(mids[dim]),
                            marks={str(low): str(low), str(high): str(high)},
                        ),
                    ],
                    style={"margin": "10px 0"},
                )
            )
        return html.Div(sliders)

    # updates 3d visualization when anything updates
    @app.callback(
        Output("graph", "figure"),
        Input("dimension-pair", "value"),
        Input("objective", "value"),
        Input({"type": "slider", "index": ALL}, "value"),
        State({"type": "slider", "index": ALL}, "id"),
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
        pts = np.tile(x_fixed, (sampling_density * sampling_density, 1))
        pts[:, i] = Xi.ravel()
        pts[:, j] = Xj.ravel()

        with torch.no_grad():
            Y = prob.evaluate(torch.from_numpy(pts.astype(np.float32)))[0]
        Z = Y.detach().cpu().numpy()[:, obj].reshape(sampling_density, sampling_density)

        surf = go.Surface(x=Xi, y=Xj, z=Z, colorscale="Viridis", opacity=0.8)
        fig = go.Figure(data=[surf])
        fig.update_layout(
            title=f"{prob.__class__.__name__} cross‐section (x{i}, x{j}) & f_{obj}",
            scene=dict(
                xaxis_title=f"x{i}", yaxis_title=f"x{j}", zaxis_title=f"f_{obj}(x)"
            ),
            width=800,
            height=700,
        )
        return fig

    print("Go to http://127.0.0.1:8050/ to view the visualization.")
    app.run(debug=False)
