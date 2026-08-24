"""Plot implicit equations of the form ``F(x, y) = 0`` using contours."""

import numpy as np
import plotly.graph_objects as go
import sympy as sp

__all__ = ["plot_implicit"]

_NUMPY_MODULES = [
    {
        "cot": lambda x: 1 / np.tan(x),
        "sec": lambda x: 1 / np.cos(x),
        "csc": lambda x: 1 / np.sin(x),
        "ln": np.log,
    },
    "numpy",
]


def plot_implicit(
    expr: sp.Expr,
    x_var: sp.Symbol,
    y_var: sp.Symbol,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid_size: int = 300,
    color: str = "#00e5ff",
    width: float = 2.2,
) -> go.Figure:
    x_min, x_max = x_range
    y_min, y_max = y_range
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid range.")

    func = sp.lambdify((x_var, y_var), expr, modules=_NUMPY_MODULES)
    x_vals = np.linspace(x_min, x_max, grid_size)
    y_vals = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    try:
        Z = func(X, Y)
        if np.isscalar(Z):
            Z = np.full_like(X, Z, dtype=float)
        else:
            Z = np.asarray(Z, dtype=float)
    except (TypeError, ValueError):
        Z = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = float(func(X[i, j], Y[i, j]))
                except (TypeError, ValueError):
                    Z[i, j] = np.nan

    contour = go.Contour(
        x=x_vals,
        y=y_vals,
        z=Z,
        contours=dict(start=0, end=0, size=0.01, coloring="lines"),
        line=dict(width=width, color=color),
        showscale=False,
        connectgaps=False,
    )

    fig = go.Figure(data=contour)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d3f5"),
        title=f"${sp.latex(expr)} = 0$",
        xaxis=dict(gridcolor="rgba(200,211,245,0.1)", zerolinecolor="rgba(200,211,245,0.2)"),
        yaxis=dict(gridcolor="rgba(200,211,245,0.1)", zerolinecolor="rgba(200,211,245,0.2)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig