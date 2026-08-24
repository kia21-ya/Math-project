"""Plot 3D surfaces of the form ``z = f(x, y)`` using Plotly."""

import numpy as np
import plotly.graph_objects as go
import sympy as sp

__all__ = ["plot_surface"]

_NUMPY_MODULES = [
    {
        "cot": lambda x: 1 / np.tan(x),
        "sec": lambda x: 1 / np.cos(x),
        "csc": lambda x: 1 / np.sin(x),
        "ln": np.log,
    },
    "numpy",
]


def plot_surface(
    expr: sp.Expr,
    x_var: sp.Symbol,
    y_var: sp.Symbol,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid_size: int = 100,
    colorscale: str = "Viridis",
    show_contours: bool = True,
    opacity: float = 1.0,
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
                except (TypeError, ValueError, ZeroDivisionError):
                    Z[i, j] = np.nan

    Z = np.ma.masked_invalid(Z)

    surface = go.Surface(
        x=x_vals,
        y=y_vals,
        z=Z,
        colorscale=colorscale,
        opacity=opacity,
        contours={
            "z": {"show": show_contours, "usecolormap": True,
                  "highlightcolor": "#ffffff", "project": {"z": True}}
        } if show_contours else None,
        name=sp.latex(expr),
    )

    fig = go.Figure(data=surface)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d3f5"),
        title=f"Surface: $z = {sp.latex(expr)}$",
        scene=dict(
            xaxis_title=str(x_var),
            yaxis_title=str(y_var),
            zaxis_title="z",
            xaxis=dict(gridcolor="rgba(200,211,245,0.1)"),
            yaxis=dict(gridcolor="rgba(200,211,245,0.1)"),
            zaxis=dict(gridcolor="rgba(200,211,245,0.1)"),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig