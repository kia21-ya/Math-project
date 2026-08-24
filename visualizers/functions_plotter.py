"""Plot parametric curves and polar curves with styling options."""

import numpy as np
import plotly.graph_objects as go
import sympy as sp

__all__ = ["plot_parametric", "plot_polar"]

_NUMPY_MODULES = [
    {
        "cot": lambda x: 1 / np.tan(x),
        "sec": lambda x: 1 / np.cos(x),
        "csc": lambda x: 1 / np.sin(x),
        "ln": np.log,
    },
    "numpy",
]


def _eval_scalar(func, value) -> float:
    try:
        return float(func(value))
    except (TypeError, ValueError):
        return np.nan


def plot_parametric(
    x_expr: sp.Expr,
    y_expr: sp.Expr,
    param: sp.Symbol,
    t_range: tuple[float, float],
    n_points: int = 1000,
    color: str = "#00e5ff",
    dash: str = "solid",
    width: float = 2.2,
) -> go.Figure:
    t_min, t_max = t_range
    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max.")

    fx = sp.lambdify(param, x_expr, modules=_NUMPY_MODULES)
    fy = sp.lambdify(param, y_expr, modules=_NUMPY_MODULES)

    t_vals = np.linspace(t_min, t_max, n_points)
    x_vals = np.array([_eval_scalar(fx, t) for t in t_vals], dtype=float)
    y_vals = np.array([_eval_scalar(fy, t) for t in t_vals], dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals[mask],
            y=y_vals[mask],
            mode="lines",
            name=f"({sp.latex(x_expr)}, {sp.latex(y_expr)})",
            line=dict(color=color, width=width, dash=dash),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d3f5"),
        title=f"Parametric curve: $(x(t), y(t)) = ({sp.latex(x_expr)}, {sp.latex(y_expr)})$",
        xaxis=dict(gridcolor="rgba(200,211,245,0.1)", zerolinecolor="rgba(200,211,245,0.2)"),
        yaxis=dict(gridcolor="rgba(200,211,245,0.1)", zerolinecolor="rgba(200,211,245,0.2)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_polar(
    r_expr: sp.Expr,
    theta: sp.Symbol,
    theta_range: tuple[float, float],
    n_points: int = 1500,
    color: str = "#ff007f",
    dash: str = "solid",
    width: float = 2.2,
) -> go.Figure:
    t_min, t_max = theta_range
    if t_min >= t_max:
        raise ValueError("theta_min must be less than theta_max.")

    fr = sp.lambdify(theta, r_expr, modules=_NUMPY_MODULES)
    theta_vals = np.linspace(t_min, t_max, n_points)
    x_vals = []
    y_vals = []
    for th in theta_vals:
        r = _eval_scalar(fr, th)
        if np.isfinite(r):
            x_vals.append(r * np.cos(th))
            y_vals.append(r * np.sin(th))
        else:
            x_vals.append(np.nan)
            y_vals.append(np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=f"r({sp.latex(theta)}) = {sp.latex(r_expr)}",
            line=dict(color=color, width=width, dash=dash),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d3f5"),
        title=f"Polar curve: $r = {sp.latex(r_expr)}$",
        xaxis=dict(gridcolor="rgba(200,211,245,0.1)", zerolinecolor="rgba(200,211,245,0.2)"),
        yaxis=dict(gridcolor="rgba(200,211,245,0.1)", zerolinecolor="rgba(200,211,245,0.2)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig