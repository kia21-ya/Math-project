"""Plot standard 2D functions of the form ``y = f(x)`` with full styling control."""

import numpy as np
import plotly.graph_objects as go
import sympy as sp

__all__ = ["make_expression_trace"]

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


def make_expression_trace(
    expr: sp.Expr,
    var: sp.Symbol,
    x_range: tuple[float, float],
    n_points: int = 1000,
    name: str | None = None,
    color: str = "#00e5ff",
    dash: str = "solid",
    width: float = 2.2,
) -> go.Scatter:
    x_min, x_max = x_range
    if x_min >= x_max:
        raise ValueError("x_min must be less than x_max.")

    func = sp.lambdify(var, expr, modules=_NUMPY_MODULES)
    x_vals = np.linspace(x_min, x_max, n_points)
    y_vals = np.array([_eval_scalar(func, x) for x in x_vals], dtype=float)
    mask = np.isfinite(y_vals)

    trace = go.Scatter(
        x=x_vals[mask],
        y=y_vals[mask],
        mode="lines",
        name=name or sp.latex(expr),
        line=dict(color=color, width=width, dash=dash),
    )
    return trace