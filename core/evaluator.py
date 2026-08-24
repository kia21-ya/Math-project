"""Calculus and numeric evaluation for the MathLab backend."""

import sympy as sp

__all__ = [
    "derivative",
    "integral",
    "evaluate_expression",
]


def derivative(expr: sp.Expr, var: sp.Symbol, order: int = 1) -> sp.Expr:
    if order < 1:
        raise ValueError("Derivative order must be at least 1.")
    return sp.diff(expr, var, order)


def integral(
    expr: sp.Expr,
    var: sp.Symbol,
    lower=None,
    upper=None,
) -> sp.Expr:
    if lower is not None and upper is not None:
        return sp.integrate(expr, (var, lower, upper))
    return sp.integrate(expr, var)


def evaluate_expression(expr: sp.Expr, substitutions: dict | None = None):
    if substitutions is None:
        substitutions = {}
    subs = {}
    for key, value in substitutions.items():
        symbol = sp.Symbol(key) if isinstance(key, str) else key
        if isinstance(value, str):
            value = sp.N(value)
        subs[symbol] = value
    return sp.N(expr.subs(subs))