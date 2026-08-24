"""Translate user text into SymPy expressions.

Supports:
  - implicit multiplication:   "2x"       -> 2*x
  - exponent caret:            "x^2"      -> x**2
  - unicode math symbols:       θ, π, ², ³, etc.
  - common constants/functions: pi, sin, cos, tan, exp, log, sqrt, etc.
  - basic LaTeX input:          \frac{a}{b}, \pi, \gamma, \left, \right, [ ], x_{0}, x^{2}
"""

import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

__all__ = [
    "parse_expression",
    "parse_equation",
    "get_free_symbols",
    "parse_variable",
]

# We no longer need convert_xor because we manually replace ^ with **
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
)

_LOCAL_DICT = {
    "x": sp.Symbol("x"),
    "y": sp.Symbol("y"),
    "z": sp.Symbol("z"),
    "t": sp.Symbol("t"),
    "theta": sp.Symbol("theta"),
    "pi": sp.pi,
    "E": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "sign": sp.sign,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "factorial": sp.factorial,
}


def _latex_to_sympy(text: str) -> str:
    """Convert common LaTeX math to SymPy‑compatible syntax."""
    # Remove surrounding \( \) or \[ \]
    if text.startswith("\\(") and text.endswith("\\)"):
        text = text[2:-2]
    elif text.startswith("\\[") and text.endswith("\\]"):
        text = text[2:-2]

    # If there is an equation L(x)=..., keep the right side
    if "=" in text:
        text = text.split("=", 1)[1]

    # Remove \left and \right (and their variants)
    text = re.sub(r"\\left\s*[.{\[]?", "", text)
    text = re.sub(r"\\right\s*[.}\]]?", "", text)

    # Replace square brackets with parentheses
    text = text.replace("[", "(").replace("]", ")")

    # Replace common LaTeX commands
    text = text.replace(r"\pi", "pi")
    text = text.replace(r"\gamma", "gamma")
    text = text.replace(r"\theta", "theta")
    text = text.replace(r"\alpha", "alpha")
    text = text.replace(r"\beta", "beta")
    text = text.replace(r"\lambda", "lambda")
    text = text.replace(r"\sigma", "sigma")
    text = text.replace(r"\omega", "omega")
    text = text.replace(r"\infty", "oo")

    # Replace \frac{a}{b} with ((a)/(b))
    def frac_repl(match):
        num = match.group(1)
        den = match.group(2)
        return f"(({num})/({den}))"
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", frac_repl, text)

    # Replace \sqrt{a} with sqrt(a)
    text = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", text)

    # Replace subscripts: x_{0} -> x0, x_{i} -> xi, etc.
    text = re.sub(r"(\w)_\{([^}]*)\}", r"\1\2", text)

    # Replace superscripts: x^{2} -> x**2
    text = re.sub(r"(\w)\^\{([^}]*)\}", r"\1**\2", text)

    # Remove any remaining curly braces
    text = text.replace("{", "").replace("}", "")

    # Remove thin spaces and other spacing commands
    text = text.replace(r"\,", "").replace(r"\;", "").replace(r"\ ", "")

    # Remove any leftover backslash
    text = text.replace("\\", "")

    return text.strip()


def _preprocess(text: str) -> str:
    """Normalize common Unicode input, superscripts, and LaTeX commands."""
    text = text.strip()
    text = text.replace("−", "-")          # Unicode minus

    # Replace common Unicode symbols with ASCII equivalents
    text = text.replace("π", "pi")
    text = text.replace("θ", "theta")
    text = text.replace("²", "^2")         # superscript two
    text = text.replace("³", "^3")         # superscript three
    # Handle other superscript digits (⁰¹⁴⁵⁶⁷⁸⁹)
    superscripts = {
        "⁰": "0", "¹": "1", "⁴": "4", "⁵": "5",
        "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"
    }
    for sup, num in superscripts.items():
        text = text.replace(sup, f"^{num}")

    # If the input looks like LaTeX, convert it
    if "\\" in text or text.startswith("\\("):
        text = _latex_to_sympy(text)

    # Directly replace caret '^' with '**' (exponent operator)
    text = text.replace("^", "**")

    return text


def parse_expression(text: str) -> sp.Expr:
    """Parse a mathematical expression string into a SymPy expression."""
    if not text or not text.strip():
        raise ValueError("Expression is empty.")

    cleaned = _preprocess(text)
    try:
        expr = parse_expr(
            cleaned,
            transformations=_TRANSFORMATIONS,
            local_dict=_LOCAL_DICT,
            evaluate=False,
        )
    except Exception as exc:
        raise ValueError(f"Could not parse expression: {text}") from exc

    return expr


def parse_equation(text: str) -> sp.Expr:
    """Parse an equation of the form 'lhs = rhs'."""
    if "=" not in text:
        raise ValueError("Equation must contain an equals sign '='.")
    lhs_text, rhs_text = text.split("=", 1)
    lhs = parse_expression(lhs_text)
    rhs = parse_expression(rhs_text)
    return sp.simplify(lhs - rhs)


def get_free_symbols(expr: sp.Expr) -> set:
    """Return the set of free symbols in a SymPy expression."""
    return expr.free_symbols


def parse_variable(name: str) -> sp.Symbol:
    """Create a SymPy symbol from a variable name."""
    name = name.strip()
    if not name:
        raise ValueError("Variable name is empty.")
    return sp.Symbol(name)