import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application
)

def parse_math_input(user_input: str):
    """
    Safely converts a string into a SymPy expression.
    Handles '^' for powers and implicit multiplication (e.g., '2x' -> '2*x').
    """
    # Replace ^ with ** for Python power syntax
    cleaned_input = user_input.replace("^", "**")
    
    # Allow 2x to be read as 2*x
    transformations = (standard_transformations + (implicit_multiplication_application,))
    
    try:
        expr = parse_expr(cleaned_input, transformations=transformations)
        return expr, None
    except Exception as e:
        return None, f"Could not understand the math: {e}"