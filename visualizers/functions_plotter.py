import numpy as np
import plotly.graph_objects as go
import sympy as sp

def plot_2d_function(expr, x_min=-10, x_max=10):
    x = sp.Symbol('x')
    
    # Create a fast numerical function
    f_num = sp.lambdify(x, expr, modules=["numpy"])
    
    # Generate 500 points between x_min and x_max
    x_vals = np.linspace(x_min, x_max, 500)
    
    try:
        y_vals = f_num(x_vals)
        # Fix for straight horizontal lines (like y = 5)
        if np.isscalar(y_vals):
            y_vals = np.full_like(x_vals, y_vals)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Math Error: {e}", showarrow=False)
        return fig

    # Draw the Plotly graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='lines', 
        name=f"f(x) = {expr}", line=dict(color='#00FFCC', width=3)
    ))
    
    # Make it look like a nice math grid
    fig.update_layout(
        title=f"Graph of Expression: {expr}",
        xaxis_title="X Axis", yaxis_title="Y Axis",
        template="plotly_dark", hovermode="x unified",
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white', gridcolor='#333'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white', gridcolor='#333')
    )
    return fig