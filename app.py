"""
MathLab – Graphical Representation of Algebraic Expressions, Equations, and Functions.

Pages:
  - Home (Interactive 3D Animation with Fullscreen)
  - Algebraic Expressions (2D & 3D with Fullscreen)
  - Algebraic Equations (Implicit & Explicit)
  - Surface Plotter (3D with Fullscreen)
  - Lorenz System (Animated 3D Attractor with Fullscreen)
  - Parametric & Polar Curves
"""

import json
import os
from pathlib import Path

import streamlit as st
import sympy as sp
import plotly
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Core modules
from core.parser import parse_expression, parse_equation, get_free_symbols, parse_variable
from core.evaluator import evaluate_expression

# Visualizer modules
from visualizers.expressions_plotter import make_expression_trace
from visualizers.equations_plotter import plot_implicit
from visualizers.functions_plotter import plot_parametric, plot_polar

# Lorenz plotter
try:
    from visualizers.lorenz_plotter import plot_lorenz_attractor
except ImportError:
    st.error("visualizers/lorenz_plotter.py not found. Please create it.")

# Inline surface plotter
_NUMPY_MODULES = [
    {
        "cot": lambda x: 1 / np.tan(x),
        "sec": lambda x: 1 / np.cos(x),
        "csc": lambda x: 1 / np.sin(x),
        "ln": np.log,
    },
    "numpy",
]

def plot_surface(expr, x_var, y_var, x_range, y_range, grid_size=100,
                 colorscale="Viridis", show_contours=True, opacity=1.0):
    """Plot 3D surface z = f(x, y)."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid range.")

    f = sp.lambdify((x_var, y_var), expr, modules=_NUMPY_MODULES)
    x_vals = np.linspace(x_min, x_max, grid_size)
    y_vals = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    try:
        Z = f(X, Y)
        if np.isscalar(Z):
            Z = np.full_like(X, Z, dtype=float)
        else:
            Z = np.asarray(Z, dtype=float)
    except Exception:
        Z = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = float(f(X[i, j], Y[i, j]))
                except Exception:
                    Z[i, j] = np.nan

    Z = np.ma.masked_invalid(Z)

    surface = go.Surface(
        x=x_vals, y=y_vals, z=Z,
        colorscale=colorscale, opacity=opacity,
        contours={"z": {"show": show_contours, "usecolormap": True,
                         "highlightcolor": "#ffffff", "project": {"z": True}}}
                 if show_contours else None,
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
            xaxis=dict(gridcolor="rgba(200,211,245,0.1)", showbackground=False),
            yaxis=dict(gridcolor="rgba(200,211,245,0.1)", showbackground=False),
            zaxis=dict(gridcolor="rgba(200,211,245,0.1)", showbackground=False),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
    )
    return fig

st.set_page_config(page_title="MathLab | Graphical Representation", page_icon="📐", layout="wide")

def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def base_layout(fig, title, is_3d=False):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d3f5"),
        title=title,
        xaxis=dict(gridcolor="rgba(200,211,245,0.1)", zerolinecolor="rgba(200,211,245,0.2)"),
        yaxis=dict(gridcolor="rgba(200,211,245,0.1)", zerolinecolor="rgba(200,211,245,0.2)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    if is_3d:
        fig.update_layout(height=700)
    return fig

def display_plot_with_fullscreen(fig, key="plot"):
    """
    Display a Plotly figure with a fullscreen toggle button.
    """
    # Use container to hold the plot
    container = st.container()
    
    # Add fullscreen toggle
    col1, col2 = container.columns([1, 8])
    with col1:
        fullscreen = st.checkbox("🔍 Fullscreen", key=f"fullscreen_{key}")
    
    # Display the plot
    if fullscreen:
        st.plotly_chart(fig, use_container_width=True, height=900, key=f"plot_{key}")
    else:
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{key}")

def get_plotly_js():
    """Return locally installed Plotly.js content for offline use."""
    try:
        plotly_dir = os.path.dirname(plotly.__file__)
        js_path = os.path.join(plotly_dir, "package_data", "plotly.min.js")
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8") as f:
                return f.read()
        for root, _, files in os.walk(plotly_dir):
            for file in files:
                if file == "plotly.min.js":
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        return f.read()
    except Exception:
        pass
    return ""

# ==================== HOME PAGE ====================
def page_home():
    st.title("📐 MathLab")
    st.markdown(
        """
        **Advanced Graphical Representation of Algebraic Expressions, Equations, and Functions.**
        Explore, visualise, and interact with mathematics like never before.
        Use the sidebar to navigate between different plotting tools.
        """
    )
    st.subheader("Interactive 3D Surface Animation")
    st.write(
        "Rotate, zoom, and pan while the animation runs. "
        "Press **Pause** to stop, **Play** to resume."
    )

    with st.form("animation_settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            grid_size = st.slider("Grid density", 30, 120, 75, 5)
        with col2:
            num_frames = st.slider("Number of frames", 20, 200, 100, 10)
        with col3:
            fps = st.slider("Animation speed (fps)", 1, 30, 10, 1)
        fullscreen = st.checkbox("🔍 Fullscreen mode", value=False)
        submitted = st.form_submit_button("Generate Animation", type="primary")

    if submitted:
        with st.spinner("Generating animation..."):
            x = np.linspace(-5, 5, grid_size)
            y = np.linspace(-5, 5, grid_size)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            z_frames = []
            for i in range(num_frames):
                t = i / num_frames * 2 * np.pi
                Z = np.sin(R - t) * np.cos(X * 0.2) / (R * 0.5 + 1)
                z_frames.append(np.ma.masked_invalid(Z).tolist())
            plotly_js = get_plotly_js()
            if plotly_js:
                height = 900 if fullscreen else 700
                html = f"""
                <!DOCTYPE html><html><head><style>
                    body {{ margin:0; padding:0; background:#111; }}
                    #plot {{ width:100%; height:{height}px; }}
                    .controls {{ position:absolute; top:10px; left:10px; z-index:10; }}
                    button {{ background:#00e5ff; border:none; color:#111; padding:8px 16px; margin:5px; cursor:pointer; border-radius:5px; }}
                </style></head><body>
                <div class="controls"><button id="playBtn">Play</button><button id="pauseBtn">Pause</button></div>
                <div id="plot"></div>
                <script>{plotly_js}</script>
                <script>
                    const x = {json.dumps(x.tolist())};
                    const y = {json.dumps(y.tolist())};
                    const zFrames = {json.dumps(z_frames)};
                    const frameCount = zFrames.length;
                    const intervalMs = {1000 // fps};
                    let currentFrame = 0, isPlaying = true, timerId;
                    const trace = {{ type:'surface', x:x, y:y, z:zFrames[0], colorscale:'Plasma', showscale:false }};
                    const layout = {{ template:'plotly_dark', paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
                        scene: {{ xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false, range:[-0.8,0.8]}} }} }};
                    Plotly.newPlot('plot', [trace], layout, {{responsive:true}}).then(() => start());
                    function start() {{ if (timerId) clearInterval(timerId); isPlaying=true;
                        timerId = setInterval(() => {{ if (!isPlaying) return; currentFrame=(currentFrame+1)%frameCount;
                        Plotly.restyle('plot', {{z:[zFrames[currentFrame]]}}, [0]); }}, intervalMs); }}
                    function stop() {{ isPlaying=false; clearInterval(timerId); timerId=null; }}
                    document.getElementById('playBtn').onclick = () => {{ if (!isPlaying) start(); }};
                    document.getElementById('pauseBtn').onclick = stop;
                </script></body></html>
                """
                st.components.v1.html(html, height=height+50, scrolling=False)
            else:
                fig = go.Figure(data=go.Surface(z=z_frames[0], colorscale="Plasma"))
                fig.update_layout(template="plotly_dark", height=700)
                display_plot_with_fullscreen(fig, key="home_fallback")
                st.warning("Interactive animation unavailable (Plotly.js not found). Displaying static surface.")

# ==================== ALGEBRAIC EXPRESSIONS PAGE ====================
def page_algebraic_expressions():
    st.header("📊 Algebraic Expressions")
    st.markdown(
        """
        Plot **expressions** in 2D or 3D. Enter an expression with:
        - One variable → 2D graph ($y = f(x)$)
        - Two variables → 3D surface ($z = f(x,y)$)
        """
    )

    expr_input = st.text_input(
        "Expression",
        value="sin(x) * cos(y)",
        help="Examples: sin(x), x**2 + y**2, exp(-x**2 - y**2)"
    )

    if expr_input:
        try:
            expr = parse_expression(expr_input)
            free_symbols = get_free_symbols(expr)
            var_names = [str(s) for s in free_symbols if str(s) not in {"pi", "E"}]

            if len(var_names) == 0:
                st.error("No variables detected.")
                return

            if len(var_names) == 1:
                x_var = sp.Symbol(var_names[0])
                with st.expander("2D Plot Settings", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        x_min = st.number_input("x min", value=-10.0, step=1.0)
                        x_max = st.number_input("x max", value=10.0, step=1.0)
                    with col2:
                        n_points = st.slider("Number of points", 100, 5000, 1000, 100)
                        color = st.color_picker("Colour", value="#00e5ff")

                fig = go.Figure()
                trace = make_expression_trace(
                    expr, x_var, (x_min, x_max),
                    n_points=n_points,
                    name=sp.latex(expr),
                    color=color,
                    dash="solid",
                    width=2.2,
                )
                fig.add_trace(trace)
                fig = base_layout(fig, f"Graph of $f({var_names[0]}) = {sp.latex(expr)}$")
                display_plot_with_fullscreen(fig, key="expr_2d")

            elif len(var_names) == 2:
                x_var = sp.Symbol(var_names[0])
                y_var = sp.Symbol(var_names[1])
                with st.expander("3D Plot Settings", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_min = st.number_input("x min", value=-5.0, step=0.5)
                        x_max = st.number_input("x max", value=5.0, step=0.5)
                    with col2:
                        y_min = st.number_input("y min", value=-5.0, step=0.5)
                        y_max = st.number_input("y max", value=5.0, step=0.5)
                    with col3:
                        grid_size = st.slider("Grid density", 20, 300, 100, 10)
                        colorscale = st.selectbox("Colour scale", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"])

                fig = plot_surface(
                    expr, x_var, y_var,
                    (x_min, x_max), (y_min, y_max),
                    grid_size=grid_size, colorscale=colorscale,
                    show_contours=True, opacity=1.0,
                )
                display_plot_with_fullscreen(fig, key="expr_3d")

            else:
                st.warning(f"Expression contains {len(var_names)} variables. Enter 1 or 2 variables.")

        except Exception as exc:
            st.error(f"Error parsing expression: {exc}")

# ==================== ALGEBRAIC EQUATIONS PAGE ====================
def page_algebraic_equations():
    st.header("🔄 Algebraic Equations")
    st.markdown(
        """
        Plot **equations** of the form $F(x, y) = 0$ (implicit) or $y = f(x)$ (explicit).
        Enter an equation with an equals sign.
        """
    )

    eq_input = st.text_input(
        "Equation",
        value="x**2 + y**2 = 25",
        help="Examples: x**2 + y**2 = 25, y = sin(x), x**2/4 + y**2/9 = 1"
    )

    if st.button("Plot Equation", type="primary"):
        try:
            expr = parse_equation(eq_input)
            free = get_free_symbols(expr)
            sym_names = [str(s) for s in free]

            if "x" in sym_names and "y" in sym_names:
                with st.expander("Implicit Plot Settings", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        x_min = st.number_input("x min", value=-8.0, step=1.0)
                        x_max = st.number_input("x max", value=8.0, step=1.0)
                    with col2:
                        y_min = st.number_input("y min", value=-8.0, step=1.0)
                        y_max = st.number_input("y max", value=8.0, step=1.0)
                    with col3:
                        grid_size = st.slider("Grid density", 50, 1000, 300, 50)
                    with col4:
                        color = st.color_picker("Contour colour", value="#00e5ff")
                        width = st.slider("Line width", 0.5, 6.0, 2.2, 0.1)
                        show_grid = st.checkbox("Show grid", value=True)

                fig = plot_implicit(
                    expr, sp.Symbol("x"), sp.Symbol("y"),
                    (x_min, x_max), (y_min, y_max),
                    grid_size=grid_size, color=color, width=width,
                )
                if not show_grid:
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=False)
                display_plot_with_fullscreen(fig, key="implicit")

            elif len(sym_names) == 1:
                var = sym_names[0]
                rhs_expr = parse_expression(eq_input.split("=", 1)[1])
                x_var = sp.Symbol(var)
                with st.expander("2D Plot Settings", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        x_min = st.number_input("x min", value=-10.0, step=1.0)
                        x_max = st.number_input("x max", value=10.0, step=1.0)
                    with col2:
                        n_points = st.slider("Number of points", 100, 5000, 1000, 100)

                fig = go.Figure()
                trace = make_expression_trace(
                    rhs_expr, x_var, (x_min, x_max),
                    n_points=n_points,
                    name=sp.latex(rhs_expr),
                    color="#00e5ff",
                    dash="solid",
                    width=2.2,
                )
                fig.add_trace(trace)
                fig = base_layout(fig, f"Graph of ${eq_input}$")
                display_plot_with_fullscreen(fig, key="explicit")

            else:
                st.error("Equation must contain x and y (implicit) or one variable (explicit).")

        except Exception as exc:
            st.error(f"Error parsing equation: {exc}")

# ==================== SURFACE PLOTTER PAGE ====================
def page_surface_plotter():
    st.header("🌐 Surface Plotter")
    st.markdown("Plot 3D surfaces of the form $z = f(x, y)$.")

    expr_input = st.text_input(
        "Expression `z = f(x, y)`",
        value="sin(sqrt(x**2 + y**2))",
        help="Example: x**2 + y**2, exp(-x**2 - y**2), sin(x)*cos(y)"
    )

    with st.expander("Plot Settings", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min = st.number_input("x min", value=-5.0, step=0.5)
            x_max = st.number_input("x max", value=5.0, step=0.5)
        with col2:
            y_min = st.number_input("y min", value=-5.0, step=0.5)
            y_max = st.number_input("y max", value=5.0, step=0.5)
        with col3:
            grid_size = st.slider("Grid density", 20, 300, 100, 10)
            opacity = st.slider("Opacity", 0.0, 1.0, 1.0, 0.05)
        with col4:
            colorscale = st.selectbox("Colour scale", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Blues", "Reds"])
            show_contours = st.checkbox("Show contour projections", value=True)

    if st.button("Plot Surface", type="primary"):
        try:
            expr = parse_expression(expr_input)
            free = get_free_symbols(expr)
            if not {"x", "y"}.issubset({str(s) for s in free}):
                st.error("Expression must contain both x and y.")
                return
            fig = plot_surface(
                expr, sp.Symbol("x"), sp.Symbol("y"),
                (x_min, x_max), (y_min, y_max),
                grid_size=grid_size, colorscale=colorscale,
                show_contours=show_contours, opacity=opacity,
            )
            display_plot_with_fullscreen(fig, key="surface")
        except Exception as exc:
            st.error(f"Error: {exc}")

# ==================== LORENZ SYSTEM PAGE ====================
def page_lorenz():
    st.header("🦋 Lorenz System Attractor")
    st.markdown(
        r"""
        The **Lorenz system** is a system of three coupled, nonlinear differential equations:
        
        $$\frac{dx}{dt} = \sigma(y - x)$$
        $$\frac{dy}{dt} = x(\rho - z) - y$$
        $$\frac{dz}{dt} = xy - \beta z$$
        
        This system exhibits chaotic behavior for certain parameter values.
        The animation traces the trajectory over time like a butterfly effect.
        """
    )

    with st.expander("Lorenz Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sigma = st.slider("σ (sigma) – Prandtl number", 0.0, 20.0, 10.0, 0.1)
        with col2:
            rho = st.slider("ρ (rho) – Rayleigh number", 0.0, 50.0, 28.0, 0.1)
        with col3:
            beta = st.slider("β (beta) – Geometric factor", 0.0, 5.0, 8.0/3.0, 0.01)

    with st.expander("Initial Conditions & Integration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            x0 = st.number_input("Initial x", value=1.0, step=0.1)
            y0 = st.number_input("Initial y", value=1.0, step=0.1)
            z0 = st.number_input("Initial z", value=1.0, step=0.1)
        with col2:
            t_max = st.slider("Time span", 10.0, 200.0, 50.0, 5.0)
            dt = st.slider("Time step (dt)", 0.001, 0.05, 0.01, 0.001)
        with col3:
            line_color = st.color_picker("Trajectory colour", value="#00e5ff")
            line_width = st.slider("Line width", 0.5, 5.0, 2.0, 0.1)
            animation_speed = st.slider("Animation speed", 0.5, 5.0, 1.0, 0.1)

    if st.button("Generate Lorenz Attractor", type="primary"):
        with st.spinner("Integrating Lorenz system..."):
            try:
                fig = plot_lorenz_attractor(
                    sigma=sigma,
                    rho=rho,
                    beta=beta,
                    initial_state=(x0, y0, z0),
                    t_max=t_max,
                    dt=dt,
                    line_color=line_color,
                    line_width=line_width,
                    animation_speed=animation_speed,
                )
                display_plot_with_fullscreen(fig, key="lorenz")

                with st.expander("ℹ️ About this System"):
                    st.markdown(
                        f"""
                        **Current Parameters:**
                        - σ = {sigma:.2f}
                        - ρ = {rho:.2f}
                        - β = {beta:.3f}
                        
                        **Classic chaotic parameters:**
                        - σ = 10.0
                        - ρ = 28.0
                        - β = 8/3 ≈ 2.667
                        
                        The system exhibits chaos when ρ > 24.74 (for σ = 10, β = 8/3).
                        """
                    )
            except Exception as exc:
                st.error(f"Error generating Lorenz attractor: {exc}")

# ==================== PARAMETRIC & POLAR PAGE ====================
def page_parametric_polar():
    st.header("🌀 Parametric & Polar Curves")
    mode = st.radio("Choose curve type", ["Parametric (x(t), y(t))", "Polar (r(θ))"])

    if mode.startswith("Parametric"):
        st.subheader("Parametric Curve")
        col1, col2 = st.columns(2)
        with col1:
            x_expr = st.text_input("x(t)", value="cos(t)")
        with col2:
            y_expr = st.text_input("y(t)", value="sin(t)")

        with st.expander("Plot Settings", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                t_min = st.number_input("t min", value=0.0, step=0.1)
                t_max = st.number_input("t max", value=2*3.14159, step=0.1)
            with col2:
                n_points = st.slider("Points", 100, 5000, 1000, 100)
            with col3:
                color = st.color_picker("Colour", value="#00e5ff")
                dash = st.selectbox("Dash", ["solid", "dash", "dot", "dashdot"], index=0)
            with col4:
                width = st.slider("Width", 0.5, 6.0, 2.2, 0.1)
                show_grid = st.checkbox("Show grid", value=True)

        if st.button("Plot Parametric Curve", type="primary"):
            try:
                x_e = parse_expression(x_expr)
                y_e = parse_expression(y_expr)
                fig = plot_parametric(
                    x_e, y_e, sp.Symbol("t"), (t_min, t_max),
                    n_points=n_points, color=color, dash=dash, width=width,
                )
                if not show_grid:
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=False)
                display_plot_with_fullscreen(fig, key="parametric")
            except Exception as exc:
                st.error(f"Error: {exc}")

    else:
        st.subheader("Polar Curve")
        r_expr = st.text_input("r(θ)", value="cos(4*theta)")

        with st.expander("Plot Settings", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                theta_min = st.number_input("θ min", value=0.0, step=0.1)
                theta_max = st.number_input("θ max", value=2*3.14159, step=0.1)
            with col2:
                n_points = st.slider("Points", 100, 5000, 1500, 100)
            with col3:
                color = st.color_picker("Colour", value="#ff007f")
                dash = st.selectbox("Dash", ["solid", "dash", "dot", "dashdot"], index=0)
            with col4:
                width = st.slider("Width", 0.5, 6.0, 2.2, 0.1)
                show_grid = st.checkbox("Show grid", value=True)

        if st.button("Plot Polar Curve", type="primary"):
            try:
                r_e = parse_expression(r_expr)
                fig = plot_polar(
                    r_e, sp.Symbol("theta"), (theta_min, theta_max),
                    n_points=n_points, color=color, dash=dash, width=width,
                )
                if not show_grid:
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=False)
                display_plot_with_fullscreen(fig, key="polar")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ==================== MAIN ROUTER ====================
def main():
    load_css()
    st.sidebar.title("📐 MathLab")
    st.sidebar.caption("Graphical Representation of Algebraic Expressions, Equations, and Functions")

    page = st.sidebar.radio("Navigation", [
        "Home",
        "Algebraic Expressions",
        "Algebraic Equations",
        "Surface Plotter",
        "Lorenz System",
        "Parametric & Polar",
    ])

    if page == "Home":
        page_home()
    elif page == "Algebraic Expressions":
        page_algebraic_expressions()
    elif page == "Algebraic Equations":
        page_algebraic_equations()
    elif page == "Surface Plotter":
        page_surface_plotter()
    elif page == "Lorenz System":
        page_lorenz()
    else:
        page_parametric_polar()

if __name__ == "__main__":
    main()