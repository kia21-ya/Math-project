"""
Lorentz System Plotter – Visualizes the famous Lorenz attractor with animated tracing.

The Lorenz system:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
"""

import numpy as np
import plotly.graph_objects as go

__all__ = ["plot_lorenz_attractor", "lorenz_system"]


def lorenz_system(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Compute derivatives for the Lorenz system."""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


def rk4_step(func, state, t, dt, **kwargs):
    """Perform one 4th-order Runge-Kutta integration step."""
    state = np.array(state, dtype=float)
    k1 = np.array(func(state, t, **kwargs))
    k2 = np.array(func(state + 0.5 * dt * k1, t + 0.5 * dt, **kwargs))
    k3 = np.array(func(state + 0.5 * dt * k2, t + 0.5 * dt, **kwargs))
    k4 = np.array(func(state + dt * k3, t + dt, **kwargs))
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def plot_lorenz_attractor(
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0,
    initial_state=(1.0, 1.0, 1.0),
    t_max=50.0,
    dt=0.01,
    line_color="#00e5ff",
    line_width=2.0,
    animation_speed=1.0,
):
    """
    Generate and plot the Lorenz attractor with animated tracing.

    Parameters:
    -----------
    sigma, rho, beta : float
        Lorenz system parameters
    initial_state : tuple
        Initial (x, y, z)
    t_max : float
        Maximum integration time
    dt : float
        Time step for integration
    line_color : str
        Color of trajectory
    line_width : float
        Width of trajectory
    animation_speed : float
        Speed multiplier for animation (higher = faster)

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure with animated tracing
    """
    # Integrate the Lorenz system
    n_steps = int(t_max / dt)
    states = np.zeros((n_steps + 1, 3))
    states[0] = initial_state
    t = 0.0

    for i in range(n_steps):
        states[i + 1] = rk4_step(lorenz_system, states[i], t, dt, 
                                 sigma=sigma, rho=rho, beta=beta)
        t += dt

    x_vals = states[:, 0]
    y_vals = states[:, 1]
    z_vals = states[:, 2]

    # Create frames for animation
    # Number of frames: aim for ~150 frames for smooth animation
    n_frames = 150
    frame_step = max(1, n_steps // n_frames)
    
    # Duration per frame (ms) - lower = faster animation
    frame_duration = int(50 / animation_speed)
    
    frames = []
    for i in range(frame_step, n_steps + 1, frame_step):
        # Add a marker at the current position (head of the trace)
        frames.append(
            go.Frame(
                data=[
                    # The traced path so far
                    go.Scatter3d(
                        x=x_vals[:i],
                        y=y_vals[:i],
                        z=z_vals[:i],
                        mode="lines",
                        line=dict(color=line_color, width=line_width),
                        name="Trajectory",
                    ),
                    # Current position marker (head)
                    go.Scatter3d(
                        x=[x_vals[i-1]],
                        y=[y_vals[i-1]],
                        z=[z_vals[i-1]],
                        mode="markers",
                        marker=dict(size=8, color="#ff007f", symbol="circle"),
                        name="Current Position",
                    )
                ],
                name=f"frame{i}"
            )
        )

    # Initial trace (single point)
    initial_traces = [
        go.Scatter3d(
            x=[x_vals[0]],
            y=[y_vals[0]],
            z=[z_vals[0]],
            mode="markers",
            marker=dict(size=8, color="#ff007f", symbol="circle"),
            name="Start",
        ),
        go.Scatter3d(
            x=[x_vals[0]],
            y=[y_vals[0]],
            z=[z_vals[0]],
            mode="lines",
            line=dict(color=line_color, width=line_width),
            name="Trajectory",
        )
    ]

    fig = go.Figure(data=initial_traces, frames=frames)

    # Create slider steps
    steps = []
    for i in range(len(frames)):
        step = dict(
            method="animate",
            args=[
                [frames[i].name],
                {
                    "frame": {"duration": frame_duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }
            ],
            label=f"{i}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        steps=steps,
        currentvalue={"prefix": "Time step: ", "font": {"color": "#c8d3f5"}},
        pad={"t": 50},
        len=0.9,
        x=0.1,
        xanchor="left",
        y=0,
        yanchor="top"
    )]

    # Play/Pause buttons
    updatemenus = [
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                            "mode": "immediate"
                        }
                    ]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                ),
                dict(
                    label="🔄 Reset",
                    method="animate",
                    args=[
                        ["frame0"],
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                )
            ],
            direction="left",
            pad={"r": 10, "t": 70},
            showactive=False,
            active=0,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#00e5ff",
            borderwidth=1,
            font=dict(color="#c8d3f5")
        )
    ]

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d3f5"),
        title=dict(
            text=f"Lorenz Attractor (σ={sigma:.1f}, ρ={rho:.1f}, β={beta:.2f})",
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(gridcolor="rgba(200,211,245,0.1)", showbackground=False),
            yaxis=dict(gridcolor="rgba(200,211,245,0.1)", showbackground=False),
            zaxis=dict(gridcolor="rgba(200,211,245,0.1)", showbackground=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        margin=dict(l=0, r=0, t=60, b=0),
        height=800,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#00e5ff",
            borderwidth=1
        )
    )

    return fig