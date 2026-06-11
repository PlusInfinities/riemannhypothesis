"""
Interactive 3D visualization of the analytic continuation of the Riemann
zeta function, recreating old-blender-files/blender-riemann-zeta.py with
Plotly instead of Blender.

A flat grid of points s = x + iy in the complex plane is morphed into the
surface (Re(zeta(s)), Im(zeta(s)), |s|), animated as a sequence of frames
the same way the original Blender keyframe animation interpolated from
frame 0 (flat grid) to frame 24 (transformed grid).

https://en.wikipedia.org/wiki/Riemann_zeta_function
"""

import os

import numpy as np
import plotly.graph_objects as go


def zeta_ac(s, n_terms=100):
    """Analytic continuation of zeta via the Dirichlet eta function, valid for Re(s) > 0."""
    n = np.arange(1, n_terms)[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.sum((-1) ** (n - 1) / n ** s, axis=0)
        return eta / (1 - 2 ** (1 - s))


def build_grid(n=50, extent=3.0, n_terms=100):
    """Build the flat grid (U, V) and its zeta-transformed counterpart (X1, Y1, Z1)."""
    u = np.linspace(-extent, extent, n)
    v = np.linspace(-extent, extent, n)
    U, V = np.meshgrid(u, v)
    S = U + 1j * V

    # Skip the pole at s = 1 and the strip immediately around x = 0,
    # mirroring the offsets used in the Blender script.
    mask = (U > 0.01) & ((U < 0.99) | (U > 1.01))

    W = zeta_ac(S, n_terms)

    X1 = np.where(mask, W.real, U)
    Y1 = np.where(mask, W.imag, V)
    Z1 = np.where(mask, np.abs(S), 0.0)

    return U, V, X1, Y1, Z1, np.abs(S)


def build_figure(n=50, extent=3.0, n_terms=100, n_frames=30):
    U, V, X1, Y1, Z1, color = build_grid(n, extent, n_terms)
    Z0 = np.zeros_like(U)
    cmax = float(np.max(color))

    def surface(t):
        return go.Surface(
            x=(1 - t) * U + t * X1,
            y=(1 - t) * V + t * Y1,
            z=(1 - t) * Z0 + t * Z1,
            surfacecolor=color,
            colorscale="Viridis",
            cmin=0,
            cmax=cmax,
            colorbar=dict(title="|s|"),
        )

    frames = [
        go.Frame(data=[surface(i / (n_frames - 1))], name=str(i))
        for i in range(n_frames)
    ]

    fig = go.Figure(
        data=[surface(0)],
        frames=frames,
        layout=go.Layout(
            title="Analytic Continuation of the Riemann Zeta Function",
            scene=dict(
                xaxis_title="Re(s) / Re(zeta(s))",
                yaxis_title="Im(s) / Im(zeta(s))",
                zaxis_title="|s|",
                aspectmode="cube",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 60, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Reset",
                            method="animate",
                            args=[
                                ["0"],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                        ),
                    ],
                )
            ],
        ),
    )
    return fig


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    fig = build_figure()
    out_path = os.path.join(out_dir, "riemann_zeta.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
