"""
Interactive 2D (C -> C) visualization of the analytic continuation of the
Riemann zeta function.

A grid of horizontal and vertical lines in the complex plane is morphed into
its image under zeta(s), scrubbable with a slider that interpolates between
the identity grid (t=0) and the fully transformed grid (t=1). The critical
line Re(s) = 1/2 is drawn separately as a dotted line.

https://en.wikipedia.org/wiki/Riemann_zeta_function
"""

import os
from math import comb

import numpy as np
import plotly.graph_objects as go


def zeta_hasse(s, n_terms=40):
    """zeta(s) for any complex s != 1, via Hasse's globally convergent series:
    zeta(s) = 1/(1 - 2^(1-s)) * sum_{n=0}^N 2^-(n+1) * sum_{k=0}^n (-1)^k C(n,k) (k+1)^-s

    Unlike the truncated Dirichlet eta series, this converges quickly and
    accurately everywhere, including near s=0 where the eta series converges
    too slowly to be usable.
    """
    s = np.asarray(s, dtype=complex)
    shape = s.shape
    sf = s.ravel()

    total = np.zeros_like(sf)
    for n in range(n_terms + 1):
        k = np.arange(n + 1)
        coeffs = np.array([(-1) ** ki * comb(n, ki) for ki in k], dtype=float)
        total += (coeffs[:, None] * (k[:, None] + 1.0) ** (-sf[None, :])).sum(axis=0) / 2 ** (n + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = total / (1 - 2 ** (1 - sf))
    return result.reshape(shape)


def transform_line(x, y, threshold):
    """Map a line of points s = x + iy through zeta. Points whose image lies
    beyond `threshold` (i.e. near the pole at s=1) become NaN gaps rather
    than huge spikes that would dominate the plot."""
    s = x + 1j * y
    w = zeta_hasse(s)
    far = ~np.isfinite(w) | (np.abs(w) > threshold)
    return np.where(far, np.nan, w.real), np.where(far, np.nan, w.imag)


def build_grid_lines(extent=3.0, n_lines=21, n_points=200, threshold=9.0):
    """Build identity (X0, Y0) and zeta-image (X1, Y1) coordinates for a grid of lines."""
    coords = np.linspace(-extent, extent, n_lines)
    fine = np.linspace(-extent, extent, n_points)
    nan = np.array([np.nan])

    def build(is_vertical):
        x0_parts, y0_parts, x1_parts, y1_parts = [], [], [], []
        for c in coords:
            x = np.full_like(fine, c) if is_vertical else fine
            y = fine if is_vertical else np.full_like(fine, c)
            wx, wy = transform_line(x, y, threshold)

            x0_parts += [x, nan]
            y0_parts += [y, nan]
            x1_parts += [wx, nan]
            y1_parts += [wy, nan]

        return (
            np.concatenate(x0_parts),
            np.concatenate(y0_parts),
            np.concatenate(x1_parts),
            np.concatenate(y1_parts),
        )

    return build(is_vertical=True), build(is_vertical=False)


def build_critical_line(extent=3.0, n_points=200, threshold=9.0):
    """Build identity and zeta-image coordinates for the critical line Re(s) = 1/2."""
    fine = np.linspace(-extent, extent, n_points)
    x = np.full_like(fine, 0.5)
    y = fine
    wx, wy = transform_line(x, y, threshold)
    return x, y, wx, wy


def build_figure(extent=3.0, n_lines=21, n_points=200, n_frames=30):
    span = 2 * extent
    threshold = 1.5 * span

    (vX0, vY0, vX1, vY1), (hX0, hY0, hX1, hY1) = build_grid_lines(extent, n_lines, n_points, threshold)
    cX0, cY0, cX1, cY1 = build_critical_line(extent, n_points, threshold)

    def lerp(a, b, t):
        return (1 - t) * a + t * b

    def traces(t):
        return [
            go.Scatter(
                x=lerp(vX0, vX1, t), y=lerp(vY0, vY1, t),
                mode="lines", line=dict(color="royalblue", width=1),
                name="constant Re(s)",
            ),
            go.Scatter(
                x=lerp(hX0, hX1, t), y=lerp(hY0, hY1, t),
                mode="lines", line=dict(color="firebrick", width=1),
                name="constant Im(s)",
            ),
            go.Scatter(
                x=lerp(cX0, cX1, t), y=lerp(cY0, cY1, t),
                mode="lines", line=dict(color="black", width=2.5, dash="dot"),
                name="Re(s) = 1/2 (critical line)",
            ),
        ]

    frames = []
    steps = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        frames.append(go.Frame(data=traces(t), name=str(i)))
        steps.append(
            dict(
                method="animate",
                args=[[str(i)], {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                }],
                label=f"{t:.2f}",
            )
        )

    fig = go.Figure(
        data=traces(0),
        frames=frames,
        layout=go.Layout(
            title="Riemann Zeta as a Map from C to C",
            xaxis=dict(title="Re", range=[-span, span], zeroline=False),
            yaxis=dict(title="Im", range=[-span, span], zeroline=False, scaleanchor="x", scaleratio=1),
            width=1000,
            height=1000,
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "t = "},
                pad={"t": 40},
                steps=steps,
            )],
        ),
    )
    return fig


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    fig = build_figure()
    out_path = os.path.join(out_dir, "riemann_zeta_2d.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
