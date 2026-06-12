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


# Imaginary parts of the first known nontrivial zeros zeta(1/2 + i*t) = 0
# (positive t; conjugates -t are also zeros by the reflection formula).
NONTRIVIAL_ZERO_ORDINATES = [
    14.134725141734693,
    21.022039638771554,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671,
    40.918719012147495,
    43.327073280914999,
]


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


def _im_sample_points(im_extent, inner_extent=5.0, n_inner=300, n_outer=250):
    """Sample points over [-im_extent, im_extent], dense near 0 (where vertical
    lines close to Re(s) = 1 vary rapidly near the pole at s = 1) and coarser
    further out (where zeta varies slowly), keeping the point count manageable
    even when im_extent is large."""
    inner_extent = min(inner_extent, im_extent)
    inner = np.linspace(-inner_extent, inner_extent, n_inner)
    if im_extent <= inner_extent:
        return inner
    outer_pos = np.linspace(inner_extent, im_extent, n_outer)[1:]
    return np.concatenate([-outer_pos[::-1], inner, outer_pos])


def line_family(fixed_axis, fixed_values, span):
    """s-plane (x, y) coordinates for a family of straight line segments,
    NaN-separated (one line per value in `fixed_values`).

    `fixed_axis` is "re" or "im" - the coordinate held constant along each
    line; `span` gives the sample points for the varying coordinate.
    """
    nan = np.array([np.nan])
    x_parts, y_parts = [], []
    for c in fixed_values:
        if fixed_axis == "re":
            x_parts += [np.full_like(span, c), nan]
            y_parts += [span, nan]
        else:
            x_parts += [span, nan]
            y_parts += [np.full_like(span, c), nan]
    return np.concatenate(x_parts), np.concatenate(y_parts)


def build_strip_boundary(re_lo, re_hi, im_extent=3.0, n_points=60, threshold=9.0, eps=0.01):
    """Build identity and zeta-image coordinates for the rectangular boundary of
    the vertical strip re_lo <= Re(s) <= re_hi, -im_extent <= Im(s) <= im_extent.

    Used to shade the strip and show (via fill="toself") the region it is
    mapped to under zeta. Values are clipped (rather than NaN'd) so the
    boundary stays a closed, finite polygon even near the pole at s=1.
    """
    if re_hi >= 1.0:
        re_hi -= eps  # nudge away from the pole at s = 1

    re_edge = np.linspace(re_lo, re_hi, n_points)
    im_edge = np.linspace(-im_extent, im_extent, n_points)

    x = np.concatenate([re_edge, np.full(n_points, re_hi), re_edge[::-1], np.full(n_points, re_lo)])
    y = np.concatenate([np.full(n_points, -im_extent), im_edge, np.full(n_points, im_extent), im_edge[::-1]])

    w = zeta_hasse(x + 1j * y)
    wx = np.clip(w.real, -threshold, threshold)
    wy = np.clip(w.imag, -threshold, threshold)
    return x, y, wx, wy


def circle_family(center, radii, n_points=200):
    """s-plane (x, y) coordinates for a family of concentric circles
    |s - center| = r, one per radius in `radii`, NaN-separated."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    nan = np.array([np.nan])
    x_parts, y_parts = [], []
    for r in radii:
        x_parts += [center[0] + r * np.cos(theta), nan]
        y_parts += [center[1] + r * np.sin(theta), nan]
    return np.concatenate(x_parts), np.concatenate(y_parts)


def sine_wave_family(amplitudes, t, axis="re", center=0.0, frequency=1.0, phase=0.0):
    """s-plane (x, y) coordinates for a family of sine curves, one per
    amplitude in `amplitudes`, sampled over parameter `t`, NaN-separated.

    `axis` selects which s-plane axis runs along `t`:
    - "re": x = t, y = center + amplitude * sin(frequency * t + phase)
        (waves run horizontally, oscillating in Im(s) around `center`).
    - "im": y = t, x = center + amplitude * sin(frequency * t + phase)
        (waves run vertically, oscillating in Re(s) around `center`).
    """
    nan = np.array([np.nan])
    x_parts, y_parts = [], []
    for a in amplitudes:
        wave = center + a * np.sin(frequency * t + phase)
        if axis == "re":
            x_parts += [t, nan]
            y_parts += [wave, nan]
        else:
            x_parts += [wave, nan]
            y_parts += [t, nan]
    return np.concatenate(x_parts), np.concatenate(y_parts)


def build_zero_markers(im_extent=3.0):
    """Identity (on the critical line) and image (at the origin, since zeta = 0
    there) coordinates for the known nontrivial zeros within +/- im_extent."""
    ordinates = [t for t in NONTRIVIAL_ZERO_ORDINATES if t <= im_extent]
    ordinates = [-t for t in reversed(ordinates)] + ordinates
    x0 = np.full(len(ordinates), 0.5)
    y0 = np.array(ordinates)
    x1 = np.zeros(len(ordinates))
    y1 = np.zeros(len(ordinates))
    return x0, y0, x1, y1, ordinates


# Keys identifying the selectable entries in the curve library built by
# build_figure(). Pass a subset of these as `selected` to choose which
# transforming curve families are computed and shown.
RE_LINES = "re_lines"
IM_LINES = "im_lines"
CIRCLES_RE0 = "circles_re0"
CIRCLES_RE_HALF = "circles_re_half"
CIRCLES_RE1 = "circles_re1"
CRITICAL_LINE = "critical_line"
SINE_WAVES = "sine_waves"
SINE_WAVES_CRITICAL = "sine_waves_critical"

ALL_CURVES = [
    RE_LINES, IM_LINES, CIRCLES_RE0, CIRCLES_RE_HALF, CIRCLES_RE1, CRITICAL_LINE,
    SINE_WAVES, SINE_WAVES_CRITICAL,
]


def build_figure(re_extent=3.0, im_extent=45.0, n_lines=21, threshold=9.0, n_frames=30,
                 n_circles=30, circle_n_points=200, h_re_extent=None, selected=None,
                 sine_amplitudes=None, sine_x_extent=10.0, sine_n_points=400, sine_frequency=1.0,
                 sine_critical_amplitudes=None, sine_critical_frequency=2 * np.pi, sine_critical_phase=0.0):
    """Build the figure. `selected` is a list of curve-library keys (see
    ALL_CURVES) choosing which transforming curve families to include as
    toggleable traces; defaults to all of them."""
    span = im_extent
    if h_re_extent is None:
        h_re_extent = im_extent
    if selected is None:
        selected = ALL_CURVES
    if sine_amplitudes is None:
        sine_amplitudes = np.arange(0.5, 5.5, 0.5)  # 10 waves, amplitudes 0.5, 1.0, ..., 5.0
    if sine_critical_amplitudes is None:
        sine_critical_amplitudes = np.arange(0.1, 0.6, 0.1)  # 5 waves, 0.1 .. 0.5 (0.5 reaches strip edges)

    re_coords = np.linspace(-re_extent, re_extent, n_lines)
    im_coords = np.linspace(-im_extent, im_extent, n_lines)
    im_fine = _im_sample_points(im_extent)
    h_re_fine = _im_sample_points(h_re_extent)
    sine_x_fine = np.linspace(-sine_x_extent, sine_x_extent, sine_n_points)

    s0X0, s0Y0, s0X1, s0Y1 = build_strip_boundary(0.0, 0.5, im_extent, threshold=threshold)
    s1X0, s1Y0, s1X1, s1Y1 = build_strip_boundary(0.5, 1.0, im_extent, threshold=threshold)
    zX0, zY0, zX1, zY1, zero_ordinates = build_zero_markers(im_extent)
    zero_text = [f"ζ(1/2 + {t:.3f}i) = 0" for t in zero_ordinates]

    # Extend the concentric circle families out to |s| of the farthest
    # displayed zero (Re(s) = 1/2, Im(s) = last ordinate), spacing them so the
    # count per family stays fixed regardless of how far that reaches.
    if zero_ordinates:
        circle_max_radius = float(np.hypot(0.5, max(abs(o) for o in zero_ordinates)))
    else:
        circle_max_radius = re_extent
    circle_step = circle_max_radius / n_circles
    circle_radii = np.arange(circle_step, circle_max_radius + circle_step / 2, circle_step)

    curve_library = {
        RE_LINES: dict(
            name="constant Re(s)", color="royalblue", width=1,
            xy=line_family("re", re_coords, im_fine),
        ),
        IM_LINES: dict(
            name="constant Im(s)", color="firebrick", width=1,
            xy=line_family("im", im_coords, h_re_fine),
        ),
        CIRCLES_RE0: dict(
            name="circles centered at Re(s) = 0", color="darkviolet", width=1,
            xy=circle_family((0.0, 0.0), circle_radii, circle_n_points),
        ),
        CIRCLES_RE_HALF: dict(
            name="circles centered at Re(s) = 1/2", color="gold", width=1,
            xy=circle_family((0.5, 0.0), circle_radii, circle_n_points),
        ),
        CIRCLES_RE1: dict(
            name="circles centered at Re(s) = 1", color="deeppink", width=1,
            xy=circle_family((1.0, 0.0), circle_radii, circle_n_points),
        ),
        CRITICAL_LINE: dict(
            name="Re(s) = 1/2 (critical line)", color="black", width=2.5, dash="dot",
            xy=line_family("re", [0.5], im_fine),
        ),
        SINE_WAVES: dict(
            name="sine waves (varying amplitude)", color="teal", width=1,
            xy=sine_wave_family(sine_amplitudes, sine_x_fine, frequency=sine_frequency),
        ),
        SINE_WAVES_CRITICAL: dict(
            name="sine waves through critical strip", color="slateblue", width=1,
            xy=sine_wave_family(
                sine_critical_amplitudes, im_fine, axis="im", center=0.5,
                frequency=sine_critical_frequency, phase=sine_critical_phase,
            ),
        ),
    }

    def lerp(a, b, t):
        if t == 0:
            return a
        return (1 - t) * a + t * b

    # Precompute s-plane and zeta-image coordinates for each selected curve.
    curves = []
    for key in selected:
        spec = curve_library[key]
        x0, y0 = spec["xy"]
        x1, y1 = transform_line(x0, y0, threshold)
        curves.append((spec, x0, y0, x1, y1))

    def shading_traces(t):
        return [
            go.Scatter(
                x=lerp(s0X0, s0X1, t), y=lerp(s0Y0, s0Y1, t),
                mode="lines", line=dict(width=0),
                fill="toself", fillcolor="rgba(255, 165, 0, 0.18)",
                name="0 < Re(s) < 1/2",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=lerp(s1X0, s1X1, t), y=lerp(s1Y0, s1Y1, t),
                mode="lines", line=dict(width=0),
                fill="toself", fillcolor="rgba(0, 153, 76, 0.18)",
                name="1/2 < Re(s) < 1",
                hoverinfo="skip",
            ),
        ]

    def curve_traces(t, initial=False):
        result = []
        for spec, x0, y0, x1, y1 in curves:
            line = dict(color=spec["color"], width=spec["width"])
            if "dash" in spec:
                line["dash"] = spec["dash"]
            result.append(go.Scatter(
                x=lerp(x0, x1, t), y=lerp(y0, y1, t),
                mode="lines", line=line,
                name=spec["name"],
                # Start hidden (toggle on via the legend); only set on the
                # initial trace, not on frames, so slider redraws don't undo
                # a toggle the user has made (see uirevision below).
                **({"visible": "legendonly"} if initial else {}),
            ))
        return result

    def zero_trace(t):
        return go.Scatter(
            x=lerp(zX0, zX1, t), y=lerp(zY0, zY1, t),
            mode="markers", marker=dict(color="red", size=10, symbol="x", line=dict(width=2, color="red")),
            name="nontrivial zeros",
            text=zero_text, hoverinfo="text",
        )

    def traces(t, initial=False):
        return shading_traces(t) + curve_traces(t, initial=initial) + [zero_trace(t)]

    # Every frame carries full data for every trace, so any curve family
    # toggled on via the legend gets the same smooth t-morph as the
    # always-visible strip shading and zero markers.
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
        data=traces(0, initial=True),
        frames=frames,
        layout=go.Layout(
            title="Riemann Zeta as a Map from C to C",
            xaxis=dict(title="Re", range=[-span, span], zeroline=False, dtick=2),
            yaxis=dict(title="Im", range=[-span, span], zeroline=False, scaleanchor="x", scaleratio=1, dtick=2),
            width=1000,
            height=1000,
            # Keep legend-driven visibility toggles (e.g. hiding "constant Re(s)"
            # / "constant Im(s)") in place across slider-driven frame redraws.
            uirevision="zeta",
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
