# Riemann Hypothesis Visualizations

Exploring the [Riemann zeta function](https://en.wikipedia.org/wiki/Riemann_zeta_function) and its
analytic continuation through interactive visualizations.

## What it does

A flat 50x50 grid of points `s = x + iy` in the complex plane is fed through the
analytically continued zeta function (via the Dirichlet eta function, valid for `Re(s) > 0`).
Each point is then plotted at `(Re(zeta(s)), Im(zeta(s)), |s|)`, producing a surface that shows
how the zeta function warps the complex plane. The visualization animates the transition from
the flat input grid to the fully transformed surface.

## Plotly versions

[riemann_zeta_plotly.py](riemann_zeta_plotly.py) generates an interactive, animated 3D plot
using [Plotly](https://plotly.com/python/), with the same `(Re(zeta(s)), Im(zeta(s)), |s|)`
mapping as the Blender version.

[riemann_zeta_plotly_2d.py](riemann_zeta_plotly_2d.py) generates a 2D (C -> C) view of the same
map: a grid of horizontal and vertical lines in the complex plane is morphed into its image
under zeta(s), with a slider to scrub between the identity grid and the fully transformed grid.
The critical line `Re(s) = 1/2` is drawn separately as a dotted line.

To compute zeta(s) accurately across the whole plane (including near `s = 0`, where the
truncated Dirichlet eta series converges too slowly to be usable), this script uses
[Hasse's globally convergent series](https://en.wikipedia.org/wiki/Riemann_zeta_function#Other_series_representations)
instead. Points whose image lands too close to the pole at `s = 1` are dropped (turned into
gaps) rather than drawn as huge spikes.

### Run with Docker

```bash
docker build -t riemann-zeta-plotly .
docker run --rm -v "${PWD}/output:/app/output" riemann-zeta-plotly
```

### Run locally

```bash
pip install -r requirements.txt
python riemann_zeta_plotly.py
python riemann_zeta_plotly_2d.py
```

Either way, the output is written to `output/riemann_zeta.html` and `output/riemann_zeta_2d.html`.
Open them in a browser to rotate/zoom/play the 3D animation, or scrub the slider on the 2D map.

## Original Blender version

The project started as a [Blender](https://www.blender.org/) Python script that produces the
same kind of visualization as a rendered animation. The original script and a sample render are
kept in [old-blender-files](old-blender-files).

![Original Blender render](old-blender-files/riemann-zeta.png)
