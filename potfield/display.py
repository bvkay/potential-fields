"""Plotting helpers: clipped colour scales, sunshading, AGC, line overlays.

Potential field grids have long-tailed histograms, so a linear stretch
between min and max hides most of the structure. clip_limits() gives
standard-deviation or percentile limits; sunshade() adds a hillshade;
agc() equalises amplitude locally so weak and strong anomalies show
together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter

from .grid import Grid


def clip_limits(values: np.ndarray, method: str = "std", n: float = 2.5) -> tuple[float, float]:
    """Colour limits. method "std": mean +- n std. method "pct": n-th and (100-n)-th percentiles."""
    v = values[np.isfinite(values)]
    if method == "std":
        m, s = v.mean(), v.std()
        return (m - n * s, m + n * s)
    if method == "pct":
        return tuple(np.percentile(v, [n, 100 - n]))
    if method == "minmax":
        return (float(v.min()), float(v.max()))
    raise ValueError("method must be std, pct or minmax")


def sunshade(grid: Grid, azimuth: float = 45.0, altitude: float = 45.0, vert_exag: float = 1.0) -> np.ndarray:
    """Hillshade in [0, 1]. azimuth clockwise from north, altitude above horizon, degrees.

    Data units are not lengths, so vert_exag scales the field before
    shading; the automatic value puts the field range on the same footing
    as the grid width.
    """
    v = grid.values
    ok = np.isfinite(v)
    filled = np.where(ok, v, np.nanmean(v))
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    dx, dy = grid.spacing()
    scale = vert_exag * grid.nx * dx / (np.ptp(filled) or 1.0)
    hs = ls.hillshade(filled, vert_exag=scale, dx=dx, dy=dy)
    return np.where(ok, hs, np.nan)


def agc(grid: Grid, sigma_cells: float = 10.0) -> Grid:
    """Automatic gain control: local residual divided by local RMS.

    Both the local mean and the RMS use a Gaussian of `sigma_cells`. Output
    is dimensionless, roughly in [-3, 3]. Reveals texture in quiet areas at
    the cost of amplitude information.
    """
    v = grid.values
    ok = np.isfinite(v)
    filled = np.where(ok, v, np.nanmean(v))
    local_mean = gaussian_filter(filled, sigma_cells)
    resid = filled - local_mean
    rms = np.sqrt(gaussian_filter(resid**2, sigma_cells))
    out = resid / np.where(rms > 0, rms, np.nan)
    out[~ok] = np.nan
    return grid.with_values(out, name=f"{grid.name}_agc", units="")


def plot(
    grid: Grid,
    ax=None,
    cmap: str = "viridis",
    clip: tuple[str, float] | tuple[float, float] | None = ("std", 2.5),
    shade: dict | None = None,
    title: str | None = None,
    colorbar: bool = True,
    **imshow_kw,
):
    """imshow a grid in its own coordinates.

    clip: ("std", 2.5), ("pct", 2), or (vmin, vmax). shade: kwargs for
    sunshade(), e.g. {"azimuth": 45, "altitude": 45}; the shade multiplies
    the colours.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if clip is None:
        vmin, vmax = None, None
    elif isinstance(clip[0], str):
        vmin, vmax = clip_limits(grid.values, clip[0], clip[1])
    else:
        vmin, vmax = clip
    im = ax.imshow(grid.values, extent=grid.extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kw)
    if shade is not None:
        hs = sunshade(grid, **shade)
        ax.imshow(hs, extent=grid.extent, origin="upper", cmap="gray", alpha=0.35, vmin=0, vmax=1)
    ax.set_aspect("equal")
    unit = "m" if grid.is_projected else "deg"
    ax.set_xlabel(f"Easting ({unit})" if grid.is_projected else "Longitude (deg)")
    ax.set_ylabel(f"Northing ({unit})" if grid.is_projected else "Latitude (deg)")
    ax.set_title(title if title is not None else grid.name)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(grid.units)
    return ax


def compare(grids: list[Grid], ncols: int = 2, figsize_per: tuple[float, float] = (6, 5), **plot_kw):
    """Small multiples of several grids with the same plot() settings."""
    n = len(grids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False)
    for ax, g in zip(axes.ravel(), grids):
        plot(g, ax=ax, **plot_kw)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


# overlays -----------------------------------------------------------------


def read_polylines(paths, src_epsg: int = 4283) -> list[tuple[np.ndarray, int]]:
    """Read two-column x y text files (one vertex per line) as polylines.

    Returns a list of (N x 2 array, epsg). Lines that do not parse are
    skipped, so headers are tolerated.
    """
    out = []
    for p in map(Path, np.atleast_1d(paths)):
        rows = []
        for line in p.read_text().splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if rows:
            out.append((np.array(rows), src_epsg))
    return out


def overlay_lines(ax, lines, grid: Grid, color: str = "k", lw: float = 0.8, **kw):
    """Plot polylines from read_polylines() on a grid's axes, transforming to the grid CRS."""
    import pyproj

    for xy, epsg in lines:
        if grid.epsg != epsg:
            tr = pyproj.Transformer.from_crs(epsg, grid.crs.to_wkt(), always_xy=True)
            x, y = tr.transform(xy[:, 0], xy[:, 1])
        else:
            x, y = xy[:, 0], xy[:, 1]
        ax.plot(x, y, color=color, lw=lw, **kw)
    w, e, s, n = grid.extent
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    return ax
