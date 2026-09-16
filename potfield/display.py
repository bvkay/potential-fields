"""Plotting helpers: clipped colour scales, sunshading, AGC, line overlays.

Potential field grids have long-tailed histograms, so a linear stretch
between min and max hides most of the structure. clip_limits() gives
standard-deviation or percentile limits; sunshade() adds a hillshade;
agc() equalises amplitude locally so weak and strong anomalies show
together.

@author: Ben Kay (ben@auscope.org.au)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter

from .grid import Grid, Profile


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
    if grid.is_projected:
        # data stay in metres so contours and overlays line up; ticks read in km
        km = FuncFormatter(lambda v, pos: f"{v / 1000:g}")
        ax.xaxis.set_major_formatter(km)
        ax.yaxis.set_major_formatter(km)
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
    else:
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
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


def plot_profiles(profiles: list[Profile], sharey: bool = False, figsize_per: float = 2.2, **plot_kw):
    """Stacked line plots of several profiles against distance in km.

    One axis per profile so different units do not share a scale. Returns
    (fig, axes).
    """
    n = len(profiles)
    fig, axes = plt.subplots(n, 1, sharex=True, sharey=sharey, figsize=(9, figsize_per * n + 0.8), squeeze=False)
    for ax, p in zip(axes[:, 0], profiles):
        ax.plot(p.distance / 1e3, p.values, **plot_kw)
        ax.set_ylabel(p.units or "")
        ax.set_title(p.name, loc="left", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("distance (km)")
    fig.tight_layout()
    return fig, axes


def plot_line(ax, profile: Profile, color: str = "k", **kw):
    """Draw a profile's track on a map axis with start and end marked."""
    ax.plot(profile.x, profile.y, color=color, lw=1.2, **kw)
    ax.plot(profile.x[0], profile.y[0], "o", color=color, ms=4)
    ax.plot(profile.x[-1], profile.y[-1], "s", color=color, ms=4)
    return ax


def plot_spectrum_2d(grid: Grid, ax=None, kmax: float | None = None, clip: tuple[str, float] = ("pct", 1), pad: int | None = None, cmap: str = "magma"):
    """Image of the 2D log power spectrum in rad/km, north up.

    kmax (rad/m) limits the axes; default is the Nyquist of the grid. A
    linear feature striking at azimuth a shows up as a spoke at azimuth
    a + 90 in this picture.
    """
    from .spectrum import power_spectrum_2d

    s = power_spectrum_2d(grid, pad=pad)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    vmin, vmax = clip_limits(s["log_power"], clip[0], clip[1])
    im = ax.imshow(s["log_power"], extent=s["extent"], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    lim = (kmax if kmax is not None else np.pi / max(grid.spacing())) * 1e3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="w", lw=0.3, alpha=0.5)
    ax.axvline(0, color="w", lw=0.3, alpha=0.5)
    ax.set_xlabel("kx east (rad/km)")
    ax.set_ylabel("ky north (rad/km)")
    ax.set_title(f"{grid.name} log power")
    plt.colorbar(im, ax=ax, shrink=0.8, label="ln P")
    return ax


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
