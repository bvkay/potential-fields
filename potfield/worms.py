"""Multiscale edges of potential field grids, the "worms".

Upward continue the field to a ladder of heights, take the total horizontal
derivative at each height, keep the pixels where it is a local maximum
along the gradient direction, threshold, and link the survivors into
strings. Stacked in three dimensions the strings show structure: a
vertical contact gives a string that stays put with height, a dipping
contact migrates down-dip, a small shallow body dies out at low heights and
a crustal structure persists to tens of kilometres.

The amplitude of a string decays with height in a way set by the source
geometry. For reduced-to-pole magnetics the horizontal derivative over a
vertical contact decays as 1/(h + z0), over a thin dyke as 1/(h + z0)^2; for
gravity a contact decays as ln and a thin sheet as 1/(h + z0). Fitting the
decay along a string therefore gives a depth to the top of the source once
the geometry is assumed.

Hornby, Boschetti and Horowitz 1999, Geophys. J. Int. 137, 175-196;
Archibald, Gow and Boschetti 1999, Expl. Geophys. 30, 38-44; Holden,
Archibald, Boschetti and Jessell 2000, Expl. Geophys. 31, 617-621. Frank
Horowitz's bsdwormer is an earlier open implementation.

@author: Ben Kay (ben@auscope.org.au)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import fft
from .filters import _spacing, continuation_response
from .grid import Grid


@dataclass
class Worms:
    """Edge points from all levels. One row per point.

    x, y in the grid CRS; height in metres above the observation surface;
    amplitude is the total horizontal derivative there (field units per
    metre); strike in degrees clockwise from north, 0 to 180; level indexes
    `heights`; worm_id is unique across levels.
    """

    x: np.ndarray
    y: np.ndarray
    height: np.ndarray
    amplitude: np.ndarray
    strike: np.ndarray
    level: np.ndarray
    worm_id: np.ndarray
    heights: np.ndarray
    name: str = ""
    epsg: int | None = None
    units: str = ""
    links: dict = field(default_factory=dict, repr=False)  # worm_id -> worm_id at the next level

    def __len__(self) -> int:
        return len(self.x)

    def at_level(self, i: int) -> np.ndarray:
        return self.level == i

    def worm(self, wid: int) -> np.ndarray:
        return self.worm_id == wid

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(
            {"x": self.x, "y": self.y, "height": self.height, "amplitude": self.amplitude,
             "strike": self.strike, "level": self.level, "worm_id": self.worm_id}
        )

    def to_csv(self, path: str | Path) -> Path:
        """Points with attributes, one header line naming the CRS. Loads into Geotools or QGIS as XYZ points."""
        path = Path(path)
        header = f"x,y,height,amplitude,strike,level,worm_id  (EPSG:{self.epsg}, height m above surface, amplitude {self.units}/m)"
        np.savetxt(
            path,
            np.column_stack([self.x, self.y, self.height, self.amplitude, self.strike, self.level, self.worm_id]),
            fmt="%.3f,%.3f,%.1f,%.6g,%.1f,%d,%d",
            header=header,
            comments="# ",
        )
        return path

    def to_vtk(self, path: str | Path, vertical: str = "up") -> Path:
        """Legacy ASCII VTK polydata of points with scalars, for ParaView.

        vertical "up" puts height above the surface on +z; "down" plots it as
        a depth proxy on -z, the usual display convention for worms.
        """
        path = Path(path)
        z = self.height if vertical == "up" else -self.height
        n = len(self)
        lines = ["# vtk DataFile Version 3.0", f"potfield worms {self.name}", "ASCII", "DATASET POLYDATA", f"POINTS {n} float"]
        lines += [f"{a:.3f} {b:.3f} {c:.3f}" for a, b, c in zip(self.x, self.y, z)]
        lines += [f"VERTICES {n} {2 * n}"] + [f"1 {i}" for i in range(n)]
        lines += [f"POINT_DATA {n}"]
        for name, arr in (("height", self.height), ("amplitude", self.amplitude), ("strike", self.strike), ("worm_id", self.worm_id.astype(float))):
            lines += [f"SCALARS {name} float 1", "LOOKUP_TABLE default"] + [f"{v:.6g}" for v in arr]
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path


# detection ------------------------------------------------------------------


def nonmax_suppression(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """True where mag is a local maximum along the gradient direction.

    gx is the east derivative (along columns), gy the north derivative;
    rows run north to south, so the gradient's row component is -gy.
    Eight-neighbour comparison in four quantised directions (Canny).
    """
    m = np.where(np.isfinite(mag), mag, -np.inf)
    u, v = np.nan_to_num(gx), -np.nan_to_num(gy)  # column and row components of the gradient
    ang = np.degrees(np.arctan2(v, u)) % 180.0
    sector = ((ang + 22.5) // 45).astype(int) % 4  # 0: cols, 1: diag (+c,+r), 2: rows, 3: diag (+c,-r)

    def shifted(dr, dc):
        s = np.full_like(m, -np.inf)
        rs = slice(max(dr, 0), m.shape[0] + min(dr, 0))
        rd = slice(max(-dr, 0), m.shape[0] + min(-dr, 0))
        cs = slice(max(dc, 0), m.shape[1] + min(dc, 0))
        cd = slice(max(-dc, 0), m.shape[1] + min(-dc, 0))
        s[rd, cd] = m[rs, cs]
        return s

    pairs = {0: ((0, -1), (0, 1)), 1: ((-1, -1), (1, 1)), 2: ((-1, 0), (1, 0)), 3: ((1, -1), (-1, 1))}
    edge = np.zeros(m.shape, dtype=bool)
    for s, ((r1, c1), (r2, c2)) in pairs.items():
        n1, n2 = shifted(r1, c1), shifted(r2, c2)
        edge |= (sector == s) & (m >= n1) & (m >= n2) & np.isfinite(mag) & (mag > 0)
    edge[0, :] = edge[-1, :] = False
    edge[:, 0] = edge[:, -1] = False
    return edge


def multiscale_edges(
    grid: Grid,
    heights: np.ndarray,
    percentile: float = 70.0,
    min_length: int = 10,
    min_fraction: float = 1e-3,
    pad: int | None = None,
) -> Worms:
    """Worms of a grid at the given continuation heights (m).

    At each height the total horizontal derivative of the continued field
    is computed from one shared FFT, thinned to its ridges, thresholded to
    the points above `percentile` of ridge amplitude at that height and
    above min_fraction of the strongest ridge, and grouped into 8-connected
    strings; strings shorter than min_length pixels are dropped. The
    min_fraction floor removes the false ridges that numerical noise
    produces wherever the field is flat, for instance over null areas
    filled with a constant. Use the RTP field for magnetics so edges sit
    over contacts.
    """
    heights = np.asarray(heights, dtype=float).ravel()
    if np.any(heights < 0):
        raise ValueError("heights must be >= 0")
    dx, dy = _spacing(grid)
    prep = fft.prepare(grid.values, dx, dy, pad)
    cols = np.arange(grid.nx)
    rows = np.arange(grid.ny)
    out = {k: [] for k in ("x", "y", "height", "amplitude", "strike", "level", "worm_id")}
    next_id = 0
    for i, h in enumerate(heights):
        Hc = prep.response(lambda k, kx, ky: continuation_response(k, h))
        gx = fft.finish(prep, Hc * (1j * prep.kx), plane="dx")
        gy = fft.finish(prep, Hc * (1j * prep.ky), plane="dy")
        mag = np.hypot(gx, gy)
        edge = nonmax_suppression(mag, gx, gy)
        if not edge.any():
            continue
        thr = max(np.nanpercentile(mag[edge], percentile), min_fraction * np.nanmax(mag[edge]))
        edge &= mag >= thr
        labels, n = ndimage.label(edge, structure=np.ones((3, 3), dtype=int))
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel())
        keep_label = sizes >= min_length
        keep_label[0] = False
        sel = keep_label[labels]
        r, c = np.nonzero(sel)
        out["x"].append(grid.x[c])
        out["y"].append(grid.y[r])
        out["height"].append(np.full(r.size, h))
        out["amplitude"].append(mag[r, c])
        out["strike"].append((np.degrees(np.arctan2(gx[r, c], gy[r, c])) + 90.0) % 180.0)
        out["level"].append(np.full(r.size, i, dtype=int))
        ids = np.cumsum(keep_label) - 1  # dense ids for kept labels
        out["worm_id"].append(ids[labels[r, c]] + next_id)
        next_id += int(keep_label.sum())
    if not out["x"]:
        raise ValueError("no edges found; lower the percentile or min_length")
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    return Worms(**arrays, heights=heights, name=grid.name, epsg=grid.epsg, units=grid.units)


# linking across levels ------------------------------------------------------


def link_levels(w: Worms, dip_max: float = 60.0) -> dict[int, int]:
    """For every worm, the worm at the next level that most of its points
    fall closest to, within a search radius set by the height step and
    dip_max (degrees): a contact dipping at dip_max migrates
    dh / tan(dip_max) between levels. Stored in w.links and returned."""
    links: dict[int, int] = {}
    cell = np.median(np.diff(np.unique(w.x))) if len(np.unique(w.x)) > 1 else 1.0
    for i in range(len(w.heights) - 1):
        lo, hi = w.at_level(i), w.at_level(i + 1)
        if not lo.any() or not hi.any():
            continue
        dh = w.heights[i + 1] - w.heights[i]
        radius = dh / np.tan(np.radians(dip_max)) + 2 * cell
        tree = cKDTree(np.column_stack([w.x[hi], w.y[hi]]))
        dist, idx = tree.query(np.column_stack([w.x[lo], w.y[lo]]), distance_upper_bound=radius)
        ok = np.isfinite(dist)
        hi_ids = w.worm_id[hi]
        for wid in np.unique(w.worm_id[lo]):
            m = (w.worm_id[lo] == wid) & ok
            if m.sum() < 3:
                continue
            targets = hi_ids[idx[m]]
            vals, counts = np.unique(targets, return_counts=True)
            if counts.max() >= 0.5 * m.sum():
                links[int(wid)] = int(vals[np.argmax(counts)])
    w.links = links
    return links


def chains(w: Worms, min_levels: int = 4) -> list[list[int]]:
    """Sequences of linked worm ids from low to high height, at least min_levels long."""
    if not w.links:
        link_levels(w)
    targets = set(w.links.values())
    starts = [wid for wid in np.unique(w.worm_id) if int(wid) not in targets]
    out = []
    for s in starts:
        chain, wid = [int(s)], int(s)
        while wid in w.links:
            wid = w.links[wid]
            chain.append(wid)
        if len(chain) >= min_levels:
            out.append(chain)
    return out


def chain_profile(w: Worms, chain: list[int]) -> dict[str, np.ndarray]:
    """height, median amplitude and centroid position of each worm in a chain."""
    h, a, x, y = [], [], [], []
    for wid in chain:
        m = w.worm(wid)
        h.append(w.height[m][0])
        a.append(np.median(w.amplitude[m]))
        x.append(w.x[m].mean())
        y.append(w.y[m].mean())
    return {"height": np.array(h), "amplitude": np.array(a), "x": np.array(x), "y": np.array(y)}


def depth_from_decay(height: np.ndarray, amplitude: np.ndarray, n: float = 1.0) -> dict[str, float]:
    """Depth to source from amplitude decay A = C / (h + z0)^n.

    Linearised as A^(-1/n) = (h + z0) / C^(1/n): a straight line in h whose
    intercept over slope is z0. n = 1 for a contact in RTP magnetics or a
    thin sheet in gravity, n = 2 for a thin dyke in magnetics.
    """
    h = np.asarray(height, dtype=float)
    a = np.asarray(amplitude, dtype=float)
    ok = np.isfinite(a) & (a > 0)
    if ok.sum() < 3:
        raise ValueError("need at least 3 positive amplitudes")
    slope, intercept = np.polyfit(h[ok], a[ok] ** (-1.0 / n), 1)
    z0 = intercept / slope if slope > 0 else np.nan
    fit = (slope * h[ok] + intercept) ** (-n)
    return {"depth": float(z0), "n": n, "misfit": float(np.sqrt(np.mean((fit - a[ok]) ** 2)) / a[ok].max())}


def contact_dip(w: Worms, chain: list[int]) -> dict[str, float]:
    """Dip of a contact from the migration of its worm with height.

    For the RTP field of a 2D contact the horizontal-derivative maximum
    moves down-dip as height increases, along a line that bisects the angle
    between the contact and the vertical. With worm_dip = atan(dh / shift)
    the contact dips at 2 * worm_dip - 90 toward the migration azimuth;
    90 means no migration, a vertical contact. Checked against the 2D
    forward model at 30, 45 and 60 degrees in tests/test_worms.py.

    Applies to single worms over contacts. A thin dyke gives a pair of
    flank worms that diverge with height; do not read dip from either.
    Returns shift (m), worm_dip, dip and azimuth (degrees), and the
    migration rate shift / dh.
    """
    p = chain_profile(w, chain)
    if len(p["height"]) < 2:
        raise ValueError("chain too short")
    dh = p["height"][-1] - p["height"][0]
    ex, ny_ = p["x"][-1] - p["x"][0], p["y"][-1] - p["y"][0]
    shift = float(np.hypot(ex, ny_))
    worm_dip = float(np.degrees(np.arctan2(dh, shift)))
    return {
        "shift": shift,
        "worm_dip": worm_dip,
        "dip": float(np.clip(2 * worm_dip - 90.0, 0.0, 90.0)),
        "azimuth": float(np.degrees(np.arctan2(ex, ny_)) % 360),
        "rate": shift / dh if dh > 0 else np.nan,
    }


# plotting -------------------------------------------------------------------


def plot_map(w: Worms, ax=None, grid: Grid | None = None, s: float = 1.0, cmap: str = "plasma", levels=None, **grid_kw):
    """Worm points coloured by height (log scale) over an optional grid image."""
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    from .display import plot as plot_grid

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    if grid is not None:
        plot_grid(grid, ax=ax, colorbar=False, **{"cmap": "gray", "clip": ("std", 2.0), **grid_kw})
    m = np.ones(len(w), dtype=bool) if levels is None else np.isin(w.level, levels)
    hmin = max(w.heights[w.heights > 0].min(), 1.0)
    sc = ax.scatter(w.x[m], w.y[m], c=np.maximum(w.height[m], hmin), s=s, cmap=cmap, norm=LogNorm(hmin, w.heights.max()), linewidths=0)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("continuation height (m)")
    ax.set_title(f"{w.name} worms, {len(w.heights)} levels")
    return ax


def plot_3d(
    w: Worms,
    ax=None,
    color: str = "height",
    vertical: str = "up",
    exaggeration: float = 3.0,
    s: float = 1.0,
    stride: int = 1,
    min_height: float = 0.0,
    alpha: float = 0.8,
    cmap: str = "plasma",
    elev: float = 25,
    azim: float = -60,
):
    """3D scatter of the worms in km.

    vertical "up" shows height above the surface; "down" plots it downwards
    as a depth proxy. exaggeration stretches the vertical axis relative to
    the map. min_height drops the lowest, most crowded levels; stride
    subsamples the rest. Static in a notebook; export to VTK for ParaView
    when you want to rotate it.
    """
    from matplotlib import pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    keep = np.nonzero(w.height >= min_height)[0][:: max(1, int(stride))]
    z = w.height[keep] if vertical == "up" else -w.height[keep]
    c = w.height[keep] if color == "height" else w.amplitude[keep]
    sc = ax.scatter(w.x[keep] / 1e3, w.y[keep] / 1e3, z / 1e3, c=c, s=s, cmap=cmap, linewidths=0, depthshade=False, alpha=alpha)
    ax.set_xlabel("easting (km)")
    ax.set_ylabel("northing (km)")
    ax.set_zlabel("height (km)" if vertical == "up" else "-height (km)")
    xr, yr = np.ptp(w.x) / 1e3, np.ptp(w.y) / 1e3
    zr = max((w.heights.max() - max(min_height, w.heights.min())) / 1e3, 1e-3)
    ax.set_box_aspect((xr, yr, zr * exaggeration))
    ax.view_init(elev=elev, azim=azim)
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label="height (m)" if color == "height" else f"amplitude ({w.units}/m)")
    return ax
