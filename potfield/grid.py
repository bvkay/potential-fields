"""Grid container, reading, subsetting and reprojection.

A Grid holds a 2D array on a regular, north-up, unrotated grid together with
an affine transform and a CRS. Row 0 is the northern edge, column 0 the
western edge. Null cells are NaN.

Reading goes through rasterio/GDAL, so any GDAL raster works: ERS, GeoTIFF,
Surfer GRD, Arc ASCII, netCDF.

@author: Ben Kay (ben@auscope.org.au)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin
from rasterio.warp import calculate_default_transform, transform as warp_transform
from rasterio.warp import reproject as _warp_reproject
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

M_PER_DEG = 111_320.0  # metres per degree of latitude, spherical approximation


@dataclass
class Grid:
    values: np.ndarray
    transform: Affine
    crs: CRS
    name: str = ""
    units: str = ""

    def __post_init__(self):
        v = np.asarray(self.values, dtype=np.float64)
        if v.ndim != 2:
            raise ValueError("values must be 2D")
        self.values = v
        if not isinstance(self.crs, CRS):
            self.crs = CRS.from_user_input(self.crs)
        t = self.transform
        if t.e >= 0:
            raise ValueError("transform must be north-up (e < 0)")
        if t.b != 0 or t.d != 0:
            raise ValueError("rotated grids are not supported")

    # geometry -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def ny(self) -> int:
        return self.values.shape[0]

    @property
    def nx(self) -> int:
        return self.values.shape[1]

    @property
    def dx(self) -> float:
        """Cell width in CRS units."""
        return self.transform.a

    @property
    def dy(self) -> float:
        """Cell height in CRS units (positive)."""
        return -self.transform.e

    @property
    def x(self) -> np.ndarray:
        """Cell-centre x coordinates, west to east."""
        return self.transform.c + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y(self) -> np.ndarray:
        """Cell-centre y coordinates, north to south (descending)."""
        return self.transform.f - (np.arange(self.ny) + 0.5) * self.dy

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) outer edges."""
        w = self.transform.c
        n = self.transform.f
        return (w, n - self.ny * self.dy, w + self.nx * self.dx, n)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(west, east, south, north) for matplotlib imshow."""
        w, s, e, n = self.bounds
        return (w, e, s, n)

    @property
    def is_projected(self) -> bool:
        return bool(self.crs.is_projected)

    @property
    def epsg(self) -> int | None:
        return self.crs.to_epsg()

    def centre(self) -> tuple[float, float]:
        w, s, e, n = self.bounds
        return (0.5 * (w + e), 0.5 * (s + n))

    def centre_lonlat(self) -> tuple[float, float]:
        cx, cy = self.centre()
        lon, lat = warp_transform(self.crs, CRS.from_epsg(4326), [cx], [cy])
        return (lon[0], lat[0])

    def spacing(self) -> tuple[float, float]:
        """Cell size in metres.

        Exact for projected grids in metres. For geographic grids this is the
        cos(lat) approximation at the grid centre: dx = dlon*111320*cos(lat),
        dy = dlat*111320. Adequate for extents under about 3 degrees; use
        reproject() when it matters.
        """
        if self.is_projected:
            return (self.dx, self.dy)
        _, lat = self.centre()
        return (self.dx * M_PER_DEG * math.cos(math.radians(lat)), self.dy * M_PER_DEG)

    def mga_epsg(self, datum: str = "GDA94") -> int:
        """EPSG code of the MGA zone containing the grid centre.

        GDA94 zones are 28349 to 28356, GDA2020 zones 7849 to 7856.
        """
        lon, _ = self.centre_lonlat()
        zone = int(math.floor((lon + 180.0) / 6.0)) + 1
        base = {"GDA94": 28300, "GDA2020": 7800}[datum]
        return base + zone

    # derived grids ------------------------------------------------------

    def with_values(self, values: np.ndarray, name: str | None = None, units: str | None = None) -> "Grid":
        """New Grid with the same geometry and different values."""
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.shape:
            raise ValueError(f"shape {values.shape} != grid shape {self.shape}")
        return Grid(
            values,
            self.transform,
            self.crs,
            self.name if name is None else name,
            self.units if units is None else units,
        )

    def subset(self, bounds: tuple[float, float, float, float], bounds_epsg: int | None = None) -> "Grid":
        """Extract cells within bounds = (west, south, east, north).

        bounds are in the grid CRS unless bounds_epsg is given. The result is
        snapped outward to whole cells and clipped to the grid.
        """
        if bounds_epsg is not None:
            src = CRS.from_epsg(bounds_epsg)
            if src != self.crs:
                bounds = transform_bounds(src, self.crs, *bounds, densify_pts=21)
        w, s, e, n = bounds
        x0, ytop = self.transform.c, self.transform.f
        c0 = max(0, int(math.floor((w - x0) / self.dx)))
        c1 = min(self.nx, int(math.ceil((e - x0) / self.dx)))
        r0 = max(0, int(math.floor((ytop - n) / self.dy)))
        r1 = min(self.ny, int(math.ceil((ytop - s) / self.dy)))
        if c1 <= c0 or r1 <= r0:
            raise ValueError("bounds do not intersect the grid")
        transform = from_origin(x0 + c0 * self.dx, ytop - r0 * self.dy, self.dx, self.dy)
        return Grid(self.values[r0:r1, c0:c1].copy(), transform, self.crs, self.name, self.units)

    def clip(self, geometry, crop: bool = True) -> "Grid":
        """Set cells outside a shapely geometry (grid CRS) to NaN.

        With crop=True the grid is also cut to the geometry bounds.
        """
        g = self.subset(geometry.bounds) if crop else self
        inside = geometry_mask([geometry], g.shape, g.transform, invert=True)
        values = np.where(inside, g.values, np.nan)
        return Grid(values, g.transform, g.crs, g.name, g.units)

    def reproject(
        self,
        epsg: int | None = None,
        res: float | None = None,
        resampling: Resampling = Resampling.bilinear,
    ) -> "Grid":
        """Warp to another CRS on square cells.

        epsg defaults to the GDA94 MGA zone at the grid centre. res is the
        output cell size in metres; default preserves the input cell count.
        """
        dst_crs = CRS.from_epsg(epsg if epsg is not None else self.mga_epsg())
        if dst_crs == self.crs and res is None:
            return self
        transform, width, height = calculate_default_transform(
            self.crs, dst_crs, self.nx, self.ny, *self.bounds, resolution=res
        )
        out = np.full((height, width), np.nan)
        _warp_reproject(
            source=self.values,
            destination=out,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return Grid(out, transform, dst_crs, self.name, self.units)

    def regrid(self, like: "Grid", resampling: Resampling = Resampling.bilinear) -> "Grid":
        """Resample onto the geometry (CRS, transform, shape) of another grid."""
        if like.crs == self.crs and like.transform.almost_equals(self.transform) and like.shape == self.shape:
            return self
        out = np.full(like.shape, np.nan)
        _warp_reproject(
            source=self.values,
            destination=out,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=np.nan,
            dst_transform=like.transform,
            dst_crs=like.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return Grid(out, like.transform, like.crs, self.name, self.units)

    # inspection ---------------------------------------------------------

    def stats(self) -> dict:
        v = self.values[np.isfinite(self.values)]
        return {
            "n": int(v.size),
            "null": int(self.values.size - v.size),
            "min": float(v.min()) if v.size else np.nan,
            "max": float(v.max()) if v.size else np.nan,
            "mean": float(v.mean()) if v.size else np.nan,
            "std": float(v.std()) if v.size else np.nan,
        }

    def __repr__(self) -> str:
        w, s, e, n = self.bounds
        sx, sy = self.spacing()
        unit = "m" if self.is_projected else "deg"
        return (
            f"Grid({self.name!r}, {self.ny}x{self.nx}, cell {self.dx:g}x{self.dy:g} {unit}"
            f" (~{sx:.0f}x{sy:.0f} m), EPSG:{self.epsg}, "
            f"bounds W{w:.4f} S{s:.4f} E{e:.4f} N{n:.4f}, units {self.units!r})"
        )


def read_grid(
    path: str | Path,
    bounds: tuple[float, float, float, float] | None = None,
    bounds_epsg: int | None = None,
    epsg: int | None = None,
    band: int = 1,
    name: str | None = None,
    units: str = "",
) -> Grid:
    """Read a raster with rasterio.

    bounds (west, south, east, north) reads only that window, in the raster
    CRS unless bounds_epsg is given. epsg reprojects the result. Only the
    window is read from disk, so large national grids are fine if bounds are
    given.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path.name} has no CRS")
        if bounds is not None:
            if bounds_epsg is not None and CRS.from_epsg(bounds_epsg) != src.crs:
                bounds = transform_bounds(CRS.from_epsg(bounds_epsg), src.crs, *bounds, densify_pts=21)
            win = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
            win = win.intersection(Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                raise ValueError("bounds do not intersect the raster")
            data = src.read(band, window=win)
            transform = src.window_transform(win)
        else:
            data = src.read(band)
            transform = src.transform
        data = data.astype(np.float64)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        crs = src.crs
    data[~np.isfinite(data)] = np.nan
    g = Grid(data, transform, crs, name if name is not None else path.stem, units)
    if epsg is not None:
        g = g.reproject(epsg)
    return g


read_ers = read_grid
