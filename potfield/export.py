"""Write grids to GeoTIFF, Surfer GRD, Arc ASCII, ERS, netCDF and XYZ.

Driver is chosen from the file extension. Pass epsg to reproject before
writing; Geotools needs the file CRS to match its database CRS.

Sidecar files: Arc ASCII gets a .prj, Surfer GRD gets a .aux.xml with the CRS
(Surfer's own format has no CRS field). Surfer null cells are written with
Surfer's blank value 1.70141e38; the nodata argument is ignored for .grd.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from .grid import Grid

DRIVERS = {
    ".tif": "GTiff",
    ".tiff": "GTiff",
    ".grd": "GSBG",  # Surfer 6 binary (DSBB). Use driver="GS7BG" for Surfer 7.
    ".asc": "AAIGrid",
    ".ers": "ERS",
}

SURFER_BLANK = 1.70141e38  # Surfer's blank value; GDAL does not translate nodata for GSBG


def _prepare(grid: Grid, epsg: int | None) -> Grid:
    if epsg is not None and grid.epsg != epsg:
        grid = grid.reproject(epsg)
    return grid


def write_grid(
    grid: Grid,
    path: str | Path,
    epsg: int | None = None,
    nodata: float = -99999.0,
    driver: str | None = None,
    **creation_options,
) -> Path:
    """Write a grid. Format from extension: .tif .grd .asc .ers .nc .xyz."""
    path = Path(path)
    grid = _prepare(grid, epsg)
    ext = path.suffix.lower()
    if ext == ".xyz":
        return write_xyz(grid, path)
    if ext == ".nc":
        return write_netcdf(grid, path)
    driver = driver or DRIVERS.get(ext)
    if driver is None:
        raise ValueError(f"no driver for extension {ext!r}")
    if driver in ("GSBG", "GS7BG"):
        nodata = SURFER_BLANK
    data = np.where(np.isfinite(grid.values), grid.values, nodata).astype(np.float32)
    profile = dict(
        driver=driver,
        height=grid.ny,
        width=grid.nx,
        count=1,
        dtype="float32",
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
    )
    if driver == "GTiff":
        profile.setdefault("compress", "deflate")
    if driver == "AAIGrid":
        creation_options.setdefault("DECIMAL_PRECISION", 6)
    profile.update(creation_options)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        if grid.units:
            dst.update_tags(1, units=grid.units)
    return path


def write_xyz(grid: Grid, path: str | Path, skip_null: bool = True, header: bool = True, epsg: int | None = None) -> Path:
    """Three-column ASCII: x y value, one cell per line, north row first.

    Coordinates are cell centres. Null cells are dropped unless skip_null is
    False, in which case they are written as -99999.
    """
    path = Path(path)
    grid = _prepare(grid, epsg)
    xx, yy = np.meshgrid(grid.x, grid.y)
    v = grid.values
    if skip_null:
        keep = np.isfinite(v)
        cols = np.column_stack([xx[keep], yy[keep], v[keep]])
    else:
        cols = np.column_stack([xx.ravel(), yy.ravel(), np.where(np.isfinite(v), v, -99999.0).ravel()])
    coord_fmt = "%.3f" if grid.is_projected else "%.7f"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        cols,
        fmt=f"{coord_fmt} {coord_fmt} %.5f",
        header=f"x y {grid.units or 'value'} EPSG:{grid.epsg}" if header else "",
        comments="# " if header else "",
    )
    return path


def write_netcdf(grid: Grid, path: str | Path, epsg: int | None = None) -> Path:
    """CF-style netCDF via xarray with x, y coordinates and crs_wkt attribute."""
    import xarray as xr

    path = Path(path)
    grid = _prepare(grid, epsg)
    da = xr.DataArray(
        grid.values,
        dims=("y", "x"),
        coords={"y": grid.y, "x": grid.x},
        name=grid.name or "value",
        attrs={"units": grid.units, "crs_wkt": grid.crs.to_wkt(), "epsg": grid.epsg or -1},
    )
    da.x.attrs["units"] = "m" if grid.is_projected else "degrees_east"
    da.y.attrs["units"] = "m" if grid.is_projected else "degrees_north"
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(path)
    return path


def write_all(grid: Grid, stem: str | Path, formats=("tif", "grd", "asc", "xyz"), epsg: int | None = None) -> list[Path]:
    """Write the same grid in several formats: stem.tif, stem.grd, ..."""
    stem = Path(stem)
    grid = _prepare(grid, epsg)
    return [write_grid(grid, stem.with_suffix(f".{f}")) for f in formats]
