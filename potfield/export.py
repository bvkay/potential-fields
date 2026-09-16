"""Write grids to GeoTIFF, Surfer GRD, Arc ASCII, ERS, netCDF and XYZ.

Driver is chosen from the file extension. Pass epsg to reproject before
writing; Geotools needs the file CRS to match its database CRS.

Sidecar files: Arc ASCII gets a .prj, Surfer GRD gets a .aux.xml with the CRS
(Surfer's own format has no CRS field). Surfer null cells are written with
Surfer's blank value 1.70141e38; the nodata argument is ignored for .grd.
ERS is written directly (write_ers) so lat/lon grids get a LATLONG header
like the Geoscience Australia grids; GDAL's ERS driver writes EN for
everything.

@author: Ben Kay (ben@auscope.org.au)
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
    if ext == ".ers" and driver is None:
        return write_ers(grid, path, nodata=nodata)
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


def _dms(deg: float) -> str:
    """Decimal degrees to ER Mapper D:M:S with the sign on the degrees field."""
    sign = "-" if deg < 0 else ""
    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return f"{sign}{d}:{m}:{s:.6f}"


def _ers_coordinate_space(crs: CRS) -> tuple[str, str, str]:
    """(Datum, Projection, CoordinateType) for an ERS header.

    Uses ER Mapper names where they are certain (GDA94 geodetic and MGA
    zones, WGS84 geodetic) and "EPSG:code" otherwise, which GDAL and recent
    ER Mapper-family software read back.
    """
    import pyproj

    pc = pyproj.CRS(crs.to_wkt())
    epsg = pc.to_epsg(min_confidence=25)
    if pc.is_geographic:
        datum = (pc.datum.name if pc.datum else "") or ""
        if epsg == 4283 or "GDA94" in datum or "Australia 1994" in datum:
            return "GDA94", "GEODETIC", "LATLONG"
        if epsg == 4326 or "WGS 84" in datum or "WGS84" in datum:
            return "WGS84", "GEODETIC", "LATLONG"
        if epsg is not None:
            return f"EPSG:{epsg}", "GEODETIC", "LATLONG"
        raise ValueError("cannot name this geographic CRS for ERS; reproject to a known EPSG first")
    if epsg is not None and 28349 <= epsg <= 28356:
        return "GDA94", f"MGA{epsg - 28300}", "EN"
    if epsg is not None:
        return f"EPSG:{epsg}", f"EPSG:{epsg}", "EN"
    raise ValueError("cannot name this projected CRS for ERS; reproject to a known EPSG first")


def write_ers(grid: Grid, path: str | Path, nodata: float = -99999.0, epsg: int | None = None) -> Path:
    """ER Mapper grid: text header `name.ers` plus flat little-endian float32 `name`.

    The header follows the Geoscience Australia layout: LATLONG grids carry
    the registration point as Longitude/Latitude in D:M:S, projected grids
    as Eastings/Northings in metres. Registration cell (0, 0) is the top-left
    corner of the top-left cell, which is what GDAL assumes on read.
    """
    path = Path(path)
    if path.suffix.lower() != ".ers":
        path = path.with_suffix(".ers")
    grid = _prepare(grid, epsg)
    datum, projection, ctype = _ers_coordinate_space(grid.crs)
    w, s, e, n = grid.bounds
    if ctype == "LATLONG":
        reg = f"\t\t\tLongitude\t= {_dms(w)}\n\t\t\tLatitude\t= {_dms(n)}\n"
        units = ""
    else:
        reg = f"\t\t\tEastings\t= {w:.6f}\n\t\t\tNorthings\t= {n:.6f}\n"
        units = '\t\tUnits\t= "METERS"\n'
    header = (
        "DatasetHeader Begin\n"
        '\tVersion\t= "6.0"\n'
        f'\tName\t= "{path.name}"\n'
        "\tDataSetType\t= ERStorage\n"
        "\tDataType\t= Raster\n"
        "\tByteOrder\t= LSBFirst\n"
        "\tCoordinateSpace Begin\n"
        f'\t\tDatum\t= "{datum}"\n'
        f'\t\tProjection\t= "{projection}"\n'
        f"\t\tCoordinateType\t= {ctype}\n"
        f"{units}"
        "\t\tRotation\t= 0:0:0.0\n"
        "\tCoordinateSpace End\n"
        "\tRasterInfo Begin\n"
        "\t\tCellType\t= IEEE4ByteReal\n"
        f"\t\tNullCellValue\t= {nodata:g}\n"
        "\t\tCellInfo Begin\n"
        f"\t\t\tXdimension\t= {grid.dx:.12g}\n"
        f"\t\t\tYdimension\t= {grid.dy:.12g}\n"
        "\t\tCellInfo End\n"
        f"\t\tNrOfLines\t= {grid.ny}\n"
        f"\t\tNrOfCellsPerLine\t= {grid.nx}\n"
        "\t\tNrOfBands\t= 1\n"
        "\t\tRegistrationCellX\t= 0\n"
        "\t\tRegistrationCellY\t= 0\n"
        "\t\tRegistrationCoord Begin\n"
        f"{reg}"
        "\t\tRegistrationCoord End\n"
        "\t\tBandId Begin\n"
        f'\t\t\tValue\t= "{grid.units or grid.name or "value"}"\n'
        "\t\tBandId End\n"
        "\tRasterInfo End\n"
        "DatasetHeader End\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header, encoding="ascii", newline="\n")
    data = np.where(np.isfinite(grid.values), grid.values, nodata).astype("<f4")
    data.tofile(path.with_suffix(""))
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
