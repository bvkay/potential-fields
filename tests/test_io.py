"""Read, write, subset and reprojection tests.

Each test states what would make it fail. All compare against independently
computed expectations, not against the code under test.

@author: Ben Kay (ben@auscope.org.au)
"""

from pathlib import Path

import numpy as np
import pyproj
import pytest
from rasterio.crs import CRS

from potfield import Grid, read_grid, write_grid, write_xyz
from potfield.export import write_netcdf

REAL_ERS = Path("E:/Ben_Documents/DATA/Curnamona_Magnetics/magmap_v6_2015.ers")


@pytest.mark.parametrize("ext", ["tif", "grd", "asc", "ers"])
def test_roundtrip_formats(mga_grid, tmp_path, ext):
    # Fails if any finite cell differs beyond float32 precision, any NaN moves,
    # the geotransform changes, or the CRS is lost.
    p = write_grid(mga_grid, tmp_path / f"g.{ext}")
    back = read_grid(p)
    assert back.shape == mga_grid.shape
    assert np.array_equal(np.isnan(back.values), np.isnan(mga_grid.values))
    finite = np.isfinite(mga_grid.values)
    assert np.max(np.abs(back.values[finite] - mga_grid.values[finite])) < 1e-4
    assert back.transform.almost_equals(mga_grid.transform, precision=1e-6)
    assert back.crs.is_projected
    assert pyproj.CRS(back.crs.to_wkt()).equals(pyproj.CRS.from_epsg(28354), ignore_axis_order=True)


def test_ers_header_geographic_and_projected(geo_grid, mga_grid, tmp_path):
    # Fails if a lat/lon grid is not written as LATLONG with D:M:S
    # registration, if GDAL cannot read the header back to the same
    # geotransform, or if the MGA grid does not get the ER Mapper zone name.
    p = write_grid(geo_grid, tmp_path / "geo.ers")
    text = p.read_text()
    assert "CoordinateType\t= LATLONG" in text and "Longitude\t= 140:0:0" in text
    assert 'Projection\t= "GEODETIC"' in text and 'Datum\t= "GDA94"' in text
    back = read_grid(p)
    assert back.transform.almost_equals(geo_grid.transform, precision=1e-9)
    assert not back.is_projected
    finite = np.isfinite(geo_grid.values)
    assert np.max(np.abs(back.values[finite] - geo_grid.values[finite])) < 1e-3

    p2 = write_grid(mga_grid, tmp_path / "mga.ers")
    text2 = p2.read_text()
    assert 'Projection\t= "MGA54"' in text2 and "CoordinateType\t= EN" in text2
    assert (tmp_path / "mga").stat().st_size == mga_grid.values.size * 4


def test_roundtrip_netcdf(mga_grid, tmp_path):
    import xarray as xr

    p = write_netcdf(mga_grid, tmp_path / "g.nc")
    da = xr.open_dataarray(p)
    assert da.shape == mga_grid.shape
    assert np.allclose(da.x.values, mga_grid.x)
    assert np.allclose(da.y.values, mga_grid.y)
    assert np.array_equal(np.isnan(da.values), np.isnan(mga_grid.values))
    assert da.attrs["epsg"] == 28354


def test_xyz(mga_grid, tmp_path):
    # Fails if the row count differs from the finite cell count or the first
    # line is not the north-west finite cell centre.
    p = write_xyz(mga_grid, tmp_path / "g.xyz")
    arr = np.loadtxt(p, comments="#")
    assert arr.shape[0] == np.isfinite(mga_grid.values).sum()
    assert np.allclose(arr[0], [mga_grid.x[0], mga_grid.y[0], mga_grid.values[0, 0]], atol=1e-3)


def test_write_reprojects_when_epsg_given(geo_grid, tmp_path):
    p = write_grid(geo_grid, tmp_path / "g.tif", epsg=28354)
    back = read_grid(p)
    assert back.epsg == 28354
    assert back.is_projected


def test_subset_indices(mga_grid):
    # Bounds covering columns 10..29 and rows 20..39 exactly. Fails if the
    # slice or origin is off by a cell.
    w, s, e, n = mga_grid.bounds
    d = mga_grid.dx
    sub = mga_grid.subset((w + 10 * d, n - 40 * d, w + 30 * d, n - 20 * d))
    assert sub.shape == (20, 20)
    assert np.array_equal(sub.values, mga_grid.values[20:40, 10:30], equal_nan=True)
    assert sub.transform.c == pytest.approx(w + 10 * d)
    assert sub.transform.f == pytest.approx(n - 20 * d)


def test_subset_partial_cells_snap_outward(mga_grid):
    w, s, e, n = mga_grid.bounds
    d = mga_grid.dx
    sub = mga_grid.subset((w + 10.5 * d, n - 40.5 * d, w + 29.5 * d, n - 19.5 * d))
    assert sub.shape == (22, 20)


def test_subset_in_other_crs(mga_grid):
    # A lat/lon rectangle built from the SW and NE corners of an MGA box is
    # slightly larger than the box (grid convergence), so the geographic
    # subset must contain the MGA subset and be at most 2 cells larger per
    # axis. Fails if the window is shifted or the CRS conversion is skipped.
    w, s, e, n = mga_grid.bounds
    d = mga_grid.dx
    box = (w + 10 * d, n - 40 * d, w + 30 * d, n - 20 * d)
    tr = pyproj.Transformer.from_crs(28354, 4283, always_xy=True)
    lon0, lat0 = tr.transform(box[0], box[1])
    lon1, lat1 = tr.transform(box[2], box[3])
    a = mga_grid.subset(box)
    b = mga_grid.subset((lon0, lat0, lon1, lat1), bounds_epsg=4283)
    aw, as_, ae, an = a.bounds
    bw, bs, be, bn = b.bounds
    assert bw <= aw and bs <= as_ and be >= ae and bn >= an
    assert 0 <= b.nx - a.nx <= 2 and 0 <= b.ny - a.ny <= 2


def test_clip_polygon(mga_grid):
    from shapely.geometry import box

    w, s, e, n = mga_grid.bounds
    d = mga_grid.dx
    poly = box(w + 10 * d, n - 40 * d, w + 30 * d, n - 20 * d)
    c = mga_grid.clip(poly)
    assert c.shape == (20, 20)
    assert np.isfinite(c.values).sum() == np.isfinite(mga_grid.values[20:40, 10:30]).sum()
    c2 = mga_grid.clip(poly, crop=False)
    assert c2.shape == mga_grid.shape
    assert np.isnan(c2.values[0, 0]) and np.isfinite(c2.values[30, 20])


def test_mga_zone(geo_grid, mga_grid):
    assert geo_grid.mga_epsg() == 28354  # 140.15E is in zone 54 (138 to 144E)
    assert geo_grid.mga_epsg("GDA2020") == 7854
    assert mga_grid.mga_epsg() == 28354


def test_spacing_geographic_matches_geodesic(geo_grid):
    # Fails if the cos(lat) spacing is more than 1% from the geodesic cell size.
    sx, sy = geo_grid.spacing()
    lon, lat = geo_grid.centre()
    geod = pyproj.Geod(ellps="GRS80")
    _, _, ex = geod.inv(lon, lat, lon + geo_grid.dx, lat)
    _, _, ny_ = geod.inv(lon, lat, lon, lat + geo_grid.dy)
    assert abs(sx - ex) / ex < 0.01
    assert abs(sy - ny_) / ny_ < 0.01


def test_reproject_geographic_to_mga(geo_grid):
    # After warping, each output cell centre converted back to lon must match
    # the stored value lon*1000 to within half a cell (0.001 deg -> 1.0).
    g = geo_grid.reproject()
    assert g.epsg == 28354
    assert g.dx == pytest.approx(g.dy)
    assert 150 < g.dx < 250  # 0.002 deg at 31S is ~190 m east, ~222 m north
    tr = pyproj.Transformer.from_crs(28354, 4283, always_xy=True)
    xx, yy = np.meshgrid(g.x, g.y)
    lon, _ = tr.transform(xx, yy)
    finite = np.isfinite(g.values)
    assert finite.mean() > 0.8
    err = np.abs(g.values[finite] - lon[finite] * 1000.0)
    assert np.percentile(err, 99) < 1.0


def test_reproject_noop_same_crs(mga_grid):
    assert mga_grid.reproject(28354) is mga_grid


@pytest.mark.skipif(not REAL_ERS.exists(), reason="Curnamona ERS not on this machine")
def test_read_real_ers_window():
    g = read_grid(REAL_ERS, bounds=(140.0, -31.5, 140.2, -31.3), units="nT")
    assert g.shape == (240, 240)
    assert g.epsg in (4283, None) and not g.is_projected
    assert np.isfinite(g.values).mean() > 0.9
    sx, sy = g.spacing()
    assert 70 < sx < 90 and 85 < sy < 100
    m = g.reproject()
    assert m.epsg == 28354 and 70 < m.dx < 100
