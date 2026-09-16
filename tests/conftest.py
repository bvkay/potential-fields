import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from potfield import Grid


@pytest.fixture
def mga_grid() -> Grid:
    """60 x 80 cells, 100 m, GDA94 MGA zone 54, smooth values in [-100, 100] with a NaN block."""
    ny, nx, d = 60, 80, 100.0
    t = from_origin(600_000.0, 6_500_000.0, d, d)
    j, i = np.mgrid[0:ny, 0:nx]
    v = 100.0 * np.sin(2 * np.pi * i / 40) * np.cos(2 * np.pi * j / 30)
    v[5:10, 5:12] = np.nan
    return Grid(v, t, CRS.from_epsg(28354), name="synthetic", units="nT")


@pytest.fixture
def geo_grid() -> Grid:
    """Geographic grid over the Curnamona area, values = lon * 1000."""
    ny, nx, d = 120, 150, 0.002
    t = from_origin(140.0, -31.0, d, d)
    lon = 140.0 + (np.arange(nx) + 0.5) * d
    v = np.tile(lon * 1000.0, (ny, 1))
    return Grid(v, t, CRS.from_epsg(4283), name="geo", units="test")
