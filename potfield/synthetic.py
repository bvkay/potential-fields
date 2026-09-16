"""Analytic fields of simple sources, for tests and for checking filters.

Coordinates: x east, y north, in metres relative to the source, which sits at
depth z0 below the observation plane. Inclination is positive down,
declination positive clockwise from north. Field units: gravity in m/s^2 for
G*m = 1 (relative), or mGal when a real mass is given; TMI in nT.
"""

from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from .grid import Grid

G = 6.674e-11  # m^3 kg^-1 s^-2
CM = 1e-7  # mu0 / 4pi, T m A^-1
MGAL = 1e5  # mGal per m/s^2
NT = 1e9  # nT per T


def unit_vector(inc: float, dec: float) -> np.ndarray:
    """(east, north, up) unit vector for inclination and declination in degrees."""
    i, d = np.radians(inc), np.radians(dec)
    return np.array([np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), -np.sin(i)])


def point_mass(x: np.ndarray, y: np.ndarray, z0: float, gm: float = 1.0, height: float = 0.0) -> dict[str, np.ndarray]:
    """Vertical gravity of a point mass and its derivatives at `height` above the plane.

    gm is G times mass. Returns gz, dx, dy (horizontal derivatives, y north),
    vd1 and vd2 (first and second vertical derivatives, z positive down).
    """
    z = z0 + height
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    return {
        "gz": gm * z / r**3,
        "dx": -3 * gm * z * x / r**5,
        "dy": -3 * gm * z * y / r**5,
        "vd1": gm * (3 * z**2 - r2) / r**5,
        "vd2": 3 * gm * z * (2 * z**2 - 3 * (x**2 + y**2)) / r**7,
    }


def dipole_tmi(
    x: np.ndarray,
    y: np.ndarray,
    z0: float,
    moment: float,
    inc: float,
    dec: float,
    minc: float | None = None,
    mdec: float | None = None,
) -> np.ndarray:
    """Total field anomaly (nT) of a point dipole.

    moment in A m^2. inc, dec give the ambient field direction; minc, mdec
    the magnetisation direction (default: induced, same as the field).
    Uses B = Cm m [3 (m.r) r - m] / r^3 and dT = f . B (Blakely 1996, 4.13
    and 8.11).
    """
    f = unit_vector(inc, dec)
    m = unit_vector(inc if minc is None else minc, dec if mdec is None else mdec)
    rz = np.full_like(x, z0, dtype=float)
    r = np.sqrt(x**2 + y**2 + rz**2)
    mr = (m[0] * x + m[1] * y + m[2] * rz) / r
    bx = (3 * mr * x / r - m[0]) / r**3
    by = (3 * mr * y / r - m[1]) / r**3
    bz = (3 * mr * rz / r - m[2]) / r**3
    return NT * CM * moment * (f[0] * bx + f[1] * by + f[2] * bz)


def _blank_grid(nx: int, ny: int, dx: float, epsg: int, origin: tuple[float, float], name: str, units: str) -> Grid:
    t = from_origin(origin[0], origin[1], dx, dx)
    return Grid(np.zeros((ny, nx)), t, CRS.from_epsg(epsg), name=name, units=units)


def point_mass_grid(
    nx: int = 128,
    ny: int = 128,
    dx: float = 100.0,
    z0: float = 1500.0,
    gm: float = 1.0,
    epsg: int = 28354,
    origin: tuple[float, float] = (600_000.0, 6_500_000.0),
    height: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[Grid, dict[str, np.ndarray]]:
    """Grid of point-mass gravity with the source under the grid centre
    (plus `offset` metres east, north).

    Returns (grid, analytic) where analytic holds the fields from point_mass()
    on the same cells.
    """
    g0 = _blank_grid(nx, ny, dx, epsg, origin, "point_mass", "mGal")
    cx, cy = g0.centre()
    xx, yy = np.meshgrid(g0.x - cx - offset[0], g0.y - cy - offset[1])
    fields = point_mass(xx, yy, z0, gm, height)
    return g0.with_values(fields["gz"]), fields


def dipole_grid(
    nx: int = 128,
    ny: int = 128,
    dx: float = 100.0,
    z0: float = 1500.0,
    moment: float = 1e6,
    inc: float = -60.0,
    dec: float = 5.0,
    minc: float | None = None,
    mdec: float | None = None,
    epsg: int = 28354,
    origin: tuple[float, float] = (600_000.0, 6_500_000.0),
    offset: tuple[float, float] = (0.0, 0.0),
) -> Grid:
    """Grid of dipole TMI (nT) with the source under the grid centre plus `offset`."""
    g0 = _blank_grid(nx, ny, dx, epsg, origin, "dipole", "nT")
    cx, cy = g0.centre()
    xx, yy = np.meshgrid(g0.x - cx - offset[0], g0.y - cy - offset[1])
    return g0.with_values(dipole_tmi(xx, yy, z0, moment, inc, dec, minc, mdec))
