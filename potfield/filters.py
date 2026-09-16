"""Wavenumber-domain filters for potential field grids.

Every function takes a Grid and returns a Grid (or a tuple of Grids). Cell
sizes come from Grid.spacing() in metres. k is the angular wavenumber
|k| = 2*pi*sqrt(u^2 + v^2) in rad/m; kx, ky are its east and north components.

Filter responses, Blakely (1996) chapter 12 unless noted:

    continuation to height h        H = exp(-h k)                    (12.7)
    nth vertical derivative         H = k^n                          (12.20)
    nth horizontal derivative       H = (i kx)^n, (i ky)^n           (12.17)
    Butterworth low pass            H = 1 / (1 + (k/k0)^(2n))
    Gaussian low pass               H = exp(-(k/k0)^2 / 2)
    directional cosine              H = |cos(theta - alpha)|^n

with k0 = 2*pi / cutoff wavelength. Derived products:

    total horizontal derivative     THDR = sqrt(dx^2 + dy^2)
    analytic signal amplitude       AS   = sqrt(dx^2 + dy^2 + dz^2)  Roest et al. 1992
    tilt angle                      T    = atan(dz / THDR)           Miller and Singh 1994

Vertical derivatives use z positive down, the usual convention for potential
field maps: the first vertical derivative is positive over a positive peak.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import fft
from .grid import Grid


def _spacing(grid: Grid) -> tuple[float, float]:
    if not grid.is_projected:
        warnings.warn(
            "geographic grid: cell size from cos(lat) approximation. Use Grid.reproject() for exact spacing.",
            stacklevel=3,
        )
    return grid.spacing()


def _apply(grid: Grid, response, pad, name: str, units: str | None = None, plane: str = "keep") -> Grid:
    dx, dy = _spacing(grid)
    out = fft.apply(grid.values, dx, dy, response, pad=pad, plane=plane)
    return grid.with_values(out, name=f"{grid.name}_{name}", units=grid.units if units is None else units)


def _k0(wavelength: float) -> float:
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    return 2 * np.pi / wavelength


# responses ---------------------------------------------------------------


def continuation_response(k: np.ndarray, height: float) -> np.ndarray:
    """exp(-h k). h > 0 upward (attenuates), h < 0 downward (amplifies)."""
    return np.exp(-height * k)


def butterworth_lowpass_response(k: np.ndarray, wavelength: float, order: int = 4) -> np.ndarray:
    return 1.0 / (1.0 + (k / _k0(wavelength)) ** (2 * order))


def gaussian_lowpass_response(k: np.ndarray, wavelength: float) -> np.ndarray:
    return np.exp(-0.5 * (k / _k0(wavelength)) ** 2)


# continuation ------------------------------------------------------------


def upward_continue(grid: Grid, height: float, pad: int | None = None) -> Grid:
    """Field at `height` metres above the observation surface."""
    if height < 0:
        raise ValueError("height must be >= 0; use downward_continue()")
    return _apply(grid, lambda k, kx, ky: continuation_response(k, height), pad, f"uc{height:g}")


def downward_continue(grid: Grid, depth: float, cutoff_wavelength: float, order: int = 4, pad: int | None = None) -> Grid:
    """Field `depth` metres below the observation surface.

    exp(+depth k) amplifies noise without bound, so it is combined with a
    Butterworth low pass at cutoff_wavelength. Keep depth below about
    2 to 3 cell sizes and the cutoff above about 4 cells unless the data are
    very clean.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")

    def response(k, kx, ky):
        return continuation_response(k, -depth) * butterworth_lowpass_response(k, cutoff_wavelength, order)

    return _apply(grid, response, pad, f"dc{depth:g}")


def regional_residual(grid: Grid, height: float, pad: int | None = None) -> tuple[Grid, Grid]:
    """(regional, residual) with regional = upward continuation to `height`."""
    regional = upward_continue(grid, height, pad)
    residual = grid.with_values(grid.values - regional.values, name=f"{grid.name}_res{height:g}")
    return regional, residual


# derivatives -------------------------------------------------------------


def vertical_derivative(grid: Grid, order: float = 1, pad: int | None = None) -> Grid:
    """nth vertical derivative, z positive down. order may be fractional.

    Negative order gives the vertical integral; the DC term is set to zero.
    """

    def response(k, kx, ky):
        H = np.where(k > 0, k**order, 0.0)
        return H

    units = f"{grid.units}/m^{order:g}" if grid.units else ""
    return _apply(grid, response, pad, f"vd{order:g}", units=units, plane="drop")


def horizontal_derivatives(grid: Grid, order: float = 1, pad: int | None = None) -> tuple[Grid, Grid]:
    """(d/dx, d/dy) with x east and y north.

    For order 1 the gradient of the removed boundary plane is added back;
    for other orders the plane contributes nothing.
    """
    units = f"{grid.units}/m^{order:g}" if grid.units else ""
    px, py = ("dx", "dy") if order == 1 else ("drop", "drop")
    gx = _apply(grid, lambda k, kx, ky: (1j * kx) ** order, pad, f"dx{order:g}", units, plane=px)
    gy = _apply(grid, lambda k, kx, ky: (1j * ky) ** order, pad, f"dy{order:g}", units, plane=py)
    return gx, gy


def total_horizontal_derivative(grid: Grid, pad: int | None = None) -> Grid:
    """THDR = sqrt(dx^2 + dy^2). Peaks over edges of sources."""
    gx, gy = horizontal_derivatives(grid, 1, pad)
    return grid.with_values(np.hypot(gx.values, gy.values), name=f"{grid.name}_thdr", units=gx.units)


def analytic_signal(grid: Grid, pad: int | None = None) -> Grid:
    """Analytic signal amplitude sqrt(dx^2 + dy^2 + dz^2).

    Peaks over source edges and, for magnetics, is close to independent of
    field direction, so it is often used instead of reduction to the pole.
    """
    gx, gy = horizontal_derivatives(grid, 1, pad)
    gz = vertical_derivative(grid, 1, pad)
    a = np.sqrt(gx.values**2 + gy.values**2 + gz.values**2)
    return grid.with_values(a, name=f"{grid.name}_as", units=gx.units)


def tilt_angle(grid: Grid, degrees: bool = True, pad: int | None = None) -> Grid:
    """Tilt angle atan(dz / THDR).

    Positive over sources, negative outside, zero close to the source edge.
    Bounded to +-90 deg so weak and strong anomalies get equal weight.
    """
    thdr = total_horizontal_derivative(grid, pad)
    gz = vertical_derivative(grid, 1, pad)
    t = np.arctan2(gz.values, thdr.values)
    if degrees:
        t = np.degrees(t)
    return grid.with_values(t, name=f"{grid.name}_tilt", units="deg" if degrees else "rad")


# wavelength filters ------------------------------------------------------


def butterworth_lowpass(grid: Grid, wavelength: float, order: int = 4, pad: int | None = None) -> Grid:
    """Pass wavelengths longer than `wavelength` (metres)."""
    return _apply(grid, lambda k, kx, ky: butterworth_lowpass_response(k, wavelength, order), pad, f"lp{wavelength:g}")


def butterworth_highpass(grid: Grid, wavelength: float, order: int = 4, pad: int | None = None) -> Grid:
    """Pass wavelengths shorter than `wavelength` (metres)."""
    return _apply(
        grid,
        lambda k, kx, ky: 1.0 - butterworth_lowpass_response(k, wavelength, order),
        pad,
        f"hp{wavelength:g}",
        plane="drop",
    )


def bandpass(grid: Grid, long_wavelength: float, short_wavelength: float, order: int = 4, pad: int | None = None) -> Grid:
    """Pass wavelengths between short_wavelength and long_wavelength (metres)."""
    if not short_wavelength < long_wavelength:
        raise ValueError("need short_wavelength < long_wavelength")

    def response(k, kx, ky):
        lp = butterworth_lowpass_response(k, short_wavelength, order)
        hp = 1.0 - butterworth_lowpass_response(k, long_wavelength, order)
        return lp * hp

    return _apply(grid, response, pad, f"bp{short_wavelength:g}-{long_wavelength:g}", plane="drop")


def gaussian_lowpass(grid: Grid, wavelength: float, pad: int | None = None) -> Grid:
    """Gaussian low pass, no ringing, gentle roll-off; cutoff at 1/sqrt(e)."""
    return _apply(grid, lambda k, kx, ky: gaussian_lowpass_response(k, wavelength), pad, f"glp{wavelength:g}")


def gaussian_highpass(grid: Grid, wavelength: float, pad: int | None = None) -> Grid:
    return _apply(
        grid, lambda k, kx, ky: 1.0 - gaussian_lowpass_response(k, wavelength), pad, f"ghp{wavelength:g}", plane="drop"
    )


def directional_cosine(grid: Grid, azimuth: float, degree: float = 2.0, reject: bool = True, pad: int | None = None) -> Grid:
    """Remove (or keep) linear features striking at `azimuth` degrees clockwise from north.

    A feature striking at azimuth a has its spectral energy along the
    wavenumber direction a + 90. H = |cos(theta_k - (a + 90))|^degree is the
    weight along that direction; reject=True applies 1 - H. Higher degree
    gives a narrower rejection wedge and less damage to other features;
    degree 1 removes a broad wedge. The mean is always kept.
    """
    target = np.radians(azimuth + 90.0)

    def response(k, kx, ky):
        theta = np.arctan2(kx, ky)  # azimuth of the wavenumber vector, clockwise from north
        H = np.abs(np.cos(theta - target)) ** degree
        H = 1.0 - H if reject else H
        return np.where(k > 0, H, 1.0)

    return _apply(grid, response, pad, f"dcos{azimuth:g}")


# trend --------------------------------------------------------------------


def remove_trend(grid: Grid, order: int = 1) -> tuple[Grid, Grid]:
    """Least-squares polynomial surface of `order` (0 to 3) fitted to finite
    cells. Returns (residual, trend). Coordinates are normalised to [-1, 1]
    before fitting so the normal equations stay well conditioned.
    """
    if not 0 <= order <= 3:
        raise ValueError("order must be 0 to 3")
    xx, yy = np.meshgrid(grid.x, grid.y)
    xn = (xx - xx.mean()) / (np.ptp(xx) / 2 or 1.0)
    yn = (yy - yy.mean()) / (np.ptp(yy) / 2 or 1.0)
    terms = [xn**i * yn**j for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.stack([t.ravel() for t in terms], axis=1)
    v = grid.values.ravel()
    ok = np.isfinite(v)
    coef, *_ = np.linalg.lstsq(A[ok], v[ok], rcond=None)
    trend = (A @ coef).reshape(grid.shape)
    trend[~np.isfinite(grid.values)] = np.nan
    return (
        grid.with_values(grid.values - trend, name=f"{grid.name}_detrend{order}"),
        grid.with_values(trend, name=f"{grid.name}_trend{order}"),
    )
