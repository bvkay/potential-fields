"""Transforms that change the physical meaning of the field.

Reduction to the pole, pseudogravity, pseudomagnetic and moving-window
Poisson analysis. Blakely (1996) chapter 12 and section 5.6 unless noted.

Direction factor for a unit vector with inclination I (positive down) and
declination D (clockwise from north), evaluated along wavenumber azimuth
theta = atan2(kx, ky):

    Theta(I, D) = sin I + i cos I cos(theta - D)                    (11.29)

TMI of a body magnetised along (Im, Dm) in a field along (If, Df) has
spectrum proportional to Theta_m Theta_f |k| F[potential]. Dividing by
Theta_m Theta_f gives the field at the pole where both are 1:

    RTP              H = 1 / (Theta_m Theta_f)                      (12.31)
    pseudogravity    H = G rho / (Cm M) * 1 / (Theta_m Theta_f |k|) (12.36)
    pseudomagnetic   H = Cm M / (G rho) * Theta_m Theta_f |k|

Poisson's relation for a body with uniform density rho and magnetisation M
also holds point by point at the pole: dT_pole = Cm M / (G rho) * dg/dz.
Regressing RTP magnetics against the first vertical derivative of gravity in
a moving window gives that ratio where the two fields share a source
(Chandler and Malek 1991).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from . import fft
from .filters import _spacing, vertical_derivative
from .grid import Grid
from .synthetic import CM, G, MGAL, NT


def direction_factor(kx: np.ndarray, ky: np.ndarray, inc: float, dec: float) -> np.ndarray:
    """Theta(I, D) on the wavenumber grid. Undefined at k = 0 (set to sin I)."""
    i, d = np.radians(inc), np.radians(dec)
    theta = np.arctan2(kx, ky)
    return np.sin(i) + 1j * np.cos(i) * np.cos(theta - d)


def _theta_product(kx, ky, inc, dec, minc, mdec, amp_inclination):
    """Theta_m Theta_f, optionally with the amplitude stabilised.

    For |I| below amp_inclination the modulus is taken from a direction with
    that inclination while the phase is kept. This bounds 1/|Theta|^2 near
    the magnetic equator (Geosoft's amplitude-correction inclination).
    """
    tf = direction_factor(kx, ky, inc, dec)
    tm = direction_factor(kx, ky, inc if minc is None else minc, dec if mdec is None else mdec)
    if amp_inclination is None:
        return tf * tm
    prod = tf * tm
    ia = np.sign(inc) * max(abs(inc), abs(amp_inclination))
    ima = np.sign(inc if minc is None else minc) * max(abs(inc if minc is None else minc), abs(amp_inclination))
    amp = np.abs(direction_factor(kx, ky, ia, dec)) * np.abs(
        direction_factor(kx, ky, ima, dec if mdec is None else mdec)
    )
    return amp * np.exp(1j * np.angle(prod))


def reduce_to_pole(
    grid: Grid,
    inclination: float,
    declination: float,
    mag_inclination: float | None = None,
    mag_declination: float | None = None,
    amp_inclination: float | None = None,
    pad: int | None = None,
) -> Grid:
    """Reduction to the pole.

    inclination, declination: ambient field at the survey, degrees.
    mag_*: magnetisation direction if not induced.
    amp_inclination: stabilise amplitudes for |I| below this value (try 20).
    Not needed for South Australia where I is about -60 to -65.
    """
    dx, dy = _spacing(grid)

    def response(k, kx, ky):
        H = 1.0 / _theta_product(kx, ky, inclination, declination, mag_inclination, mag_declination, amp_inclination)
        return np.where(k > 0, H, 1.0)

    out = fft.apply(grid.values, dx, dy, response, pad=pad, plane="keep")
    return grid.with_values(out, name=f"{grid.name}_rtp")


def pseudogravity(
    grid: Grid,
    inclination: float,
    declination: float,
    density: float = 1000.0,
    magnetisation: float = 1.0,
    mag_inclination: float | None = None,
    mag_declination: float | None = None,
    amp_inclination: float | None = None,
    pad: int | None = None,
) -> Grid:
    """Pseudogravity (mGal) from TMI (nT).

    density in kg/m^3 and magnetisation in A/m set the scale; only their
    ratio matters. The result is the gravity a body would produce if its
    magnetisation M were replaced by density rho.
    """
    dx, dy = _spacing(grid)
    scale = G * density / (CM * magnetisation) * MGAL / NT  # nT -> mGal

    def response(k, kx, ky):
        tp = _theta_product(kx, ky, inclination, declination, mag_inclination, mag_declination, amp_inclination)
        H = scale / (tp * k)
        return np.where(k > 0, H, 0.0)

    out = fft.apply(grid.values, dx, dy, response, pad=pad, plane="drop")
    return grid.with_values(out, name=f"{grid.name}_pseudograv", units="mGal")


def pseudomagnetic(
    grid: Grid,
    inclination: float,
    declination: float,
    density: float = 1000.0,
    magnetisation: float = 1.0,
    mag_inclination: float | None = None,
    mag_declination: float | None = None,
    pad: int | None = None,
) -> Grid:
    """Pseudomagnetic TMI (nT) from gravity (mGal): the inverse of pseudogravity."""
    dx, dy = _spacing(grid)
    scale = CM * magnetisation / (G * density) * NT / MGAL  # mGal -> nT

    def response(k, kx, ky):
        tp = _theta_product(kx, ky, inclination, declination, mag_inclination, mag_declination, None)
        return scale * tp * k

    out = fft.apply(grid.values, dx, dy, response, pad=pad, plane="drop")
    return grid.with_values(out, name=f"{grid.name}_pseudomag", units="nT")


# Poisson analysis -----------------------------------------------------------


def _window_mean(a: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    frac = uniform_filter(valid.astype(float), size, mode="constant")
    s = uniform_filter(np.where(valid, a, 0.0), size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(frac > 0, s / frac, np.nan), frac


def windowed_regression(x: np.ndarray, y: np.ndarray, size: int, min_fraction: float = 0.5) -> dict[str, np.ndarray]:
    """Sliding-window least squares y = slope * x + intercept.

    size is the window width in cells (odd). Windows with fewer than
    min_fraction of their cells finite in both inputs return NaN.
    Returns slope, intercept, correlation (Pearson r).
    """
    valid = np.isfinite(x) & np.isfinite(y)
    mx, frac = _window_mean(x, valid, size)
    my, _ = _window_mean(y, valid, size)
    mxx, _ = _window_mean(x * x, valid, size)
    myy, _ = _window_mean(y * y, valid, size)
    mxy, _ = _window_mean(x * y, valid, size)
    var_x = mxx - mx**2
    var_y = myy - my**2
    cov = mxy - mx * my
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = cov / var_x
        r = cov / np.sqrt(var_x * var_y)
    intercept = my - slope * mx
    bad = (frac < min_fraction) | ~valid | ~(var_x > 0) | ~(var_y > 0)
    for arr in (slope, intercept, r):
        arr[bad] = np.nan
    return {"slope": slope, "intercept": intercept, "correlation": r}


def poisson_analysis(
    gravity: Grid,
    magnetics: Grid,
    inclination: float | None = None,
    declination: float | None = None,
    window: float = 15,
    magnetics_is_rtp: bool = False,
    pad: int | None = None,
) -> dict[str, Grid]:
    """Moving-window Poisson analysis (Chandler and Malek 1991).

    gravity in mGal, magnetics in nT. magnetics is reduced to the pole with
    the given field direction unless magnetics_is_rtp. Both grids are put on
    the gravity geometry. window is the regression window in cells (odd).

    Returns Grids: slope (nT per mGal/m), intercept (nT), correlation, and
    m_over_rho, the magnetisation to density ratio (A/m per kg/m^3) implied
    by the slope: M/rho = slope * 1e-4 * G / Cm.
    """
    if not magnetics_is_rtp:
        if inclination is None or declination is None:
            raise ValueError("inclination and declination are needed unless magnetics_is_rtp")
        magnetics = reduce_to_pole(magnetics, inclination, declination, pad=pad)
    mag = magnetics.regrid(gravity)
    gz = vertical_derivative(gravity, 1, pad=pad)
    size = int(window) | 1  # force odd
    reg = windowed_regression(gz.values, mag.values, size)
    ratio = reg["slope"] * (NT / MGAL) ** -1 * G / CM  # nT/(mGal/m) -> (A/m)/(kg/m^3)
    return {
        "slope": gravity.with_values(reg["slope"], name="poisson_slope", units="nT/(mGal/m)"),
        "intercept": gravity.with_values(reg["intercept"], name="poisson_intercept", units="nT"),
        "correlation": gravity.with_values(reg["correlation"], name="poisson_correlation", units=""),
        "m_over_rho": gravity.with_values(ratio, name="poisson_m_over_rho", units="(A/m)/(kg/m^3)"),
    }
