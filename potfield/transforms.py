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

@author: Ben Kay (ben@auscope.org.au)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from . import fft
from .filters import _spacing, butterworth_highpass, butterworth_lowpass, vertical_derivative
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


PG_DENSITY = 1000.0  # nominal rho0 (kg/m^3) used for pseudogravity inside poisson_analysis
PG_MAGNETISATION = 1.0  # nominal M0 (A/m)


def poisson_analysis(
    gravity: Grid,
    magnetics: Grid,
    inclination: float | None = None,
    declination: float | None = None,
    window: int = 21,
    window_m: float | None = None,
    magnetics_is_rtp: bool = False,
    mode: str = "pseudogravity",
    lowpass_wavelength: float | None = None,
    regional_wavelength: float | None = None,
    pad: int | None = None,
) -> dict[str, Grid]:
    """Moving-window Poisson analysis (Chandler and Malek 1991).

    gravity in mGal, magnetics in nT. magnetics is reduced to the pole with
    the given field direction unless magnetics_is_rtp. The regression pair
    is formed, put on the gravity geometry, band passed identically, and
    regressed in a sliding window.

    mode "pseudogravity" (default): y = pseudogravity of the RTP field
        (nominal rho0 = 1000 kg/m^3, M0 = 1 A/m), x = gravity. Both are
        smooth, so the fit is controlled by the wavelengths the gravity
        grid resolves. slope = (rho0/M0) * (M/rho).
    mode "derivative": y = RTP field, x = dg/dz (the form in Chandler and
        Malek). Sharper, but dg/dz amplifies gridding noise in gravity
        interpolated from sparse stations. slope = Cm M / (G rho) in
        nT per mGal/m.

    lowpass_wavelength (m): Butterworth low pass on both inputs; set to about
        twice the gravity station spacing.
    regional_wavelength (m): Butterworth high pass on both inputs. Without it
        two smooth trends inside a window correlate whatever their sources.
    window: regression window in cells (odd); window_m in metres overrides it.

    Returns Grids: slope, intercept, correlation (Pearson r), m_over_rho
    (A/m per kg/m^3), signal (window std of x over its global std, a
    measure of whether the window holds an anomaly at all), and the two
    regressed inputs x and y.
    """
    if mode not in ("pseudogravity", "derivative"):
        raise ValueError("mode must be 'pseudogravity' or 'derivative'")
    if not magnetics_is_rtp:
        if inclination is None or declination is None:
            raise ValueError("inclination and declination are needed unless magnetics_is_rtp")
        magnetics = reduce_to_pole(magnetics, inclination, declination, pad=pad)
    if mode == "pseudogravity":
        y = pseudogravity(magnetics, 90.0, 0.0, PG_DENSITY, PG_MAGNETISATION, pad=pad).regrid(gravity)
        x = gravity
        slope_units, ratio_scale = "mGal/mGal", PG_MAGNETISATION / PG_DENSITY
    else:
        y = magnetics.regrid(gravity)
        x = vertical_derivative(gravity, 1, pad=pad)
        slope_units, ratio_scale = "nT/(mGal/m)", (MGAL / NT) * G / CM
    if lowpass_wavelength:
        x = butterworth_lowpass(x, lowpass_wavelength, pad=pad)
        y = butterworth_lowpass(y, lowpass_wavelength, pad=pad)
    if regional_wavelength:
        x = butterworth_highpass(x, regional_wavelength, pad=pad)
        y = butterworth_highpass(y, regional_wavelength, pad=pad)

    size = int(round(window_m / gravity.dx)) if window_m is not None else int(window)
    size = max(3, size) | 1  # odd, at least 3
    reg = windowed_regression(x.values, y.values, size)
    valid = np.isfinite(x.values)
    mx, _ = _window_mean(x.values, valid, size)
    mxx, _ = _window_mean(x.values**2, valid, size)
    signal = np.sqrt(np.maximum(mxx - mx**2, 0.0)) / np.nanstd(x.values)
    signal[~np.isfinite(reg["correlation"])] = np.nan
    return {
        "slope": gravity.with_values(reg["slope"], name="poisson_slope", units=slope_units),
        "intercept": gravity.with_values(reg["intercept"], name="poisson_intercept", units=y.units),
        "correlation": gravity.with_values(reg["correlation"], name="poisson_correlation", units=""),
        "m_over_rho": gravity.with_values(reg["slope"] * ratio_scale, name="poisson_m_over_rho", units="(A/m)/(kg/m^3)"),
        "signal": gravity.with_values(signal, name="poisson_signal", units=""),
        "x": x.with_values(x.values, name=f"{x.name} (regression x)"),
        "y": y.with_values(y.values, name=f"{y.name} (regression y)"),
    }
