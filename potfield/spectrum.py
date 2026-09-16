"""Radially averaged power spectrum and depth-to-source estimates.

For an ensemble of sources at depth h the power spectrum falls off as
exp(-2 h |k|) with k the angular wavenumber in rad/m (Spector and Grant
1970; Blakely 1996 section 11.3). On a plot of ln P against k a straight
segment of slope s therefore gives h = -s / 2. With k in cycles/km the
depth in km is -s / (4 pi).

Deep sources dominate the low wavenumbers, shallow ones the high, so a
spectrum is usually read as two or three straight segments.

@author: Ben Kay (ben@auscope.org.au)
"""

from __future__ import annotations

import numpy as np

from . import fft
from .filters import _spacing
from .grid import Grid


def radial_power_spectrum(grid: Grid, nbins: int | None = None, pad: int | None = None) -> dict[str, np.ndarray]:
    """Radial average of the 2D power spectrum.

    Returns k (rad/m, bin centres), wavelength (m), log_power (ln of mean
    power per bin), power and count. Bins with no samples are dropped.
    """
    dx, dy = _spacing(grid)
    P, k = fft.power_spectrum(grid.values, dx, dy, pad=pad)
    kmax = np.pi / max(dx, dy)  # Nyquist of the coarser axis
    if nbins is None:
        nbins = max(16, min(grid.shape) // 2)
    edges = np.linspace(0.0, kmax, nbins + 1)
    sel = (k > 0) & (k <= kmax)
    idx = np.clip(np.digitize(k[sel], edges) - 1, 0, nbins - 1)
    count = np.bincount(idx, minlength=nbins)
    power = np.bincount(idx, weights=P[sel], minlength=nbins)
    keep = count > 0
    power = power[keep] / count[keep]
    kc = 0.5 * (edges[:-1] + edges[1:])[keep]
    return {
        "k": kc,
        "wavelength": 2 * np.pi / kc,
        "log_power": np.log(power),
        "power": power,
        "count": count[keep],
    }


def power_spectrum_2d(grid: Grid, pad: int | None = None) -> dict[str, np.ndarray]:
    """Two-dimensional log power spectrum, zero wavenumber at the centre.

    Returns log_power (2D, north up), kx and ky (1D, rad/m, ascending) and
    extent for imshow in rad/km. Linear features in the grid appear as a
    spoke through the centre perpendicular to their strike: north-south
    stripes put energy along the kx axis.
    """
    dx, dy = _spacing(grid)
    P, _ = fft.power_spectrum(grid.values, dx, dy, pad=pad)
    ny, nx = P.shape
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    logP = np.log(np.fft.fftshift(P) + np.finfo(float).tiny)
    # rows of the shifted array run from most negative fy to most positive.
    # Our ky is positive north, i.e. the negative of fy, so flip to put north up.
    logP = logP[::-1, :]
    dkx, dky = kx[1] - kx[0], ky[1] - ky[0]
    extent = ((kx[0] - dkx / 2) * 1e3, (kx[-1] + dkx / 2) * 1e3, (ky[0] - dky / 2) * 1e3, (ky[-1] + dky / 2) * 1e3)
    return {"log_power": logP, "kx": kx, "ky": ky, "extent": extent}


def depth_from_slope(k: np.ndarray, log_power: np.ndarray, kmin: float, kmax: float) -> dict[str, float]:
    """Straight-line fit to ln P over kmin <= k <= kmax (rad/m).

    Returns depth = -slope / 2 in metres, plus slope, intercept and the
    number of points used.
    """
    sel = (k >= kmin) & (k <= kmax) & np.isfinite(log_power)
    if sel.sum() < 3:
        raise ValueError("fewer than 3 spectrum points in the wavenumber range")
    slope, intercept = np.polyfit(k[sel], log_power[sel], 1)
    return {"depth": -slope / 2.0, "slope": float(slope), "intercept": float(intercept), "n": int(sel.sum())}


def depth_from_wavelengths(spec: dict[str, np.ndarray], long_wavelength: float, short_wavelength: float) -> dict[str, float]:
    """depth_from_slope() with the range given as wavelengths in metres."""
    return depth_from_slope(spec["k"], spec["log_power"], 2 * np.pi / long_wavelength, 2 * np.pi / short_wavelength)
