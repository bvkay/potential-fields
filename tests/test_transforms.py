"""RTP, pseudogravity, pseudomagnetic and Poisson analysis against analytic dipole
and point-mass fields. Comparisons are within 2*z0 of the source, where the
fields are well above edge noise. Each test fails at the stated RMS misfit
relative to the analytic peak.

@author: Ben Kay (ben@auscope.org.au)
"""

import numpy as np
import pytest

from potfield import transforms
from potfield.synthetic import CM, G, MGAL, dipole_grid, point_mass_grid

Z0 = 1500.0
DX = 100.0
N = 128
MOMENT = 1e6  # A m^2


def near_mask(shape, radius):
    jj, ii = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.hypot((jj - (shape[0] - 1) / 2) * DX, (ii - (shape[1] - 1) / 2) * DX) < radius


def rms_near(got, want, radius=2 * Z0):
    m = near_mask(got.shape, radius)
    return float(np.sqrt(np.nanmean((got[m] - want[m]) ** 2)) / np.nanmax(np.abs(want)))


@pytest.mark.parametrize("dec", [5.0, 40.0, -70.0])
def test_rtp_matches_pole_field(dec):
    # Fails if the reduced field differs from the analytic pole anomaly by
    # more than 3% RMS, or if its peak is not over the source. A wrong sign
    # on the declination term leaves a skewed anomaly and fails both.
    tmi = dipole_grid(N, N, DX, Z0, MOMENT, inc=-60.0, dec=dec)
    pole = dipole_grid(N, N, DX, Z0, MOMENT, inc=90.0, dec=0.0)
    rtp = transforms.reduce_to_pole(tmi, -60.0, dec)
    assert rms_near(rtp.values, pole.values) < 0.03
    j, i = np.unravel_index(np.nanargmax(rtp.values), rtp.shape)
    assert abs(j - 63.5) <= 1 and abs(i - 63.5) <= 1
    # input anomaly is skewed (peak displaced) so the test is not trivial
    j0, i0 = np.unravel_index(np.nanargmax(tmi.values), tmi.shape)
    assert np.hypot(j0 - 63.5, i0 - 63.5) > 2


def test_rtp_at_pole_is_identity():
    pole = dipole_grid(N, N, DX, Z0, MOMENT, inc=90.0, dec=0.0)
    rtp = transforms.reduce_to_pole(pole, 90.0, 0.0)
    assert rms_near(rtp.values, pole.values) < 0.005


def test_rtp_remanent_direction():
    tmi = dipole_grid(N, N, DX, Z0, MOMENT, inc=-60.0, dec=5.0, minc=20.0, mdec=-30.0)
    pole = dipole_grid(N, N, DX, Z0, MOMENT, inc=90.0, dec=0.0)
    rtp = transforms.reduce_to_pole(tmi, -60.0, 5.0, mag_inclination=20.0, mag_declination=-30.0)
    assert rms_near(rtp.values, pole.values) < 0.05


def test_rtp_amplitude_stabilisation_bounds_low_latitude():
    tmi = dipole_grid(N, N, DX, Z0, MOMENT, inc=-8.0, dec=0.0)
    raw = transforms.reduce_to_pole(tmi, -8.0, 0.0)
    stab = transforms.reduce_to_pole(tmi, -8.0, 0.0, amp_inclination=20.0)
    pole = dipole_grid(N, N, DX, Z0, MOMENT, inc=90.0, dec=0.0)
    assert np.nanmax(np.abs(stab.values)) < np.nanmax(np.abs(raw.values))
    assert np.nanmax(np.abs(stab.values)) < 3 * np.nanmax(np.abs(pole.values))


def test_pseudogravity_matches_point_mass():
    # Poisson: a dipole of moment m with magnetisation M is equivalent to a
    # point mass m*rho/M. Fails if the 1/|k| integration or the constants
    # are wrong by more than 3%.
    rho, M = 500.0, 1.0
    tmi = dipole_grid(N, N, DX, Z0, MOMENT, inc=-60.0, dec=5.0)
    pg = transforms.pseudogravity(tmi, -60.0, 5.0, density=rho, magnetisation=M)
    mass = MOMENT * rho / M
    _, want = point_mass_grid(N, N, DX, Z0, gm=G * mass * MGAL)
    assert pg.units == "mGal"
    assert rms_near(pg.values, want["gz"]) < 0.03
    assert np.nanmax(pg.values) == pytest.approx(want["gz"].max(), rel=0.03)


def test_pseudomagnetic_matches_dipole():
    rho, M = 500.0, 1.0
    mass = MOMENT * rho / M
    grav, _ = point_mass_grid(N, N, DX, Z0, gm=G * mass * MGAL)
    pm = transforms.pseudomagnetic(grav, -60.0, 5.0, density=rho, magnetisation=M)
    want = dipole_grid(N, N, DX, Z0, MOMENT, inc=-60.0, dec=5.0)
    assert pm.units == "nT"
    assert rms_near(pm.values, want.values) < 0.03


def test_pseudogravity_pseudomagnetic_roundtrip():
    tmi = dipole_grid(N, N, DX, Z0, MOMENT, inc=-60.0, dec=5.0)
    pg = transforms.pseudogravity(tmi, -60.0, 5.0)
    back = transforms.pseudomagnetic(pg, -60.0, 5.0)
    assert rms_near(back.values, tmi.values) < 0.02


def test_poisson_analysis_recovers_ratio_and_correlation():
    # Common source: point mass and the equivalent pole dipole. Expected
    # slope of RTP (nT) against dg/dz (mGal/m) is Cm M/(G rho) in those units.
    # Fails if the slope is off by more than 10% or r < 0.95 over the source,
    # or if a gravity-only source 4 km away still shows r > 0.7.
    rho, M = 500.0, 1.0
    mass = MOMENT * rho / M
    grav, _ = point_mass_grid(N, N, DX, Z0, gm=G * mass * MGAL)
    grav2, _ = point_mass_grid(N, N, DX, Z0, gm=G * mass * MGAL, offset=(4000.0, 0.0))
    grav = grav.with_values(grav.values + grav2.values)
    pole = dipole_grid(N, N, DX, Z0, MOMENT, inc=90.0, dec=0.0)
    out = transforms.poisson_analysis(grav, pole, window=15, magnetics_is_rtp=True)

    expected_slope = CM * M / (G * rho) * 1e-4**-1  # T per (m/s^2 / m) -> nT per (mGal/m)
    near = near_mask(grav.shape, Z0 / 2)
    assert np.nanmedian(out["slope"].values[near]) == pytest.approx(expected_slope, rel=0.10)
    assert np.nanmin(out["correlation"].values[near]) > 0.95
    assert np.nanmedian(out["m_over_rho"].values[near]) == pytest.approx(M / rho, rel=0.10)
    # gravity-only source: centre column shifted 40 cells east
    far = near_mask(grav.shape, Z0 / 3)
    far = np.roll(far, 40, axis=1)
    assert np.nanmedian(np.abs(out["correlation"].values[far])) < 0.7


def test_poisson_analysis_regrids_and_reduces():
    rho, M = 500.0, 1.0
    mass = MOMENT * rho / M
    grav, _ = point_mass_grid(N, N, DX, Z0, gm=G * mass * MGAL)
    tmi = dipole_grid(N, N, DX, Z0, MOMENT, inc=-60.0, dec=5.0)
    coarse = tmi.reproject(28354, res=200.0)  # different geometry, same CRS
    assert coarse.shape != grav.shape
    out = transforms.poisson_analysis(grav, coarse, inclination=-60.0, declination=5.0, window=15)
    assert out["correlation"].shape == grav.shape
    near = near_mask(grav.shape, Z0 / 2)
    assert np.nanmedian(out["correlation"].values[near]) > 0.9


def test_windowed_regression_exact_line():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 50))
    y = 2.5 * x - 1.0
    y[3:6, 3:6] = np.nan
    reg = transforms.windowed_regression(x, y, 7)
    ok = np.isfinite(reg["slope"])
    assert np.allclose(reg["slope"][ok], 2.5) and np.allclose(reg["intercept"][ok], -1.0)
    assert np.allclose(reg["correlation"][ok], 1.0)
    assert np.isnan(reg["slope"][4, 4])
