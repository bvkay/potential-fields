"""Filter tests against analytic point-mass fields.

Comparisons exclude a margin of 10% of the grid on each side. Each test
fails if the RMS misfit over the interior exceeds the stated fraction of the
analytic peak amplitude. Both an even and an odd grid size are run.
"""

import numpy as np
import pytest

from potfield import filters
from potfield.fft import wavenumbers
from potfield.synthetic import point_mass_grid

SHAPES = [(128, 128), (127, 131)]
Z0 = 1500.0
DX = 100.0


def interior(a: np.ndarray, frac: float = 0.1) -> np.ndarray:
    my, mx = int(a.shape[0] * frac), int(a.shape[1] * frac)
    return a[my:-my, mx:-mx]


def rms_rel(got: np.ndarray, want: np.ndarray) -> float:
    d = interior(got) - interior(want)
    return float(np.sqrt(np.nanmean(d**2)) / np.nanmax(np.abs(want)))


@pytest.mark.parametrize("shape", SHAPES)
def test_wavenumber_layout(shape):
    ny, nx = shape
    k, kx, ky = wavenumbers(shape, DX, DX)
    assert k[0, 0] == 0.0
    assert kx[0, 1] == pytest.approx(2 * np.pi / (nx * DX))
    assert ky[1, 0] == pytest.approx(-2 * np.pi / (ny * DX))  # row 1 is south of row 0
    assert k.max() <= np.pi / DX * np.sqrt(2) + 1e-12


@pytest.mark.parametrize("shape", SHAPES)
def test_upward_continuation(shape):
    ny, nx = shape
    g, _ = point_mass_grid(nx, ny, DX, Z0)
    h = 800.0
    _, want = point_mass_grid(nx, ny, DX, Z0, height=h)
    up = filters.upward_continue(g, h)
    assert rms_rel(up.values, want["gz"]) < 0.01


@pytest.mark.parametrize("shape", SHAPES)
def test_regional_residual_identity(shape):
    ny, nx = shape
    g, _ = point_mass_grid(nx, ny, DX, Z0)
    reg, res = filters.regional_residual(g, 500.0)
    assert np.allclose(reg.values + res.values, g.values)


@pytest.mark.parametrize("shape", SHAPES)
def test_first_vertical_derivative(shape):
    ny, nx = shape
    g, want = point_mass_grid(nx, ny, DX, Z0)
    vd = filters.vertical_derivative(g, 1)
    assert rms_rel(vd.values, want["vd1"]) < 0.01
    # z positive down: positive over the peak
    assert vd.values[ny // 2, nx // 2] > 0


@pytest.mark.parametrize("shape", SHAPES)
def test_second_vertical_derivative(shape):
    ny, nx = shape
    g, want = point_mass_grid(nx, ny, DX, Z0)
    vd = filters.vertical_derivative(g, 2)
    assert rms_rel(vd.values, want["vd2"]) < 0.02


def test_fractional_vertical_derivative_between_orders():
    g, want = point_mass_grid(128, 128, DX, Z0)
    v1 = filters.vertical_derivative(g, 1).values
    v15 = filters.vertical_derivative(g, 1.5).values
    v2 = filters.vertical_derivative(g, 2).values
    c = (64, 64)
    assert v1[c] < v15[c] < v2[c] or v1[c] > v15[c] > v2[c]


@pytest.mark.parametrize("shape", SHAPES)
def test_horizontal_derivatives_and_signs(shape):
    ny, nx = shape
    g, want = point_mass_grid(nx, ny, DX, Z0)
    gx, gy = filters.horizontal_derivatives(g)
    assert rms_rel(gx.values, want["dx"]) < 0.01
    assert rms_rel(gy.values, want["dy"]) < 0.01
    # east of the source gz decreases with x; north of it gz decreases with y
    assert gx.values[ny // 2, nx // 2 + 5] < 0
    assert gy.values[ny // 2 - 5, nx // 2] < 0


def test_regional_plane_on_anomaly():
    # anomaly + (a x + b y_north + c). Derivatives must be the analytic ones
    # plus a or b; continuation must return anomaly_up + the same plane.
    # Fails if the y sign convention is wrong or the plane is mishandled.
    g, want = point_mass_grid(128, 128, DX, Z0)
    peak = np.nanmax(g.values)
    a, b, c = 2e-4 * peak / DX, -1e-4 * peak / DX, 3 * peak
    xx, yy = np.meshgrid(g.x - g.x.mean(), g.y - g.y.mean())
    plane = a * xx + b * yy + c
    v = g.with_values(g.values + plane)
    gx, gy = filters.horizontal_derivatives(v)
    assert rms_rel(gx.values, want["dx"] + a) < 0.01
    assert rms_rel(gy.values, want["dy"] + b) < 0.01
    vd = filters.vertical_derivative(v)
    assert rms_rel(vd.values, want["vd1"]) < 0.01
    _, up_want = point_mass_grid(128, 128, DX, Z0, height=800.0)
    up = filters.upward_continue(v, 800.0)
    assert rms_rel(up.values, up_want["gz"] + plane) < 0.01
    hp = filters.butterworth_highpass(v, 20 * DX)
    assert abs(np.nanmean(interior(hp.values))) < 0.05 * peak


@pytest.mark.parametrize("shape", SHAPES)
def test_analytic_signal(shape):
    ny, nx = shape
    g, want = point_mass_grid(nx, ny, DX, Z0)
    a_want = np.sqrt(want["dx"] ** 2 + want["dy"] ** 2 + want["vd1"] ** 2)
    a = filters.analytic_signal(g)
    assert rms_rel(a.values, a_want) < 0.01
    # peak over the source
    j, i = np.unravel_index(np.nanargmax(a.values), a.shape)
    assert abs(j - ny // 2) <= 1 and abs(i - nx // 2) <= 1


def test_total_horizontal_derivative_ring():
    g, want = point_mass_grid(128, 128, DX, Z0)
    thdr = filters.total_horizontal_derivative(g)
    assert rms_rel(thdr.values, np.hypot(want["dx"], want["dy"])) < 0.01
    # Near zero over the source (the nearest cell centre is 71 m off it, where
    # the analytic value is 16% of the ring max) and maximum on a ring at
    # r = z0/2 for a point mass.
    assert thdr.values[64, 64] < 0.2 * np.nanmax(thdr.values)
    j, i = np.unravel_index(np.nanargmax(thdr.values), thdr.shape)
    r = np.hypot((j - 63.5) * DX, (i - 63.5) * DX)
    assert abs(r - Z0 / 2) < 1.5 * DX


def test_tilt_angle_range_and_peak():
    g, want = point_mass_grid(128, 128, DX, Z0)
    t = filters.tilt_angle(g)
    assert t.units == "deg"
    assert np.nanmax(t.values) > 85 and np.nanmax(t.values) <= 90
    assert np.nanmin(interior(t.values)) < 0
    # Tilt is a ratio of two derivatives, so far from the source, where both
    # are a fraction of a percent of their peaks, edge noise dominates it.
    # Compare within 2*z0 of the source. Fails if the RMS error there exceeds
    # 2 degrees (a sign or scale error gives tens of degrees).
    thdr_want = np.hypot(want["dx"], want["dy"])
    want_tilt = np.degrees(np.arctan2(want["vd1"], thdr_want))
    jj, ii = np.mgrid[0:128, 0:128]
    near = np.hypot((jj - 63.5) * DX, (ii - 63.5) * DX) < 2 * Z0
    err = (t.values - want_tilt)[near]
    assert np.sqrt(np.mean(err**2)) < 2.0


def test_lowpass_highpass_on_sinusoids():
    # Fails if a 40-cell wavelength is attenuated by a 10-cell low pass or a
    # 4-cell wavelength gets through it.
    g, _ = point_mass_grid(128, 128, DX, Z0)
    xx, _ = np.meshgrid(g.x, g.y)
    long = np.sin(2 * np.pi * xx / (40 * DX))
    short = np.sin(2 * np.pi * xx / (4 * DX))
    lp_long = filters.butterworth_lowpass(g.with_values(long), 10 * DX, order=8)
    lp_short = filters.butterworth_lowpass(g.with_values(short), 10 * DX, order=8)
    assert np.nanstd(interior(lp_long.values)) / np.std(interior(long)) > 0.95
    assert np.nanstd(interior(lp_short.values)) / np.std(interior(short)) < 0.05
    hp_short = filters.butterworth_highpass(g.with_values(short), 10 * DX, order=8)
    assert np.nanstd(interior(hp_short.values)) / np.std(interior(short)) > 0.95
    bp = filters.bandpass(g.with_values(long + short), 20 * DX, 8 * DX, order=8)
    assert np.nanstd(interior(bp.values)) / np.std(interior(long)) < 0.1


def test_gaussian_lowpass_halfpower_point():
    # At k = k0 the Gaussian response is exp(-1/2) = 0.607.
    g, _ = point_mass_grid(128, 128, DX, Z0)
    xx, _ = np.meshgrid(g.x, g.y)
    lam = 16 * DX
    s = np.sin(2 * np.pi * xx / lam)
    out = filters.gaussian_lowpass(g.with_values(s), lam)
    ratio = np.nanstd(interior(out.values)) / np.std(interior(s))
    assert abs(ratio - np.exp(-0.5)) < 0.03


def test_directional_cosine():
    # North-south stripes vary in x, so their energy sits at wavenumber
    # azimuth 90. Rejecting strike azimuth 0 must remove them; azimuth 90
    # must not. On an isotropic anomaly a narrow wedge (degree 8) must keep
    # more than a broad one (degree 1) and leave the peak in place.
    g, _ = point_mass_grid(128, 128, DX, Z0)
    xx, _ = np.meshgrid(g.x, g.y)
    stripes = g.with_values(np.sin(2 * np.pi * xx / (6 * DX)))
    clean = filters.directional_cosine(stripes, azimuth=0.0, degree=8.0)
    assert np.nanstd(interior(clean.values)) / np.std(interior(stripes.values)) < 0.05
    wrong = filters.directional_cosine(stripes, azimuth=90.0, degree=8.0)
    assert np.nanstd(interior(wrong.values)) / np.std(interior(stripes.values)) > 0.9

    narrow = filters.directional_cosine(g, azimuth=0.0, degree=8.0)
    broad = filters.directional_cosine(g, azimuth=0.0, degree=1.0)
    peak = np.nanmax(g.values)
    assert np.nanmax(narrow.values) > np.nanmax(broad.values)
    assert np.nanmax(narrow.values) > 0.6 * peak
    j, i = np.unravel_index(np.nanargmax(narrow.values), narrow.shape)
    assert abs(j - 64) <= 1 and abs(i - 64) <= 1
    # a constant background passes through unchanged
    shifted = filters.directional_cosine(g.with_values(g.values + 5 * peak), azimuth=0.0, degree=8.0)
    assert np.allclose(shifted.values - narrow.values, 5 * peak, rtol=1e-6)


def test_remove_trend_recovers_quadratic():
    g, _ = point_mass_grid(100, 90, DX, Z0)
    xx, yy = np.meshgrid(g.x - g.x.mean(), g.y - g.y.mean())
    trend = 5.0 + 1e-4 * xx - 2e-4 * yy + 3e-8 * xx**2 - 1e-8 * xx * yy
    v = g.values * 1e6 + trend
    v[10:20, 10:25] = np.nan
    resid, fit = filters.remove_trend(g.with_values(v), order=2)
    ok = np.isfinite(v)
    assert np.nanmax(np.abs(fit.values[ok] - trend[ok])) < 0.05 * np.ptp(trend)
    assert np.isnan(resid.values[12, 12])
    with pytest.raises(ValueError):
        filters.remove_trend(g, order=4)


def test_nan_hole_is_preserved_and_local():
    g, want = point_mass_grid(128, 128, DX, Z0)
    v = g.values.copy()
    v[20:30, 90:100] = np.nan
    up_hole = filters.upward_continue(g.with_values(v), 800.0)
    up_full = filters.upward_continue(g, 800.0)
    assert np.isnan(up_hole.values[25, 95])
    assert np.isfinite(up_hole.values[19, 89])
    # far from the hole the result must be unaffected
    far = (slice(50, 110), slice(10, 70))
    diff = np.abs(up_hole.values[far] - up_full.values[far])
    assert diff.max() < 0.01 * np.nanmax(up_full.values)


def test_geographic_grid_warns(geo_grid):
    with pytest.warns(UserWarning, match="geographic"):
        filters.upward_continue(geo_grid, 500.0)
