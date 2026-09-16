"""2D forward modelling against closed-form solutions.

A regular polygon with many sides stands in for a horizontal cylinder,
whose external field is exactly that of a line mass (gravity) or a line
dipole (magnetics). A thin rectangle stands in for a thin sheet. The
Poisson cross-check compares the magnetic kernel with derivatives of the
gravity kernel, which are independent code paths. Each test fails at the
stated RMS misfit relative to the analytic peak.

@author: Ben Kay (ben@auscope.org.au)
"""

import numpy as np
import pytest

from potfield import model2d
from potfield.synthetic import CM, G, MGAL, NT

X = np.linspace(-8000.0, 8000.0, 321)
ZC, R = 1500.0, 400.0
F_NT = 56700.0


def rel_rms(got, want):
    return float(np.sqrt(np.mean((got - want) ** 2)) / np.max(np.abs(want)))


def line_dipole_tmi(x, zc, mx, mz, fx, fz):
    """Total field of a line dipole (mx, mz per unit length) at depth zc, observation at z = 0."""
    rx, rz = x, -zc * np.ones_like(x)
    r2 = rx**2 + rz**2
    mr = (mx * rx + mz * rz) / r2
    bx = 2 * CM * (2 * mr * rx - mx) / r2
    bz = 2 * CM * (2 * mr * rz - mz) / r2
    return NT * (fx * bx + fz * bz)


def test_body_orientation_and_area():
    b = model2d.Body([0, 100, 100, 0], [10, 10, 60, 60])
    assert b.area == pytest.approx(5000.0)
    r = model2d.Body([0, 0, 100, 100], [10, 60, 60, 10])  # clockwise input, closing handled
    assert r.area == pytest.approx(5000.0)
    assert np.allclose(sorted(r.x), sorted(b.x))
    with pytest.raises(ValueError):
        model2d.Body([0, 100, 200], [10, 10, 10])  # zero area


def test_dyke_geometry():
    v = model2d.dyke(1000, 100, 50, 90, 400)
    assert v.area == pytest.approx(50 * 400)
    assert np.ptp(v.x) == pytest.approx(50)
    d = model2d.dyke(1000, 100, 50, 45, 400)
    assert d.area == pytest.approx(50 * 400)
    assert d.x.max() == pytest.approx(1000 + 25 + 400)  # dips toward +x
    e = model2d.dyke(1000, 100, 50, 135, 400)
    assert e.x.min() == pytest.approx(1000 - 25 - 400)  # dips toward -x


def test_cylinder_gravity_matches_line_mass():
    # Fails if the Green's theorem sum has the wrong sign, orientation or
    # scale: gz of a cylinder is 2 G rho A zc / (x^2 + zc^2).
    rho = 500.0
    body = model2d.circle(0.0, ZC, R, n=180, density=rho)
    got = model2d.gravity([body], X)
    want = MGAL * 2 * G * rho * body.area * ZC / (X**2 + ZC**2)
    assert rel_rms(got, want) < 0.005
    assert np.argmax(got) == np.argmin(np.abs(X))
    gx, _ = model2d.gravity_components([body], X)
    assert gx[X > 0].mean() < 0 < gx[X < 0].mean()  # attraction toward the body


def test_thin_sheet_gravity():
    # Thin vertical sheet, thickness t from z1 to z2: gz = G rho t ln((x^2+z2^2)/(x^2+z1^2)).
    rho, t, z1, z2 = 1000.0, 10.0, 200.0, 2200.0
    body = model2d.rectangle(-t / 2, t / 2, z1, z2, density=rho)
    got = model2d.gravity([body], X)
    want = MGAL * G * rho * t * np.log((X**2 + z2**2) / (X**2 + z1**2))
    assert rel_rms(got, want) < 0.005


@pytest.mark.parametrize("inc,dec,azimuth", [(90.0, 0.0, 0.0), (-60.0, 7.0, 0.0), (-60.0, 7.0, 45.0), (-60.0, 7.0, 90.0), (-30.0, 20.0, 160.0)])
def test_cylinder_tmi_matches_line_dipole(inc, dec, azimuth):
    # Fails if the pole-density kernel, the outward normal, the in-plane
    # projection of field and magnetisation, or the nT scaling is wrong.
    kappa = 0.02
    body = model2d.circle(0.0, ZC, R, n=180, susceptibility=kappa)
    got = model2d.magnetic([body], X, inc, dec, azimuth, F_NT)
    fx, fz = model2d.in_plane_direction(inc, dec, azimuth)
    m = model2d.induced_magnetisation(kappa, F_NT) * body.area
    want = line_dipole_tmi(X, ZC, m * fx, m * fz, fx, fz)
    assert np.max(np.abs(want)) > 0
    assert rel_rms(got, want) < 0.005


def test_cylinder_tmi_with_remanence():
    body = model2d.circle(0.0, ZC, R, n=180, susceptibility=0.01, magnetisation=0.8, mag_inc=20.0, mag_dec=-40.0)
    inc, dec, az = -62.0, 7.0, 30.0
    got = model2d.magnetic([body], X, inc, dec, az, F_NT)
    fx, fz = model2d.in_plane_direction(inc, dec, az)
    rx, rz = model2d.in_plane_direction(20.0, -40.0, az)
    mi = model2d.induced_magnetisation(0.01, F_NT)
    mx = (mi * fx + 0.8 * rx) * body.area
    mz = (mi * fz + 0.8 * rz) * body.area
    want = line_dipole_tmi(X, ZC, mx, mz, fx, fz)
    assert rel_rms(got, want) < 0.005


def test_poisson_cross_check_between_kernels():
    # For vertical magnetisation Mz: Bx = Cm Mz/(G rho) d(gz)/dx and
    # Bz = -Cm Mz/(G rho) d(gx)/dx. Gravity and magnetic kernels are
    # independent code, so agreement to 1% checks both.
    rho, kappa = 400.0, 0.03
    body = model2d.dyke(500.0, 300.0, 600.0, 60.0, 2500.0, density=rho, susceptibility=kappa)
    x = np.linspace(-6000.0, 7000.0, 1301)
    gx, gz = model2d.gravity_components([body], x)
    dgz_dx = np.gradient(gz, x)
    dgx_dx = np.gradient(gx, x)
    bx, bz = model2d.magnetic_components([body], x, 90.0, 0.0, 0.0, F_NT)
    mz = model2d.induced_magnetisation(kappa, F_NT)
    scale = CM * mz / (G * rho) * NT
    inner = slice(5, -5)
    assert rel_rms(bx[inner], (scale * dgz_dx)[inner]) < 0.01
    assert rel_rms(bz[inner], (-scale * dgx_dx)[inner]) < 0.01


def test_superposition_zero_and_far_field():
    a = model2d.rectangle(-500, 500, 200, 800, density=300, susceptibility=0.01)
    b = model2d.rectangle(2000, 2600, 400, 1400, density=-200, susceptibility=0.02)
    both = model2d.forward([a, b], X, inc=-62, dec=7, azimuth=0, field_nt=F_NT)
    sep = model2d.forward([a], X, inc=-62, dec=7, azimuth=0, field_nt=F_NT)
    sep2 = model2d.forward([b], X, inc=-62, dec=7, azimuth=0, field_nt=F_NT)
    for k in ("gravity", "tmi"):
        assert np.allclose(both[k], sep[k] + sep2[k])
    zero = model2d.forward([model2d.rectangle(-500, 500, 200, 800)], X, inc=-62, dec=7, azimuth=0, field_nt=F_NT)
    assert np.all(zero["gravity"] == 0) and np.all(zero["tmi"] == 0)
    # far field decays: anomaly at 8 km is a small fraction of the peak
    assert abs(both["gravity"][0]) < 0.1 * np.max(np.abs(both["gravity"]))


def test_observation_must_be_above_bodies():
    body = model2d.rectangle(-500, 500, 200, 800, density=300)
    with pytest.raises(ValueError):
        model2d.gravity([body], X, z_obs=250.0)
    # a sensor 80 m above ground (z = -80) is fine
    assert np.isfinite(model2d.gravity([body], X, z_obs=-80.0)).all()


def test_induced_magnetisation_value():
    assert model2d.induced_magnetisation(0.01, 56700.0) == pytest.approx(0.4512, rel=1e-3)


def test_fit_recovers_dyke_parameters():
    # Synthetic TMI from a known dyke plus a linear regional and noise.
    # Fails if the fit does not recover position, top, width and
    # susceptibility within 10% from a perturbed start.
    inc, dec, az = -62.0, 7.0, 30.0
    true = np.array([1200.0, 150.0, 300.0, 70.0, 0.02])  # x_centre, top, width, dip, kappa

    def build(p):
        return [model2d.dyke(p[0], p[1], p[2], p[3], 3000.0, susceptibility=p[4], name="dyke")]

    x = np.linspace(-5000.0, 7000.0, 241)
    clean = model2d.magnetic(build(true), x, inc, dec, az, F_NT)
    rng = np.random.default_rng(1)
    obs = clean + 20.0 + 0.004 * (x - x.mean()) + rng.normal(0, 0.01 * np.ptp(clean), x.size)
    p0 = np.array([800.0, 300.0, 500.0, 90.0, 0.01])
    lo = [-2000.0, 10.0, 20.0, 20.0, 0.0]
    hi = [4000.0, 2000.0, 3000.0, 160.0, 0.5]
    res = model2d.fit(x, obs, build, p0, field_name="tmi", bounds=(lo, hi), regional_order=1, inc=inc, dec=dec, azimuth=az, field_nt=F_NT)
    assert abs(res.params[0] - true[0]) < 60.0
    assert res.params[1] == pytest.approx(true[1], rel=0.10)
    assert res.params[2] == pytest.approx(true[2], rel=0.10)
    assert res.params[4] == pytest.approx(true[4], rel=0.10)
    assert res.rms < 0.03 * np.ptp(clean)
    assert np.all(res.params >= lo) and np.all(res.params <= hi)
    assert "rms" in res.summary(["x", "top", "width", "dip", "kappa"])


def test_fit_gravity_density_linear_case():
    body = model2d.rectangle(-800, 800, 300, 2300, density=250.0)
    x = np.linspace(-6000.0, 6000.0, 121)
    obs = model2d.gravity([body], x) + 3.0

    def build(p):
        return [model2d.rectangle(-800, 800, 300, 2300, density=p[0])]

    res = model2d.fit(x, obs, build, [50.0], field_name="gravity", regional_order=0)
    assert res.params[0] == pytest.approx(250.0, rel=1e-3)
    assert res.rms < 1e-6


def test_save_load_roundtrip(tmp_path):
    bodies = [model2d.dyke(100, 50, 30, 80, 500, susceptibility=0.05, name="d"), model2d.rectangle(0, 10, 20, 30, density=100)]
    p = model2d.save_bodies(bodies, tmp_path / "m.json")
    back = model2d.load_bodies(p)
    assert len(back) == 2 and back[0].name == "d" and back[0].susceptibility == 0.05
    assert np.allclose(back[1].x, bodies[1].x) and back[1].density == 100


def test_plot_model_runs():
    import matplotlib

    matplotlib.use("Agg")
    body = model2d.dyke(0, 100, 200, 80, 1000, density=200, susceptibility=0.02, name="dyke")
    x = np.linspace(-3000, 3000, 61)
    f = model2d.forward([body], x, inc=-62, dec=7, azimuth=0, field_nt=F_NT)
    fig, axes = model2d.plot_model(x, f, f, [body], regional_part={"tmi": np.zeros_like(x)})
    assert len(axes) == 3
