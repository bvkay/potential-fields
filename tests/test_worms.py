"""Multiscale edge tests.

Synthetic RTP grids are built by tiling 2D profiles from model2d along
the grid's y axis, so the same geometry evaluated directly by model2d at
each height is an independent check on continuation, derivative and edge
detection together. Each test states what would make it fail.

@author: Ben Kay (ben@auscope.org.au)
"""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from potfield import Grid, model2d, worms

DX = 100.0
NX, NY = 240, 160
F_NT = 56700.0


def tiled_grid(profile_fn, name="syn"):
    """Grid whose value at column i is profile_fn(x_i), constant along rows (features strike north-south)."""
    t = from_origin(600_000.0, 6_500_000.0, DX, DX)
    g0 = Grid(np.zeros((NY, NX)), t, CRS.from_epsg(28354), name=name, units="nT")
    xrel = g0.x - g0.x.mean()
    return g0, xrel, g0.with_values(np.tile(profile_fn(xrel), (NY, 1)))


def test_nonmax_suppression_ridge():
    # A ridge along a column must be kept only at its crest.
    mag = np.zeros((20, 30))
    mag[:, 10] = 2.0
    mag[:, 9] = mag[:, 11] = 1.0
    gx = np.where(np.arange(30)[None, :] < 10, 1.0, -1.0) * np.ones((20, 1))
    gy = np.zeros_like(mag)
    edge = worms.nonmax_suppression(mag, gx, gy)
    assert edge[1:-1, 10].all()
    assert not edge[:, 9].any() and not edge[:, 11].any()


def test_vertical_contact_edges_stay_put_and_decay_as_contact():
    # RTP field of a vertical contact (very wide, very deep block) with its
    # west edge at x = 0. Every level must put a north-south string at the
    # contact within one cell, its strike within 5 degrees of north, and the
    # amplitude decay must return the depth to top within 15%.
    z_top = 400.0
    block = model2d.rectangle(0.0, 400e3, z_top, 400e3, susceptibility=0.02)
    g0, xrel, g = tiled_grid(lambda x: model2d.magnetic([block], x, 90.0, 0.0, 90.0, F_NT))
    heights = np.array([0.0, 200.0, 400.0, 800.0, 1600.0])
    w = worms.multiscale_edges(g, heights, percentile=50.0, min_length=20)
    x_contact = g0.x.mean()
    for i in range(len(heights)):
        m = w.at_level(i)
        assert m.any(), f"no edges at level {i}"
        # the strongest string at this level sits at the contact
        top = w.amplitude[m] > 0.8 * w.amplitude[m].max()
        assert np.all(np.abs(w.x[m][top] - x_contact) <= DX)
        assert np.all(np.minimum(w.strike[m][top], 180 - w.strike[m][top]) < 5.0)
    # amplitude at the contact against the direct model at each height
    for i, h in enumerate(heights):
        m = w.at_level(i) & (np.abs(w.x - x_contact) <= DX / 2)
        direct = model2d.magnetic([block], np.array([0.0, DX]), 90.0, 0.0, 90.0, F_NT, z_obs=-h)
        slope = abs(direct[1] - direct[0]) / DX
        assert np.median(w.amplitude[m]) == pytest.approx(slope, rel=0.10)
    worms.link_levels(w)
    ch = [c for c in worms.chains(w, min_levels=4) if abs(worms.chain_profile(w, c)["x"][0] - x_contact) <= DX]
    assert ch, "the contact string was not linked across levels"
    prof = worms.chain_profile(w, ch[0])
    est = worms.depth_from_decay(prof["height"], prof["amplitude"], n=1.0)
    assert est["depth"] == pytest.approx(z_top, rel=0.15)


def test_dipping_dyke_worms_follow_the_model_and_diverge():
    # A thin dyke dipping toward +x. At each height the strongest worm must
    # sit where model2d's own horizontal derivative at that height peaks,
    # within 1.5 cells: this checks continuation, derivative and edge
    # picking together. A dyke gives two flank worms whose separation grows
    # with height.
    dyke = model2d.dyke(0.0, 300.0, 200.0, 45.0, 3000.0, susceptibility=0.05)
    inc, dec, az = 90.0, 0.0, 90.0
    g0, xrel, g = tiled_grid(lambda x: model2d.magnetic([dyke], x, inc, dec, az, F_NT))
    heights = np.array([0.0, 300.0, 600.0, 1200.0])
    w = worms.multiscale_edges(g, heights, percentile=50.0, min_length=20)
    sep = []
    for i, h in enumerate(heights):
        fine = np.linspace(-3000.0, 4000.0, 1401)
        t = model2d.magnetic([dyke], fine, inc, dec, az, F_NT, z_obs=-h)
        thdr = np.abs(np.gradient(t, fine))
        x_peak = fine[np.argmax(thdr)] + g0.x.mean()
        m = w.at_level(i)
        strongest = w.x[m][w.amplitude[m] > 0.9 * w.amplitude[m].max()]
        assert np.abs(np.median(strongest) - x_peak) <= 1.5 * DX, f"level {i}"
        xs = np.unique(np.round(w.x[m]))
        sep.append(np.ptp(xs[np.abs(xs - g0.x.mean()) < 3000]))
    assert sep[-1] > sep[0]


@pytest.mark.parametrize("dip", [30.0, 45.0, 60.0])
def test_dipping_contact_migrates_down_dip_and_dip_is_recovered(dip):
    # Block east of x = 0 whose western contact dips toward +x. The worm
    # must migrate east with height and contact_dip must return the true
    # dip within 5 degrees from the bisector relation. Fails if the
    # migration direction, the linking or the relation is wrong.
    z_top, z_bot = 400.0, 400e3
    shift = (z_bot - z_top) / np.tan(np.radians(dip))
    block = model2d.Body([0.0, 400e3, 400e3 + shift, shift], [z_top, z_top, z_bot, z_bot], susceptibility=0.02)
    g0, xrel, g = tiled_grid(lambda x: model2d.magnetic([block], x, 90.0, 0.0, 90.0, F_NT))
    heights = np.array([0.0, 300.0, 600.0, 1200.0, 2400.0])
    w = worms.multiscale_edges(g, heights, percentile=50.0, min_length=20)
    worms.link_levels(w)
    ch = worms.chains(w, min_levels=5)
    assert ch
    best = max(ch, key=lambda c: np.median(w.amplitude[w.worm(c[0])]))
    p = worms.chain_profile(w, best)
    assert np.all(np.diff(p["x"]) >= 0) and p["x"][-1] > p["x"][0] + 2 * DX  # eastward, never backwards
    d = worms.contact_dip(w, best)
    assert 45.0 < d["azimuth"] < 135.0
    assert d["dip"] == pytest.approx(dip, abs=5.0)


def test_threshold_and_min_length_and_export(tmp_path):
    block = model2d.rectangle(0.0, 400e3, 400.0, 400e3, susceptibility=0.02)
    _, _, g = tiled_grid(lambda x: model2d.magnetic([block], x, 90.0, 0.0, 90.0, F_NT))
    rng = np.random.default_rng(0)
    noisy = g.with_values(g.values + rng.normal(0, 0.02 * np.ptp(g.values), g.shape))
    heights = np.array([0.0, 500.0])
    loose = worms.multiscale_edges(noisy, heights, percentile=20.0, min_length=1)
    strict = worms.multiscale_edges(noisy, heights, percentile=90.0, min_length=30)
    assert len(strict) < len(loose)
    ids, counts = np.unique(strict.worm_id, return_counts=True)
    assert counts.min() >= 30
    p = strict.to_csv(tmp_path / "w.csv")
    arr = np.loadtxt(p, delimiter=",", comments="#")
    assert arr.shape == (len(strict), 7)
    v = strict.to_vtk(tmp_path / "w.vtk")
    text = v.read_text()
    assert f"POINTS {len(strict)} float" in text and "SCALARS amplitude" in text
    df = strict.to_dataframe()
    assert list(df.columns) == ["x", "y", "height", "amplitude", "strike", "level", "worm_id"]


def test_flat_regions_give_no_edges():
    # Over an exactly constant part of the grid the gradient direction is
    # numerical noise and every pixel would pass the plateau test; the
    # amplitude floor must remove them. Fails if worms appear more than
    # 1 km into the flat region at any height.
    dyke = model2d.dyke(6000.0, 300.0, 200.0, 90.0, 3000.0, susceptibility=0.05)
    g0, xrel, g = tiled_grid(lambda x: np.where(x < -3000.0, 0.0, model2d.magnetic([dyke], x, 90.0, 0.0, 90.0, F_NT)))
    w = worms.multiscale_edges(g, [0.0, 300.0, 1000.0], percentile=30.0, min_length=5)
    flat = w.x < g0.x.mean() - 4000.0
    assert not flat.any()
    assert len(w) > 0


def test_depth_from_decay_exact_and_errors():
    h = np.array([0.0, 100.0, 300.0, 700.0, 1500.0])
    a = 5.0 / (h + 250.0)
    assert worms.depth_from_decay(h, a, 1.0)["depth"] == pytest.approx(250.0)
    a2 = 5.0 / (h + 250.0) ** 2
    assert worms.depth_from_decay(h, a2, 2.0)["depth"] == pytest.approx(250.0)
    with pytest.raises(ValueError):
        worms.depth_from_decay(h[:2], a[:2])
    with pytest.raises(ValueError):
        worms.multiscale_edges(Grid(np.zeros((10, 10)), from_origin(0, 1000, 100, 100), CRS.from_epsg(28354)), [-1.0])


def test_plots_run():
    import matplotlib

    matplotlib.use("Agg")
    block = model2d.rectangle(0.0, 400e3, 400.0, 400e3, susceptibility=0.02)
    _, _, g = tiled_grid(lambda x: model2d.magnetic([block], x, 90.0, 0.0, 90.0, F_NT))
    w = worms.multiscale_edges(g, [0.0, 500.0, 1000.0], percentile=50.0, min_length=20)
    worms.plot_map(w, grid=g)
    worms.plot_3d(w, stride=2)
