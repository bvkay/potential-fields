"""Radial spectrum depth estimate against a point mass at known depth.

The spectrum of point-mass gravity is proportional to exp(-2 k z0), so the
slope of ln P over any wavenumber band gives z0. The bands stop at a
wavelength of 20 cells: beyond that a point-mass spectrum has dropped more
than 20 orders of magnitude and the taper's leakage floor takes over, which
no real grid approaches. Fails if the recovered depth is more than 5% off in
the 60 to 20 cell band or 10% off in the 120 to 40 cell band.

@author: Ben Kay (ben@auscope.org.au)
"""

import numpy as np
import pytest

from potfield import spectrum
from potfield.synthetic import point_mass_grid

Z0 = 2000.0
DX = 100.0


def test_point_mass_depth_from_spectrum():
    g, _ = point_mass_grid(256, 256, DX, Z0)
    spec = spectrum.radial_power_spectrum(g)
    assert np.all(np.diff(spec["k"]) > 0)
    assert spec["wavelength"][0] > spec["wavelength"][-1]
    mid = spectrum.depth_from_wavelengths(spec, long_wavelength=60 * DX, short_wavelength=20 * DX)
    assert mid["depth"] == pytest.approx(Z0, rel=0.05)
    low = spectrum.depth_from_wavelengths(spec, long_wavelength=120 * DX, short_wavelength=40 * DX)
    assert low["depth"] == pytest.approx(Z0, rel=0.10)


def test_depth_scales_with_source_depth():
    d = []
    for z in (1000.0, 3000.0):
        g, _ = point_mass_grid(256, 256, DX, z)
        spec = spectrum.radial_power_spectrum(g)
        d.append(spectrum.depth_from_wavelengths(spec, 60 * DX, 20 * DX)["depth"])
    assert d[1] / d[0] == pytest.approx(3.0, rel=0.05)


def test_depth_from_slope_requires_points():
    with pytest.raises(ValueError):
        spectrum.depth_from_slope(np.array([1.0, 2.0]), np.array([0.0, -1.0]), 0.0, 10.0)
