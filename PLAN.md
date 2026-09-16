# potfield: MATLAB to Python port of the Potential Fields app

Status: approved 2026-09-16, implementation in progress. Git: commits under
Ben's name only, one-line messages.

## 1. What exists

Source: `E:\Ben_Documents\TO_SORT_2024\MATLAB\Potential_Fields`

- `Potential_Field_App_V5.mlapp` (2019 App Designer GUI). Implemented filters:
  upward continuation, UC residual, analytic signal amplitude, nth vertical
  derivative. Tilt and horizontal derivative buttons exist but were never
  coded. N/E/S/W bounds fields exist but are never read. Export: GeoTIFF only.
  Input: two hard-coded `.mat` files (Curnamona VRTP mag at 80 m, Bouguer
  gravity at 800 m), both GDA94 lat/lon.
- `Other/` is Peter Kovesi's MATLAB toolbox (MIT licence): `wavenumbergrid`,
  `freqderiv`, `vertderivativeintegral`, `upwardcontinue`, `analyticsignal`,
  `removetrend`, `agc`, `ppdrc`, `dealias`, `orientationfilter`. Reference
  implementation, not something to port wholesale.
- `Domains/*.txt`: lon lat polylines of Curnamona geological domains, used
  as map overlays.

Data: `E:\Ben_Documents\DATA`, 1543 `.ers` grids. All sampled headers are
IEEE4ByteReal, LSBFirst, single band. Two coordinate cases:

| CoordinateType | Example | Cell |
|---|---|---|
| LATLONG / LL, GDA94 (most GA grids) | `Curnamona_Magnetics/magmap_v6_2015.ers` | 0.000833 deg (~80 m) |
| EN, EPSG:3107 (SA Lambert), metres | `SA/.../SA_TMI_VRTP_UC1000.ers` | 80 m |

Null values are `-99999` or `-999999`. Read from header, never hard-code.
Other grid formats already in the DATA tree: GeoTIFF (420), XYZ (216), Surfer
GRD binary DSBB (151), Arc ASCII (37), netCDF (7). No Geosoft GRD seen.

Environment: Python 3.13, numpy 2.3, scipy 1.16, matplotlib, rasterio 1.4
(GDAL 3.9 with ERS, GTiff, GSBG/Surfer, netCDF, GXF drivers), xarray, pyproj,
geopandas, shapely, jupyterlab, pytest. Not installed: harmonica, verde.

## 2. Equation review of the MATLAB app

Filter shapes are right. Scaling and grid geometry are not.

1. Upward continuation: `dd = cell_size*(lon(2)-lon(1))` multiplies metres by
   degrees. Result is ~0.067 and is treated as km. Continuation height is
   off by a factor of ~0.84 on the 80 m grid and a different factor on the
   800 m grid. Also `[kx fliplr(-kx)]` gives a vector of length N+1 for odd N
   and fails.
2. Analytic signal: `kx = 2*pi*u1/Nx` divides normalised frequency by the
   number of columns instead of the cell size. Should be `2*pi*u1/dx`.
   Amplitude units are meaningless and x and y are scaled differently when
   Nx != Ny.
3. Vertical derivative: `(2*pi*radius)^n` with radius in cycles per cell.
   Output is nT per cell, not nT per metre. Fine for a picture, wrong for a
   number.
4. All three: grid is in degrees but treated as square. At -31 deg one degree
   of longitude is 95 km and one degree of latitude is 111 km, so every
   "isotropic" filter is 14% anisotropic.
5. No NaN handling, no edge padding or taper, no detrend before FFT. Nulls
   propagate to the whole grid; edges ring.
6. UC residual: `data - UC(data)` is correct.

Kovesi's versions in `Other/` handle 2, 3 and 5 correctly (dimensional
wavenumbers, NaN fill and mask, optional taper). They still assume the grid is
in length units, so item 4 stands.

## 3. Proposed structure

Package holds the maths. Notebooks stay thin and import it. Scripts cover the
batch case. Students read one module per method.

```
potfield/                     repo root
  potfield/
    __init__.py
    grid.py        Grid dataclass: values (float64, NaN nulls), x, y, dx, dy, crs
                   read_ers(path, bounds=None, polygon=None), subset, reproject
    fft.py         wavenumber grids, pad/taper, NaN fill, fft2/ifft2 wrapper
    filters.py     continuation, derivatives, THDR, analytic signal, tilt,
                   Butterworth/Gaussian band-pass, directional cosine, trend removal
    transforms.py  RTP, pseudogravity, pseudomagnetic (Poisson), Poisson window correlation
    spectrum.py    radially averaged power spectrum, depth-to-source slopes
    display.py     percentile / std-dev clip, sunshade, AGC, domain overlays
    export.py      GeoTIFF, ERS, Surfer GRD, netCDF, XYZ
  notebooks/
    00_read_subset_export.ipynb
    01_continuation.ipynb
    02_derivatives_thdr_as_tilt.ipynb
    03_bandpass_regional_residual.ipynb
    04_rtp_and_pseudogravity.ipynb
    05_power_spectrum_depth.ipynb
    06_gravity_magnetics_poisson.ipynb
  scripts/
    filter_grid.py   CLI: in.ers --bounds W S E N --method uc --height 1000 --out x.tif
  tests/
    test_filters.py  analytic checks (section 6)
  data/              small sample subset for the notebooks (a few MB)
  PLAN.md
  README.md
  pyproject.toml
```

Dependencies: numpy, scipy, rasterio, pyproj, matplotlib, xarray. geopandas
only for polygon clipping. No harmonica: the filters are five to ten lines
each and writing them out is the teaching content. harmonica can be a dev-only
cross-check in tests.

## 4. Grid geometry

Every filter takes `dx, dy` in metres. Two ways to get there from a lat/lon
grid:

- `Grid.reproject(epsg)` via `rasterio.warp.reproject` onto square cells in
  the MGA zone of the grid centre (default), or a user-supplied EPSG.
- `Grid.local_spacing()` approximation: dx = dlon * 111320 * cos(lat_mid),
  dy = dlat * 111320. Cheap, adequate for grids under ~3 deg extent.

Decision: reproject by default. `epsg` is a parameter on read, reproject and
every export, because Geotools requires the file CRS to match its database
CRS (for example MGA zone 53, EPSG:28353 for GDA94 or 7853 for GDA2020).
The cos(lat) approximation stays as an explicit opt-in for quick looks.

## 5. Methods

Tier 1 (port and fix):
- Upward and downward continuation. Downward needs a stabilising low-pass;
  expose the cutoff wavelength.
- nth-order vertical derivative, fractional order allowed.
- Horizontal derivatives dx, dy; total horizontal derivative THDR.
- Analytic signal amplitude sqrt(dx^2 + dy^2 + dz^2).
- Tilt angle atan(dz / THDR). Tilt of THDR optional.
- Regional / residual: UC residual (existing), Gaussian and Butterworth
  low/high/band-pass by wavelength.
- Polynomial trend removal, order 0 to 3.
- Bounds extraction (W S E N) and polygon clip on read.

Tier 2 (new, gravity + magnetics):
- RTP with user-supplied inclination and declination. Include the standard
  amplitude clamp for low inclinations even though SA sits at I ~ -65 deg.
  Optional IGRF lookup via `ppigrf` if wanted.
- Pseudogravity (Baranov 1957, Poisson's relation): RTP then vertical
  integration, scaled by G rho / (Cm M). Puts magnetics in gravity units so the
  two grids compare directly.
- Pseudomagnetic: the inverse, gravity to equivalent RTP mag.
- Moving-window Poisson analysis (Chandler and Malek 1991): regress the
  vertical derivative of pseudogravity against the vertical derivative of
  gravity in a sliding window. Outputs a slope grid (density to susceptibility
  ratio) and a correlation grid. This is the standard product for combining
  the two fields and maps where sources are common to both.
- Radially averaged power spectrum with fitted linear segments for depth to
  source ensemble (Spector and Grant 1970). Cheap and good for teaching.

Tier 3 (display, from Kovesi):
- Std-dev and percentile clipping (existing), sunshading, AGC.
- Skip PPDRC, monogenic and orientation filters unless asked.

Not planned: Euler deconvolution, inversion, terrain corrections. Different
class of tool and a scope-creep risk. Revisit after Tiers 1 and 2 are solid.

Decision: Tier 2 in full.

## 6. Tests (each must be able to fail)

- Continuation: field of a point mass or dipole at z0, continue by h, compare
  against the analytic field at z0 + h. Fails if interior RMS misfit exceeds
  1% of anomaly amplitude.
- Vertical derivative: same source, compare with analytic dz. Same criterion.
- RTP: dipole grid at I = -65, D = 5, apply RTP, result must be symmetric
  about the source to within 2% and peak over it.
- Pseudogravity: dipole with known M and rho ratio, compare with analytic
  gravity of the equivalent sphere.
- Round trip: read ERS, write ERS and GeoTIFF, read back. Values and
  geotransform identical to float32 precision. Fails on any cell mismatch.
- Odd and even grid sizes for every wavenumber test.

## 7. Export

Targets: QGIS and Geotools. Geotools takes Surfer GRD and ASCII grids and
needs the CRS to match its database, so `write_*` functions take an optional
`epsg` and reproject before writing.

Formats, all via GDAL through rasterio unless noted: GeoTIFF (QGIS), Surfer
GRD binary GSBG (Geotools), Arc ASCII (Geotools, QGIS), ERS (round trip to
the source format), netCDF, XYZ via numpy with a header line. No Geosoft GRD.

## 8. Writing style for notes and comments

Technical, short. State the equation, cite the page (Blakely 1996), name the
units. No filler.

## 9. Order of work

1. `grid.py` + `export.py` + notebook 00. Read, subset, reproject, write.
   Round-trip tests pass.
2. `fft.py` + `filters.py` Tier 1 + analytic tests + notebooks 01 to 03.
3. `transforms.py` + `spectrum.py` + notebooks 04 to 06.
4. `display.py`, CLI script, README.

## References

- Blakely, R.J. 1996. Potential Theory in Gravity and Magnetic Applications.
  CUP. Ch. 12, pp 313-346.
- Baranov, V. 1957. A new method for interpretation of aeromagnetic maps:
  pseudo-gravimetric anomalies. Geophysics 22, 359-383.
- Chandler, V.W. and Malek, K.C. 1991. Moving-window Poisson analysis of
  gravity and magnetic data from the Penokean orogen. Geophysics 56, 123-132.
- Miller, H.G. and Singh, V. 1994. Potential field tilt. J. Appl. Geophys. 32, 213-217.
- Spector, A. and Grant, F.S. 1970. Statistical models for interpreting
  aeromagnetic data. Geophysics 35, 293-302.
- Kovesi, P. MATLAB functions for geophysical image processing, peterkovesi.com.
