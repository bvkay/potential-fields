# potential-fields

Frequency-domain filters and transforms for gridded gravity and magnetic
data, written for teaching. Python package `potfield`, notebooks that use it,
and a command-line script for batch work.

Port of a 2019 MATLAB App Designer tool. The filters are written out in full
(five to ten lines each) rather than imported from a geophysics library so
the equations can be read next to the code.

## Install

```bash
pip install -e .
```

Needs numpy, scipy, rasterio (GDAL), pyproj, shapely, matplotlib, xarray.
`pip install -e .[dev]` adds pytest and JupyterLab.

## Quick start

```python
from potfield import read_grid, write_grid, filters, transforms, display

# read a window of an ER Mapper grid in lat/lon and warp it to MGA zone 54
g = read_grid("magmap_v6_2015.ers", bounds=(140.0, -31.8, 140.6, -31.3), epsg=28354, units="nT")

uc = filters.upward_continue(g, 1000)          # metres
vd = filters.vertical_derivative(g, 1)
tilt = filters.tilt_angle(g)
rtp = transforms.reduce_to_pole(g, inclination=-63, declination=7)

write_grid(rtp, "rtp.grd")                      # Surfer, for Geotools
write_grid(rtp, "rtp.tif", epsg=28353)          # reprojected on the way out
display.plot(tilt, cmap="RdBu_r", shade={"azimuth": 45})
```

Any GDAL raster reads: ERS, GeoTIFF, Surfer GRD, Arc ASCII, netCDF. Cell
sizes for the filters come from the grid CRS in metres; a lat/lon grid warns
and uses a cos(lat) approximation, so reproject first.

## Modules

| module | contents |
|---|---|
| `grid` | `Grid` container, `read_grid`, subset by bounds or polygon, `reproject`, `regrid`, `profile` along a line |
| `export` | `write_grid` to .tif .grd .asc .ers .nc .xyz, optional target EPSG |
| `fft` | wavenumber grids, edge plane removal, padding and taper, `apply` |
| `filters` | continuation, derivatives, THDR, analytic signal, tilt, Butterworth and Gaussian band filters, directional cosine, trend removal |
| `transforms` | reduction to pole, pseudogravity, pseudomagnetic, moving-window Poisson analysis |
| `spectrum` | radially averaged and 2D power spectra, depth from slope |
| `display` | clipped colour scales, sunshade, AGC, profile plots, 2D spectrum image, polyline overlays |
| `synthetic` | analytic point mass and dipole fields used by the tests |

Every filter takes and returns a `Grid`. Filter responses are documented at
the top of `filters.py` and `transforms.py` with page references to Blakely
(1996).

## Sample data

`data/` holds ER Mapper grids for two areas, cut from Geoscience Australia
and Geological Survey of South Australia compilations. About 24 MB in total.

| file | area | source | cell | CRS |
|---|---|---|---|---|
| `frome_tmi.ers` | Frome Embayment, northern Flinders Ranges | GA magmap_v6_2015 TMI | 80 m | GDA94 lat/lon |
| `frome_grav.ers` | same | SA GRAVITY 2016 Bouguer anomaly | 200 m | GDA94 MGA zone 54 |
| `eyre_tmi.ers` | Eyre Peninsula | SA TMI 2011 compilation, averaged | 200 m | GDA94 MGA zone 53 |
| `eyre_grav.ers` | same | SA GRAVITY 2016 Bouguer anomaly | 400 m | GDA94 MGA zone 53 |

The Frome TMI is stored in lat/lon so the notebooks show the warp to
metres; the Eyre grids are stored projected and have null cells over the
sea. Gravity is in mGal. Note that the GA national gravity grids (2016 and
2019) are in um/s^2, ten times the mGal value. `scripts/make_sample_data.py`
rebuilds everything from the full grids.

## Notebooks

Run in order. They use the grids in `data/`.

| | |
|---|---|
| 00 | read, subset, reproject, export |
| 01 | upward and downward continuation, regional and residual |
| 02 | derivatives, THDR, analytic signal, tilt |
| 03 | band pass, directional cosine, trend removal |
| 04 | reduction to pole and pseudogravity |
| 05 | power spectrum and depth to source |
| 06 | combining gravity and magnetics with Poisson analysis |
| 07 | the workflow on a second area, Eyre Peninsula, from projected grids |

## Command line

```bash
python scripts/filter_grid.py in.ers --bounds 140.0 -31.8 140.6 -31.3 --bounds-epsg 4283 --epsg 28354 --method uc --height 1000 --out out/uc.grd --out out/uc.tif
```

`--method` list: `python scripts/filter_grid.py -h`.

## Export for Geotools and QGIS

Geotools needs the file CRS to match its database. Pass `epsg=` to
`write_grid` (or `--epsg` on the command line) and the grid is warped before
writing. Surfer `.grd` (binary, Surfer 6) and Arc ASCII `.asc` both load in
Geotools; GeoTIFF is the natural choice for QGIS. Surfer has no CRS field, so
a `.aux.xml` sidecar carries it.

## EPSG codes

Common Australian codes for `epsg=`. MGA zones are 6 degrees wide
(zone 53: 132 to 138 E, zone 54: 138 to 144 E). `Grid.mga_epsg()` returns
the zone under a grid's centre. Notebook 00 has the full list with a
pyproj check.

| CRS | EPSG |
|---|---|
| GDA94, GDA2020, WGS 84 geographic | 4283, 7844, 4326 |
| GDA94 / MGA zones 49 to 56 | 28349 to 28356 |
| GDA2020 / MGA zones 49 to 56 | 7849 to 7856 |
| AGD66 / AMG, AGD84 / AMG zones 49 to 56 | 20249 to 20256, 20349 to 20356 |
| GDA94 / Geoscience Australia Lambert, GDA2020 / GA LCC | 3112, 7845 |
| GDA94 / Australian Albers, GDA2020 / Australian Albers | 3577, 9473 |
| GDA94 / SA Lambert, GDA2020 / SA Lambert | 3107, 8059 |
| GDA94 / NSW Lambert, GDA2020 / NSW Lambert | 3308, 8058 |
| GDA94 / Vicgrid, GDA2020 / Vicgrid | 3111, 7899 |

## Tests

```bash
pytest
```

Filters are checked against analytic point-mass and dipole fields: upward
continuation, vertical and horizontal derivatives, analytic signal, tilt,
RTP at several declinations and with remanence, pseudogravity via Poisson's
relation, the Poisson regression slope, and power-spectrum depth. Each test
states what would make it fail.

## References

- Blakely, R.J. 1996. Potential Theory in Gravity and Magnetic Applications. Cambridge University Press.
- Baranov, V. 1957. A new method for interpretation of aeromagnetic maps: pseudo-gravimetric anomalies. Geophysics 22, 359-383.
- Chandler, V.W. and Malek, K.C. 1991. Moving-window Poisson analysis of gravity and magnetic data from the Penokean orogen, east-central Minnesota. Geophysics 56, 123-132.
- Miller, H.G. and Singh, V. 1994. Potential field tilt: a new concept for location of potential field sources. Journal of Applied Geophysics 32, 213-217.
- Roest, W.R., Verhoef, J. and Pilkington, M. 1992. Magnetic interpretation using the 3-D analytic signal. Geophysics 57, 116-125.
- Spector, A. and Grant, F.S. 1970. Statistical models for interpreting aeromagnetic data. Geophysics 35, 293-302.
