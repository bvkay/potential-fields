"""Build the sample ERS grids in data/ from the full state and national grids on Ben's machine.

Not needed to use the package; the outputs are committed. Two areas:

Frome Embayment and northern Flinders Ranges (Arkaroola, Paralana, Beverley,
Lake Frome, Mount Painter and Mount Babbage inliers), GDA94:
  frome_tmi.ers   GA magmap_v6_2015 TMI, 80 m, lat/lon as supplied      ~11 MB
  frome_grav.ers  SA GRAVITY 2016 Bouguer anomaly, 200 m, MGA zone 54    ~2 MB

Eyre Peninsula (Port Lincoln to Kimba, Streaky Bay to Whyalla), GDA94:
  eyre_tmi.ers    SA TMI 2011 compilation, averaged to 200 m, MGA zone 53  ~9 MB
  eyre_grav.ers   SA GRAVITY 2016 Bouguer anomaly, 400 m, MGA zone 53      ~2 MB

The Eyre grids are stored projected so the notebooks show both cases: a
lat/lon grid that has to be warped and a projected grid ready to filter.
Ocean cells in the Eyre TMI are null.

Units: SA GRAVITY 2016 is in mGal (its metadata says so). The GA national
2016 and 2019 gravity grids are in um/s^2, ten times the mGal value; do not
label them mGal.

@author: Ben Kay (ben@auscope.org.au)
"""

from pathlib import Path
import sys

import numpy as np
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from potfield import read_grid, write_grid  # noqa: E402

D = Path("E:/Ben_Documents/DATA")
SRC_CURNAMONA_TMI = D / "Curnamona_Magnetics/magmap_v6_2015.ers"
SRC_SA_TMI = D / "SA/Geophysics_State_Images/Pre-2020/SA_TMI_ERS/SA_TMI.ers"
SRC_SA_GRAV = D / "SA/Geophysics_State_Images/2020/SA_GRAV_ERS/SA_GRAV_200M.ers"
SRC_NAT_CBA = D / "Australian_Gravity_2019/2019_A4_CBA.ers"
OUT = Path(__file__).resolve().parents[1] / "data"

FROME = (139.00, -31.20, 140.40, -29.80)  # W S E N, degrees
EYRE = (134.50, -34.95, 137.60, -32.30)


def size_mb(path: Path) -> float:
    return path.with_suffix("").stat().st_size / 1e6


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("frome_cba*"):
        old.unlink()

    tmi = read_grid(SRC_CURNAMONA_TMI, bounds=FROME, name="frome_tmi", units="nT")
    p = write_grid(tmi, OUT / "frome_tmi.ers")
    print(tmi, f"{size_mb(p):.1f} MB")

    grav = read_grid(SRC_SA_GRAV, bounds=FROME, bounds_epsg=4283, name="frome_grav", units="mGal")
    grav = grav.reproject(28354, res=200.0, resampling=Resampling.average)
    p = write_grid(grav, OUT / "frome_grav.ers")
    print(grav, f"{size_mb(p):.1f} MB")

    eyre_tmi = read_grid(SRC_SA_TMI, bounds=EYRE, name="eyre_tmi", units="nT")
    eyre_tmi = eyre_tmi.reproject(28353, res=200.0, resampling=Resampling.average)
    p = write_grid(eyre_tmi, OUT / "eyre_tmi.ers")
    print(eyre_tmi, f"{size_mb(p):.1f} MB")

    eyre_grav = read_grid(SRC_SA_GRAV, bounds=EYRE, bounds_epsg=4283, name="eyre_grav", units="mGal")
    eyre_grav = eyre_grav.reproject(28353, res=400.0, resampling=Resampling.average)
    p = write_grid(eyre_grav, OUT / "eyre_grav.ers")
    print(eyre_grav, f"{size_mb(p):.1f} MB")

    # units cross-check: SA (mGal) against the GA national grid (um/s^2) should give slope ~0.1
    nat = read_grid(SRC_NAT_CBA, bounds=FROME, name="nat")
    sa_on_nat = grav.regrid(nat)
    ok = np.isfinite(sa_on_nat.values) & np.isfinite(nat.values)
    slope = np.polyfit(nat.values[ok], sa_on_nat.values[ok], 1)[0]
    print(f"SA gravity vs GA national: slope {slope:.3f} (0.1 means national is in um/s^2)")


if __name__ == "__main__":
    main()
