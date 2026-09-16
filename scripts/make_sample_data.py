"""Build the small sample grids in data/ from the full GA grids on Ben's machine.

Not needed to use the package; the outputs are committed. Sources:
  GA magmap_v6_2015 TMI (80 m, GDA94 lat/lon)
  GA onshore Complete Bouguer 2016 (~800 m, GDA94 lat/lon)
Both are cut to part of the Curnamona Province and written as GeoTIFF.
"""

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from potfield import read_grid, write_grid  # noqa: E402

SRC_TMI = Path("E:/Ben_Documents/DATA/Curnamona_Magnetics/magmap_v6_2015.ers")
SRC_GRAV = Path("E:/Ben_Documents/DATA/Australia_Gravity/onshore_geodetic_Complete_Bouguer_2016.ers")
SRC_DOMAINS = Path("E:/Ben_Documents/TO_SORT_2024/MATLAB/Potential_Fields/Domains")
OUT = Path(__file__).resolve().parents[1] / "data"

TMI_BOX = (139.7, -31.9, 140.5, -31.3)  # W S E N, degrees
GRAV_BOX = (139.0, -33.0, 142.0, -29.5)


def main():
    OUT.mkdir(exist_ok=True)
    tmi = read_grid(SRC_TMI, bounds=TMI_BOX, name="curnamona_tmi", units="nT")
    print(tmi)
    write_grid(tmi, OUT / "curnamona_tmi.tif")
    grav = read_grid(SRC_GRAV, bounds=GRAV_BOX, name="curnamona_bouguer", units="mGal")
    print(grav)
    write_grid(grav, OUT / "curnamona_bouguer.tif")
    dom = OUT / "domains"
    dom.mkdir(exist_ok=True)
    for f in sorted(SRC_DOMAINS.glob("*.txt")):
        if not f.name.startswith("._"):
            shutil.copy(f, dom / f.name)
    print("domains:", len(list(dom.glob("*.txt"))))


if __name__ == "__main__":
    main()
