"""Build the sample ERS grids in data/ from the full GA grids on Ben's machine.

Not needed to use the package; the outputs are committed. Area: Frome
Embayment and the northern Flinders Ranges around Arkaroola and Paralana,
South Australia.

Sources (Geoscience Australia, GDA94 lat/lon):
  magmap_v6_2015            TMI, 80 m           -> data/frome_tmi.ers
  2019_A4_CBA               Complete Bouguer, 400 m -> data/frome_cba.ers

@author: Ben Kay (ben@auscope.org.au)
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from potfield import read_grid, write_grid  # noqa: E402

SRC_TMI = Path("E:/Ben_Documents/DATA/Curnamona_Magnetics/magmap_v6_2015.ers")
SRC_CBA = Path("E:/Ben_Documents/DATA/Australian_Gravity_2019/2019_A4_CBA.ers")
OUT = Path(__file__).resolve().parents[1] / "data"

TMI_BOX = (139.20, -30.75, 139.95, -30.05)  # W S E N, degrees
CBA_BOX = (138.70, -31.50, 140.70, -29.50)


def main():
    OUT.mkdir(exist_ok=True)
    tmi = read_grid(SRC_TMI, bounds=TMI_BOX, name="frome_tmi", units="nT")
    print(tmi)
    write_grid(tmi, OUT / "frome_tmi.ers")
    cba = read_grid(SRC_CBA, bounds=CBA_BOX, name="frome_cba", units="mGal")
    print(cba)
    write_grid(cba, OUT / "frome_cba.ers")


if __name__ == "__main__":
    main()
