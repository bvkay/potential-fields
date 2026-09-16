"""Command-line filtering: read a grid, subset, reproject, filter, export.

Examples (run from the repo root):

  python scripts/filter_grid.py in.ers --bounds 140.0 -31.8 140.6 -31.3 --bounds-epsg 4283 \
      --epsg 28354 --method uc --height 1000 --out out/uc1000.grd --out out/uc1000.tif

  python scripts/filter_grid.py tmi.tif --method rtp --inc -63 --dec 7 --out out/rtp.asc

  python scripts/filter_grid.py tmi.tif --method bp --long 20000 --short 2000 --out out/bp.xyz

Output format follows the extension: .tif .grd (Surfer) .asc (Arc ASCII)
.ers .nc .xyz. --epsg applies to the whole run, so the output CRS matches
whatever database the file is going into.

@author: Ben Kay (ben@auscope.org.au)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from potfield import filters, read_grid, transforms, write_grid  # noqa: E402

METHODS = {
    "none": "no filter, just subset / reproject / convert",
    "uc": "upward continuation (--height m)",
    "dc": "downward continuation (--height m, --long cutoff wavelength m)",
    "residual": "input minus upward continuation (--height m)",
    "vd": "vertical derivative (--order)",
    "thdr": "total horizontal derivative",
    "as": "analytic signal amplitude",
    "tilt": "tilt angle (deg)",
    "lp": "Butterworth low pass (--long wavelength m, --order)",
    "hp": "Butterworth high pass (--short wavelength m, --order)",
    "bp": "Butterworth band pass (--long, --short m, --order)",
    "dcos": "directional cosine reject (--azimuth deg, --degree)",
    "detrend": "remove polynomial trend (--order 0-3)",
    "rtp": "reduction to pole (--inc, --dec)",
    "pseudograv": "pseudogravity mGal (--inc, --dec, --density, --magnetisation)",
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--bounds", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    p.add_argument("--bounds-epsg", type=int, help="CRS of --bounds if not the grid CRS")
    p.add_argument("--epsg", type=int, help="reproject to this CRS before filtering (default: MGA zone at centre)")
    p.add_argument("--no-reproject", action="store_true", help="keep a geographic grid as is (cos(lat) spacing)")
    p.add_argument("--method", choices=METHODS, default="none", help="; ".join(f"{k}: {v}" for k, v in METHODS.items()))
    p.add_argument("--height", type=float, default=1000.0)
    p.add_argument("--order", type=float, default=1)
    p.add_argument("--long", type=float, help="long wavelength m")
    p.add_argument("--short", type=float, help="short wavelength m")
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--degree", type=float, default=2.0)
    p.add_argument("--inc", type=float, help="field inclination deg (positive down)")
    p.add_argument("--dec", type=float, help="field declination deg")
    p.add_argument("--density", type=float, default=1000.0)
    p.add_argument("--magnetisation", type=float, default=1.0)
    p.add_argument("--units", default="", help="data units label, e.g. nT")
    p.add_argument("--out", action="append", required=True, help="output path; repeat for several formats")
    p.add_argument("--plot", action="store_true", help="show the result")
    a = p.parse_args(argv)

    g = read_grid(a.input, bounds=a.bounds, bounds_epsg=a.bounds_epsg, units=a.units)
    if not g.is_projected and not a.no_reproject:
        g = g.reproject(a.epsg)
    elif a.epsg is not None and g.epsg != a.epsg:
        g = g.reproject(a.epsg)
    print(g)

    m = a.method
    if m == "none":
        out = g
    elif m == "uc":
        out = filters.upward_continue(g, a.height)
    elif m == "dc":
        out = filters.downward_continue(g, a.height, a.long or 4 * g.dx, int(a.order) or 4)
    elif m == "residual":
        out = filters.regional_residual(g, a.height)[1]
    elif m == "vd":
        out = filters.vertical_derivative(g, a.order)
    elif m == "thdr":
        out = filters.total_horizontal_derivative(g)
    elif m == "as":
        out = filters.analytic_signal(g)
    elif m == "tilt":
        out = filters.tilt_angle(g)
    elif m == "lp":
        out = filters.butterworth_lowpass(g, a.long, int(a.order) if a.order > 1 else 4)
    elif m == "hp":
        out = filters.butterworth_highpass(g, a.short, int(a.order) if a.order > 1 else 4)
    elif m == "bp":
        out = filters.bandpass(g, a.long, a.short, int(a.order) if a.order > 1 else 4)
    elif m == "dcos":
        out = filters.directional_cosine(g, a.azimuth, a.degree)
    elif m == "detrend":
        out = filters.remove_trend(g, int(a.order))[0]
    elif m == "rtp":
        out = transforms.reduce_to_pole(g, a.inc, a.dec)
    elif m == "pseudograv":
        out = transforms.pseudogravity(g, a.inc, a.dec, a.density, a.magnetisation)
    else:
        raise SystemExit(f"unknown method {m}")

    for path in a.out:
        print("wrote", write_grid(out, path))

    if a.plot:
        from matplotlib import pyplot as plt

        from potfield import display

        display.plot(out)
        plt.show()


if __name__ == "__main__":
    main()
