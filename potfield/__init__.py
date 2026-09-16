"""potfield: frequency-domain processing of gridded gravity and magnetic data.

Conventions used throughout:
- Grid rows run north to south, columns west to east (raster order).
- Null cells are NaN.
- Filter functions take cell sizes dx, dy in metres.
- Wavenumbers are angular (rad/m): k = 2*pi*f. kx is positive east, ky
  positive north.

@author: Ben Kay (ben@auscope.org.au)
"""

from . import filters, model2d, spectrum, synthetic, transforms
from .export import write_grid, write_netcdf, write_xyz
from .grid import Grid, Profile, read_ers, read_grid

__all__ = [
    "Grid",
    "Profile",
    "read_grid",
    "read_ers",
    "write_grid",
    "write_xyz",
    "write_netcdf",
    "filters",
    "transforms",
    "spectrum",
    "model2d",
    "synthetic",
]

__version__ = "0.1.0"
