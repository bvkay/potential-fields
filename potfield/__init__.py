"""potfield: frequency-domain processing of gridded gravity and magnetic data.

Conventions used throughout:
- Grid rows run north to south, columns west to east (raster order).
- Null cells are NaN.
- Filter functions take cell sizes dx, dy in metres.
- Wavenumbers are angular (rad/m): k = 2*pi*f. kx is positive east, ky
  positive north.
"""

from .grid import Grid, read_grid, read_ers
from .export import write_grid, write_xyz, write_netcdf

__all__ = [
    "Grid",
    "read_grid",
    "read_ers",
    "write_grid",
    "write_xyz",
    "write_netcdf",
]

__version__ = "0.1.0"
