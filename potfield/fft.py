"""FFT plumbing shared by all frequency-domain filters.

Pipeline in apply():
1. Fill NaN cells with the nearest finite value and remember the mask.
2. Fit a plane a*x + b*y + c to the boundary band of the grid and remove it.
   The boundary is the best estimate of the regional level and gradient the
   anomalies sit on; removing it makes the edges near zero so the taper does
   not pull the padded region toward the wrong level.
3. Pad by half the grid size on every side, replicating the edge value and
   tapering it to zero with a cosine so the periodic wrap is continuous.
   (Mirror padding was tested and is worse: the reflected anomaly re-enters
   the window as a fake source.)
4. fft2, multiply by the filter response H(k, kx, ky), ifft2, take real part.
5. Trim the padding, restore the plane as the filter would have transformed
   it (see `plane` below), restore NaNs.

Wavenumber conventions: k = 2*pi*f in rad/m. kx is positive east (along
columns). Rows run north to south, so ky = -2*pi*fy makes ky positive north.

@author: Ben Kay (ben@auscope.org.au)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import distance_transform_edt

Response = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

PLANE_MODES = ("keep", "drop", "dx", "dy")


def wavenumbers(shape: tuple[int, int], dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Angular wavenumber grids (k, kx, ky) in rad/m for an array of `shape`.

    Zero frequency sits at index [0, 0] (unshifted FFT layout). Odd and even
    sizes are handled by fftfreq.
    """
    ny, nx = shape
    kx = 2 * np.pi * sfft.fftfreq(nx, d=dx)[None, :]
    ky = -2 * np.pi * sfft.fftfreq(ny, d=dy)[:, None]  # rows go south, so flip sign
    kx, ky = np.broadcast_arrays(kx, ky)
    k = np.hypot(kx, ky)
    return k, kx, ky


def fill_nan(values: np.ndarray) -> np.ndarray:
    """Replace NaN with the nearest finite value (Euclidean distance)."""
    mask = np.isnan(values)
    if not mask.any():
        return values.copy()
    if mask.all():
        raise ValueError("grid has no finite values")
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return values[tuple(idx)]


def local_coords(shape: tuple[int, int], dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre x (east) and y (north) in metres relative to the grid centre."""
    ny, nx = shape
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = ((ny - 1) / 2 - np.arange(ny)) * dy
    return np.meshgrid(x, y)


def fit_plane(values: np.ndarray, dx: float, dy: float, band: int | None = None) -> tuple[float, float, float]:
    """Least-squares plane a*x + b*y + c through the outer `band` cells.

    band defaults to 10% of the shorter side, at least 2 cells. NaN cells are
    ignored. Coordinates are those of local_coords().
    """
    ny, nx = values.shape
    if band is None:
        band = max(2, int(0.1 * min(ny, nx)))
    sel = np.zeros(values.shape, dtype=bool)
    sel[:band, :] = sel[-band:, :] = True
    sel[:, :band] = sel[:, -band:] = True
    sel &= np.isfinite(values)
    if sel.sum() < 3:
        sel = np.isfinite(values)
    xx, yy = local_coords(values.shape, dx, dy)
    A = np.column_stack([xx[sel], yy[sel], np.ones(sel.sum())])
    coef, *_ = np.linalg.lstsq(A, values[sel], rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def default_pad(shape: tuple[int, int]) -> int:
    """Half of the larger dimension, at least 16 cells."""
    return max(16, max(shape) // 2)


def pad_taper(values: np.ndarray, width: int) -> np.ndarray:
    """Pad by `width` cells on every side with the edge value, then
    cosine-taper the padded band from 1 at the data edge to 0 at the outer
    edge. Input should be near zero at its edges (plane removed) so the taper
    lands on the regional level."""
    ny, nx = values.shape
    width = int(width)
    if width <= 0:
        return values.copy()
    padded = np.pad(values, width, mode="edge")
    ramp = np.cos(np.linspace(0, np.pi / 2, width + 1)[1:])  # width values, ~1 -> 0

    def window(n: int) -> np.ndarray:
        w = np.ones(n + 2 * width)
        w[:width] = ramp[::-1]
        w[n + width :] = ramp
        return w

    return padded * np.outer(window(ny), window(nx))


class Prepared:
    """A grid taken through steps 1 to 3 of the pipeline with its FFT stored,
    so several filters can be applied without repeating the forward
    transform. Created by prepare(); consumed by finish()."""

    def __init__(self, values: np.ndarray, dx: float, dy: float, pad: int | None):
        values = np.asarray(values, dtype=np.float64)
        self.shape = values.shape
        self.mask = np.isnan(values)
        v = fill_nan(values)
        self.plane = fit_plane(v, dx, dy)
        self.xx, self.yy = local_coords(v.shape, dx, dy)
        a, b, c = self.plane
        v = v - (a * self.xx + b * self.yy + c)
        self.width = default_pad(v.shape) if pad is None else max(0, int(pad))
        p = pad_taper(v, self.width)
        fast = (sfft.next_fast_len(p.shape[0]), sfft.next_fast_len(p.shape[1]))
        p = np.pad(p, ((0, fast[0] - p.shape[0]), (0, fast[1] - p.shape[1])))
        self.F = sfft.fft2(p, workers=-1)
        self.k, self.kx, self.ky = wavenumbers(p.shape, dx, dy)

    def response(self, fn: Response) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            H = np.asarray(fn(self.k, self.kx, self.ky))
        return np.where(np.isfinite(H), H, 0.0)


def prepare(values: np.ndarray, dx: float, dy: float, pad: int | None = None) -> Prepared:
    return Prepared(values, dx, dy, pad)


def finish(prep: Prepared, H: np.ndarray, plane: str = "keep") -> np.ndarray:
    """Steps 4 and 5 for one response H: inverse FFT, trim, restore the plane and the NaNs."""
    if plane not in PLANE_MODES:
        raise ValueError(f"plane must be one of {PLANE_MODES}")
    ny, nx = prep.shape
    w = prep.width
    out = np.real(sfft.ifft2(prep.F * H, workers=-1))[w : w + ny, w : w + nx]
    a, b, c = prep.plane
    if plane == "keep":
        out = out + (a * prep.xx + b * prep.yy + c)
    elif plane == "dx":
        out = out + a
    elif plane == "dy":
        out = out + b
    out[prep.mask] = np.nan
    return out


def apply(
    values: np.ndarray,
    dx: float,
    dy: float,
    response: Response,
    pad: int | None = None,
    plane: str = "keep",
) -> np.ndarray:
    """Filter a 2D array in the wavenumber domain.

    response(k, kx, ky) returns the filter H on the padded wavenumber grid.

    plane says what the filter does to the removed boundary plane
    a*x + b*y + c, which the FFT cannot represent:
      "keep"  add it back unchanged (continuation, low pass, directional)
      "drop"  leave it out (high pass, vertical derivatives)
      "dx"    add back its x-gradient a (first horizontal derivative in x)
      "dy"    add back its y-gradient b
    Returns an array the same shape as values with NaNs where the input had
    them.
    """
    prep = prepare(values, dx, dy, pad)
    return finish(prep, prep.response(response), plane)


def apply_many(
    values: np.ndarray,
    dx: float,
    dy: float,
    responses: list[Response],
    planes: list[str],
    pad: int | None = None,
) -> list[np.ndarray]:
    """apply() for several responses sharing one forward FFT."""
    prep = prepare(values, dx, dy, pad)
    return [finish(prep, prep.response(r), p) for r, p in zip(responses, planes)]


def power_spectrum(
    values: np.ndarray, dx: float, dy: float, pad: int | None = None, window: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided power spectrum |F|^2 and matching |k| grid.

    Same NaN fill and boundary-plane removal as apply(), then a 2D Tukey
    window (flat centre, cosine taper over `window` of each side) and zero
    padding by `pad` cells. For spectral estimates a window is the right
    tool: the straight grid edges otherwise leak energy into a bright cross
    along the kx and ky axes that hides real spokes. Zero padding only
    refines the wavenumber sampling.
    """
    from scipy.signal.windows import tukey

    values = np.asarray(values, dtype=np.float64)
    v = fill_nan(values)
    a, b, c = fit_plane(v, dx, dy)
    xx, yy = local_coords(v.shape, dx, dy)
    v = v - (a * xx + b * yy + c)
    ny, nx = v.shape
    v = v * np.outer(tukey(ny, alpha=window), tukey(nx, alpha=window))
    width = default_pad(v.shape) if pad is None else max(0, int(pad))
    fast = (sfft.next_fast_len(ny + 2 * width), sfft.next_fast_len(nx + 2 * width))
    p = np.zeros(fast)
    p[width : width + ny, width : width + nx] = v
    F = sfft.fft2(p, workers=-1)
    k, _, _ = wavenumbers(p.shape, dx, dy)
    return np.abs(F) ** 2, k
