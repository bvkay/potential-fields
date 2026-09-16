"""Two-dimensional forward modelling and fitting of gravity and magnetic profiles.

Bodies are polygons of infinite strike length perpendicular to the profile.
x is distance along the profile (m, increasing toward the profile azimuth),
z is depth (m, positive down). Observation points must lie above every
vertex.

Gravity (Talwani, Worzel and Landisman 1959, written here in complex form):
with q = x + i z the position of a boundary point relative to the
observation point, a and b the end points of an edge and d = b - a,

    g_x + i g_z = -i G rho  sum_edges [ (d / conj d) conj(b ln b - a ln a) - d ]

for counterclockwise vertices. This is Green's theorem applied to the area
integral of 1/conj(q); the -d terms sum to zero round a closed polygon.

Magnetics: a uniformly magnetised body is equivalent to poles of surface
density sigma = M . n on its boundary, n the outward normal. In two
dimensions an edge with unit tangent e contributes

    B_x + i B_z = -2 Cm sigma e conj(ln(b / a))

so the field is a sum over edges of ln|b/a| and the angle the edge
subtends. The total field anomaly is dT = f . B with f the ambient field
direction projected onto the profile plane (Blakely 1996 section 9.3); the
along-strike components of M and f produce nothing in 2D.

Units: lengths m, density contrast kg/m^3, susceptibility SI, magnetisation
A/m, ambient field nT. Outputs mGal and nT.

References: Talwani, Worzel and Landisman 1959, J. Geophys. Res. 64;
Talwani and Heirtzler 1964; Won and Bevis 1987, Geophysics 52, 232-238;
Blakely 1996 chapter 9 and appendix B.

@author: Ben Kay (ben@auscope.org.au)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from .synthetic import CM, G, MGAL, NT

MU0 = 4e-7 * np.pi


# bodies -------------------------------------------------------------------


@dataclass
class Body:
    """Polygon cross-section with physical properties.

    x, z: vertices (m), any order; stored counterclockwise in the (x, z)
    plane with a closing vertex removed. density is the contrast in kg/m^3.
    susceptibility (SI) gives induced magnetisation kappa F / mu0 along the
    ambient field. magnetisation (A/m) with mag_inc, mag_dec adds a remanent
    component; with mag_inc None it is taken along the ambient field.
    """

    x: Sequence[float]
    z: Sequence[float]
    density: float = 0.0
    susceptibility: float = 0.0
    magnetisation: float = 0.0
    mag_inc: float | None = None
    mag_dec: float | None = None
    name: str = ""

    def __post_init__(self):
        x = np.asarray(self.x, dtype=float).ravel()
        z = np.asarray(self.z, dtype=float).ravel()
        if x.shape != z.shape or x.size < 3:
            raise ValueError("a body needs at least 3 vertices with matching x and z")
        if x[0] == x[-1] and z[0] == z[-1] and x.size > 3:
            x, z = x[:-1], z[:-1]
        area = 0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)
        if abs(area) < 1e-12:
            raise ValueError("body has zero area")
        if area < 0:
            x, z = x[::-1], z[::-1]
        self.x, self.z = x, z

    @property
    def area(self) -> float:
        return float(0.5 * np.sum(self.x * np.roll(self.z, -1) - np.roll(self.x, -1) * self.z))

    @property
    def centroid(self) -> tuple[float, float]:
        x, z = self.x, self.z
        xn, zn = np.roll(x, -1), np.roll(z, -1)
        cross = x * zn - xn * z
        a = 0.5 * cross.sum()
        return (float(np.sum((x + xn) * cross) / (6 * a)), float(np.sum((z + zn) * cross) / (6 * a)))

    def to_dict(self) -> dict:
        return {
            "x": self.x.tolist(),
            "z": self.z.tolist(),
            "density": self.density,
            "susceptibility": self.susceptibility,
            "magnetisation": self.magnetisation,
            "mag_inc": self.mag_inc,
            "mag_dec": self.mag_dec,
            "name": self.name,
        }


def rectangle(x0: float, x1: float, z_top: float, z_bottom: float, **props) -> Body:
    """Axis-aligned rectangle between x0 and x1, z_top and z_bottom."""
    return Body([x0, x1, x1, x0], [z_top, z_top, z_bottom, z_bottom], **props)


def dyke(x_centre: float, top: float, width: float, dip: float, extent: float, **props) -> Body:
    """Parallelogram: horizontal width at the top, vertical extent, dip in
    degrees from horizontal (90 vertical, less than 90 dips toward +x,
    more than 90 toward -x)."""
    shift = 0.0 if abs(dip - 90.0) < 1e-9 else extent / np.tan(np.radians(dip))
    h = width / 2.0
    return Body(
        [x_centre - h, x_centre + h, x_centre + h + shift, x_centre - h + shift],
        [top, top, top + extent, top + extent],
        **props,
    )


def circle(x_centre: float, z_centre: float, radius: float, n: int = 72, **props) -> Body:
    """Regular polygon approximating a horizontal cylinder."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Body(x_centre + radius * np.cos(t), z_centre + radius * np.sin(t), **props)


def save_bodies(bodies: Sequence[Body], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps([b.to_dict() for b in bodies], indent=1), encoding="utf-8")
    return path


def load_bodies(path: str | Path) -> list[Body]:
    return [Body(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


# directions ----------------------------------------------------------------


def in_plane_direction(inc: float, dec: float, azimuth: float) -> tuple[float, float]:
    """(fx, fz): components along the profile (+x toward `azimuth`) and down
    of a unit vector with inclination inc (positive down) and declination dec.
    The along-strike component is dropped."""
    i, d = np.radians(inc), np.radians(dec - azimuth)
    return (float(np.cos(i) * np.cos(d)), float(np.sin(i)))


def induced_magnetisation(susceptibility: float, field_nt: float) -> float:
    """kappa F / mu0 in A/m for F in nT."""
    return susceptibility * field_nt * 1e-9 / MU0


def magnetisation_in_plane(body: Body, inc: float, dec: float, azimuth: float, field_nt: float) -> tuple[float, float]:
    """(Mx, Mz) of a body: induced along the ambient field plus remanent."""
    fx, fz = in_plane_direction(inc, dec, azimuth)
    m_ind = induced_magnetisation(body.susceptibility, field_nt)
    mx, mz = m_ind * fx, m_ind * fz
    if body.magnetisation:
        rx, rz = (
            in_plane_direction(body.mag_inc, body.mag_dec, azimuth)
            if body.mag_inc is not None
            else (fx, fz)
        )
        mx += body.magnetisation * rx
        mz += body.magnetisation * rz
    return (mx, mz)


# kernels -------------------------------------------------------------------


def _edges(body: Body, x: np.ndarray, z_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Complex positions a, b of each edge's end points relative to every
    observation point. Shape (n_edges, n_obs)."""
    if np.any(body.z[:, None] <= z_obs[None, :]):
        raise ValueError(f"body {body.name!r} has vertices at or above an observation point; bodies must lie below the sensor")
    qx = body.x[:, None] - x[None, :]
    qz = body.z[:, None] - z_obs[None, :]
    q = qx + 1j * qz
    return q, np.roll(q, -1, axis=0)


def gravity_components(bodies: Sequence[Body], x: np.ndarray, z_obs: float | np.ndarray = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(gx, gz) in m/s^2 along and down. gz positive toward the body."""
    x = np.asarray(x, dtype=float).ravel()
    z_obs = np.broadcast_to(np.asarray(z_obs, dtype=float), x.shape)
    total = np.zeros(x.shape, dtype=complex)
    for body in bodies:
        if body.density == 0.0:
            continue
        a, b = _edges(body, x, z_obs)
        d = b - a
        term = (d / np.conj(d)) * np.conj(b * np.log(b) - a * np.log(a)) - d
        total += -1j * G * body.density * term.sum(axis=0)
    return total.real, total.imag


def gravity(bodies: Sequence[Body], x: np.ndarray, z_obs: float | np.ndarray = 0.0) -> np.ndarray:
    """Vertical gravity anomaly in mGal."""
    _, gz = gravity_components(bodies, x, z_obs)
    return gz * MGAL


def magnetic_components(
    bodies: Sequence[Body], x: np.ndarray, inc: float, dec: float, azimuth: float, field_nt: float, z_obs: float | np.ndarray = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """(Bx, Bz) anomaly in nT along the profile and down."""
    x = np.asarray(x, dtype=float).ravel()
    z_obs = np.broadcast_to(np.asarray(z_obs, dtype=float), x.shape)
    total = np.zeros(x.shape, dtype=complex)
    for body in bodies:
        mx, mz = magnetisation_in_plane(body, inc, dec, azimuth, field_nt)
        if mx == 0.0 and mz == 0.0:
            continue
        a, b = _edges(body, x, z_obs)
        d = b - a
        e = d / np.abs(d)  # unit tangent, counterclockwise
        nx, nz = e.imag, -e.real  # outward normal for counterclockwise vertices
        sigma = mx * nx + mz * nz  # pole density on each edge (n_edges, n_obs; constant along n_obs)
        total += (-2 * CM * sigma * e * np.conj(np.log(b / a))).sum(axis=0)
    return total.real * NT, total.imag * NT


def magnetic(
    bodies: Sequence[Body], x: np.ndarray, inc: float, dec: float, azimuth: float, field_nt: float, z_obs: float | np.ndarray = 0.0
) -> np.ndarray:
    """Total field anomaly in nT: the anomalous field projected onto the ambient field."""
    bx, bz = magnetic_components(bodies, x, inc, dec, azimuth, field_nt, z_obs)
    fx, fz = in_plane_direction(inc, dec, azimuth)
    return fx * bx + fz * bz


def forward(
    bodies: Sequence[Body],
    x: np.ndarray,
    z_obs: float | np.ndarray = 0.0,
    inc: float | None = None,
    dec: float | None = None,
    azimuth: float | None = None,
    field_nt: float | None = None,
) -> dict[str, np.ndarray]:
    """gravity (mGal) and, when the field is given, tmi (nT)."""
    out = {"gravity": gravity(bodies, x, z_obs)}
    if inc is not None:
        if dec is None or azimuth is None or field_nt is None:
            raise ValueError("tmi needs inc, dec, azimuth and field_nt")
        out["tmi"] = magnetic(bodies, x, inc, dec, azimuth, field_nt, z_obs)
    return out


# fitting -------------------------------------------------------------------


def regional(x: np.ndarray, residual: np.ndarray, order: int | None) -> np.ndarray:
    """Least-squares polynomial of `order` through residual; zeros if order is None."""
    if order is None:
        return np.zeros_like(x, dtype=float)
    ok = np.isfinite(residual)
    xn = (x - x.mean()) / (np.ptp(x) / 2 or 1.0)
    coef = np.polyfit(xn[ok], residual[ok], order)
    return np.polyval(coef, xn)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.nanmean(d**2)))


@dataclass
class FitResult:
    params: np.ndarray
    bodies: list[Body]
    x: np.ndarray
    observed: np.ndarray
    body_field: np.ndarray
    regional: np.ndarray
    predicted: np.ndarray
    rms: float
    result: object = field(repr=False, default=None)

    def summary(self, names: Sequence[str] | None = None) -> str:
        names = names or [f"p{i}" for i in range(len(self.params))]
        lines = [f"{n:>16s} = {v:12.4g}" for n, v in zip(names, self.params)]
        lines.append(f"{'rms':>16s} = {self.rms:12.4g}")
        return "\n".join(lines)


def fit(
    x: np.ndarray,
    observed: np.ndarray,
    build: Callable[[np.ndarray], Sequence[Body]],
    p0: Sequence[float],
    field_name: str = "tmi",
    bounds: tuple = (-np.inf, np.inf),
    regional_order: int | None = 1,
    z_obs: float | np.ndarray = 0.0,
    **field_kw,
) -> FitResult:
    """Least-squares fit of body parameters to an observed profile.

    build(p) returns the bodies for parameter vector p. field_name is
    "tmi" (needs inc, dec, azimuth, field_nt in field_kw) or "gravity".
    A polynomial regional of regional_order is solved linearly inside every
    evaluation, so only the body parameters are nonlinear. Uses
    scipy.optimize.least_squares with bounds (trust region reflective).
    Non-uniqueness is real: try several starting points and compare rms.
    """
    x = np.asarray(x, dtype=float).ravel()
    observed = np.asarray(observed, dtype=float).ravel()
    ok = np.isfinite(observed)

    def model(p):
        f = forward(build(p), x, z_obs, **field_kw)[field_name]
        reg = regional(x, np.where(ok, observed - f, np.nan), regional_order)
        return f, reg

    def residual(p):
        f, reg = model(p)
        return (observed - f - reg)[ok]

    res = least_squares(residual, np.asarray(p0, dtype=float), bounds=bounds, x_scale="jac", method="trf")
    f, reg = model(res.x)
    return FitResult(res.x, list(build(res.x)), x, observed, f, reg, f + reg, rms(observed[ok], (f + reg)[ok]), res)


# plotting ------------------------------------------------------------------


def plot_model(
    x: np.ndarray,
    observed: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
    bodies: Sequence[Body],
    z_obs: float | np.ndarray = 0.0,
    regional_part: dict[str, np.ndarray] | None = None,
    depth_max: float | None = None,
    equal_aspect: bool = False,
    axes=None,
):
    """Observed and predicted profiles on top, cross-section below.

    observed and predicted map field name ("tmi", "gravity") to arrays on
    x. Bodies are filled and labelled; the section has depth positive down
    and, if equal_aspect, no vertical exaggeration.
    """
    from matplotlib import pyplot as plt

    names = [k for k in ("tmi", "gravity") if k in observed or k in predicted]
    n = len(names)
    if axes is None:
        fig, axes = plt.subplots(n + 1, 1, figsize=(9, 2.4 * n + 3.2), sharex=True, gridspec_kw={"height_ratios": [1] * n + [1.4]})
    else:
        fig = axes[0].figure
    units = {"tmi": "nT", "gravity": "mGal"}
    xk = np.asarray(x) / 1e3
    for ax, name in zip(axes[:n], names):
        if name in observed:
            ax.plot(xk, observed[name], ".", color="k", ms=3, label="observed")
        if name in predicted:
            ax.plot(xk, predicted[name], "-", color="tab:red", lw=1.5, label="predicted")
        if regional_part and name in regional_part:
            ax.plot(xk, regional_part[name], "--", color="tab:gray", lw=1, label="regional")
        if name in observed and name in predicted:
            ax.set_title(f"{name}: rms {rms(observed[name], predicted[name]):.3g} {units[name]}", loc="left", fontsize=9)
        ax.set_ylabel(units[name])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    sec = axes[-1]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, b in enumerate(bodies):
        sec.fill(b.x / 1e3, b.z / 1e3, color=colours[i % len(colours)], alpha=0.6, ec="k", lw=0.8)
        cx, cz = b.centroid
        label = b.name or f"body {i + 1}"
        props = []
        if b.density:
            props.append(f"{b.density:+.3g} kg/m3")
        if b.susceptibility:
            props.append(f"k={b.susceptibility:.3g}")
        if b.magnetisation:
            props.append(f"M={b.magnetisation:.3g} A/m")
        sec.annotate("\n".join([label] + props), (cx / 1e3, cz / 1e3), ha="center", va="center", fontsize=8)
    zo = np.broadcast_to(np.asarray(z_obs, dtype=float), np.shape(x))
    sec.plot(xk, zo / 1e3, color="k", lw=0.8)
    zmax = depth_max if depth_max is not None else max(b.z.max() for b in bodies) * 1.2
    sec.set_ylim(zmax / 1e3, min(zo.min() / 1e3, 0) - 0.02 * zmax / 1e3)
    sec.set_xlim(xk.min(), xk.max())
    sec.set_xlabel("distance (km)")
    sec.set_ylabel("depth (km)")
    if equal_aspect:
        sec.set_aspect("equal")
    sec.grid(alpha=0.3)
    return fig, axes
