"""Footprint and margin safety predicates (NAV-015).

Implements the geometry contract frozen in ``config/geometry_contract_v1.json``:
a rotation-invariant circumscribed disc ``r_fp`` at the base_link origin plus the
position margin ``delta``, applied exactly once.

Three quantities are kept strictly apart, because collapsing them is how a
conservative constraint gets mistaken for a safety certificate:

======================  =====================================================  ==========================
field                   meaning                                                who uses it
======================  =====================================================  ==========================
``physical_collision``  the frozen physical footprint actually touches an       CR / MC reporting
                        obstacle polygon or a boundary wall
``nominal_violation``   the nominal position entered the inflated set          policy constraint, y
                        ``obstacle ⊕ disc(r_fp + delta)``
``safety_y``            irreversible history of ``nominal_violation``          value network, budget
======================  =====================================================  ==========================

Two things deliberately are **not** taken from the frozen modules:

1. ``camra_calibration.obstacles.polygon_distance`` decides intersection by
   testing whether one polygon contains a vertex of the other.  For two convex
   polygons that is *not* sufficient: a long thin rectangle crossing another
   leaves every vertex outside, yet the two overlap.  ``NAV-015`` therefore uses a
   separating-axis test (exact for convex polygons) and the contract test keeps
   the frozen function's miss as a recorded counterexample.
2. ``rotating_sweep_hull`` is documented as a lower bound of the continuous
   sweep, so it cannot bound a collision.  :func:`physical_collision_during_step`
   instead uses ``hull(F0, F1) ⊕ disc(r_fp * |dpsi| / 2)``, which is a proven
   superset of the swept region (see the docstring there).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from camra_calibration.geometry import Polygon

from ..frames import wrap_angle

__all__ = [
    "SafetyGeometry",
    "StepSafety",
    "convex_polygons_intersect",
    "polygon_distance_exact",
    "segment_intersects_polygon",
    "footprint_at",
    "nominal_margin",
    "nominal_violation",
    "nominal_violation_during_step",
    "physical_clearance",
    "physical_collision",
    "physical_collision_during_step",
    "boundary_margin",
    "boundary_violation",
    "update_safety_history",
    "legacy_avoid_to_y",
    "y_to_legacy_avoid",
]

_EPS = 1e-12


def _byte_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class SafetyGeometry:
    """Frozen footprint plus the margins that turn it into a constraint."""

    footprint: Polygon
    r_fp_m: float
    delta_m: float
    contract_version: str
    footprint_sha256: str
    margins_sha256: str
    attitude_treatment: str = "rotation_invariant_circumscribed_disc"

    @classmethod
    def from_frozen(
        cls,
        footprint_path: str,
        margins_path: str,
        contract_path: str,
    ) -> "SafetyGeometry":
        fp = json.load(open(footprint_path, encoding="utf-8"))
        poly = Polygon(np.asarray(fp["polygon"]["vertices"], dtype=np.float64))
        r_fp = poly.circumscribed_radius
        declared = float(fp["polygon"]["circumscribed_radius"])
        if abs(r_fp - declared) > 1e-12:
            raise ValueError(
                f"recomputed circumscribed radius {r_fp} disagrees with the frozen "
                f"value {declared}; refusing to continue with two different radii"
            )
        margins = json.load(open(margins_path, encoding="utf-8"))
        contract = json.load(open(contract_path, encoding="utf-8"))
        return cls(
            footprint=poly,
            r_fp_m=r_fp,
            delta_m=float(margins["delta"]),
            contract_version=str(contract["version"]),
            footprint_sha256=_byte_sha256(footprint_path),
            margins_sha256=_byte_sha256(margins_path),
        )

    @property
    def inflation_radius_m(self) -> float:
        """``r_fp + delta``: the only radius that may be applied to an obstacle."""
        return self.r_fp_m + self.delta_m

    def metadata(self) -> dict:
        """Provenance block that must travel with every run using this geometry."""
        return {
            "schema": "camra-navigation-safety-geometry-v1",
            "contract_version": self.contract_version,
            "r_fp_m": self.r_fp_m,
            "r_fp_reference": "base_link origin",
            "delta_m": self.delta_m,
            "delta_semantics": "90% quantile of planar position error; not a heading "
                               "bound and not a hard upper bound",
            "inflation_radius_m": self.inflation_radius_m,
            "attitude_treatment": self.attitude_treatment,
            "delta_applications": 1,
            "footprint_sha256": self.footprint_sha256,
            "margins_sha256": self.margins_sha256,
            "not_a_safety_certificate": True,
            "note": "The disc is the nominal constraint only; CR/MC use the physical "
                    "footprint against raw obstacles.",
        }


@dataclass(frozen=True)
class StepSafety:
    """Per-step outcome, with the three fields deliberately separate."""

    physical_collision: bool
    nominal_violation: bool
    safety_y: float

    def as_dict(self) -> dict:
        return {
            "physical_collision": self.physical_collision,
            "nominal_violation": self.nominal_violation,
            "safety_y": self.safety_y,
        }


# ---------------------------------------------------------------- primitives
def _edge_axes(poly: Polygon) -> np.ndarray:
    v = poly.vertices
    e = np.roll(v, -1, axis=0) - v
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, _EPS)


def convex_polygons_intersect(a: Polygon, b: Polygon, tol: float = 0.0) -> bool:
    """Separating-axis test; exact for convex polygons.

    ``tol > 0`` reports contact for a gap up to ``tol``, which is how "touching
    counts as collision" is expressed without relying on exact arithmetic.
    """
    axes = np.vstack([_edge_axes(a), _edge_axes(b)])
    for axis in axes:
        pa = a.vertices @ axis
        pb = b.vertices @ axis
        if pa.min() > pb.max() + tol or pb.min() > pa.max() + tol:
            return False
    return True


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= _EPS:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return (o1 * o2 <= 0.0) and (o3 * o4 <= 0.0)


def _segment_segment_distance(a1, a2, b1, b2) -> float:
    if _segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(
        _point_segment_distance(a1, b1, b2),
        _point_segment_distance(a2, b1, b2),
        _point_segment_distance(b1, a1, a2),
        _point_segment_distance(b2, a1, a2),
    )


def polygon_distance_exact(a: Polygon, b: Polygon) -> float:
    """Exact distance between two convex polygons; 0 when they overlap or touch.

    Edge-to-edge distance is required: vertex-to-polygon distance alone misses the
    thin-crossing case, which is the whole point of having this function here.
    """
    if convex_polygons_intersect(a, b, tol=0.0):
        return 0.0
    best = math.inf
    ea = np.roll(a.vertices, -1, axis=0)
    eb = np.roll(b.vertices, -1, axis=0)
    for i in range(a.vertices.shape[0]):
        for j in range(b.vertices.shape[0]):
            best = min(
                best,
                _segment_segment_distance(
                    a.vertices[i], ea[i], b.vertices[j], eb[j]
                ),
            )
    return best


def segment_intersects_polygon(
    p0: Sequence[float], p1: Sequence[float], poly: Polygon, tol: float = 0.0
) -> bool:
    """Does the straight segment ``p0 -> p1`` touch or enter a convex polygon?

    Endpoint containment alone is not enough: a thin obstacle can sit entirely
    between two 10 Hz samples.  Both endpoints plus every edge crossing are
    tested, which is exact for a straight segment and a convex polygon.
    """
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    if poly.contains(a.reshape(1, 2), eps=tol)[0] or poly.contains(b.reshape(1, 2), eps=tol)[0]:
        return True
    v = poly.vertices
    nxt = np.roll(v, -1, axis=0)
    for i in range(v.shape[0]):
        if _segments_intersect(a, b, v[i], nxt[i]):
            return True
    return False


def footprint_at(geom: SafetyGeometry, pos: Sequence[float], yaw: float) -> Polygon:
    return geom.footprint.rotate(yaw).translate(np.asarray(pos, dtype=np.float64))


# ---------------------------------------------------------------- nominal side
def nominal_margin(
    pos: Sequence[float], obstacle: Polygon, geom: SafetyGeometry
) -> float:
    """Signed margin to the inflated obstacle; ``> 0`` means the constraint holds.

    ``obstacle.distance`` is a signed point distance (negative inside), computed
    from the frozen polygon, so this is exact for the point-vs-inflated-set test.
    """
    d = float(obstacle.distance(np.asarray(pos, dtype=np.float64).reshape(1, 2))[0])
    return d - geom.inflation_radius_m


def nominal_violation(
    pos: Sequence[float], obstacle: Polygon, geom: SafetyGeometry, tol: float = 0.0
) -> bool:
    """Inside or on the inflated set counts as violation (``d <= delta``)."""
    return nominal_margin(pos, obstacle, geom) <= tol


def nominal_violation_during_step(
    p0: Sequence[float],
    p1: Sequence[float],
    obstacle: Polygon,
    geom: SafetyGeometry,
    tol: float = 0.0,
) -> bool:
    """Did the nominal point enter the inflated set at any time during the step?

    Uses the inflated polygon and tests the whole segment, so a thin obstacle
    passed between two samples is still caught.
    """
    inflated = obstacle.inflate(geom.inflation_radius_m)
    return segment_intersects_polygon(p0, p1, inflated, tol=tol)


# ---------------------------------------------------------------- physical side
def physical_clearance(
    pos: Sequence[float], yaw: float, obstacle: Polygon, geom: SafetyGeometry
) -> float:
    """Distance between the physical footprint at ``(pos, yaw)`` and an obstacle."""
    return polygon_distance_exact(footprint_at(geom, pos, yaw), obstacle)


def physical_collision(
    pos: Sequence[float], yaw: float, obstacle: Polygon, geom: SafetyGeometry,
    tol: float = 0.0,
) -> bool:
    """Exact convex-polygon intersection, including edge crossings."""
    return convex_polygons_intersect(footprint_at(geom, pos, yaw), obstacle, tol=tol)


def physical_collision_during_step(
    p0: Sequence[float],
    yaw0: float,
    p1: Sequence[float],
    yaw1: float,
    obstacle: Polygon,
    geom: SafetyGeometry,
    tol: float = 0.0,
) -> bool:
    """Conservative collision test over a whole step, rotation included.

    With ``dpsi = |wrap(yaw1 - yaw0)|`` and ``rho = r_fp`` (every body point is
    within ``rho`` of the rotation centre), rotating by ``theta`` moves any point
    at most ``rho * theta``.  Hence for ``t in [0, 1]``

        F(t) ⊆ F(0) ⊕ disc(rho * t * dpsi)      and      F(t) ⊆ F(1) ⊕ disc(rho * (1-t) * dpsi),

    so for ``t <= 1/2`` the first bound and for ``t >= 1/2`` the second both fit
    inside ``hull(F(0), F(1)) ⊕ disc(rho * dpsi / 2)``.  That super-set is what is
    tested here: it can report a collision that did not happen (never the
    reverse), and for ``dpsi = 0`` it degenerates to the exact translation hull.
    """
    f0 = footprint_at(geom, p0, yaw0)
    f1 = footprint_at(geom, p1, yaw1)
    swept = f0.union(f1)
    dpsi = abs(wrap_angle(yaw1 - yaw0))
    if dpsi > 0.0:
        swept = swept.inflate(geom.r_fp_m * dpsi / 2.0)
    return convex_polygons_intersect(swept, obstacle, tol=tol)


# ---------------------------------------------------------------- boundary
def boundary_margin(
    pos: Sequence[float], extent: Sequence[float], geom: SafetyGeometry
) -> float:
    """Signed margin to the rectangular boundary, walls treated like obstacles.

    A wall is a half-plane, and inflating a half-plane by a disc is exactly a
    shifted half-plane, so taking the minimum over the four walls is exact for
    the disc constraint (corner rounding included).
    """
    x_min, x_max, y_min, y_max = (float(v) for v in extent)
    px, py = float(pos[0]), float(pos[1])
    return min(px - x_min, x_max - px, py - y_min, y_max - py) - geom.inflation_radius_m


def boundary_violation(
    pos: Sequence[float], extent: Sequence[float], geom: SafetyGeometry, tol: float = 0.0
) -> bool:
    return boundary_margin(pos, extent, geom) <= tol


# ---------------------------------------------------------------- history
def update_safety_history(y_prev: float, violation: bool) -> float:
    """Irreversible update: once unsafe, always unsafe within the episode."""
    prev = 1.0 if float(y_prev) >= 0.5 else 0.0
    return 1.0 if (violation or prev >= 0.5) else 0.0


def legacy_avoid_to_y(avoid: float) -> float:
    """Map the frozen RC-PPO ``avoid`` sign to the navigation ``y``.

    In the reference implementation ``avoid_value = where(is_avoid, -1, 1)``, i.e.
    ``-1`` means unsafe and ``+1`` means safe.  The navigation history is the
    opposite polarity and unsigned, so ``avoid=-1 -> y=1`` and ``avoid=+1 -> y=0``.
    """
    value = float(avoid)
    if value == -1.0:
        return 1.0
    if value == 1.0:
        return 0.0
    raise ValueError(f"legacy avoid must be -1 or +1, got {avoid!r}")


def y_to_legacy_avoid(y: float) -> float:
    """Inverse of :func:`legacy_avoid_to_y`, for reading old logs."""
    return -1.0 if float(y) >= 0.5 else 1.0
