"""Coordinate frames, quaternion order and heading conventions (NAV-013).

Frozen conventions (authoritative copy lives in
``config/navigation_contract_v1.json`` under ``frames_and_conventions`` /
``observation_spec``, version ``obs-v1``).  This module is the single
implementation; the contract test asserts the two agree.

World frame
    Right-handed.  ``+x`` and ``+y`` are the map plane, ``+z`` up.  Motion is
    planar, so only ``(px, py, yaw)`` is integrated; ``z`` is not part of the
    navigation state.

Body frame
    The Go2 ``base_link`` frame: ``x`` forward, ``y`` left, ``z`` up.  This is
    the frame the velocity command ``[vx, vy, wz]`` is expressed in and the frame
    ``footprint_v1.json`` uses.  The footprint centroid is **not** at the
    origin (``[-0.0456, -0.0110]`` m), so the navigation position is always the
    base_link origin -- never the footprint centroid.

Quaternion order
    ``(w, x, y, z)``, matching Isaac Lab.  Reordering to ``(x, y, z, w)``
    silently rotates the robot and is not detectable downstream, so this is
    frozen here and asserted in the contract test.

Yaw
    The *projected heading*: rotate the body ``+x`` axis by the quaternion,
    project onto the world ``xy`` plane, normalise, take ``atan2(y, x)``.  With
    zero roll/pitch this equals the planar yaw.  With nonzero roll/pitch it is
    the heading of the projected axis and is **not** a ZYX yaw decomposition;
    the two differ and only the projected heading is used, because it is the one
    that stays defined and continuous when the body tilts.

Angles are radians.  Distances are metres.  Nothing in this module filters,
smooths or otherwise modifies a measurement.
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = [
    "QUAT_ORDER",
    "HeadingUndefined",
    "wrap_angle",
    "quat_from_yaw",
    "quat_rotate",
    "heading_from_quat",
    "yaw_from_quat",
    "body_to_world_vector",
    "world_to_body_vector",
    "body_to_world_point",
    "world_to_body_point",
]

#: Quaternion component order used everywhere in this project.
QUAT_ORDER = "wxyz"

_EPS = 1e-12
TWO_PI = 2.0 * math.pi


class HeadingUndefined(ValueError):
    """Raised when the body ``+x`` axis projects to (almost) zero length.

    This happens when the robot points straight up or straight down, so no
    planar heading exists.  Callers must treat it as an ``internal_violation``
    (see the event table) rather than substituting an arbitrary angle.
    """


def wrap_angle(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]`` without changing the represented heading."""
    a = math.fmod(a + math.pi, TWO_PI)
    if a <= 0.0:
        a += TWO_PI
    return a - math.pi


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Return the ``(w, x, y, z)`` quaternion of a pure yaw rotation."""
    h = 0.5 * yaw
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def quat_rotate(
    q: Sequence[float], v: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate vector ``v`` by quaternion ``q`` (both in frozen conventions)."""
    w, x, y, z = (float(c) for c in q)
    vx, vy, vz = (float(c) for c in v)
    # t = 2 * (q_vec x v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def heading_from_quat(q: Sequence[float]) -> tuple[float, float]:
    """Return the unit ``(cos, sin)`` of the projected heading of the body ``+x``.

    Raises:
        HeadingUndefined: if the projected axis has no usable planar direction.
    """
    fx, fy, _fz = quat_rotate(q, (1.0, 0.0, 0.0))
    norm = math.hypot(fx, fy)
    if norm < 1e-9:
        raise HeadingUndefined(
            "body +x axis projects to zero length; planar heading undefined"
        )
    return (fx / norm, fy / norm)


def yaw_from_quat(q: Sequence[float]) -> float:
    """Return the projected heading as an angle in ``(-pi, pi]``."""
    c, s = heading_from_quat(q)
    return math.atan2(s, c)


def body_to_world_vector(
    yaw: float, vx: float, vy: float
) -> tuple[float, float]:
    """Rotate a body-frame planar vector into the world frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (vx * c - vy * s, vx * s + vy * c)


def world_to_body_vector(
    yaw: float, vx: float, vy: float
) -> tuple[float, float]:
    """Rotate a world-frame planar vector into the body frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (vx * c + vy * s, -vx * s + vy * c)


def body_to_world_point(
    yaw: float, px: float, py: float, bx: float, by: float
) -> tuple[float, float]:
    """Map a body-frame point to world coordinates given the base_link pose."""
    dx, dy = body_to_world_vector(yaw, bx, by)
    return (px + dx, py + dy)


def world_to_body_point(
    yaw: float, px: float, py: float, wx: float, wy: float
) -> tuple[float, float]:
    """Map a world point to body coordinates given the base_link pose."""
    return world_to_body_vector(yaw, wx - px, wy - py)
