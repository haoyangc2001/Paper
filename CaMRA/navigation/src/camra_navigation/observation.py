"""Frozen observation and action encoding (NAV-013, ``obs-v1``).

Two ends, one encoder
---------------------
The nominal environment and the Isaac Lab environment must produce *the same
numbers from the same physical state*.  That is achieved structurally, not by
review: both ends build an :class:`EgoState` / :class:`TaskState` and call
:func:`encode_observation`.  The physical entry point
:func:`encode_from_physical` only converts ``base_pos`` + ``base_quat`` +
measured body velocities into the same dataclasses; it does not encode anything
itself.  ``tests/test_observation_contract.py`` asserts bitwise equality of the
two paths on identical artificial states.

What is deliberately *not* here
-------------------------------
No filtering, smoothing, frame-rate adaptation or clipping of a measurement
beyond the frozen normalisation ranges.  The command domain is not modified
here: :data:`COMMAND_MAX` is loaded from the NAV-055 resolution and the applied
command is clipped to exactly that domain, once.

Command single-source rule
--------------------------
:func:`command_from_action` returns a :class:`CommandRecord`.  Only
``record.applied`` may be fed to the nominal integrator, the power model and the
low-level command bridge.  ``record.requested`` exists for logging and for the
budget/telemetry audit, never for dynamics or cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .frames import body_to_world_vector, heading_from_quat, world_to_body_vector, yaw_from_quat

__all__ = [
    "OBS_VERSION",
    "COMMAND_VERSION",
    "N_OBSTACLE_BINS",
    "OBSTACLE_CAPACITY",
    "N_WALL_DISTS",
    "R_SENSE_M",
    "MAP_HALF_M",
    "T_MAX_S",
    "V_REF_MPS",
    "W_REF_RADPS",
    "V_BODY_CLIP",
    "Z_REF_J",
    "COMMAND_MAX",
    "LAYOUT",
    "OBS_DIM",
    "OBSTACLE_SORT_RULE",
    "EgoState",
    "Obstacle",
    "PolygonObstacle",
    "obstacle_from_polygon",
    "MapExtent",
    "TaskState",
    "CommandRecord",
    "encode_observation",
    "encode_from_physical",
    "surface_distance",
    "action_to_requested",
    "clip_applied",
    "command_from_action",
    "field_slices",
]

#: Version of the observation layout and of every constant below.
#: ``obs-v1`` supported circles only.  NAV-016 froze rectangular islands, and the
#: contract's ``map_encoding.extension_rule`` forbids reusing ``radius`` for a
#: different primitive, so ``obs-v2`` widens the primitive set to circles *and*
#: convex polygons.  The layout, dimension and every constant are unchanged;
#: only the accepted primitive set grew, which is why the 28 obs-v1 checks still
#: hold bit-for-bit on circle maps.
OBS_VERSION = "obs-v2"
#: Version of the action/normalisation mapping.
COMMAND_VERSION = "cmd-v1"

# --- map encoding -----------------------------------------------------------
#: Radial occupancy bins in the body frame, ``[-pi, pi)``.
N_OBSTACLE_BINS = 16
#: Obstacle slots available to the network; excess primitives are dropped by the
#: documented rule and counted in the step record.
OBSTACLE_CAPACITY = 8
#: Signed distances to the four boundary walls: ``x_min, x_max, y_min, y_max``.
N_WALL_DISTS = 4
#: Sensing range for the occupancy bins, metres.
R_SENSE_M = 6.0
#: Map half-extent used to normalise world position and goal distance, metres.
#: Provisional: owned by NAV-017 (map generation).  Saturation is counted in the
#: step record so a wrong choice is visible rather than silent.
MAP_HALF_M = 15.0

# --- task scales ------------------------------------------------------------
#: Episode horizon used by the frozen calibration protocol, seconds.
T_MAX_S = 30.0
#: Reference speeds for the measured body velocity, metres/second, rad/second.
V_REF_MPS = 1.0
W_REF_RADPS = 1.0
#: Body velocity is clipped to this many multiples of the reference speed.
V_BODY_CLIP = 2.0
#: Reference budget for ``z_ratio``, joules.
#:
#: ``P_nom(0.30986, 0, 0) * 30 s`` with the frozen ``power-v1`` weights, i.e. the
#: 30 s energy of the analytically energy-optimal straight-line speed.  It is a
#: *scale*, not a task budget: NAV-056 owns ``z_J`` / ``z_ref_J`` and the critic
#: scale, and must confirm or replace this value with a version bump.
Z_REF_J = 225.6069087767049

#: Validated commanded velocity domain ``(vx, vy, wz)`` from NAV-055: protocol,
#: frozen controller contract, power-model domain, generator limits and 800
#: measured applied commands all agree on these values.
COMMAND_MAX: tuple[float, float, float] = (1.0, 0.4, 1.0)

#: Canonical obstacle ordering used to build ``obstacle_mask``.  Sorting makes
#: the encoding invariant to the order in which the map lists primitives.
OBSTACLE_SORT_RULE = "ascending surface distance, then ascending body-frame bearing"

# --- layout -----------------------------------------------------------------
#: ``(name, dim)`` in concatenation order.  Offsets are derived, never typed.
LAYOUT: tuple[tuple[str, int], ...] = (
    ("sin_yaw", 1),
    ("cos_yaw", 1),
    ("vx_body_n", 1),
    ("vy_body_n", 1),
    ("wz_body_n", 1),
    ("px_n", 1),
    ("py_n", 1),
    ("goal_dir_x_body", 1),
    ("goal_dir_y_body", 1),
    ("goal_dist_n", 1),
    ("obstacle_bins", N_OBSTACLE_BINS),
    ("wall_dist_n", N_WALL_DISTS),
    ("obstacle_mask", OBSTACLE_CAPACITY),
    ("safety_y", 1),
    ("z_ratio", 1),
    ("time_remaining_n", 1),
)

OBS_DIM = sum(dim for _name, dim in LAYOUT)


def field_slices() -> dict[str, slice]:
    """Return the frozen slice of every field in the observation vector."""
    out: dict[str, slice] = {}
    start = 0
    for name, dim in LAYOUT:
        out[name] = slice(start, start + dim)
        start += dim
    assert start == OBS_DIM
    return out


# --- state containers -------------------------------------------------------
@dataclass(frozen=True)
class EgoState:
    """Base_link planar pose plus measured body-frame velocities.

    ``px``/``py`` are the base_link origin in world coordinates.  They are never
    the footprint centroid (offset ``[-0.0456, -0.0110]`` m in
    ``footprint_v1.json``).
    """

    px: float
    py: float
    yaw: float
    vx_body: float
    vy_body: float
    wz_body: float


@dataclass(frozen=True)
class Obstacle:
    """A circular obstacle primitive in world coordinates, metres.

    Circles are one of the two primitives ``obs-v2`` accepts.  ``radius`` is only
    ever a radius; turning a polygon into a circle is a separate, declared step
    (:func:`obstacle_from_polygon`) and never a reinterpretation of this field.
    """

    cx: float
    cy: float
    radius: float


@dataclass(frozen=True)
class PolygonObstacle:
    """A convex polygon primitive in world coordinates, metres.

    Added with ``obs-v2`` because NAV-016 froze rectangular islands and the
    contract forbids encoding a polygon through a ``radius``.  Coverage of a
    rectangle by circles would either need many primitives (capacity is 8) or one
    circumscribed circle that is ~0.4 m too fat per side — enough to close the
    0.76 m free lane a 7 m corridor leaves after the delta inflation, so the
    polygon has to be carried through honestly instead of approximated.
    """

    vertices: tuple[tuple[float, float], ...]
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError(f"a polygon needs at least 3 vertices, got {len(self.vertices)}")
        for x, y in self.vertices:
            if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                raise ValueError("polygon vertices must be finite")

    @property
    def centroid(self) -> tuple[float, float]:
        n = float(len(self.vertices))
        return (
            sum(float(v[0]) for v in self.vertices) / n,
            sum(float(v[1]) for v in self.vertices) / n,
        )


def obstacle_from_polygon(vertices, *, label: str = "") -> PolygonObstacle:
    """Wrap polygon vertices as an observation primitive (no simplification)."""
    pts = tuple((float(v[0]), float(v[1])) for v in vertices)
    return PolygonObstacle(vertices=pts, label=label)


@dataclass(frozen=True)
class MapExtent:
    """Axis-aligned rectangular boundary of the map, metres, world frame."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class TaskState:
    """Per-episode goal and progress quantities."""

    goal_x: float
    goal_y: float
    safety_y: float  # 0.0 or 1.0, irreversible
    z_J: float  # signed remaining budget, joules (negative = over budget)
    time_remaining_s: float


@dataclass(frozen=True)
class CommandRecord:
    """One command application: what was asked, what was actually held."""

    requested: tuple[float, float, float]
    applied: tuple[float, float, float]
    clipped: bool
    invalid: bool


# --- geometry kernel --------------------------------------------------------
def _nearest_on_boundary(px: float, py: float, verts):
    """Closest point on the polygon boundary, as ``(distance, qx, qy)``."""
    best_d = math.inf
    best_q = (px, py)
    n = len(verts)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        seg2 = ex * ex + ey * ey
        if seg2 <= 0.0:
            qx, qy = ax, ay
        else:
            t = ((px - ax) * ex + (py - ay) * ey) / seg2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            qx, qy = ax + t * ex, ay + t * ey
        d = math.hypot(px - qx, py - qy)
        if d < best_d:
            best_d, best_q = d, (qx, qy)
    return best_d, best_q[0], best_q[1]


def _point_in_convex(px: float, py: float, verts) -> bool:
    """Cross-product sign test; the polygon is convex by contract."""
    n = len(verts)
    pos = neg = False
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > 1e-12:
            pos = True
        elif cross < -1e-12:
            neg = True
        if pos and neg:
            return False
    return True


def _polygon_surface_distance(ego: EgoState, obstacle: "PolygonObstacle") -> tuple[float, float]:
    verts = [(float(v[0]), float(v[1])) for v in obstacle.vertices]
    d, qx, qy = _nearest_on_boundary(ego.px, ego.py, verts)
    inside = _point_in_convex(ego.px, ego.py, verts)
    surf = 0.0 if inside else d
    if d <= 1e-12:
        # base_link sits on the boundary: the contact direction is degenerate,
        # so fall back to the centroid direction rather than emitting noise.
        cx, cy = obstacle.centroid
        bx, by = world_to_body_vector(ego.yaw, cx - ego.px, cy - ego.py)
        return (surf, math.atan2(by, bx))
    bx, by = world_to_body_vector(ego.yaw, qx - ego.px, qy - ego.py)
    return (surf, math.atan2(by, bx))


def surface_distance(
    ego: EgoState, obstacle
) -> tuple[float, float]:
    """Return ``(surface_distance_m, bearing_rad)`` of one obstacle from base_link.

    ``surface_distance`` is the gap between the base_link origin and the
    primitive surface, clamped at zero: a point inside the primitive yields
    ``0.0``.  The bearing is the body-frame angle of the **nearest surface
    point**, in ``[-pi, pi)``.  For a circle seen from outside the nearest point
    lies on the centre ray, so this reproduces the ``obs-v1`` circle behaviour
    exactly; for a long rectangle it reports the direction of the face you would
    actually hit, not the direction of the centroid.
    """
    if isinstance(obstacle, PolygonObstacle):
        return _polygon_surface_distance(ego, obstacle)
    dx, dy = world_to_body_vector(ego.yaw, obstacle.cx - ego.px, obstacle.cy - ego.py)
    centre = math.hypot(dx, dy)
    return (max(0.0, centre - max(0.0, float(obstacle.radius))), math.atan2(dy, dx))


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _finite(x: float, default: float = 0.0) -> float:
    return float(x) if math.isfinite(float(x)) else default


# --- encoder ----------------------------------------------------------------
def encode_observation(
    ego: EgoState,
    obstacles: Iterable[Obstacle],
    extent: MapExtent,
    task: TaskState,
    *,
    z_ref_J: float = Z_REF_J,
) -> list[float]:
    """Encode one state into the frozen ``obs-v1`` vector.

    The same function is called by the nominal environment and by the Isaac Lab
    environment.  Order of ``obstacles`` does not affect the result.
    """
    sin_yaw = math.sin(ego.yaw)
    cos_yaw = math.cos(ego.yaw)

    vx_n = _clip(_finite(ego.vx_body) / V_REF_MPS, -V_BODY_CLIP, V_BODY_CLIP)
    vy_n = _clip(_finite(ego.vy_body) / V_REF_MPS, -V_BODY_CLIP, V_BODY_CLIP)
    wz_n = _clip(_finite(ego.wz_body) / W_REF_RADPS, -V_BODY_CLIP, V_BODY_CLIP)

    px_n = _clip(_finite(ego.px) / MAP_HALF_M, -1.0, 1.0)
    py_n = _clip(_finite(ego.py) / MAP_HALF_M, -1.0, 1.0)

    # Goal in the body frame: unit direction plus normalised distance.
    gx_b, gy_b = world_to_body_vector(
        ego.yaw, task.goal_x - ego.px, task.goal_y - ego.py
    )
    gdist = math.hypot(gx_b, gy_b)
    if gdist > 1e-9:
        gdir_x, gdir_y = gx_b / gdist, gy_b / gdist
    else:
        gdir_x, gdir_y = 0.0, 0.0
    gdist_n = _clip(gdist / MAP_HALF_M, 0.0, 1.5)

    # Occupancy bins: max intensity per angular bin, surface-distance based.
    bins = [0.0] * N_OBSTACLE_BINS
    bin_width = 2.0 * math.pi / N_OBSTACLE_BINS
    measured: list[tuple[float, float]] = []
    for ob in obstacles:
        surf, bearing = surface_distance(ego, ob)
        if not math.isfinite(bearing):
            continue
        measured.append((surf, bearing))
        if surf > R_SENSE_M:
            continue
        intensity = 1.0 - _clip(surf / R_SENSE_M, 0.0, 1.0)
        idx = int(math.floor((bearing + math.pi) / bin_width))
        if idx < 0:
            idx = 0
        elif idx >= N_OBSTACLE_BINS:
            idx = N_OBSTACLE_BINS - 1
        if intensity > bins[idx]:
            bins[idx] = intensity

    # Canonical order -> mask.  Excess primitives are dropped by rule; the
    # caller records the dropped count (see step record), never silently.
    measured.sort(key=lambda item: (item[0], item[1]))
    mask = [0.0] * OBSTACLE_CAPACITY
    for slot in range(min(len(measured), OBSTACLE_CAPACITY)):
        mask[slot] = 1.0

    walls = (
        _clip(_finite(ego.px - extent.x_min) / MAP_HALF_M, 0.0, 1.0),
        _clip(_finite(extent.x_max - ego.px) / MAP_HALF_M, 0.0, 1.0),
        _clip(_finite(ego.py - extent.y_min) / MAP_HALF_M, 0.0, 1.0),
        _clip(_finite(extent.y_max - ego.py) / MAP_HALF_M, 0.0, 1.0),
    )

    safety_y = 1.0 if float(task.safety_y) >= 0.5 else 0.0
    z_ratio = _clip(_finite(task.z_J) / z_ref_J, -1.0, 1.0)
    t_rem_n = _clip(_finite(task.time_remaining_s) / T_MAX_S, 0.0, 1.0)

    vec = (
        [sin_yaw, cos_yaw, vx_n, vy_n, wz_n, px_n, py_n]
        + [gdir_x, gdir_y, gdist_n]
        + bins
        + list(walls)
        + mask
        + [safety_y, z_ratio, t_rem_n]
    )
    assert len(vec) == OBS_DIM, (len(vec), OBS_DIM)
    return vec


def encode_from_physical(
    *,
    base_pos_xy: Sequence[float],
    base_quat_wxyz: Sequence[float],
    lin_vel_body: Sequence[float],
    ang_vel_body: Sequence[float],
    obstacles: Iterable[Obstacle],
    extent: MapExtent,
    task: TaskState,
    z_ref_J: float = Z_REF_J,
) -> list[float]:
    """Encode from raw simulator quantities through the same encoder.

    ``base_pos_xy`` is the base_link origin projected onto the world plane;
    ``base_quat_wxyz`` follows :data:`camra_navigation.frames.QUAT_ORDER`.  Yaw is
    the projected heading, so nonzero roll/pitch does not break the encoding.
    """
    yaw = yaw_from_quat(base_quat_wxyz)
    ego = EgoState(
        px=float(base_pos_xy[0]),
        py=float(base_pos_xy[1]),
        yaw=yaw,
        vx_body=float(lin_vel_body[0]),
        vy_body=float(lin_vel_body[1]),
        wz_body=float(ang_vel_body[2]),
    )
    return encode_observation(ego, obstacles, extent, task, z_ref_J=z_ref_J)


# --- action pipeline --------------------------------------------------------
def action_to_requested(
    action: Sequence[float],
) -> tuple[tuple[float, float, float], bool, bool]:
    """Map a normalised action ``a in [-1, 1]^3`` to a requested command.

    The mapping is affine, ``u_req = clip(a, -1, 1) * COMMAND_MAX``, not
    ``tanh``: the validated domain must be exactly reachable, and NAV-059 probes
    speeds close to the lower end where a squashing nonlinearity would distort
    the commanded value.  Boundedness inside the network is a network concern,
    not part of this contract.

    Returns ``(requested, clipped, invalid)``.  ``invalid`` is separate from
    ``clipped`` because a non-finite action is not "out of range": it means the
    network produced garbage, which must reach the run log as such.
    """
    vals: list[float] = []
    clipped = False
    invalid = False
    for a in action:
        af = float(a)
        if not math.isfinite(af):
            af = 0.0
            invalid = True
            clipped = True
        elif not -1.0 <= af <= 1.0:
            clipped = True
        vals.append(_clip(af, -1.0, 1.0))
    return (
        (vals[0] * COMMAND_MAX[0], vals[1] * COMMAND_MAX[1], vals[2] * COMMAND_MAX[2]),
        clipped,
        invalid,
    )


def clip_applied(requested: Sequence[float]) -> tuple[tuple[float, float, float], bool, bool]:
    """Clip a requested command to the validated domain, once.

    Returns ``(applied, clipped, invalid)``.  A non-finite request is replaced by
    zeros and flagged ``invalid``; it is never propagated.
    """
    applied: list[float] = []
    invalid = False
    clipped = False
    for value, hi in zip(requested, COMMAND_MAX):
        v = float(value)
        if not math.isfinite(v):
            v = 0.0
            invalid = True
            clipped = True
        elif abs(v) > hi:
            clipped = True
        applied.append(_clip(v, -hi, hi))
    return ((applied[0], applied[1], applied[2]), clipped, invalid)


def command_from_action(action: Sequence[float]) -> CommandRecord:
    """Full action -> requested -> applied pipeline, returning the record."""
    requested, action_clipped, action_invalid = action_to_requested(action)
    applied, applied_clipped, applied_invalid = clip_applied(requested)
    return CommandRecord(
        requested=requested,
        applied=applied,
        clipped=action_clipped or applied_clipped,
        invalid=action_invalid or applied_invalid,
    )
