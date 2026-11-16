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
    "SUPPORTED_OBS_VERSIONS",
    "Z_RATIO_CLIP_BY_VERSION",
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
    "z_ratio_value",
    "z_ratio_clip",
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
#:
#: ``obs-v3`` (NAV-062) changes exactly one constant: the ``z_ratio`` clip widens
#: from +-1.0 to :data:`Z_RATIO_CLIP` (1.5).  Reason, measured rather than
#: assumed: the training budget used to be a single k_train, while the evaluated
#: and sampled budgets span 0.6-1.3 x ``E_ref``; with ``E_ref`` up to 489.4 J the
#: ratio reaches 1.16 > 1.0, so the observation reported "no budget left" while
#: the internal budget was still positive, and ``W`` (built on the clipped value)
#: stopped being additive.  Widening the clip fixes the encoding; the internal
#: budget stays unclipped and is what the objective uses.
#:
#: Legacy encodings stay reconstructible: :func:`encode_observation` takes
#: ``version=`` and every version keeps its own clip, so ``obs-v2`` can be
#: replayed bit-for-bit on the same state.
OBS_VERSION = "obs-v5"
#: Observation versions this module can still encode (old reports stay checkable).
SUPPORTED_OBS_VERSIONS: tuple[str, ...] = ("obs-v2", "obs-v3", "obs-v4", "obs-v5", "obs-v6")
#: ``|z_ratio|`` upper bound per observation version.  Not a budget: a scale.
#:
#: NAV-064 r14 (obs-v4, D9 fix, user-authorized 2026-11-15T17:32): every scene
#: has ``E_ref > Z_REF`` so under obs-v3 the cost channel saturated for the critic
#: before the robot ever overspent -- ``V_c - z`` could not be learned near z=0.
#: Clip 3.0 covers the full training band ``k in [0.6, 1.3]`` plus the overspend
#: excursions actually observed (NAV-064: z down to -1.6 during a 30 s run).
Z_RATIO_CLIP_BY_VERSION: dict[str, float] = {
    "obs-v2": 1.0, "obs-v3": 1.5, "obs-v4": 3.0, "obs-v5": 3.0, "obs-v6": 3.0,
}
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
# Explicit opt-in: old checkpoints keep their original 30-second encoding.
# NAV-064's task deadline is 45 seconds; v6 represents the full remaining time.
TIME_REF_S_BY_VERSION = {v: (45.0 if v == "obs-v6" else T_MAX_S)
                         for v in SUPPORTED_OBS_VERSIONS}
#: Reference speeds for the measured body velocity, metres/second, rad/second.
V_REF_MPS = 1.0
W_REF_RADPS = 1.0
#: Body velocity is clipped to this many multiples of the reference speed.
V_BODY_CLIP = 2.0
#: Reference budget for ``z_ratio``, joules.
#:
#: NAV-035 replaced the original value (225.6069087767049 J = the 30 s energy at
#: the analytically energy-optimal speed 0.30986 m/s) with 550 J.  The old scale
#: was below the reference route energy of *every* scenario (362-489 J, see
#: reports/budget_fairness_check.json), and ``z_ratio`` has to stay unsaturated
#: for the residual to mean anything -- so no policy could ever be judged
#: feasible, the observation blinded the policy to its own budget for the first
#: part of each episode, and FAIL_PENALTY (1.0) ended up cheaper than a
#: successful route (1.61-2.05 on the old scale).
#:
#: 550 J = 1.05 x the largest reference energy (489.40 J, L2) rounded up: the
#: hardest route sits at 0.89 of the scale and reference costs span 0.66-0.89,
#: all strictly below FAIL_PENALTY.  It is a *scale*, not a task budget: the
#: per-episode budget ``z0_J`` stays a task parameter chosen elsewhere.
Z_REF_J = 550.0

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


def z_ratio_clip(version: str | None = None) -> float:
    """``|z_ratio|`` bound of one observation version (default: current)."""
    v = OBS_VERSION if version is None else str(version)
    if v not in Z_RATIO_CLIP_BY_VERSION:
        raise ValueError(
            f"unknown observation version {v!r}; supported: {SUPPORTED_OBS_VERSIONS}")
    return Z_RATIO_CLIP_BY_VERSION[v]


def z_ratio_value(z_J: float, z_ref_J: float = Z_REF_J, *,
                  version: str | None = None) -> float:
    """The encoded ``z_ratio`` of a signed budget (clip is version-specific).

    Single implementation for the encoder, the environment records and the
    training buffer.  The *internal* budget is never clipped; only this encoded
    number is, and the objective must use the unclipped quotient instead.
    """
    lim = z_ratio_clip(version)
    return _clip(_finite(z_J) / float(z_ref_J), -lim, lim)


# --- encoder ----------------------------------------------------------------
def encode_observation(
    ego: EgoState,
    obstacles: Iterable[Obstacle],
    extent: MapExtent,
    task: TaskState,
    *,
    z_ref_J: float = Z_REF_J,
    version: str | None = None,
) -> list[float]:
    """Encode one state into the frozen observation vector.

    The same function is called by the nominal environment and by the Isaac Lab
    environment.  Order of ``obstacles`` does not affect the result.

    ``version`` selects the observation version (default :data:`OBS_VERSION`);
    the only thing it changes today is the ``z_ratio`` clip
    (:func:`z_ratio_clip`), so an ``obs-v2`` vector can be reproduced exactly.
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

    # v5 preserves the graded, irreversible history used by the objective.
    # Earlier versions retain their original threshold for checkpoint replay.
    selected_version = OBS_VERSION if version is None else version
    safety_y = (_clip(_finite(task.safety_y), 0.0, 1.0)
                if selected_version in ("obs-v5", "obs-v6")
                else (1.0 if float(task.safety_y) >= 0.5 else 0.0))
    z_ratio = z_ratio_value(task.z_J, z_ref_J, version=version)
    time_ref_s = TIME_REF_S_BY_VERSION[selected_version]
    if selected_version == "obs-v6" and (
            not math.isfinite(float(task.time_remaining_s))
            or float(task.time_remaining_s) > time_ref_s + 1e-9):
        raise ValueError("obs-v6 requires finite remaining time <= 45 seconds")
    t_rem_n = _clip(_finite(task.time_remaining_s) / time_ref_s, 0.0, 1.0)

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
    version: str | None = None,
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
    return encode_observation(ego, obstacles, extent, task, z_ref_J=z_ref_J,
                              version=version)


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
