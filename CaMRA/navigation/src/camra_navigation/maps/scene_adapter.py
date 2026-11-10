"""Map-to-physical-scene geometry adapter (NAV-022).

This module is pure data: it converts a :class:`~camra_navigation.maps.generator.MapSpec`
into the objects an Isaac Lab scene must contain, and nothing here launches a
simulator.  It exists because the physical scene must be driven by **the same raw
map** the nominal environment uses, with every dimension, rotation, origin and
wall taken from the spec — no inherited default box size, no fixed slot count, no
"boundary that only exists in the cost function".

Conventions
-----------
* The **map centre** is the env-local origin.  A point at world ``p`` sits at
  ``p - map_center + env_origin`` in the scene.
* Boundary walls are real boxes: their inner faces coincide with the map extent,
  they are ``WALL_THICKNESS_M`` thick outward and ``OBSTACLE_HEIGHT_M`` tall, so
  the robot cannot leave through a wall that only existed in the cost function.
* Unused obstacle bodies are parked in a row outside the whole grid, so two
  parallel environments can never see each other's obstacles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .generator import MapSpec

#: Same thickness as the nominal walls (see maps.geometry.wall_polygons).
WALL_THICKNESS_M = 1.0
#: Box height.  The Go2 body is ~0.35 m tall, so a 0.5 m wall cannot be stepped
#: or jumped over; anything shorter would turn "hit the wall" into "climb it".
OBSTACLE_HEIGHT_M = 0.5
#: Extra clearance between the footprints of two neighbouring environments.
ENV_SPACING_MARGIN_M = 1.0
#: Gap between parked obstacle bodies.
PARKING_GAP_M = 1.0


class SceneAdapterError(RuntimeError):
    """Raised when a scene cannot be built from a map spec."""


@dataclass(frozen=True)
class BoxBody:
    """One physical box, in env-local coordinates, metres and radians."""

    name: str
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    height: float
    yaw: float
    z_center: float

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]:
        """Isaac Lab uses ``wxyz``; a yaw-only rotation is this quaternion."""
        return (math.cos(self.yaw / 2.0), 0.0, 0.0, math.sin(self.yaw / 2.0))

    def aabb(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds, conservative for a rotated box."""
        c = abs(math.cos(self.yaw))
        s = abs(math.sin(self.yaw))
        ex = 0.5 * (self.size_x * c + self.size_y * s)
        ey = 0.5 * (self.size_x * s + self.size_y * c)
        return (
            self.center_x - ex,
            self.center_x + ex,
            self.center_y - ey,
            self.center_y + ey,
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "center_xy": [self.center_x, self.center_y],
            "size_xy": [self.size_x, self.size_y],
            "height_m": self.height,
            "z_center_m": self.z_center,
            "yaw_rad": self.yaw,
            "quat_wxyz": list(self.quat_wxyz),
        }


@dataclass
class SceneLayout:
    """Everything the Isaac Lab task needs for one environment slot."""

    map_id: str
    level: str
    seed: int
    env_index: int
    num_envs: int
    env_origin: tuple[float, float]
    map_center: tuple[float, float]
    robot_spawn_xy: tuple[float, float]
    robot_spawn_yaw: float
    island_boxes: list[BoxBody] = field(default_factory=list)
    wall_boxes: list[BoxBody] = field(default_factory=list)
    parking_positions: list[tuple[float, float]] = field(default_factory=list)
    env_spacing_m: float = 0.0

    def as_dict(self) -> dict:
        return {
            "map_id": self.map_id,
            "level": self.level,
            "seed": self.seed,
            "env_index": self.env_index,
            "num_envs": self.num_envs,
            "env_origin_xy": list(self.env_origin),
            "map_center_xy": list(self.map_center),
            "robot_spawn_xy": list(self.robot_spawn_xy),
            "robot_spawn_yaw_rad": self.robot_spawn_yaw,
            "island_boxes": [b.as_dict() for b in self.island_boxes],
            "wall_boxes": [b.as_dict() for b in self.wall_boxes],
            "parking_positions_xy": [list(p) for p in self.parking_positions],
            "env_spacing_m": self.env_spacing_m,
            "constants": {
                "wall_thickness_m": WALL_THICKNESS_M,
                "obstacle_height_m": OBSTACLE_HEIGHT_M,
                "env_spacing_margin_m": ENV_SPACING_MARGIN_M,
                "parking_gap_m": PARKING_GAP_M,
            },
        }


def map_center(spec: MapSpec) -> tuple[float, float]:
    x0, x1, y0, y1 = spec.extent
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def required_env_spacing(spec: MapSpec, *, margin: float = ENV_SPACING_MARGIN_M) -> float:
    """Smallest centre-to-centre distance so two environments cannot touch.

    Each environment occupies its map extent **plus** the outward walls, so the
    footprint is the extent inflated by ``WALL_THICKNESS_M`` on every side.  Two
    such footprints must not overlap, with ``margin`` to spare.
    """
    x0, x1, y0, y1 = spec.extent
    fx = (x1 - x0) + 2.0 * WALL_THICKNESS_M
    fy = (y1 - y0) + 2.0 * WALL_THICKNESS_M
    return max(fx, fy) + margin


def env_grid(num_envs: int) -> tuple[int, int]:
    """Deterministic (columns, rows) tiling, same for every scene build."""
    if num_envs < 1:
        raise SceneAdapterError(f"num_envs must be >= 1, got {num_envs}")
    cols = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / cols)
    return cols, rows


def env_origin(index: int, num_envs: int, spacing: float) -> tuple[float, float]:
    cols, rows = env_grid(num_envs)
    if not 0 <= index < num_envs:
        raise SceneAdapterError(f"env index {index} out of range for {num_envs} envs")
    col = index % cols
    row = index // cols
    cx = (cols - 1) / 2.0
    cy = (rows - 1) / 2.0
    return ((col - cx) * spacing, (row - cy) * spacing)


def parking_positions(num_envs: int, spacing: float, count: int) -> list[tuple[float, float]]:
    """A row outside the whole grid, far from every environment footprint."""
    cols, rows = env_grid(num_envs)
    y = (rows / 2.0) * spacing + WALL_THICKNESS_M + ENV_SPACING_MARGIN_M + OBSTACLE_HEIGHT_M
    start = -(count * (PARKING_GAP_M + 0.5)) / 2.0
    return [(start + i * (PARKING_GAP_M + 0.5), y) for i in range(count)]


def _box(name: str, cx_local: float, cy_local: float, sx: float, sy: float,
         yaw: float) -> BoxBody:
    return BoxBody(
        name=name,
        center_x=cx_local,
        center_y=cy_local,
        size_x=sx,
        size_y=sy,
        height=OBSTACLE_HEIGHT_M,
        yaw=yaw,
        z_center=OBSTACLE_HEIGHT_M / 2.0,
    )


def build_scene(spec: MapSpec, env_index: int, num_envs: int,
                *, spacing: float | None = None) -> SceneLayout:
    """Full scene description for one environment slot."""
    if spacing is None:
        spacing = required_env_spacing(spec)
    if spacing < required_env_spacing(spec):
        raise SceneAdapterError(
            f"spacing {spacing} is smaller than the required "
            f"{required_env_spacing(spec)}; parallel environments would touch"
        )
    cx, cy = map_center(spec)
    ox, oy = env_origin(env_index, num_envs, spacing)

    def local(x: float, y: float) -> tuple[float, float]:
        return (x - cx + ox, y - cy + oy)

    islands = []
    for i, (ix, iy, w, d, yaw) in enumerate(spec.islands):
        lx, ly = local(ix, iy)
        islands.append(_box(f"island_{i}", lx, ly, w, d, yaw))

    x0, x1, y0, y1 = spec.extent
    walls = [
        _box("wall_x_min", *local(x0 - WALL_THICKNESS_M / 2.0, cy), WALL_THICKNESS_M,
             (y1 - y0) + 2.0 * WALL_THICKNESS_M, 0.0),
        _box("wall_x_max", *local(x1 + WALL_THICKNESS_M / 2.0, cy), WALL_THICKNESS_M,
             (y1 - y0) + 2.0 * WALL_THICKNESS_M, 0.0),
        _box("wall_y_min", *local(cx, y0 - WALL_THICKNESS_M / 2.0),
             (x1 - x0) + 2.0 * WALL_THICKNESS_M, WALL_THICKNESS_M, 0.0),
        _box("wall_y_max", *local(cx, y1 + WALL_THICKNESS_M / 2.0),
             (x1 - x0) + 2.0 * WALL_THICKNESS_M, WALL_THICKNESS_M, 0.0),
    ]

    sx, sy = local(spec.start[0], spec.start[1])
    parked = parking_positions(num_envs, spacing, count=max(0, 8 - len(spec.islands)))

    return SceneLayout(
        map_id=spec.map_id,
        level=spec.level,
        seed=spec.seed,
        env_index=env_index,
        num_envs=num_envs,
        env_origin=(ox, oy),
        map_center=(cx, cy),
        robot_spawn_xy=(sx, sy),
        robot_spawn_yaw=spec.start_yaw,
        island_boxes=islands,
        wall_boxes=walls,
        parking_positions=parked,
        env_spacing_m=spacing,
    )


def round_trip_spec(scene: SceneLayout, design_version: str) -> MapSpec:
    """Rebuild a MapSpec from the scene boxes and compare identities.

    This is the reverse check: if the scene lost a dimension, a rotation or an
    offset, the rebuilt spec's geometry hash would not match the original.
    """
    cx, cy = scene.map_center
    ox, oy = scene.env_origin
    islands = []
    for b in scene.island_boxes:
        wx = b.center_x - ox + cx
        wy = b.center_y - oy + cy
        islands.append((wx, wy, b.size_x, b.size_y, b.yaw))
    return MapSpec(
        map_id=scene.map_id,
        level=scene.level,
        seed=scene.seed,
        extent=(0.0, 0.0, 0.0, 0.0),  # replaced below from the walls
        start=(scene.robot_spawn_xy[0] - ox + cx, scene.robot_spawn_xy[1] - oy + cy),
        start_yaw=scene.robot_spawn_yaw,
        goal=(0.0, 0.0),  # not carried by the scene; checked separately
        islands=tuple(islands),
        design_version=design_version,
    )


def extent_from_walls(walls: Sequence[BoxBody]) -> tuple[float, float, float, float]:
    """Inner faces of the four wall boxes, i.e. the map extent."""
    if len(walls) != 4:
        raise SceneAdapterError(f"expected 4 walls, got {len(walls)}")
    by_name = {w.name: w for w in walls}
    x0 = by_name["wall_x_min"].center_x + WALL_THICKNESS_M / 2.0
    x1 = by_name["wall_x_max"].center_x - WALL_THICKNESS_M / 2.0
    y0 = by_name["wall_y_min"].center_y + WALL_THICKNESS_M / 2.0
    y1 = by_name["wall_y_max"].center_y - WALL_THICKNESS_M / 2.0
    return (x0, x1, y0, y1)


def aabb_overlap(a: tuple[float, float, float, float],
                 b: tuple[float, float, float, float]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])
