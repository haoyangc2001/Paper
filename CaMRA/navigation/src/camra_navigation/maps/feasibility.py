"""Inflated-map connectivity and deadline feasibility (NAV-017).

Admission is deliberately conservative about what it proves:

* the grid is used **only to search**.  Every candidate route is then verified
  continuously: dense sampling against the inflated geometry, the physical
  footprint against the raw obstacles, and a swept check between consecutive
  samples.  A grid path that clears the grid but clips a corner is rejected.
* the deadline test separates two failure modes.  If the route is too long even
  at ``v_max`` it is a plain ``timeout``; if it would fit at ``v_max`` but only
  at a speed where the tightest turn exceeds ``wz_max``, it is
  ``turn_limited``.  Reporting both matters because the second one looks fine
  until you notice the turn rate.
* nothing here is a measurement.  Times and energies are nominal predictions
  from the frozen power model; whether the frozen v2 actually tracks the
  reference profile is NAV-028/NAV-059's question, and the field
  ``physical_status`` says so on every record.

A map that fails is not "fixed" by shrinking ``delta``; the failure is recorded
with its reason.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from camra_calibration.power import PowerModel

from ..frames import wrap_angle
from .generator import MapSpec, reference_commands, route_energy
from .geometry import (
    SafetyGeometry,
    boundary_margin,
    nominal_margin,
    physical_clearance,
    physical_collision_during_step,
)

__all__ = [
    "FreeSpace",
    "REASONS",
    "build_free_space",
    "grid_path",
    "shortcut",
    "path_stats",
    "max_feasible_speed",
    "evaluate_path",
    "check_map",
    "check_catalogue",
]

#: Rejection reasons, all recorded verbatim in the report.
REASONS = (
    "start_outside_free",
    "goal_outside_free",
    "no_connected_path",
    "corner_clip_after_shortcut",
    "sweep_collision",
    "timeout",
    "turn_limited",
)


# ---------------------------------------------------------------- free space
@dataclass(frozen=True)
class FreeSpace:
    """Boolean occupancy of the *nominal constraint* on a regular grid."""

    x_min: float
    y_min: float
    resolution: float
    mask: np.ndarray  # (ny, nx) True = usable

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    def index(self, x: float, y: float) -> tuple[int, int] | None:
        ix = int(round((x - self.x_min) / self.resolution))
        iy = int(round((y - self.y_min) / self.resolution))
        ny, nx = self.mask.shape
        if 0 <= ix < nx and 0 <= iy < ny:
            return (iy, ix)
        return None

    def xy(self, iy: int, ix: int) -> tuple[float, float]:
        return (self.x_min + ix * self.resolution, self.y_min + iy * self.resolution)

    def is_free(self, x: float, y: float) -> bool:
        idx = self.index(x, y)
        return bool(idx is not None and self.mask[idx])


def build_free_space(
    spec: MapSpec, geometry: SafetyGeometry, resolution: float = 0.1
) -> FreeSpace:
    """Rasterise the nominal constraint; ``delta`` is applied once, inside it."""
    x_min, x_max, y_min, y_max = spec.extent
    xs = np.arange(x_min, x_max + 1e-9, resolution)
    ys = np.arange(y_min, y_max + 1e-9, resolution)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    margin = np.full(grid.shape[0], np.inf)
    for ob in spec.obstacles():
        margin = np.minimum(margin, ob.polygon().distance(grid) - geometry.inflation_radius_m)
    wall = np.min(
        np.stack(
            [
                grid[:, 0] - x_min,
                x_max - grid[:, 0],
                grid[:, 1] - y_min,
                y_max - grid[:, 1],
            ]
        ),
        axis=0,
    ) - geometry.inflation_radius_m
    free = (margin > 0.0) & (wall > 0.0)
    return FreeSpace(x_min=float(xs[0]), y_min=float(ys[0]), resolution=float(resolution),
                     mask=free.reshape(len(ys), len(xs)))


# ---------------------------------------------------------------- search
_NEIGHBOURS = (
    (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)), (1, 0, 1.0), (1, 1, math.sqrt(2.0)),
)


def grid_path(
    fs: FreeSpace, start: Sequence[float], goal: Sequence[float]
) -> list[tuple[float, float]] | None:
    """Shortest 8-connected path in cell units; ``None`` if disconnected.

    The grid is a search device only.  Its staircase is later replaced by a
    shortcut polyline and verified continuously.
    """
    s = fs.index(*start)
    g = fs.index(*goal)
    if s is None or g is None or not fs.mask[s] or not fs.mask[g]:
        return None
    ny, nx = fs.shape
    dist = {s: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    heap = [(0.0, s)]
    while heap:
        d, node = heapq.heappop(heap)
        if node == g:
            break
        if d > dist.get(node, math.inf):
            continue
        iy, ix = node
        for dy, dx, step in _NEIGHBOURS:
            jy, jx = iy + dy, ix + dx
            if not (0 <= jy < ny and 0 <= jx < nx) or not fs.mask[jy, jx]:
                continue
            nxt = (jy, jx)
            nd = d + step
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(heap, (nd, nxt))
    if g not in dist:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    cells.reverse()
    return [fs.xy(iy, ix) for iy, ix in cells]


def _segment_clear(
    a: Sequence[float], b: Sequence[float], spec: MapSpec, geometry: SafetyGeometry,
    step_m: float = 0.05,
) -> bool:
    """Continuous clearance test along a straight segment."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(math.ceil(length / step_m)) + 1)
    obstacles = spec.obstacles()
    for k in range(n):
        t = k / (n - 1)
        p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if boundary_margin(p, spec.extent, geometry) <= 0.0:
            return False
        if any(nominal_margin(p, ob.polygon(), geometry) <= 0.0 for ob in obstacles):
            return False
    return True


def shortcut(
    path: Sequence[Sequence[float]], spec: MapSpec, geometry: SafetyGeometry
) -> list[tuple[float, float]]:
    """Greedy line-of-sight shortening, each shortcut verified continuously."""
    pts = [tuple(float(v) for v in p) for p in path]
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_clear(pts[i], pts[j], spec, geometry):
            j -= 1
        out.append(pts[j])
        i = j
    return out


# ---------------------------------------------------------------- path math
def path_stats(polyline: Sequence[Sequence[float]]) -> dict:
    """Length, per-vertex turn and the curvature that limits the speed."""
    pts = [tuple(float(v) for v in p) for p in polyline]
    segs = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:])]
    length = float(sum(segs))
    headings = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(pts[:-1], pts[1:])]
    turns = [
        abs(wrap_angle(headings[i + 1] - headings[i])) for i in range(len(headings) - 1)
    ]
    total_turn = float(sum(turns))
    # Discrete curvature at interior vertex i+1 uses half the adjacent segments.
    kappas = []
    for i, turn in enumerate(turns):
        span = 0.5 * (segs[i] + segs[i + 1])
        kappas.append(turn / span if span > 1e-9 else math.inf)
    kappa_max = float(max(kappas)) if kappas else 0.0
    return {
        "length_m": length,
        "n_vertices": len(pts),
        "total_turn_rad": total_turn,
        "kappa_max_per_m": kappa_max,
        "segments_m": segs,
    }


def max_feasible_speed(stats: dict, v_max: float, wz_max: float) -> float:
    """Largest cruise speed at which the tightest turn stays inside ``wz_max``."""
    kappa = stats["kappa_max_per_m"]
    if not math.isfinite(kappa):
        return 0.0
    if kappa <= 1e-9:
        return v_max
    return min(v_max, wz_max / kappa)


# ---------------------------------------------------------------- evaluation
def evaluate_path(
    spec: MapSpec,
    geometry: SafetyGeometry,
    polyline: Sequence[Sequence[float]],
    model: PowerModel | None = None,
    *,
    v_max_mps: float = 1.0,
    wz_max_radps: float = 1.0,
    t_deadline_s: float = 30.0,
    dt_s: float = 0.1,
    sample_step_m: float = 0.1,
) -> dict:
    """One candidate route: clearance, sweep, commands, time and energy."""
    pts = [tuple(float(v) for v in p) for p in polyline]
    stats = path_stats(pts)
    v_cap = max_feasible_speed(stats, v_max_mps, wz_max_radps)
    length = stats["length_m"]
    t_min = length / v_cap if v_cap > 0.0 else math.inf
    v_deadline = length / t_deadline_s if t_deadline_s > 0.0 else math.inf
    feasible_in_time = v_cap > 0.0 and v_deadline <= v_cap

    # dense continuous verification of the *polyline* itself
    samples = []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(seg / sample_step_m)))
        for k in range(n):
            t = k / n
            samples.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    samples.append(pts[-1])

    obstacles = spec.obstacles()
    nominal_min = (
        min(min(nominal_margin(p, ob.polygon(), geometry) for ob in obstacles) for p in samples)
        if obstacles else None
    )
    boundary_min = min(boundary_margin(p, spec.extent, geometry) for p in samples)
    physical_min = (
        min(min(physical_clearance(p, 0.0, ob.polygon(), geometry) for ob in obstacles)
            for p in samples)
        if obstacles else None
    )

    # heading along the route, then a swept check between consecutive samples
    sweep_hits = 0
    if obstacles:
        heads = []
        for a, b in zip(samples[:-1], samples[1:]):
            heads.append(math.atan2(b[1] - a[1], b[0] - a[0]))
        heads.append(heads[-1] if heads else 0.0)
        for k in range(len(samples) - 1):
            for ob in obstacles:
                if physical_collision_during_step(
                    samples[k], heads[k], samples[k + 1], heads[k + 1],
                    ob.polygon(), geometry,
                ):
                    sweep_hits += 1
                    break

    # Two prices, both useful: the fastest run costs the most, the deadline-speed
    # run is the cheapest that still arrives in time.  Neither is a measurement.
    commands = None
    energy_fast = None
    energy_deadline = None
    if model is not None and v_cap > 0.0:
        commands = reference_commands(spec, pts, v_cap, dt_s)
        energy_fast = route_energy(commands, model)
        if feasible_in_time:
            energy_deadline = route_energy(
                reference_commands(spec, pts, min(v_deadline, v_max_mps), dt_s), model
            )

    reasons: list[str] = []
    if (nominal_min is not None and nominal_min <= 0.0) or boundary_min <= 0.0:
        reasons.append("corner_clip_after_shortcut")
    if sweep_hits:
        reasons.append("sweep_collision")
    if not feasible_in_time:
        if length / v_max_mps <= t_deadline_s:
            reasons.append("turn_limited")
        else:
            reasons.append("timeout")

    return {
        "polyline": [[p[0], p[1]] for p in pts],
        "length_m": stats["length_m"],
        "n_vertices": stats["n_vertices"],
        "total_turn_rad": stats["total_turn_rad"],
        "kappa_max_per_m": stats["kappa_max_per_m"],
        "v_cap_mps": v_cap,
        "v_deadline_mps": v_deadline,
        "t_min_s": t_min,
        "t_at_vmax_s": length / v_max_mps if v_max_mps > 0.0 else math.inf,
        "t_turn_only_s": stats["total_turn_rad"] / wz_max_radps,
        "min_nominal_margin_m": nominal_min,
        "min_boundary_margin_m": boundary_min,
        "min_physical_clearance_m": physical_min,
        "sweep_collisions": sweep_hits,
        "n_samples": len(samples),
        "n_commands": 0 if commands is None else int(commands.shape[0]),
        "feasible_in_time": feasible_in_time,
        "predicted_energy_fast_J": energy_fast,
        "predicted_energy_deadline_J": energy_deadline,
        "physical_status": "not_measured",
        "reasons": reasons,
        "admitted": not reasons,
    }


def check_map(
    spec: MapSpec,
    geometry: SafetyGeometry,
    model: PowerModel | None = None,
    *,
    resolution: float = 0.1,
    **kwargs,
) -> dict:
    """Full admission record for one map."""
    x_min, x_max, y_min, y_max = spec.extent
    fs = build_free_space(spec, geometry, resolution)
    obstacles = spec.obstacles()

    start_reasons: list[str] = []
    goal_reasons: list[str] = []
    for label, pt, bucket in (
        ("start", spec.start, start_reasons),
        ("goal", spec.goal, goal_reasons),
    ):
        if boundary_margin(pt, spec.extent, geometry) <= 0.0:
            bucket.append(f"{label}_outside_free")
        if any(nominal_margin(pt, ob.polygon(), geometry) <= 0.0 for ob in obstacles):
            bucket.append(f"{label}_outside_free")

    path = grid_path(fs, spec.start, spec.goal) if not start_reasons and not goal_reasons else None
    route = None
    if path is not None:
        smoothed = shortcut(path, spec, geometry)
        route = evaluate_path(spec, geometry, smoothed, model, **kwargs)

    reasons = list(dict.fromkeys(start_reasons + goal_reasons))
    if not start_reasons and not goal_reasons:
        if path is None:
            reasons.append("no_connected_path")
        elif route is not None:
            reasons.extend(route["reasons"])

    admitted = not reasons
    return {
        "map_id": spec.map_id,
        "level": spec.level,
        "seed": spec.seed,
        "geometry_hash": spec.geometry_hash(),
        "extent": list(spec.extent),
        "free_cells": int(fs.mask.sum()),
        "free_fraction": float(fs.mask.mean()),
        "resolution_m": resolution,
        "grid_connected": path is not None,
        "start_margin_m": min(
            [nominal_margin(spec.start, ob.polygon(), geometry) for ob in obstacles]
        ) if obstacles else None,
        "goal_margin_m": min(
            [nominal_margin(spec.goal, ob.polygon(), geometry) for ob in obstacles]
        ) if obstacles else None,
        "route": route,
        "reasons": reasons,
        "admitted": admitted,
        "physical_status": "not_measured",
        "note": "times and energies are nominal predictions from the frozen power "
                "model, not measurements; the grid is only a search device",
    }


def check_catalogue(
    specs: Iterable[MapSpec],
    geometry: SafetyGeometry,
    model: PowerModel | None = None,
    **kwargs,
) -> list[dict]:
    return [check_map(spec, geometry, model, **kwargs) for spec in specs]
