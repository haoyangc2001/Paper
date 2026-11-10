"""Wide-corridor graded map generator (NAV-016).

Builds the three levels frozen in ``config/map_design_v1.json`` from one raw
geometry description, so the nominal constraint and the physical obstacles read
exactly the same numbers:

* nominal side: :class:`~camra_navigation.maps.geometry.SafetyGeometry` inflates
  each island and the four walls by ``r_fp + delta``;
* physical side: the frozen footprint polygon against the raw island polygons.

Levels
------
``L1_empty``        a 7.6 m straight corridor: cost accounting with no detour.
``L2_single_detour``a 16.4 m hall with one central island, two equal 7.2 m gaps.
``L3_multi_path``   two staggered gates, four gap combinations with different
                    lengths and different turn counts.

Gaps are computed from the obstacle list by slicing the field at each gate
column, not copied from the design intent, and every route candidate is checked
against the inflated geometry before it is reported.  Route length is a polyline
through gap centres; it is labelled as such, not as a planner output.  Energy is
computed with the frozen ``power-v1`` model on a feed-forward reference profile,
so shortest distance is never assumed to be the cheapest.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from camra_calibration.obstacles import Obstacle
from camra_calibration.power import PowerModel

from ..frames import wrap_angle
from .geometry import SafetyGeometry, boundary_margin, boundary_violation, nominal_margin, nominal_violation

__all__ = [
    "LEVELS",
    "MapSpec",
    "MapError",
    "load_design",
    "build",
    "free_intervals_at_x",
    "gate_columns",
    "routes",
    "reference_commands",
    "route_energy",
    "evaluate",
    "catalogue",
    "render_svg",
]

LEVELS = ("L1_empty", "L2_single_detour", "L3_multi_path")


class MapError(RuntimeError):
    """Map definition or validation failed."""


# ---------------------------------------------------------------- spec
@dataclass(frozen=True)
class MapSpec:
    """One task instance: fixed field geometry plus a start/goal pair."""

    map_id: str
    level: str
    seed: int
    extent: tuple[float, float, float, float]
    start: tuple[float, float]
    start_yaw: float
    goal: tuple[float, float]
    islands: tuple[tuple[float, float, float, float, float], ...]  # x, y, width, depth, yaw
    design_version: str

    # ---- conversion ----
    def obstacles(self) -> tuple[Obstacle, ...]:
        return tuple(Obstacle(x, y, w, d, yaw) for x, y, w, d, yaw in self.islands)

    def to_dict(self) -> dict:
        return {
            "map_id": self.map_id,
            "level": self.level,
            "seed": self.seed,
            "extent": list(self.extent),
            "start": list(self.start),
            "start_yaw": self.start_yaw,
            "goal": list(self.goal),
            "islands": [
                {"x": x, "y": y, "width": w, "depth": d, "yaw": yaw}
                for x, y, w, d, yaw in self.islands
            ],
            "design_version": self.design_version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MapSpec":
        return cls(
            map_id=str(payload["map_id"]),
            level=str(payload["level"]),
            seed=int(payload["seed"]),
            extent=tuple(float(v) for v in payload["extent"]),
            start=tuple(float(v) for v in payload["start"]),
            start_yaw=float(payload["start_yaw"]),
            goal=tuple(float(v) for v in payload["goal"]),
            islands=tuple(
                (float(o["x"]), float(o["y"]), float(o["width"]),
                 float(o["depth"]), float(o.get("yaw", 0.0)))
                for o in payload["islands"]
            ),
            design_version=str(payload.get("design_version", "")),
        )

    # ---- identity ----
    def geometry_dict(self) -> dict:
        """Geometry only, no ``map_id`` or seed label: what actually matters."""
        return {
            "extent": [round(v, 9) for v in self.extent],
            "start": [round(v, 9) for v in self.start],
            "start_yaw": round(self.start_yaw, 12),
            "goal": [round(v, 9) for v in self.goal],
            "islands": sorted(
                [[round(v, 9) for v in island] for island in self.islands]
            ),
        }

    def geometry_hash(self) -> str:
        payload = json.dumps(self.geometry_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------- design
def load_design(path: str) -> dict:
    design = json.load(open(path, encoding="utf-8"))
    if design.get("schema") != "camra-navigation-map-design-v1":
        raise MapError(f"unexpected design schema {design.get('schema')!r}")
    known = {lv["level"] for lv in design["levels"]}
    if known != set(LEVELS):
        raise MapError(f"design levels {sorted(known)} do not match {sorted(LEVELS)}")
    return design


def _level_entry(design: dict, level: str) -> dict:
    for entry in design["levels"]:
        if entry["level"] == level:
            return entry
    raise MapError(f"unknown level {level!r}")


def build(level: str, seed: int, design: dict) -> MapSpec:
    """Deterministic map for ``(level, seed)``.

    The seed perturbs only the start/goal and the initial heading: moving an island
    could make an instance impassable, which is a property of the field, not of the
    task instance.
    """
    entry = _level_entry(design, level)
    jitter = design["seeds"]["jitter"]
    rng = random.Random(f"{level}:{seed}")
    sx, sy = entry["start"]
    gx, gy = entry["goal"]
    start = (
        float(sx),
        float(sy + rng.uniform(-jitter["start_y_m"], jitter["start_y_m"])),
    )
    goal = (
        float(gx),
        float(gy + rng.uniform(-jitter["goal_y_m"], jitter["goal_y_m"])),
    )
    yaw = float(entry.get("start_yaw", 0.0)) + rng.uniform(
        -jitter["start_yaw_rad"], jitter["start_yaw_rad"]
    )
    islands = tuple(
        (float(o["x"]), float(o["y"]), float(o["width"]), float(o["depth"]),
         float(o.get("yaw", 0.0)))
        for o in entry["islands"]
    )
    return MapSpec(
        map_id=f"{level.lower()}-s{seed}",
        level=level,
        seed=int(seed),
        extent=tuple(float(v) for v in entry["extent"]),
        start=start,
        start_yaw=yaw,
        goal=goal,
        islands=islands,
        design_version=str(design["version"]),
    )


# ---------------------------------------------------------------- gaps
def _y_span_at_x(poly_vertices: np.ndarray, x: float) -> tuple[float, float] | None:
    """Vertical extent of a convex polygon at ``x`` (None if the line misses it)."""
    ys: list[float] = []
    v = poly_vertices
    n = v.shape[0]
    for i in range(n):
        a = v[i]
        b = v[(i + 1) % n]
        if abs(b[0] - a[0]) < 1e-15:
            if abs(a[0] - x) < 1e-12:
                ys.extend([float(a[1]), float(b[1])])
            continue
        if (a[0] - x) * (b[0] - x) <= 0.0:
            t = (x - a[0]) / (b[0] - a[0])
            ys.append(float(a[1] + t * (b[1] - a[1])))
    if not ys:
        return None
    return (min(ys), max(ys))


def free_intervals_at_x(
    spec: MapSpec, x: float
) -> list[tuple[float, float]]:
    """Free ``y`` intervals at column ``x``, computed from the obstacle polygons."""
    _, _, y_min, y_max = spec.extent
    blocked: list[tuple[float, float]] = []
    for ob in spec.obstacles():
        span = _y_span_at_x(ob.polygon().vertices, x)
        if span is not None:
            blocked.append(span)
    blocked.sort()
    merged: list[list[float]] = []
    for lo, hi in blocked:
        if merged and lo <= merged[-1][1] + 1e-12:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    gaps: list[tuple[float, float]] = []
    cursor = y_min
    for lo, hi in merged:
        if lo > cursor + 1e-12:
            gaps.append((cursor, lo))
        cursor = max(cursor, hi)
    if y_max > cursor + 1e-12:
        gaps.append((cursor, y_max))
    return gaps


def gate_columns(spec: MapSpec, tol: float = 1e-6) -> list[float]:
    """Distinct island ``x`` centres, ordered along the corridor."""
    xs: list[float] = []
    for x, _y, _w, _d, _yaw in spec.islands:
        if not any(abs(x - seen) < tol for seen in xs):
            xs.append(x)
    return sorted(xs)


# ---------------------------------------------------------------- routes
def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def _sample_polyline(points: Sequence[Sequence[float]], step_m: float) -> np.ndarray:
    out: list[list[float]] = []
    for a, b in zip(points[:-1], points[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(seg / step_m)))
        for k in range(n):
            t = k / n
            out.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
    out.append([points[-1][0], points[-1][1]])
    return np.asarray(out, dtype=np.float64)


def routes(
    spec: MapSpec, geometry: SafetyGeometry, *, sample_step_m: float = 0.1
) -> list[dict]:
    """All gap combinations, keeping only those that clear the inflated geometry.

    A route is a polyline ``start -> [gate near edge, gate far edge]... -> goal``.
    The two edge points per gate matter: the inflated island extends
    ``r_fp + delta`` in ``x`` as well, so a straight line from the start to a gap
    centre would cut the inflated corner.  They are placed just outside the
    inflated ``x`` extent of the gate, at the gap centre's ``y``.
    """
    columns = gate_columns(spec)
    inflation = geometry.inflation_radius_m
    if not columns:
        combos: list[list[tuple[float, float]]] = [[]]
    else:
        per_column = [free_intervals_at_x(spec, x) for x in columns]
        if any(not gaps for gaps in per_column):
            return []
        combos = [[]]
        for gaps in per_column:
            combos = [prev + [gap] for prev in combos for gap in gaps]

    half_widths = {
        x: max(
            (w / 2.0 for ix, _iy, w, _d, _yaw in spec.islands if abs(ix - x) < 1e-6),
            default=0.0,
        )
        for x in columns
    }

    obstacles = spec.obstacles()
    out: list[dict] = []
    for combo in combos:
        points = [spec.start]
        labels: list[str] = []
        for (x, (lo, hi)) in zip(columns, combo):
            yc = 0.5 * (lo + hi)
            dx = half_widths[x] + inflation + 0.02
            points.append((float(x - dx), yc))
            points.append((float(x + dx), yc))
            labels.append(f"x={x:+.0f} gap {hi - lo:.2f} m")
        points.append(spec.goal)
        samples = _sample_polyline(points, sample_step_m)
        feasible = not any(
            nominal_violation(p, ob.polygon(), geometry) for p in samples for ob in obstacles
        ) and not any(boundary_violation(p, spec.extent, geometry) for p in samples)
        out.append(
            {
                "gaps": labels,
                "polyline": [[float(p[0]), float(p[1])] for p in points],
                "length_m": _polyline_length(points),
                "feasible": bool(feasible),
                "n_gates": len(columns),
            }
        )
    out.sort(key=lambda r: r["length_m"])
    return out


# ---------------------------------------------------------------- energy
def reference_commands(
    spec: MapSpec, polyline: Sequence[Sequence[float]], v_mps: float, dt_s: float
) -> np.ndarray:
    """Feed-forward profile: constant speed, heading rate from the polyline.

    The profile is a *reference*, not a trackable trajectory: the heading rate is
    clipped to the validated domain, so the integrated path can deviate from the
    polyline.  It exists to price a route with the frozen power model.
    """
    pts = _sample_polyline(polyline, max(v_mps * dt_s, 1e-3))
    if pts.shape[0] < 2:
        raise MapError("polyline too short for a reference profile")
    headings = []
    for a, b in zip(pts[:-1], pts[1:]):
        headings.append(math.atan2(b[1] - a[1], b[0] - a[0]))
    headings.append(headings[-1])
    cmd = np.zeros((pts.shape[0], 3), dtype=np.float64)
    cmd[:, 0] = min(v_mps, 1.0)
    for i in range(pts.shape[0] - 1):
        wz = wrap_angle(headings[i + 1] - headings[i]) / dt_s
        cmd[i, 2] = max(-1.0, min(1.0, wz))
    cmd[-1, 2] = cmd[-2, 2]
    return cmd


def route_energy(commands: np.ndarray, model: PowerModel) -> float:
    """``sum P_nom(u) * 0.1 s`` over the reference profile, in joules."""
    if commands.shape[0] < 2:
        return 0.0
    points = np.vstack([commands, commands[-1:]])
    power = model.step_power(points)
    return float(power.sum() * model.step_cost_dt)


# ---------------------------------------------------------------- evaluation
def evaluate(
    spec: MapSpec,
    geometry: SafetyGeometry,
    model: PowerModel | None = None,
    design: dict | None = None,
) -> dict:
    """Design-intent numbers for one map: gaps, routes, lengths and energy."""
    columns = gate_columns(spec)
    gaps = {f"{x:+.3f}": [round(hi - lo, 9) for lo, hi in free_intervals_at_x(spec, x)]
            for x in columns}
    flat_gaps = [g for values in gaps.values() for g in values]
    straight = math.hypot(spec.goal[0] - spec.start[0], spec.goal[1] - spec.start[1])
    route_list = routes(spec, geometry)
    feasible = [r for r in route_list if r["feasible"]]

    ref = (design or {}).get("reference_profile", {})
    dt_s = float(ref.get("dt_s", 0.1))
    deadline = float(ref.get("t_deadline_s", 30.0))
    v_max = float(ref.get("v_max_mps", 1.0))

    for route in route_list:
        v_ref = route["length_m"] / deadline
        route["v_ref_mps"] = v_ref
        route["within_deadline"] = v_ref <= v_max
        route["energy_J"] = (
            route_energy(reference_commands(spec, route["polyline"], min(v_ref, v_max), dt_s), model)
            if (model is not None and route["within_deadline"])
            else None
        )
        route["turns"] = max(0, len(route["polyline"]) - 2)

    ranked_len = [r for r in feasible if r["length_m"] > 0]
    ranked_len.sort(key=lambda r: r["length_m"])
    ranked_J = [r for r in feasible if r.get("energy_J") is not None]
    ranked_J.sort(key=lambda r: r["energy_J"])
    agreement = None
    if len(ranked_J) >= 2:
        agreement = ranked_J[0]["polyline"] == ranked_len[0]["polyline"]

    return {
        "map_id": spec.map_id,
        "level": spec.level,
        "seed": spec.seed,
        "geometry_hash": spec.geometry_hash(),
        "extent": list(spec.extent),
        "start": list(spec.start),
        "goal": list(spec.goal),
        "net_width_m": round(spec.extent[3] - spec.extent[2], 9),
        "island_polygons": [
            [[float(p[0]), float(p[1])] for p in ob.polygon().vertices]
            for ob in spec.obstacles()
        ],
        "gaps_by_column": gaps,
        "min_gap_m": min(flat_gaps) if flat_gaps else None,
        "straight_distance_m": straight,
        "n_routes_total": len(route_list),
        "n_routes_feasible": len(feasible),
        "start_margin_m": min(
            [nominal_margin(spec.start, ob.polygon(), geometry) for ob in spec.obstacles()]
            or [math.inf]
        ),
        "start_boundary_margin_m": boundary_margin(spec.start, spec.extent, geometry),
        "goal_margin_m": min(
            [nominal_margin(spec.goal, ob.polygon(), geometry) for ob in spec.obstacles()]
            or [math.inf]
        ),
        "goal_boundary_margin_m": boundary_margin(spec.goal, spec.extent, geometry),
        "routes": route_list,
        "shortest_feasible_m": ranked_len[0]["length_m"] if ranked_len else None,
        "detour_ratio": (ranked_len[0]["length_m"] / straight) if ranked_len and straight else None,
        "length_rank": [r["polyline"] for r in ranked_len],
        "energy_rank": [r["polyline"] for r in ranked_J],
        "length_and_energy_agree": agreement,
    }


def catalogue(design: dict, geometry: SafetyGeometry, model: PowerModel | None = None) -> list[dict]:
    return [
        evaluate(build(level, seed, design), geometry, model, design)
        for level in LEVELS
        for seed in design["seeds"]["values"]
    ]


# ---------------------------------------------------------------- svg
def render_svg(
    reports: Sequence[dict], geometry: SafetyGeometry, width_px: int = 900
) -> str:
    """Light-background SVG of every map, for eyeballing the geometry."""
    pad = 18
    label_h = 34
    panels: list[str] = []
    y_cursor = pad
    colours = ["#1f6feb", "#8250df", "#bf3989", "#1a7f37", "#9a6700", "#cf222e"]

    def caption_for(report: dict) -> str:
        gap = "none" if report["min_gap_m"] is None else f'{report["min_gap_m"]:.2f} m'
        shortest = report["shortest_feasible_m"] or 0.0
        return (
            f'{report["map_id"]} · net {report["net_width_m"]:.2f} m · min gap {gap} · '
            f'routes {report["n_routes_feasible"]}/{report["n_routes_total"]} · '
            f'shortest {shortest:.2f} m · straight {report["straight_distance_m"]:.2f} m'
        )

    for report in reports:
        x_min, x_max, y_min, y_max = report["extent"]
        scale = (width_px - 2 * pad) / (x_max - x_min)
        h = (y_max - y_min) * scale
        ox = pad - x_min * scale
        oy = y_cursor + label_h + y_max * scale

        def to_px(x: float, y: float, _ox=ox, _oy=oy, _s=scale) -> tuple[float, float]:
            return (_ox + x * _s, _oy - y * _s)

        parts: list[str] = []
        parts.append(
            f'<text x="{pad}" y="{y_cursor + 20}" font-family="ui-sans-serif, system-ui" '
            f'font-size="13" fill="#1f2328">{caption_for(report)}</text>'
        )
        bx, by = to_px(x_min, y_max)
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{(x_max - x_min) * scale:.1f}" '
            f'height="{(y_max - y_min) * scale:.1f}" fill="#ffffff" stroke="#1f2328" '
            f'stroke-width="1.6"/>'
        )
        inset = geometry.inflation_radius_m
        if (x_max - x_min) > 2 * inset and (y_max - y_min) > 2 * inset:
            ix, iy = to_px(x_min + inset, y_max - inset)
            parts.append(
                f'<rect x="{ix:.1f}" y="{iy:.1f}" width="{(x_max - x_min - 2 * inset) * scale:.1f}" '
                f'height="{(y_max - y_min - 2 * inset) * scale:.1f}" fill="none" stroke="#9a6700" '
                f'stroke-width="1" stroke-dasharray="5 4"/>'
            )
        for poly in report.get("island_polygons", []):
            pts = " ".join("%.1f,%.1f" % to_px(p[0], p[1]) for p in poly)
            parts.append(
                f'<polygon points="{pts}" fill="#6e7781" stroke="#1f2328" stroke-width="1"/>'
            )
        for i, route in enumerate(r for r in report["routes"] if r["feasible"]):
            pts = " ".join("%.1f,%.1f" % to_px(p[0], p[1]) for p in route["polyline"])
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{colours[i % len(colours)]}" '
                f'stroke-width="1.4" opacity="0.9"/>'
            )
        sx, sy = to_px(*report["start"])
        gx, gy = to_px(*report["goal"])
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="#1a7f37"/>')
        parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" fill="#cf222e"/>')
        parts.append(
            f'<text x="{pad + 8}" y="{y_cursor + label_h + h - 6}" font-family="ui-sans-serif, '
            f'system-ui" font-size="10" fill="#57606a">green = start, red = goal, dashed = '
            f'r_fp+delta inset ({inset:.2f} m)</text>'
        )
        panels.extend(parts)
        y_cursor += label_h + h + 12

    total_h = y_cursor + pad
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{total_h:.0f}" '
        f'viewBox="0 0 {width_px} {total_h:.0f}">'
        f'<rect width="100%" height="100%" fill="#f6f8fa"/>' + "".join(panels) + "</svg>"
    )
