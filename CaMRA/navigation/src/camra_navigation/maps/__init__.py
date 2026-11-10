"""Safety geometry: footprint, margin and history predicates (NAV-015)."""

from __future__ import annotations

from .geometry import (  # noqa: F401
    SafetyGeometry,
    StepSafety,
    boundary_margin,
    boundary_violation,
    convex_polygons_intersect,
    footprint_at,
    legacy_avoid_to_y,
    nominal_margin,
    nominal_violation,
    nominal_violation_during_step,
    physical_clearance,
    physical_collision,
    physical_collision_during_step,
    polygon_distance_exact,
    segment_intersects_polygon,
    update_safety_history,
    y_to_legacy_avoid,
)

__all__ = [
    "SafetyGeometry",
    "StepSafety",
    "boundary_margin",
    "boundary_violation",
    "convex_polygons_intersect",
    "footprint_at",
    "legacy_avoid_to_y",
    "nominal_margin",
    "nominal_violation",
    "nominal_violation_during_step",
    "physical_clearance",
    "physical_collision",
    "physical_collision_during_step",
    "polygon_distance_exact",
    "segment_intersects_polygon",
    "update_safety_history",
    "y_to_legacy_avoid",
]
