"""Deterministic nominal navigation environment (NAV-018).

One episode is a sequence of 0.1 s intervals.  Each interval

1. maps the normalised action to a **requested** and then an **applied** command,
2. advances the planar pose with the frozen closed-form integrator
   (``camra_calibration.nominal.step_exact``): a circular arc when ``wz != 0``,
   a straight line otherwise,
3. charges the frozen power model on the *applied* command,
4. updates the irreversible safety history, and
5. labels the interval at most once, following the frozen priority order.

Three things this class deliberately does **not** do:

* it never simulates contact, so a collision ends the episode at the end-of-step
  pose instead of producing a post-impact state;
* it never resets itself.  After an episode ends, :meth:`step` raises; the caller
  must call :meth:`reset` explicitly, so a finished episode can never absorb the
  fresh state of the next one (the HiRA defect recorded in NAV-005);
* it never reports measured values as if they were measured.  Body velocities are
  the applied command because the nominal model assumes ideal tracking, and
  ``physical_status`` stays ``not_measured`` throughout.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .. import observation as obs
from ..costs import DT_NAV_S, JCostModel, SignedBudget, StepCostRecord
from ..maps import geometry as G
from ..maps.generator import MapSpec

try:  # frozen calibration package, read-only use
    from camra_calibration.geometry import Polygon
    from camra_calibration.nominal import PlanarState, step_exact, wrap_angle

    _FROZEN_INTEGRATOR = "camra_calibration.nominal.step_exact"
except Exception as exc:  # pragma: no cover - import guard
    PlanarState = None  # type: ignore[assignment]
    step_exact = None  # type: ignore[assignment]
    Polygon = None  # type: ignore[assignment]
    wrap_angle = None  # type: ignore[assignment]
    _FROZEN_INTEGRATOR = f"unavailable: {exc}"

#: Terminal labels, in the frozen priority order (contract ``event_table``).
LABELS: tuple[str, ...] = (
    "fall",
    "collision",
    "out_of_bounds",
    "goal_reached",
    "budget_exhausted",
    "stagnation",
    "timeout",
)

#: Labels this environment can actually produce.  ``fall`` needs physics
#: (base contact, bad orientation) and is unreachable in a planar model, so it is
#: reported as ``not_applicable`` rather than silently omitted.
NOMINAL_LABELS: tuple[str, ...] = tuple(x for x in LABELS if x != "fall")

WALL_THICKNESS_M = 1.0


class EpisodeOverError(RuntimeError):
    """Raised when ``step`` is called on a finished episode."""


class NominalEnvError(RuntimeError):
    """Raised when the environment is configured or used inconsistently."""


# --------------------------------------------------------------------- helpers
def wall_polygons(extent: Sequence[float], thickness: float = WALL_THICKNESS_M):
    """Four thin rectangles hugging the map boundary from the outside.

    Walls are physical objects in the nominal model too, so that "hit the wall"
    is a ``collision`` and not a label invented by a coordinate comparison.
    """
    x0, x1, y0, y1 = (float(v) for v in extent)
    t = float(thickness)
    bands = (
        ((x0 - t, y0 - t), (x0, y0 - t), (x0, y1 + t), (x0 - t, y1 + t)),
        ((x1, y0 - t), (x1 + t, y0 - t), (x1 + t, y1 + t), (x1, y1 + t)),
        ((x0 - t, y0 - t), (x1 + t, y0 - t), (x1 + t, y0), (x0 - t, y0)),
        ((x0 - t, y1), (x1 + t, y1), (x1 + t, y1 + t), (x0 - t, y1 + t)),
    )
    return tuple(Polygon(np.asarray(b, dtype=np.float64)) for b in bands)


def _require_frozen_integrator() -> None:
    if step_exact is None or PlanarState is None:  # pragma: no cover
        raise NominalEnvError(
            "the frozen integrator camra_calibration.nominal.step_exact is not "
            f"importable ({_FROZEN_INTEGRATOR}); the nominal env must not "
            "substitute a home-made integrator"
        )


def integrate(px: float, py: float, yaw: float, command: Sequence[float], dt: float):
    """Advance the planar pose with the **frozen** closed-form integrator."""
    _require_frozen_integrator()
    return step_exact(PlanarState(float(px), float(py), float(yaw)), command, dt)


# ------------------------------------------------------------------- records
@dataclass(frozen=True)
class StepRecord:
    """Everything one 0.1 s interval produced.  Frozen; never mutated in place."""

    index: int
    t_s: float
    action: tuple[float, float, float]
    command: obs.CommandRecord
    pose_before: tuple[float, float, float]
    pose_after: tuple[float, float, float]
    cost: StepCostRecord
    z_J: float
    z_ratio: float
    safety_y: float
    danger_violation: bool
    internal_violation: bool
    goal_distance_m: float
    min_nominal_margin_m: float | None
    min_physical_clearance_m: float | None
    dropped_obstacles: int
    label: str | None
    done: bool

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "t_s": self.t_s,
            "action": list(self.action),
            "requested": list(self.command.requested),
            "applied": list(self.command.applied),
            "clipped": self.command.clipped,
            "invalid_action": self.command.invalid,
            "pose_before": list(self.pose_before),
            "pose_after": list(self.pose_after),
            "cost": self.cost.as_dict(),
            "z_J": self.z_J,
            "z_ratio": self.z_ratio,
            "safety_y": self.safety_y,
            "danger_violation": self.danger_violation,
            "internal_violation": self.internal_violation,
            "goal_distance_m": self.goal_distance_m,
            "min_nominal_margin_m": self.min_nominal_margin_m,
            "min_physical_clearance_m": self.min_physical_clearance_m,
            "dropped_obstacles": self.dropped_obstacles,
            "label": self.label,
            "done": self.done,
        }


@dataclass
class EpisodeResult:
    """Summary of one finished (or still running) episode."""

    map_id: str
    level: str
    seed: int
    label: str | None
    done: bool
    n_steps: int
    t_s: float
    travelled_m: float
    goal_distance_m: float
    cost_nom_J: float
    cost_model_J: float
    z0_J: float
    z_J: float
    safety_y: float
    danger_steps: int
    internal_violation_steps: int
    alpha: float
    variant: str
    physical_status: str = "not_measured"

    def as_dict(self) -> dict:
        return {
            "map_id": self.map_id,
            "level": self.level,
            "seed": self.seed,
            "label": self.label,
            "done": self.done,
            "n_steps": self.n_steps,
            "t_s": self.t_s,
            "travelled_m": self.travelled_m,
            "goal_distance_m": self.goal_distance_m,
            "cost_nom_J": self.cost_nom_J,
            "cost_model_J": self.cost_model_J,
            "z0_J": self.z0_J,
            "z_J": self.z_J,
            "safety_y": self.safety_y,
            "danger_steps": self.danger_steps,
            "internal_violation_steps": self.internal_violation_steps,
            "alpha": self.alpha,
            "variant": self.variant,
            "physical_status": self.physical_status,
        }


# ----------------------------------------------------------------------- env
class NominalNavEnv:
    """Deterministic 10 Hz nominal reach-avoid environment."""

    def __init__(
        self,
        spec: MapSpec,
        geometry: G.SafetyGeometry,
        cost_model: JCostModel,
        *,
        z0_J: float,
        r_goal_m: float = 0.5,
        t_max_s: float = 30.0,
        stagnation_window_s: float = 3.0,
        stagnation_min_displacement_m: float = 0.05,
        initial_pose_jitter_m: float = 0.0,
    ) -> None:
        if abs(cost_model.dt_s - DT_NAV_S) > 1e-12:
            raise NominalEnvError(
                f"cost model dt={cost_model.dt_s} is not the 0.1 s navigation clock"
            )
        if not math.isfinite(z0_J):
            raise NominalEnvError(f"z0_J must be finite, got {z0_J}")
        self.spec = spec
        self.geometry = geometry
        self.cost_model = cost_model
        self.z0_J = float(z0_J)
        self.r_goal_m = float(r_goal_m)
        self.t_max_s = float(t_max_s)
        self.stagnation_window_s = float(stagnation_window_s)
        self.stagnation_min_displacement_m = float(stagnation_min_displacement_m)
        self.initial_pose_jitter_m = float(initial_pose_jitter_m)

        self._obstacle_polys = tuple(ob.polygon() for ob in spec.obstacles())
        self._obstacle_obs = tuple(
            obs.obstacle_from_polygon(poly.vertices, label=f"island-{i}")
            for i, poly in enumerate(self._obstacle_polys)
        )
        self._walls = wall_polygons(spec.extent)
        self._extent = obs.MapExtent(*spec.extent)

        self.budget = SignedBudget(z0_J=self.z0_J)
        self._seed: int | None = None
        self._rng: np.random.Generator | None = None
        self.reset()

    # ---- episode lifecycle ----
    def reset(self, *, seed: int | None = None) -> list[float]:
        """Start a fresh episode.  The only way to start one; never implicit."""
        self._seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self._seed)
        jitter = self.initial_pose_jitter_m
        if jitter > 0.0:
            dx, dy = self._rng.uniform(-jitter, jitter, size=2)
            dyaw = float(self._rng.uniform(-0.05, 0.05))
        else:
            dx = dy = dyaw = 0.0
        self._pose = (
            float(self.spec.start[0]) + float(dx),
            float(self.spec.start[1]) + float(dy),
            float(self.spec.start_yaw) + dyaw,
        )
        self._reset_fingerprint = self._pose

        self.t_s = 0.0
        self.index = 0
        self.done = False
        self.label: str | None = None
        self.safety_y = 0.0
        self.budget.reset(self.z0_J)
        self.cost_model.reset()
        self._history: list[tuple[float, float, float]] = [(self.t_s, self._pose[0], self._pose[1])]
        self._records: list[StepRecord] = []
        self.travelled_m = 0.0
        self.danger_steps = 0
        self.internal_violation_steps = 0
        self._applied_last = (0.0, 0.0, 0.0)
        return self.observation()

    # ---- state ----
    @property
    def pose(self) -> tuple[float, float, float]:
        return self._pose

    @property
    def ego(self) -> obs.EgoState:
        """Nominal body velocity equals the applied command (declared assumption)."""
        vx, vy, wz = self._applied_last
        return obs.EgoState(
            px=self._pose[0],
            py=self._pose[1],
            yaw=self._pose[2],
            vx_body=vx,
            vy_body=vy,
            wz_body=wz,
        )

    @property
    def task(self) -> obs.TaskState:
        return obs.TaskState(
            goal_x=float(self.spec.goal[0]),
            goal_y=float(self.spec.goal[1]),
            safety_y=self.safety_y,
            z_J=self.budget.z_J,
            time_remaining_s=max(0.0, self.t_max_s - self.t_s),
        )

    def observation(self) -> list[float]:
        return obs.encode_observation(self.ego, self._obstacle_obs, self._extent, self.task)

    @property
    def steps(self) -> tuple[StepRecord, ...]:
        return tuple(self._records)

    # ---- one interval ----
    def step(self, action: Sequence[float]) -> StepRecord:
        if self.done:
            raise EpisodeOverError(
                f"episode already ended with label={self.label!r}; call reset() "
                "instead of stepping a finished episode"
            )
        _require_frozen_integrator()
        act = tuple(float(v) for v in action)
        if len(act) != 3:
            raise NominalEnvError(f"action must be [3], got {act}")

        command = obs.command_from_action(act)
        p0, y0 = self._pose[0], self._pose[1]
        yaw0 = self._pose[2]
        nxt = integrate(p0, self._pose[1], yaw0, command.applied, DT_NAV_S)
        p1, y1 = float(nxt.px), float(nxt.py)
        # The frozen integrator accumulates yaw without folding it, so a long
        # episode can drift to 5*pi.  sin/cos stay correct, but the contract says
        # the stored heading lives in (-pi, pi], so the fold happens here, at the
        # state boundary, not inside the integrator (whose identity with
        # step_exact must stay checkable).  Only out-of-range values are folded:
        # wrap_angle is not an identity on in-range angles (it perturbs them by
        # ~1e-16), and folding a still-valid heading would make an idle step move
        # the recorded pose for no reason.
        raw_yaw = float(nxt.yaw)
        if wrap_angle is not None and not (-math.pi < raw_yaw <= math.pi):
            yaw1 = float(wrap_angle(raw_yaw))
        else:
            yaw1 = raw_yaw

        # --- cost on the applied command (charges the budget exactly once) ---
        cost = self.cost_model.step(command.applied, budget=self.budget)

        # --- safety history: obstacle inflation plus the wall bands ----------
        danger = any(
            G.nominal_violation_during_step((p0, y0), (p1, y1), poly, self.geometry)
            for poly in self._obstacle_polys
        ) or G.boundary_violation((p1, y1), self.spec.extent, self.geometry)
        # ``internal_violation`` is the nominal constraint being crossed.  It is
        # recorded, never used as a label (frozen contract: orthogonal).
        internal = bool(danger)
        self.safety_y = G.update_safety_history(self.safety_y, danger)
        if danger:
            self.danger_steps += 1
        if internal:
            self.internal_violation_steps += 1

        # --- geometry facts for the record ----------------------------------
        min_nom = (
            min(G.nominal_margin((p1, y1), poly, self.geometry) for poly in self._obstacle_polys)
            if self._obstacle_polys
            else None
        )
        min_phys = (
            min(G.physical_clearance((p1, y1), yaw1, poly, self.geometry) for poly in self._obstacle_polys)
            if self._obstacle_polys
            else None
        )

        # --- label, in the frozen priority order ----------------------------
        self.t_s += DT_NAV_S
        self.travelled_m += math.hypot(p1 - p0, y1 - y0)
        self._pose = (p1, y1, yaw1)
        self._applied_last = command.applied
        self._history.append((self.t_s, p1, y1))

        goal_d = math.hypot(float(self.spec.goal[0]) - p1, float(self.spec.goal[1]) - y1)

        hit = any(
            G.physical_collision_during_step((p0, y0), yaw0, (p1, y1), yaw1, poly, self.geometry)
            for poly in self._obstacle_polys + self._walls
        )
        outside = not (
            self.spec.extent[0] <= p1 <= self.spec.extent[1]
            and self.spec.extent[2] <= y1 <= self.spec.extent[3]
        )

        label: str | None = None
        if hit:
            label = "collision"
        elif outside:
            label = "out_of_bounds"
        elif goal_d <= self.r_goal_m:
            label = "goal_reached"
        elif self.budget.z_J <= 0.0:
            label = "budget_exhausted"
        elif self._stagnated():
            label = "stagnation"
        elif self.t_s >= self.t_max_s - 1e-12:
            label = "timeout"

        if label is not None:
            self.done = True
            self.label = label

        record = StepRecord(
            index=self.index,
            t_s=self.t_s,
            action=act,
            command=command,
            pose_before=(p0, y0, yaw0),
            pose_after=(p1, y1, yaw1),
            cost=cost,
            z_J=self.budget.z_J,
            z_ratio=self.budget.z_ratio,
            safety_y=self.safety_y,
            danger_violation=bool(danger),
            internal_violation=internal,
            goal_distance_m=goal_d,
            min_nominal_margin_m=min_nom,
            min_physical_clearance_m=min_phys,
            dropped_obstacles=max(0, len(self._obstacle_obs) - obs.OBSTACLE_CAPACITY),
            label=label,
            done=self.done,
        )
        self._records.append(record)
        self.index += 1
        return record

    def _stagnated(self) -> bool:
        """Displacement over the trailing window, per the frozen procedure."""
        window = self.stagnation_window_s
        if self.t_s < window - 1e-12:
            return False
        anchor = None
        for t, x, y in self._history:
            if self.t_s - t <= window + 1e-9:
                anchor = (x, y)
                break
        if anchor is None:
            return False
        moved = math.hypot(self._pose[0] - anchor[0], self._pose[1] - anchor[1])
        return moved < self.stagnation_min_displacement_m

    # ---- reporting ----
    def result(self) -> EpisodeResult:
        return EpisodeResult(
            map_id=self.spec.map_id,
            level=self.spec.level,
            seed=self.spec.seed,
            label=self.label,
            done=self.done,
            n_steps=self.index,
            t_s=self.t_s,
            travelled_m=self.travelled_m,
            goal_distance_m=math.hypot(
                float(self.spec.goal[0]) - self._pose[0],
                float(self.spec.goal[1]) - self._pose[1],
            ),
            cost_nom_J=sum(r.cost.cost_nom_J for r in self._records),
            cost_model_J=sum(r.cost.cost_model_J for r in self._records),
            z0_J=self.z0_J,
            z_J=self.budget.z_J,
            safety_y=self.safety_y,
            danger_steps=self.danger_steps,
            internal_violation_steps=self.internal_violation_steps,
            alpha=self.cost_model.alpha,
            variant=self.cost_model.variant,
        )

    def trace(self) -> dict:
        """Full episode ledger, JSON-ready."""
        return {
            "schema": "camra-nominal-episode-v1",
            "map": self.spec.to_dict(),
            "map_geometry_hash": self.spec.geometry_hash(),
            "obs_version": obs.OBS_VERSION,
            "command_version": obs.COMMAND_VERSION,
            "integrator": _FROZEN_INTEGRATOR,
            "controller_id": "camra_go2_velocity_it2999_v2",
            "cost_unit": "J",
            "variant": self.cost_model.variant,
            "alpha": self.cost_model.alpha,
            "alpha_source": self.cost_model.alpha_source,
            "rng_seed": self._seed,
            "initial_pose": list(self._reset_fingerprint),
            "r_goal_m": self.r_goal_m,
            "t_max_s": self.t_max_s,
            "stagnation_window_s": self.stagnation_window_s,
            "stagnation_min_displacement_m": self.stagnation_min_displacement_m,
            "result": self.result().as_dict(),
            "steps": [r.as_dict() for r in self._records],
        }


# ------------------------------------------------------------------ presets
def preset_actions(preset: str, n_steps: int) -> list[tuple[float, float, float]]:
    """Small hand-written command sequences, not a controller.

    They exist so the environment can be exercised end to end without implying
    any navigation performance.  ``forward`` drives straight, ``creep`` idles at
    a low command, ``turn_drift`` arcs, and ``mixed`` alternates.
    """
    if preset == "forward":
        return [(0.6, 0.0, 0.0)] * n_steps
    if preset == "creep":
        return [(0.05, 0.0, 0.0)] * n_steps
    if preset == "turn_drift":
        return [(0.4, 0.0, 0.3)] * n_steps
    if preset == "mixed":
        out = []
        for i in range(n_steps):
            if i % 4 == 0:
                out.append((0.8, 0.0, 0.0))
            elif i % 4 == 1:
                out.append((0.5, 0.1, -0.2))
            elif i % 4 == 2:
                out.append((0.2, 0.0, 0.0))
            else:
                out.append((0.0, 0.0, 0.4))
        return out
    raise NominalEnvError(f"unknown preset {preset!r}")
