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
#:
#: NAV-062 removed ``budget_exhausted`` from this list.  The reference
#: implementation's ``is_terminal`` only looks at the time limit, and the energy
#: is a *cost*, never a stopping condition; making z <= 0 terminal truncated
#: 82-91 % of the old training episodes before the low-budget half of the route
#: was ever seen.  Whether an arrival happened *within* the budget is now a
#: separate reported fact (``EpisodeResult.arrived_within_budget``), not a label.
#: Old traces still carry the string, so :data:`LEGACY_LABELS` keeps it parseable.
LABELS: tuple[str, ...] = (
    "fall",
    "collision",
    "out_of_bounds",
    "goal_reached",
    "stagnation",
    "timeout",
)

#: Labels that only appear in traces recorded before NAV-062.  They are not
#: produced any more and must not be used to gate anything; they exist so old
#: ledgers keep parsing instead of being silently re-labelled.
LEGACY_LABELS: tuple[str, ...] = ("budget_exhausted",)

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
    """Everything one 0.1 s interval produced.  Frozen; never mutated in place.

    **Two ends, explicit names.**  A transition is ``(x_t, u_t, c_t, x_{t+1})``,
    so the safety flag, the signed budget and the goal distance exist in two
    versions per record and must never be confused:

    ``*_before``   state at the decision time ``t`` (the state ``obs`` was built
                   from) -- these are the ones a value at ``x_t`` may pair with;
    the unsuffixed fields
                   state *after* the step, i.e. ``t+1``.

    The unsuffixed names are kept because every existing trace and test reads
    them as "post-step"; ``compute_objectives`` now uses ``*_before`` for the
    ``h_t`` sequence and the ``next_*`` buffer fields for ``h_T``.
    """

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
    # --- the t side of the same transition (NAV-062) -----------------------
    z_J_before: float = 0.0
    z_ratio_before: float = 0.0
    safety_y_before: float = 0.0
    goal_distance_before: float = 0.0
    #: ``z_J_before >= 0``: over-budget is a reported fact now, not a terminal.
    within_budget: bool = True
    #: NAV-064 r14: severity of *this step's* violation in [0, 1] (0 = none).
    #: Binary rounds r1-r13 always recorded 0.0/1.0-equivalent facts via
    #: ``danger_violation``; this field adds the graded magnitude so logs can
    #: separate a graze from a deep hit (F1/F2 evidence).
    safety_severity: float = 0.0

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
            "z_J_before": self.z_J_before,
            "z_ratio_before": self.z_ratio_before,
            "safety_y_before": self.safety_y_before,
            "goal_distance_before": self.goal_distance_before,
            "within_budget": self.within_budget,
            "safety_severity": self.safety_severity,
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
    #: ``z_J >= 0`` at the end *and* the episode arrived with a non-negative
    #: budget path -- the two facts are reported even when the label is not
    #: ``goal_reached``, so "arrived but over budget" cannot masquerade as
    #: feasible (NAV-062 acceptance).
    within_budget: bool = True
    arrived_within_budget: bool = False
    over_budget_J: float = 0.0
    #: NAV-064 r14 diagnostics (user-authorized 2026-11-15T17:32).
    max_safety_severity: float = 0.0
    #: Full context of the stagnation trigger, or None if the episode never
    #: stagnated.  Collected so "killed while actually fine-tuning the last
    #: metre" vs "genuinely stuck" can be told apart from logs alone.
    stagnation_context: dict | None = None

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
            "within_budget": self.within_budget,
            "arrived_within_budget": self.arrived_within_budget,
            "over_budget_J": self.over_budget_J,
            "max_safety_severity": self.max_safety_severity,
            "stagnation_context": self.stagnation_context,
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
        safety_severity_enabled: bool = False,
        safety_severity_floor: float = 0.2,
        safety_severity_ref_depth_m: float = 0.25,
        observation_version: str | None = None,
    ) -> None:
        if abs(cost_model.dt_s - DT_NAV_S) > 1e-12:
            raise NominalEnvError(
                f"cost model dt={cost_model.dt_s} is not the 0.1 s navigation clock"
            )
        if not math.isfinite(z0_J):
            raise NominalEnvError(f"z0_J must be finite, got {z0_J}")
        if not (0.0 < safety_severity_floor < 1.0):
            raise NominalEnvError(
                f"safety_severity_floor must be in (0, 1), got {safety_severity_floor}")
        if safety_severity_ref_depth_m <= 0.0:
            raise NominalEnvError(
                f"safety_severity_ref_depth_m must be > 0, got {safety_severity_ref_depth_m}")
        self.spec = spec
        self.geometry = geometry
        self.cost_model = cost_model
        self.z0_J = float(z0_J)
        self.r_goal_m = float(r_goal_m)
        self.t_max_s = float(t_max_s)
        self.stagnation_window_s = float(stagnation_window_s)
        self.stagnation_min_displacement_m = float(stagnation_min_displacement_m)
        self.initial_pose_jitter_m = float(initial_pose_jitter_m)
        # NAV-064 r14 (user-authorized 2026-11-15T17:32): graded safety history.
        # Default off = binary history, byte-identical to rounds r1-r13.
        self.safety_severity_enabled = bool(safety_severity_enabled)
        self.safety_severity_floor = float(safety_severity_floor)
        self.safety_severity_ref_depth_m = float(safety_severity_ref_depth_m)
        self.observation_version = observation_version or obs.OBS_VERSION
        obs.z_ratio_clip(self.observation_version)

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
    def reset(self, *, seed: int | None = None, z0_J: float | None = None) -> list[float]:
        """Start a fresh episode.  The only way to start one; never implicit.

        ``z0_J`` overrides the construction-time budget for this episode only
        (NAV-062: budgets are sampled per episode, not fixed per map).  The
        value is stored, so the trace always says which budget it ran under.
        """
        if z0_J is not None:
            z0 = float(z0_J)
            if not math.isfinite(z0) or z0 <= 0.0:
                raise NominalEnvError(f"z0_J must be positive and finite, got {z0_J}")
            self.z0_J = z0
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
        # NAV-064 r14 diagnostics
        self.max_safety_severity = 0.0
        self.stagnation_context: dict | None = None
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
        return obs.encode_observation(self.ego, self._obstacle_obs, self._extent,
                                      self.task, version=self.observation_version)

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
        # --- the t side of this transition, captured before anything moves --
        z_before = float(self.budget.z_J)
        z_ratio_before = float(self.budget.z_ratio)
        safety_before = float(self.safety_y)
        goal_d_before = math.hypot(
            float(self.spec.goal[0]) - p0, float(self.spec.goal[1]) - y0)
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

        # --- geometry facts for the record ----------------------------------
        # NAV-064 r14: margins are computed *before* the safety update so the
        # violation severity of this very step can be priced from the depth.
        min_nom = (
            min(G.nominal_margin((p1, y1), poly, self.geometry) for poly in self._obstacle_polys)
            if self._obstacle_polys
            else None
        )

        # --- safety history: obstacle inflation plus the wall bands ----------
        danger = any(
            G.nominal_violation_during_step((p0, y0), (p1, y1), poly, self.geometry)
            for poly in self._obstacle_polys
        ) or G.boundary_violation((p1, y1), self.spec.extent, self.geometry)
        # ``internal_violation`` is the nominal constraint being crossed.  It is
        # recorded, never used as a label (frozen contract: orthogonal).
        internal = bool(danger)
        # NAV-064 r14 (user-authorized 2026-11-15T17:32): graded severity.
        # Binary mode (default, r1-r13): severity is 1 on any violation and the
        # update below is byte-identical to the old ``update_safety_history``.
        # Graded mode: severity rises linearly with penetration depth from
        # ``floor`` (grazing touch) to 1.0 (deeper than ``ref_depth_m``), so a
        # graze is priced strictly below a deep hit and post-violation steps
        # keep a position-dependent h instead of a constant 1.
        if danger:
            if self.safety_severity_enabled:
                depth = 0.0 if min_nom is None else max(0.0, -float(min_nom))
                depth = max(depth, -float(G.boundary_margin(
                    (p1, y1), self.spec.extent, self.geometry)))
                severity = G.safety_severity_from_depth(
                    depth, floor=self.safety_severity_floor,
                    ref_depth_m=self.safety_severity_ref_depth_m)
                self.safety_y = G.update_safety_history_graded(self.safety_y, severity)
            else:
                severity = 1.0
                self.safety_y = G.update_safety_history(self.safety_y, danger)
        else:
            severity = 0.0
        self.max_safety_severity = max(self.max_safety_severity, severity)
        if danger:
            self.danger_steps += 1
        if internal:
            self.internal_violation_steps += 1
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
        # No budget branch: z <= 0 is a reported fact (`within_budget`), not a
        # stopping condition.  The episode ends on arrival / collision / leaving
        # the map / stagnation / the 30 s limit, exactly like the reference
        # implementation whose `is_terminal` only looks at the horizon.
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
            z_J_before=z_before,
            z_ratio_before=z_ratio_before,
            safety_y_before=safety_before,
            goal_distance_before=goal_d_before,
            within_budget=bool(z_before >= 0.0),
            safety_severity=severity,
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
        if moved >= self.stagnation_min_displacement_m:
            return False
        # NAV-064 r14 diagnostics (user-authorized 2026-11-15T17:32): capture
        # the full scene at the trigger so logs alone can separate a false kill
        # ("slowly fine-tuning the last metre, still had budget") from a true
        # one ("genuinely out of options").  Nothing reads this during training;
        # it is evidence collection only.
        self.stagnation_context = {
            "t_s": round(self.t_s, 3),
            "pose": [round(self._pose[0], 4), round(self._pose[1], 4)],
            "goal_distance_m": round(math.hypot(
                float(self.spec.goal[0]) - self._pose[0],
                float(self.spec.goal[1]) - self._pose[1]), 4),
            "z_J": round(float(self.budget.z_J), 3),
            "window_displacement_m": round(moved, 6),
            "min_displacement_m": self.stagnation_min_displacement_m,
            "window_s": self.stagnation_window_s,
            "time_remaining_s": round(max(0.0, self.t_max_s - self.t_s), 3),
        }
        return True

    # ---- reporting ----
    def result(self) -> EpisodeResult:
        over = max(0.0, -float(self.budget.z_J))
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
            within_budget=bool(self.budget.z_J >= 0.0),
            arrived_within_budget=bool(
                self.label == "goal_reached" and self.budget.z_J >= 0.0),
            over_budget_J=over,
            max_safety_severity=self.max_safety_severity,
            stagnation_context=self.stagnation_context,
        )

    def trace(self) -> dict:
        """Full episode ledger, JSON-ready."""
        return {
            "schema": "camra-nominal-episode-v1",
            "map": self.spec.to_dict(),
            "map_geometry_hash": self.spec.geometry_hash(),
            "obs_version": self.observation_version,
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
            "safety_severity_enabled": self.safety_severity_enabled,
            "safety_severity_floor": self.safety_severity_floor,
            "safety_severity_ref_depth_m": self.safety_severity_ref_depth_m,
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
