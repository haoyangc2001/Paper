"""Vectorised nominal navigation environment (NAV-036).

Runs ``N`` independent episodes of :class:`~camra_navigation.envs.nominal.
NominalNavEnv` on the **same map**, stepping them in lockstep.  It exists for
one reason: the authorised training quota (4096 steps/update, 6 h wall clock)
needs >= 47.4 aggregate steps/s, and the scalar environment delivers 9-16.
The experiment preregistration accepts this implementation only under a
**bitwise-equivalence** gate: same (level, seed) and same command sequence
must produce identical poses, costs and labels, step by step.  That gate is
enforced by ``tests/test_vector_equivalence.py``.

Equivalence strategy, per pipeline stage:

* **integration, cost, observation, stagnation, labels** - executed per env
  with the *same frozen code* as the scalar environment
  (``camra_calibration.nominal.step_exact``, ``JCostModel.step``,
  ``obs.encode_observation``), so those outputs are identical by
  construction, not by argument.
* **geometry (the only hot spot: 22-113 ms/step in the scalar env)** - two
  batched replacements, both *exact*, not approximations:

  1. ``nominal_violation_during_step`` (segment vs inflated island): the
     frozen implementation inflates the island every step and tests the
     segment by endpoint containment + edge crossing.  The inflation is a
     deterministic function of (polygon, radius), so caching it is
     bitwise-identical.  The separating-axis test between a segment and a
     convex polygon (axis set: segment direction + polygon edge normals) is
     complete, so the boolean agrees with the frozen predicate on
     non-degenerate segments.  A bitwise-degenerate segment (p1 == p0, which
     happens when the applied command is exactly zero) falls back to the
     frozen scalar predicate, replicating its degenerate semantics exactly.
  2. ``physical_collision_during_step`` (rotating swept hull): the frozen
     test is ``convex_polygons_intersect(hull(F0, F1) (+) disc(r), obstacle)``
     with ``r = r_fp * |wrap(yaw1 - yaw0)| / 2``.  A polygon inflated by a
     disc separates from another polygon along ``n`` iff the projections
     overlap with margin ``r``, and if a separating axis exists at all, one
     can be chosen among the edge normals of the hull or the obstacle; the
     hull's edge normals are a subset of ``normals(F0) U normals(F1)``.  Hence
     testing axes ``normals(F0) U normals(F1) U normals(obstacle)`` with
     margin ``r`` returns the same boolean.  The batched footprint vertex
     sets equal the frozen ones (rotate/translate replicated with the same
     float operations); vertex *order* may differ from the frozen hull
     ordering, which cannot change min/max projections or the edge-normal set
     of a convex polygon.

Deliberately not batched: ``min_nominal_margin_m`` and
``min_physical_clearance_m`` are recorded by the scalar env but consumed by
nothing in training; they are computed here only when ``compute_margins=True``
(by calling the frozen per-env functions, so they stay bitwise-identical).
The default is ``False`` and consumers must treat the two fields as absent in
that mode.

A finished episode never steps again: done envs ignore their action row and
freeze their state (the scalar env raises ``EpisodeOverError`` instead); the
caller distinguishes the two modes via ``BatchStepRecord.active``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .. import observation as obs
from ..costs import DT_NAV_S, JCostModel
from ..maps import geometry as G
from ..maps.generator import MapSpec
from .nominal import EpisodeResult, NominalEnvError, wall_polygons

try:  # frozen calibration package, read-only use (same modules as the scalar env)
    from camra_calibration.geometry import Polygon
    from camra_calibration.nominal import PlanarState, step_exact, wrap_angle
except Exception as exc:  # pragma: no cover - import guard
    raise NominalEnvError(
        f"the frozen calibration package is not importable ({exc}); the "
        "vectorised nominal env must not substitute home-made geometry"
    )

_EPS = G._EPS


# ------------------------------------------------------------- batch helpers
def _edge_normals(vertices: np.ndarray) -> np.ndarray:
    """Edge normals, exactly the frozen ``_edge_axes`` formulation."""
    e = np.roll(vertices, -1, axis=0) - vertices
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, _EPS)


def _footprint_vertices(footprint_vertices: np.ndarray, pos: Sequence[float],
                        yaw: float) -> np.ndarray:
    """Frozen ``footprint_at`` without the (order-changing) hull rebuild.

    Same float operations: ``vertices @ rot.T`` with ``rot`` built from
    ``math.cos``/``math.sin``, then ``+ pos``.  The frozen version re-runs the
    convex hull, which may reorder vertices but never changes the vertex set;
    min/max projections and the edge-normal set of a convex polygon are
    order-independent, so downstream booleans are unaffected.
    """
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rot = np.array([[c, -s], [s, c]])
    return footprint_vertices @ rot.T + np.asarray(pos, dtype=np.float64)


def _swept_collision_batch(
    fp0: np.ndarray, fp1: np.ndarray, obstacle: np.ndarray, margin: np.ndarray
) -> np.ndarray:
    """Batched replacement of ``physical_collision_during_step``.

    ``fp0``/``fp1``: ``(N, k, 2)`` footprint vertex sets at the step endpoints.
    ``obstacle``: ``(m, 2)`` convex polygon vertices.  ``margin``: ``(N,)``
    inflation ``r_fp * |wrap(yaw1 - yaw0)| / 2`` (0 for pure translation).
    Returns the per-env boolean, equal to
    ``convex_polygons_intersect(hull(F0, F1) (+) disc(r), obstacle)``.
    """
    axes = np.concatenate(
        [_edge_normals(fp0[0]), _edge_normals(fp1[0]), _edge_normals(obstacle)],
        axis=0,
    )                                                    # (A, 2)
    proj0 = fp0 @ axes.T                                 # (N, k, A)
    proj1 = fp1 @ axes.T
    proj_o = obstacle @ axes.T                           # (m, A)
    lo_hull = np.minimum(proj0.min(axis=1), proj1.min(axis=1))   # (N, A)
    hi_hull = np.maximum(proj0.max(axis=1), proj1.max(axis=1))
    r = margin[:, None]
    sep = (hi_hull + r < proj_o.min(axis=0)[None, :]) | \
          (lo_hull - r > proj_o.max(axis=0)[None, :])
    return ~sep.any(axis=1)


def _segment_polygon_batch(
    p0: np.ndarray, p1: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """Batched SAT between non-degenerate segments and one convex polygon.

    Axis set: the segment direction (the normal of a 2-vertex polygon's single
    edge) plus the polygon's edge normals -- complete for segment-vs-polygon.
    Margin 0, so axis scaling is irrelevant.
    """
    d = p1 - p0
    d_axis = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), _EPS)
    normals = _edge_normals(vertices)                    # (m, 2)
    pr0 = p0 @ normals.T                                 # (N, m)
    pr1 = p1 @ normals.T
    lo_p = np.minimum(pr0, pr1)
    hi_p = np.maximum(pr0, pr1)
    proj_o = vertices @ normals.T                        # (m, m): vertex i on axis j
    lo_o = proj_o.min(axis=0)                            # (m,)
    hi_o = proj_o.max(axis=0)
    sep = (hi_p < lo_o[None, :]) | (lo_p > hi_o[None, :])
    # segment-direction axis
    proj_p = np.stack([(p0 * d_axis).sum(axis=1), (p1 * d_axis).sum(axis=1)], axis=1)
    lo_pd, hi_pd = proj_p.min(axis=1), proj_p.max(axis=1)
    proj_od = d_axis @ vertices.T                        # (N, m)
    sep |= (hi_pd[:, None] < proj_od.min(axis=1)[:, None]) | \
           (lo_pd[:, None] > proj_od.max(axis=1)[:, None])
    return ~sep.any(axis=1)


# ------------------------------------------------------------------ records
@dataclass(frozen=True)
class BatchStepRecord:
    """One lockstep interval for all ``N`` envs.

    For envs that had already finished (``active == False``) the state fields
    keep their frozen values, and cost/command fields are zeroed.  Nothing
    mutates after construction.
    """

    active: np.ndarray               # (N,) bool: env was still running before this step
    index: np.ndarray                # (N,) int: per-env step index BEFORE this step (frozen when done)
    requested: np.ndarray            # (N,3) float (zeros when not active)
    applied: np.ndarray              # (N,3) float (zeros when not active)
    clipped: np.ndarray              # (N,) bool
    invalid: np.ndarray              # (N,) bool
    pose_after: np.ndarray           # (N,3) float64, (px, py, yaw)
    t_s: np.ndarray                  # (N,) float, episode clock AFTER this step
    cost_t_s: np.ndarray             # (N,) float, cost-ledger clock BEFORE this step
    power_W: np.ndarray              # (N,) float
    cost_nom_J: np.ndarray           # (N,) float
    cost_model_J: np.ndarray         # (N,) float
    z_J: np.ndarray                  # (N,) float, signed budget AFTER this step
    z_ratio: np.ndarray              # (N,) float, encoded (clipped) ratio AFTER
    safety_y: np.ndarray             # (N,) float, safety history AFTER this step
    danger_violation: np.ndarray     # (N,) bool
    internal_violation: np.ndarray   # (N,) bool
    goal_distance_m: np.ndarray      # (N,) float, distance AFTER this step
    min_nominal_margin_m: np.ndarray | None
    min_physical_clearance_m: np.ndarray | None
    dropped_obstacles: int
    label: tuple[str | None, ...]    # (N,)
    done: np.ndarray                 # (N,) bool, state *after* this step
    obs: np.ndarray                  # (N, OBS_DIM) float64, post-step observation
    # --- the t side of the same transitions (NAV-062) ---------------------
    #: Signed budget / encoded ratio / safety history / goal distance at the
    #: decision time ``t`` (the state ``obs`` and the critics were evaluated at).
    z_J_before: np.ndarray = None  # type: ignore[assignment]
    z_ratio_before: np.ndarray = None  # type: ignore[assignment]
    safety_y_before: np.ndarray = None  # type: ignore[assignment]
    goal_distance_before: np.ndarray = None  # type: ignore[assignment]
    #: ``z_J_before >= 0`` per env: over-budget is reported, never terminal.
    within_budget: np.ndarray = None  # type: ignore[assignment]


# ---------------------------------------------------------------------- env
class VectorNominalNavEnv:
    """``N`` lockstep episodes of the scalar nominal env on one map."""

    def __init__(
        self,
        spec: MapSpec,
        geometry: G.SafetyGeometry,
        cost_model: JCostModel,
        *,
        n_envs: int,
        z0_J: float,
        r_goal_m: float = 0.5,
        t_max_s: float = 30.0,
        stagnation_window_s: float = 3.0,
        stagnation_min_displacement_m: float = 0.05,
        initial_pose_jitter_m: float = 0.0,
        compute_margins: bool = False,
    ) -> None:
        if n_envs < 1:
            raise NominalEnvError(f"n_envs must be >= 1, got {n_envs}")
        if abs(cost_model.dt_s - DT_NAV_S) > 1e-12:
            raise NominalEnvError(
                f"cost model dt={cost_model.dt_s} is not the 0.1 s navigation clock"
            )
        if not math.isfinite(z0_J):
            raise NominalEnvError(f"z0_J must be finite, got {z0_J}")
        self.spec = spec
        self.geometry = geometry
        self.n_envs = int(n_envs)
        self.z0_J = float(z0_J)
        self.r_goal_m = float(r_goal_m)
        self.t_max_s = float(t_max_s)
        self.stagnation_window_s = float(stagnation_window_s)
        self.stagnation_min_displacement_m = float(stagnation_min_displacement_m)
        self.initial_pose_jitter_m = float(initial_pose_jitter_m)
        self.compute_margins = bool(compute_margins)
        self.obs_dim = obs.OBS_DIM

        # --- cached map geometry (identical objects to the scalar env's) ---
        self._obstacle_polys: tuple[Polygon, ...] = tuple(
            ob.polygon() for ob in spec.obstacles()
        )
        self._obstacle_obs = tuple(
            obs.obstacle_from_polygon(poly.vertices, label=f"island-{i}")
            for i, poly in enumerate(self._obstacle_polys)
        )
        # Pre-inflated islands: the frozen env re-inflates every step; the
        # inflation is deterministic, so the cache is bitwise-identical.
        self._inflated_polys: tuple[Polygon, ...] = tuple(
            poly.inflate(geometry.inflation_radius_m) for poly in self._obstacle_polys
        )
        self._inflated_verts: tuple[np.ndarray, ...] = tuple(
            poly.vertices for poly in self._inflated_polys
        )
        self._obstacle_verts: tuple[np.ndarray, ...] = tuple(
            poly.vertices for poly in self._obstacle_polys
        )
        self._walls: tuple[Polygon, ...] = wall_polygons(spec.extent)
        self._wall_verts: tuple[np.ndarray, ...] = tuple(w.vertices for w in self._walls)
        self._extent = obs.MapExtent(*spec.extent)
        self._fp_vertices = geometry.footprint.vertices
        self._r_fp = geometry.r_fp_m
        self._dropped = max(0, len(self._obstacle_obs) - obs.OBSTACLE_CAPACITY)

        # per-env frozen cost models (shared frozen PowerModel, private clock)
        self._cost_models: list[JCostModel] = [
            JCostModel(
                cost_model.model, cost_model.alpha,
                variant=cost_model.variant, alpha_source=cost_model.alpha_source,
            )
            for _ in range(self.n_envs)
        ]

        self._seeds: list[int | None] = [None] * self.n_envs
        #: Per-env initial budget, joules.  ``z0_J`` in the constructor is the
        #: default; :meth:`reset` may set one value per env (NAV-062: budgets are
        #: sampled per episode, so two episodes on one map may differ).
        self._z0 = np.full(self.n_envs, float(z0_J), dtype=np.float64)
        self.reset()

    # ---- lifecycle ----
    def reset(self, seeds: Sequence[int | None] | None = None,
              z0_J: float | Sequence[float] | None = None) -> np.ndarray:
        """Start fresh episodes; the only way to start them, never implicit.

        ``z0_J`` may be one scalar (applied to all envs) or one value per env.
        Passing it changes what the *next* episode runs with and is recorded in
        the step records, so a trace always states its own budget.
        """
        if seeds is None:
            seeds = [None] * self.n_envs
        if len(seeds) != self.n_envs:
            raise NominalEnvError(
                f"seeds must have length n_envs={self.n_envs}, got {len(seeds)}"
            )
        if z0_J is not None:
            arr = np.asarray(z0_J, dtype=np.float64).reshape(-1)
            if arr.size == 1:
                arr = np.full(self.n_envs, float(arr[0]), dtype=np.float64)
            if arr.size != self.n_envs:
                raise NominalEnvError(
                    f"z0_J must be one value or n_envs={self.n_envs} values, got {arr.size}")
            if not np.isfinite(arr).all() or (arr <= 0.0).any():
                raise NominalEnvError(f"z0_J must be positive and finite, got {arr}")
            self._z0 = arr.copy()
        self._seeds = [None if s is None else int(s) for s in seeds]
        n = self.n_envs
        self._pose = np.zeros((n, 3), dtype=np.float64)
        self._t_s = np.zeros(n, dtype=np.float64)
        self._index = np.zeros(n, dtype=np.int64)
        self._done = np.zeros(n, dtype=bool)
        self._label: list[str | None] = [None] * n
        self._safety_y = np.zeros(n, dtype=np.float64)
        self._spent_J = [0.0] * n
        self._travelled_m = [0.0] * n
        self._danger_steps = [0] * n
        self._internal_steps = [0] * n
        self._applied_last = np.zeros((n, 3), dtype=np.float64)
        self._history: list[list[tuple[float, float, float]]] = []
        self._obs = np.zeros((n, self.obs_dim), dtype=np.float64)
        for i, seed in enumerate(self._seeds):
            # Same rng construction and call order as the scalar env's reset.
            rng = np.random.default_rng(seed)
            jitter = self.initial_pose_jitter_m
            if jitter > 0.0:
                dx, dy = rng.uniform(-jitter, jitter, size=2)
                dyaw = float(rng.uniform(-0.05, 0.05))
            else:
                dx = dy = dyaw = 0.0
            self._pose[i] = (
                float(self.spec.start[0]) + float(dx),
                float(self.spec.start[1]) + float(dy),
                float(self.spec.start_yaw) + dyaw,
            )
            self._history.append(
                [(0.0, float(self._pose[i, 0]), float(self._pose[i, 1]))]
            )
        for m in self._cost_models:
            m.reset()
        self._refresh_obs()
        return self._obs.copy()

    # ---- state ----
    @property
    def done(self) -> np.ndarray:
        return self._done.copy()

    @property
    def pose(self) -> np.ndarray:
        return self._pose.copy()

    @property
    def observation(self) -> np.ndarray:
        return self._obs.copy()

    def z_J(self, i: int) -> float:
        return float(self._z0[i] - self._spent_J[i])

    def _refresh_obs(self) -> None:
        for i in range(self.n_envs):
            ego = obs.EgoState(
                px=float(self._pose[i, 0]), py=float(self._pose[i, 1]),
                yaw=float(self._pose[i, 2]),
                vx_body=float(self._applied_last[i, 0]),
                vy_body=float(self._applied_last[i, 1]),
                wz_body=float(self._applied_last[i, 2]),
            )
            task = obs.TaskState(
                goal_x=float(self.spec.goal[0]), goal_y=float(self.spec.goal[1]),
                safety_y=float(self._safety_y[i]),
                z_J=float(self._z0[i] - self._spent_J[i]),
                time_remaining_s=max(0.0, self.t_max_s - float(self._t_s[i])),
            )
            self._obs[i] = obs.encode_observation(
                ego, self._obstacle_obs, self._extent, task
            )

    # ---- geometry (batched) ----
    def _danger_and_collision(
        self,
        p0: np.ndarray, yaw0: np.ndarray,
        p1: np.ndarray, yaw1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched (danger, collision) for the active subset.

        ``p0``/``p1``: ``(Na, 2)`` world positions at the step endpoints.
        """
        n = p0.shape[0]
        danger = np.zeros(n, dtype=bool)
        collision = np.zeros(n, dtype=bool)

        # -- nominal violation: segment vs pre-inflated islands (SAT) --------
        degenerate = (p1[:, 0] == p0[:, 0]) & (p1[:, 1] == p0[:, 1])
        alive = ~degenerate
        for verts in self._inflated_verts:
            if alive.any():
                danger[alive] |= _segment_polygon_batch(p0[alive], p1[alive], verts)
        # degenerate segments: frozen scalar predicate, which handles the
        # zero-length case with its own exact semantics (containment plus the
        # on-edge-line crossing test of ``_segments_intersect``).
        if degenerate.any():
            for k in np.nonzero(degenerate)[0]:
                for poly in self._inflated_polys:
                    if G.segment_intersects_polygon(
                        (float(p0[k, 0]), float(p0[k, 1])),
                        (float(p1[k, 0]), float(p1[k, 1])),
                        poly, tol=0.0,
                    ):
                        danger[k] = True
                        break
        # -- boundary violation at the end pose (cheap half-plane test) -------
        for k in range(n):
            if G.boundary_violation(
                (float(p1[k, 0]), float(p1[k, 1])),
                self.spec.extent, self.geometry,
            ):
                danger[k] = True

        # -- physical collision: swept footprint with rotation margin (SAT) --
        fp0 = np.empty((n, self._fp_vertices.shape[0], 2), dtype=np.float64)
        fp1 = np.empty_like(fp0)
        for k in range(n):
            fp0[k] = _footprint_vertices(self._fp_vertices, (p0[k, 0], p0[k, 1]), yaw0[k])
            fp1[k] = _footprint_vertices(self._fp_vertices, (p1[k, 0], p1[k, 1]), yaw1[k])
        dpsi = np.abs(np.array(
            [wrap_angle(float(yaw1[k]) - float(yaw0[k])) for k in range(n)]
        ))
        margin = self._r_fp * dpsi / 2.0
        for verts in self._obstacle_verts + self._wall_verts:
            collision |= _swept_collision_batch(fp0, fp1, verts, margin)
        return danger, collision

    # ---- one lockstep interval ----
    def step(self, actions: np.ndarray) -> BatchStepRecord:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.n_envs, 3):
            raise NominalEnvError(
                f"actions must be ({self.n_envs}, 3), got {actions.shape}"
            )
        n = self.n_envs
        active = ~self._done
        idx = np.nonzero(active)[0]

        requested = np.zeros((n, 3), dtype=np.float64)
        applied = np.zeros((n, 3), dtype=np.float64)
        clipped = np.zeros(n, dtype=bool)
        invalid = np.zeros(n, dtype=bool)
        pose_after = self._pose.copy()
        t_after = self._t_s.copy()
        index_before = self._index.copy()
        cost_t_before = self._t_s.copy()
        power_W = np.zeros(n, dtype=np.float64)
        cost_nom = np.zeros(n, dtype=np.float64)
        cost_model = np.zeros(n, dtype=np.float64)
        z_after = np.array([self._z0[i] - self._spent_J[i] for i in range(n)])
        safety_after = self._safety_y.copy()
        goal_dist = np.array([
            math.hypot(float(self.spec.goal[0]) - float(self._pose[i, 0]),
                       float(self.spec.goal[1]) - float(self._pose[i, 1]))
            for i in range(n)
        ])
        # the t side of these transitions, captured before anything moves
        z_before = z_after.copy()
        z_ratio_before = np.array([obs.z_ratio_value(float(z)) for z in z_before])
        safety_before = self._safety_y.copy()
        goal_dist_before = goal_dist.copy()
        danger_out = np.zeros(n, dtype=bool)
        internal_out = np.zeros(n, dtype=bool)
        margins_nom = np.full(n, np.nan) if self.compute_margins else None
        margins_phys = np.full(n, np.nan) if self.compute_margins else None

        if idx.size:
            # --- per-env: command, integration, cost (frozen code) ---------
            na = idx.size
            p0 = np.empty((na, 2), dtype=np.float64)
            p1 = np.empty((na, 2), dtype=np.float64)
            yaw0 = np.empty(na, dtype=np.float64)
            yaw1 = np.empty(na, dtype=np.float64)
            for k, ii in enumerate(idx):
                i = int(ii)
                act = tuple(float(v) for v in actions[i])
                command = obs.command_from_action(act)
                requested[i] = command.requested
                applied[i] = command.applied
                clipped[i] = command.clipped
                invalid[i] = command.invalid

                p0[k] = (float(self._pose[i, 0]), float(self._pose[i, 1]))
                yaw0[k] = float(self._pose[i, 2])
                nxt = step_exact(
                    PlanarState(p0[k, 0], p0[k, 1], yaw0[k]),
                    command.applied, DT_NAV_S,
                )
                p1[k] = (float(nxt.px), float(nxt.py))
                raw_yaw = float(nxt.yaw)
                yaw1[k] = (
                    float(wrap_angle(raw_yaw))
                    if not (-math.pi < raw_yaw <= math.pi) else raw_yaw
                )
                # --- cost on the applied command (frozen model) ---
                rec = self._cost_models[i].step(command.applied)
                cost_t_before[i] = float(rec.t_s)
                power_W[i] = rec.power_W
                cost_nom[i] = rec.cost_nom_J
                cost_model[i] = rec.cost_model_J
                # budget.charge, replicated (one float add per step, in order)
                self._spent_J[i] += float(rec.cost_model_J)
                z_after[i] = self._z0[i] - self._spent_J[i]

            # --- geometry, batched (the only hot spot) ---
            danger, collision = self._danger_and_collision(p0, yaw0, p1, yaw1)
            danger_out[idx] = danger
            internal_out[idx] = danger

            if self.compute_margins:
                for k, ii in enumerate(idx):
                    i = int(ii)
                    pos = (float(p1[k, 0]), float(p1[k, 1]))
                    if self._obstacle_polys:
                        margins_nom[i] = min(
                            G.nominal_margin(pos, poly, self.geometry)
                            for poly in self._obstacle_polys
                        )
                        margins_phys[i] = min(
                            G.physical_clearance(pos, float(yaw1[k]), poly, self.geometry)
                            for poly in self._obstacle_polys
                        )
                    else:
                        # no obstacles on this map: the scalar env records None;
                        # NaN is the array-safe stand-in for "not applicable"
                        margins_nom[i] = np.nan
                        margins_phys[i] = np.nan

            # --- per-env state update and labelling (frozen semantics) -----
            for k, ii in enumerate(idx):
                i = int(ii)
                p1x, p1y, y1 = float(p1[k, 0]), float(p1[k, 1]), float(yaw1[k])
                p0x, p0y = float(p0[k, 0]), float(p0[k, 1])
                d = bool(danger[k])
                self._safety_y[i] = G.update_safety_history(float(self._safety_y[i]), d)
                if d:
                    self._danger_steps[i] += 1
                    self._internal_steps[i] += 1
                self._t_s[i] = float(self._t_s[i]) + DT_NAV_S
                self._travelled_m[i] += math.hypot(p1x - p0x, p1y - p0y)
                self._pose[i] = (p1x, p1y, y1)
                self._applied_last[i] = applied[i]
                self._history[i].append((float(self._t_s[i]), p1x, p1y))
                gd = math.hypot(
                    float(self.spec.goal[0]) - p1x, float(self.spec.goal[1]) - p1y
                )
                goal_dist[i] = gd

                label: str | None = None
                if bool(collision[k]):
                    label = "collision"
                elif not (
                    self.spec.extent[0] <= p1x <= self.spec.extent[1]
                    and self.spec.extent[2] <= p1y <= self.spec.extent[3]
                ):
                    label = "out_of_bounds"
                elif gd <= self.r_goal_m:
                    label = "goal_reached"
                # No budget branch (NAV-062): z <= 0 is reported through
                # `within_budget`, exactly like the scalar env and the reference
                # implementation, whose `is_terminal` only looks at the horizon.
                elif self._stagnated(i):
                    label = "stagnation"
                elif float(self._t_s[i]) >= self.t_max_s - 1e-12:
                    label = "timeout"
                if label is not None:
                    self._done[i] = True
                    self._label[i] = label
                t_after[i] = float(self._t_s[i])
                pose_after[i] = (p1x, p1y, y1)
                safety_after[i] = float(self._safety_y[i])
                self._index[i] += 1

        self._refresh_obs()
        return BatchStepRecord(
            active=active,
            index=index_before,
            requested=requested,
            applied=applied,
            clipped=clipped,
            invalid=invalid,
            pose_after=pose_after,
            t_s=t_after,
            cost_t_s=cost_t_before,
            power_W=power_W,
            cost_nom_J=cost_nom,
            cost_model_J=cost_model,
            z_J=z_after,
            z_ratio=np.array([self._z_ratio_of(z) for z in z_after]),
            safety_y=safety_after,
            danger_violation=danger_out,
            internal_violation=internal_out,
            goal_distance_m=goal_dist,
            min_nominal_margin_m=margins_nom,
            min_physical_clearance_m=margins_phys,
            dropped_obstacles=self._dropped,
            label=tuple(self._label),
            done=self._done.copy(),
            obs=self._obs.copy(),
            z_J_before=z_before,
            z_ratio_before=z_ratio_before,
            safety_y_before=safety_before,
            goal_distance_before=goal_dist_before,
            within_budget=z_before >= 0.0,
        )

    def _z_ratio_of(self, z: float) -> float:
        return obs.z_ratio_value(float(z))

    def _stagnated(self, i: int) -> bool:
        """Frozen scalar procedure, per env."""
        window = self.stagnation_window_s
        t = float(self._t_s[i])
        if t < window - 1e-12:
            return False
        anchor = None
        for tt, x, y in self._history[i]:
            if t - tt <= window + 1e-9:
                anchor = (x, y)
                break
        if anchor is None:
            return False
        moved = math.hypot(float(self._pose[i, 0]) - anchor[0],
                           float(self._pose[i, 1]) - anchor[1])
        return moved < self.stagnation_min_displacement_m

    # ---- reporting ----
    def result(self, i: int) -> EpisodeResult:
        """Scalar-shaped :class:`EpisodeResult` for env ``i``.

        ``cost_nom_J`` is not accumulated here (per-step costs live in the
        trainer's buffers); ``cost_model_J`` equals the total spend, which is
        what every downstream consumer of the scalar field uses it for.
        """
        return EpisodeResult(
            map_id=self.spec.map_id,
            level=self.spec.level,
            seed=self.spec.seed,
            label=self._label[i],
            done=bool(self._done[i]),
            n_steps=int(self._index[i]),
            t_s=float(self._t_s[i]),
            travelled_m=float(self._travelled_m[i]),
            goal_distance_m=math.hypot(
                float(self.spec.goal[0]) - float(self._pose[i, 0]),
                float(self.spec.goal[1]) - float(self._pose[i, 1]),
            ),
            cost_nom_J=float("nan"),  # not accumulated; see docstring
            cost_model_J=float(self._spent_J[i]),
            z0_J=float(self._z0[i]),
            z_J=float(self._z0[i] - self._spent_J[i]),
            safety_y=float(self._safety_y[i]),
            danger_steps=self._danger_steps[i],
            internal_violation_steps=self._internal_steps[i],
            alpha=self._cost_models[i].alpha,
            variant=self._cost_models[i].variant,
            within_budget=bool(self._z0[i] - self._spent_J[i] >= 0.0),
            arrived_within_budget=bool(
                self._label[i] == "goal_reached"
                and self._z0[i] - self._spent_J[i] >= 0.0),
            over_budget_J=max(0.0, float(-(self._z0[i] - self._spent_J[i]))),
        )
