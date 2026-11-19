"""Independent finite-trajectory targets for NAV-065 (obs-v7/terminal-tail-v5).

The deterministic full-return targets use no network predictions. They are
reference values for deployment checks, not the approximate training lambda
targets. Episode feasibility is read separately from the full trajectory.
"""

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from camra_navigation.observation import MAP_HALF_M, field_slices, observation_dim


@dataclass(frozen=True)
class TrajectoryTruth:
    observations: np.ndarray
    budget_J: np.ndarray
    stop_index: np.ndarray
    cost_target: np.ndarray
    reach_target: np.ndarray
    path_residual: np.ndarray
    feasible: bool
    budget_feature_clipped: np.ndarray


def joint_residual(cost_norm, reach, budget_J, z_ref_J: float):
    """Critics output normalized values; the signed budget is still in J."""
    if not math.isfinite(z_ref_J) or z_ref_J <= 0:
        raise ValueError("invalid fixed budget scale")
    cost, reach, budget = (np.asarray(x, dtype=np.float64)
                           for x in (cost_norm, reach, budget_J))
    if cost.shape != reach.shape or reach.shape != budget.shape:
        raise ValueError("value and budget rows must align")
    if not all(np.isfinite(x).all() for x in (cost, reach, budget)):
        raise ValueError("nonfinite budget or value")
    return np.maximum(reach, cost - budget / z_ref_J)


def reconstruct_truth(
    episode: Mapping,
    *,
    initial_goal_distance_m: float,
    value_contract: Mapping,
    horizon_s: float,
    goal_radius_m: float,
    step_s: float,
) -> TrajectoryTruth:
    """Use exact map reset distance and logged post-step states to build t..T."""
    q = value_contract["quantities"]
    est = value_contract["estimator"]
    if (est["version"] != "rcppo-terminal-tail-v5" or est["gamma_cost"] != 1.0
            or est["gamma_joint"] != 1.0 or q["beta"]["value"] != 0.0):
        raise ValueError("oracle requires the bound finite, undiscounted objective")
    z_ref = float(q["Z_REF_J"]["value"])
    clip_lo, clip_hi = q["z_ratio"]["range"]
    l_ref = q["g_norm"]["L_REF_m"]
    fail_penalty = value_contract["bootstrap"]["FAIL_PENALTY"]["value"]
    if not all(math.isfinite(x) and x > 0.0 for x in (
            z_ref, l_ref, fail_penalty, horizon_s, goal_radius_m, step_s)):
        raise ValueError("invalid task scale")

    label = episode["label"]
    failure = label in ("collision", "out_of_bounds")
    if not failure and label not in ("goal_reached", "timeout", "stagnation"):
        raise ValueError(f"not a complete v5 trajectory: {label}")
    steps = episode["trace"]
    n = len(steps)
    elapsed = n * step_s
    if (n == 0 or episode["n_steps"] != n
            or not math.isclose(elapsed, episode["t_s"], abs_tol=1e-9)
            or elapsed > horizon_s + 1e-9
            or (label == "timeout" and not math.isclose(elapsed, horizon_s, abs_tol=1e-9))):
        raise ValueError("incomplete or inconsistent task timing")
    if any(row["step"] != i for i, row in enumerate(steps)):
        raise ValueError("nonconsecutive step index")

    obs = np.asarray([r["observation"] for r in steps], dtype=np.float64)
    z = np.asarray([episode["z0_J"], *(r["z_J"] for r in steps)], dtype=np.float64)
    y = np.asarray([0.0, *(r["safety_y"] for r in steps)], dtype=np.float64)
    distance = np.asarray([initial_goal_distance_m, *(r["goal_distance_m"] for r in steps)])
    if obs.shape != (n, observation_dim("obs-v7")) or not all(
            np.isfinite(x).all() for x in (obs, z, y, distance)):
        raise ValueError("invalid obs-v7 trajectory fields")
    if (np.any(distance < 0.0) or np.any(distance[:-1] <= goal_radius_m)
            or (not failure and (distance[-1] <= goal_radius_m) != (label == "goal_reached"))):
        raise ValueError("distance and arrival event disagree")
    if any(type(r["danger_violation"]) is not bool for r in steps):
        raise ValueError("danger flag must be boolean")
    danger = np.asarray([r["danger_violation"] for r in steps])
    if (np.any(y > 1.0) or np.any(np.diff(y) < 0.0)
            or np.any((np.diff(y) > 0.0) & ~danger)
            or np.any(danger & (y[1:] <= 0.0))
            or int(danger.sum()) != episode["danger_steps"]):
        raise ValueError("inconsistent irreversible safety history")
    if (z[0] <= 0.0 or np.any(np.diff(z) > 0.0)
            or not math.isclose(z[-1], episode["z_J_end"], abs_tol=1e-8)
            or not math.isclose(z[0] - z[-1], episode["cost_model_J"], abs_tol=1e-8)):
        raise ValueError("inconsistent budget/cost ledger in J")

    pre_budget = z[:-1] / z_ref
    encoded_budget = np.clip(pre_budget, clip_lo, clip_hi)
    fields = field_slices("obs-v7")
    for name, expected in (
        ("z_ratio", encoded_budget),
        ("safety_y", y[:-1]),
        ("goal_dist_n", np.clip(distance[:-1] / MAP_HALF_M, 0.0, 1.5)),
        ("time_remaining_n", (horizon_s - np.arange(n) * step_s) / horizon_s),
    ):
        if not np.allclose(obs[:, fields[name]].ravel(), expected, atol=1e-7, rtol=0.0):
            raise ValueError(f"pre-action {name} mismatch")

    g = np.where(distance <= goal_radius_m, q["g_norm"]["G_INSIDE"],
                 (distance - goal_radius_m) / l_ref)
    h = np.where(y > 0, np.maximum(g, q["C"]["value"] * y), g)
    if failure:
        h[-1] = fail_penalty

    # With gamma=1 and a real terminal, spent(t,s)-z_t is exactly -z_s.
    # Select the earliest minimum in each suffix; hard failure forces T.
    scores = np.maximum(h, -z / z_ref)
    stop = np.full(n, n, dtype=np.int64)
    if not failure:
        best = n
        for t in range(n - 1, -1, -1):
            if scores[t] <= scores[best]:
                best = t
            stop[t] = best
    cost = (z[:-1] - z[stop]) / z_ref
    if failure:
        # Match the effective last-step replacement, not an extra physical cost.
        cost = (z[:-1] - z[-2]) / z_ref + fail_penalty

    feasible = bool(label == "goal_reached" and y[-1] == 0.0 and z[-1] >= 0.0)
    if feasible != bool(scores[stop[0]] <= 0.0) or feasible != episode["feasible"]:
        raise ValueError("trajectory feasibility, residual and event summary disagree")
    return TrajectoryTruth(obs, z[:-1].copy(), stop, cost, h[stop], scores[stop],
                           feasible, pre_budget != encoded_budget)
