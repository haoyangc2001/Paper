"""Metric definitions for locomotion evaluation (TASK-020).

Pure NumPy, no Isaac/Sim imports: the evaluator and an independent recompute
script must be able to share exactly the same formulas, so that "every metric is
recomputable from the raw step log" is a checkable claim rather than a promise.

Formulas follow ``CaMRA/locomotion/docs/eval_protocol.json`` v1.0.0 verbatim:

  rmse_*       sqrt(mean((actual - commanded)^2)) over the evaluation window
  p95_abs_err 95th percentile of |actual - commanded|
  rise_time_s  from step onset to the first sample reaching 90% of the step that
               also stays above 80% for 0.2 s
  stop_drift_m planar base displacement between command reaching 0 and +2.0 s
  overshoot    max(actual_vx) - step_magnitude, floored at 0
  stand_drift  max planar displacement over the window
  failure_rate fraction of episodes ended by base_contact / bad_orientation

All tracking metrics exclude the first ``warmup_s`` seconds after reset.
"""

from __future__ import annotations

import numpy as np

#: Step-record layout. Keep in sync with the evaluator (it imports this list).
FIELDS: list[str] = (
    ["sim_time", "episode_step", "cmd_vx", "cmd_vy", "cmd_wz", "app_vx", "app_vy", "app_wz"]
    + ["vx_body", "vy_body", "wz_body"]
    + ["roll", "pitch", "yaw"]
    + ["pos_x", "pos_y", "pos_z"]
    + [f"joint_pos_{i}" for i in range(12)]
    + [f"joint_vel_{i}" for i in range(12)]
    + [f"action_{i}" for i in range(12)]
    + ["terminated", "truncated", "reason_id"]
)

NF: int = len(FIELDS)
I: dict[str, int] = {name: k for k, name in enumerate(FIELDS)}

#: termination reason ids
REASONS: dict[int, str] = {
    0: "none",
    1: "base_contact",
    2: "bad_orientation",
    3: "time_out",
    4: "scenario_end",
    5: "cap",
}

ACTION_SLICE = slice(I["action_0"], I["action_0"] + 12)


def compute_metrics(ep: np.ndarray, spec: dict, step_dt: float, warmup_s: float) -> dict:
    """All metrics for one episode, from its ``(T, NF)`` step block."""
    ep = np.asarray(ep, dtype=np.float64)
    out: dict = {"steps": int(ep.shape[0])}

    ep_step = ep[:, I["episode_step"]]
    warm = (ep_step * step_dt) > warmup_s  # strict: the whole first settle_s window is excluded
    if warm.sum() < 2:
        warm = np.ones_like(warm, dtype=bool)

    vx, vy, wz = ep[:, I["vx_body"]], ep[:, I["vy_body"]], ep[:, I["wz_body"]]
    cx, cy, cw = ep[:, I["cmd_vx"]], ep[:, I["cmd_vy"]], ep[:, I["cmd_wz"]]

    def rmse(a, b):
        return float(np.sqrt(np.mean((a[warm] - b[warm]) ** 2)))

    def p95(a, b):
        return float(np.percentile(np.abs(a[warm] - b[warm]), 95))

    out["rmse_vx"] = rmse(vx, cx)
    out["rmse_vy"] = rmse(vy, cy)
    out["rmse_wz"] = rmse(wz, cw)
    out["p95_abs_err_vx"] = p95(vx, cx)
    out["p95_abs_err_vy"] = p95(vy, cy)
    out["p95_abs_err_wz"] = p95(wz, cw)

    roll = np.degrees(ep[:, I["roll"]])
    pitch = np.degrees(ep[:, I["pitch"]])
    out["max_roll_deg"] = float(np.max(np.abs(roll[warm])))
    out["max_pitch_deg"] = float(np.max(np.abs(pitch[warm])))

    px, py = ep[:, I["pos_x"]], ep[:, I["pos_y"]]
    out["stand_drift_m"] = float(np.max(np.hypot(px[warm] - px[warm][0], py[warm] - py[warm][0])))

    mag = float(np.max(np.abs(cx))) if cx.size else 0.0
    out["rise_time_s"] = float("nan")
    out["overshoot_mps"] = float("nan")
    out["stop_drift_m"] = float("nan")
    if mag > 1e-6:
        onset = int(np.argmax(np.abs(cx) > 0.5 * mag))
        target = 0.9 * mag
        hold = max(1, int(round(0.2 / step_dt)))
        rt = float("nan")
        for idx in np.nonzero(vx[onset:] >= target)[0]:
            i0 = onset + int(idx)
            seg = vx[i0 : i0 + hold]
            if len(seg) >= hold and np.all(seg >= 0.8 * mag):
                rt = float(i0 * step_dt - onset * step_dt)
                break
        out["rise_time_s"] = rt
        out["overshoot_mps"] = float(max(0.0, np.max(vx[warm]) - mag))
        zero_idx = np.nonzero(np.abs(cx) < 0.05 * mag)[0]
        zero_idx = zero_idx[zero_idx > onset]
        if len(zero_idx):
            i0 = int(zero_idx[0])
            i1 = min(int(round(i0 + 2.0 / step_dt)), len(px) - 1)
            out["stop_drift_m"] = float(np.hypot(px[i1] - px[i0], py[i1] - py[i0]))

    act = ep[:, ACTION_SLICE]
    out["action_rate_rms"] = float(np.mean(np.linalg.norm(np.diff(act, axis=0), axis=1))) if act.shape[0] > 1 else float("nan")

    term = ep[:, I["terminated"]].astype(bool)
    trunc = ep[:, I["truncated"]].astype(bool)
    # the reason belongs to the step where the episode actually ended. Taking the
    # last row silently reports "none" when anything is recorded after termination.
    end_idx = np.nonzero(term | trunc)[0]
    reason_row = int(end_idx[0]) if len(end_idx) else len(ep) - 1
    reason = int(ep[reason_row, I["reason_id"]])
    out["ended_by"] = REASONS.get(reason, "unknown")
    out["failed"] = int(reason in (1, 2))
    out["steps_before_end"] = int(np.argmax(term | trunc) + 1) if (term.any() or trunc.any()) else len(term)
    out["duration_s"] = float(ep[-1, I["sim_time"]] - ep[0, I["sim_time"]] + step_dt)
    out["scenario"] = spec["id"]
    return out


def aggregate(rows: list[dict], spec: dict, group: str) -> dict:
    """Scenario-level aggregation plus threshold / failure-rate judgement."""
    n = len(rows)
    relax = 1.2 if group == "perturbed" else 1.0
    agg: dict = {"scenario": spec["id"], "group": group, "episodes": n}
    agg["failure_rate"] = sum(r["failed"] for r in rows) / n if n else float("nan")

    checks: dict = {}
    for m in spec["metrics"]:
        if m == "failure_rate":
            continue
        vals = np.array([r[m] for r in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        agg[m] = float(np.mean(finite)) if len(finite) else float("nan")
        limit = spec["thresholds"].get(m)
        if limit is not None:
            lim = float(limit) * (relax if m.startswith("rmse") else 1.0)
            checks[m] = {"value": agg[m], "threshold": lim, "pass": bool(np.isfinite(agg[m]) and agg[m] <= lim)}

    fr_limit = 0.10 if group == "perturbed" else 0.05
    checks["failure_rate"] = {
        "value": agg["failure_rate"],
        "threshold": fr_limit,
        "pass": bool(agg["failure_rate"] <= fr_limit),
    }
    agg["checks"] = checks
    agg["pass"] = all(c["pass"] for c in checks.values())
    return agg


def load_episode_blocks(path: str) -> tuple[list[np.ndarray], dict]:
    """Read an evaluator ``.npz`` and split the concatenated steps back per episode."""
    data = np.load(path, allow_pickle=True)
    steps = data["steps"]
    lengths = data["lengths"].astype(int)
    blocks, off = [], 0
    for n in lengths:
        blocks.append(steps[off : off + n])
        off += n
    meta = {
        "fields": [str(x) for x in data["fields"]],
        "step_dt": float(data["step_dt"][0]),
        "group": str(data["group"][0]) if "group" in data else "",
        "scenario": str(data["scenario"][0]) if "scenario" in data else "",
    }
    return blocks, meta
