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

import json
from collections import Counter

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

#: A gated metric must have at least this fraction of its episodes measured
#: (protocol supplement D-3: 90 %, with genuinely inapplicable episodes excluded
#: from the denominator).
MIN_VALID_RATIO = 0.90

#: Non-finite reasons that legitimately remove an episode from the denominator.
#: Only "the command program does not define this metric" qualifies. Everything
#: else - no response, truncated window, short episode, missing key - is missing
#: evidence and still counts against the validity ratio.
NA_REASONS = frozenset({"undefined_program"})


def _reasons_of(row: dict) -> dict:
    """Per-episode invalid-reason map, from memory (dict) or from the CSV (JSON text)."""
    raw = row.get("invalid_reasons")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            val = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return val if isinstance(val, dict) else {}
    return {}


def variant_signs(spec: dict) -> list[tuple[int, int, int]]:
    """Sign multipliers of the command variants a scenario must cover.

    Single source of truth: the evaluator builds its programs from this list and
    the cross-seed aggregator counts variants with it, so "how many variants does
    this scenario have" cannot drift between the two.
    """
    prog = spec["command_program"]
    if prog.get("type") != "constant":
        return [(1, 1, 1)]
    n = int(prog.get("sign_combinations") or 0)
    if n:
        # the protocol's sign_combinations=4 means these four diagonal combinations,
        # not all eight sign flips
        return [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)][:n]
    if prog.get("vy_sign") == "both":
        return [(1, 1, 1), (1, -1, 1)]
    if prog.get("wz_sign") == "both":
        return [(1, 1, 1), (1, 1, -1)]
    return [(1, 1, 1)]


def program_variants_count(spec: dict) -> int:
    return len(variant_signs(spec))


def _as_bool(v) -> bool:
    """CSV cells arrive as text: "False"/"0" must not be read as truthy."""
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return bool(v)


def _as_failed(r: dict) -> bool:
    v = r.get("failed", 0)
    try:
        return int(float(v)) != 0
    except (TypeError, ValueError):
        return False


def _collect(rows: list[dict], metric: str) -> tuple[list[float], int, int, dict]:
    """Split one metric column into valid values, N/A count, valid count and reasons."""
    values: list[float] = []
    reasons: Counter = Counter()
    valid = 0
    na = 0
    for r in rows:
        if metric not in r or r[metric] is None or r[metric] == "":
            reasons["missing_key"] += 1
            continue
        try:
            v = float(r[metric])
        except (TypeError, ValueError):
            reasons["not_numeric"] += 1
            continue
        if np.isfinite(v):
            if _as_bool(r.get("warmup_short", False)):
                # the episode ended before the settle window: the sample is not a measurement
                reasons["warmup_not_reached"] += 1
                continue
            valid += 1
            values.append(v)
            continue
        reason = _reasons_of(r).get(metric, "non_finite")
        reasons[reason] += 1
        if reason in NA_REASONS:
            na += 1
    return values, valid, na, dict(reasons)


def compute_metrics(ep: np.ndarray, spec: dict, step_dt: float, warmup_s: float) -> dict:
    """All metrics for one episode, from its ``(T, NF)`` step block."""
    ep = np.asarray(ep, dtype=np.float64)
    out: dict = {"steps": int(ep.shape[0])}
    #: metric -> why the value is not a usable measurement (empty dict = all valid)
    invalid: dict[str, str] = {}
    out["invalid_reasons"] = invalid

    ep_step = ep[:, I["episode_step"]]
    warm = (ep_step * step_dt) > warmup_s  # strict: the whole first settle_s window is excluded
    out["warmup_short"] = bool(warm.sum() < 2)
    out["warmup_samples"] = int(warm.sum())
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
        if np.isnan(rt):
            # the command asked for motion but the base never held 90% of it for 0.2 s
            invalid["rise_time_s"] = "not_reached"
        out["overshoot_mps"] = float(max(0.0, np.max(vx[warm]) - mag))
        zero_idx = np.nonzero(np.abs(cx) < 0.05 * mag)[0]
        after = zero_idx[zero_idx > onset]
        if len(after):
            i0 = int(after[0])
            i1 = min(int(round(i0 + 2.0 / step_dt)), len(px) - 1)
            out["stop_drift_m"] = float(np.hypot(px[i1] - px[i0], py[i1] - py[i0]))
            if i1 - i0 + 1 < int(round(2.0 / step_dt)):
                invalid["stop_drift_m"] = "window_truncated"
        else:
            # the command program never returns to ~0, so a stopping distance does not exist
            invalid["stop_drift_m"] = "undefined_program"
    else:
        # a pure standing or rotation command has no forward step to time
        invalid["rise_time_s"] = "undefined_program"
        invalid["overshoot_mps"] = "undefined_program"
        invalid["stop_drift_m"] = "undefined_program"

    act = ep[:, ACTION_SLICE]
    out["action_rate_rms"] = float(np.mean(np.linalg.norm(np.diff(act, axis=0), axis=1))) if act.shape[0] > 1 else float("nan")
    if np.isnan(out["action_rate_rms"]):
        invalid["action_rate_rms"] = "short_trajectory"

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


def aggregate(rows: list[dict], spec: dict, group: str, min_valid_ratio: float = MIN_VALID_RATIO) -> dict:
    """Scenario-level aggregation plus threshold / failure-rate judgement.

    Every gated metric carries its own validity accounting (expected / valid /
    N/A / invalid + reasons). A metric whose measured fraction falls below
    ``min_valid_ratio`` is judged ``insufficient_evidence`` - never ``pass`` -
    so "1 valid value + 29 missing" cannot produce a PASS any more.
    """
    n = len(rows)
    relax = 1.2 if group == "perturbed" else 1.0
    agg: dict = {"scenario": spec["id"], "group": group, "episodes": n, "min_valid_ratio": float(min_valid_ratio)}
    agg["failure_rate"] = sum(_as_failed(r) for r in rows) / n if n else float("nan")

    checks: dict = {}
    for m in spec["metrics"]:
        if m == "failure_rate":
            continue
        values, valid, na, reasons = _collect(rows, m)
        denom = n - na
        ratio = (valid / denom) if denom > 0 else 0.0
        agg[m] = float(np.mean(values)) if values else float("nan")
        limit = spec["thresholds"].get(m)
        entry: dict = {
            "value": agg[m],
            "expected_count": n,
            "valid_count": valid,
            "na_count": na,
            "invalid_count": max(0, denom - valid),
            "valid_ratio": ratio,
            "invalid_reasons": reasons,
        }
        if limit is None:
            entry["threshold"] = None
            entry["status"] = "reported_only"
            entry["pass"] = True
        else:
            lim = float(limit) * (relax if m.startswith("rmse") else 1.0)
            entry["threshold"] = lim
            if denom <= 0 or valid == 0 or ratio < min_valid_ratio - 1e-12:
                entry["status"] = "insufficient_evidence"
            elif agg[m] > lim:
                entry["status"] = "fail"
            else:
                entry["status"] = "pass"
            entry["pass"] = entry["status"] == "pass"
        checks[m] = entry

    fr_limit = 0.10 if group == "perturbed" else 0.05
    fr_ok = bool(n > 0 and np.isfinite(agg["failure_rate"]) and agg["failure_rate"] <= fr_limit)
    checks["failure_rate"] = {
        "value": agg["failure_rate"],
        "threshold": fr_limit,
        "expected_count": n,
        "valid_count": n,
        "na_count": 0,
        "invalid_count": 0,
        "valid_ratio": 1.0 if n else 0.0,
        "invalid_reasons": {} if n else {"no_episodes": 1},
        "status": "pass" if fr_ok else ("fail" if n else "insufficient_evidence"),
        "pass": fr_ok,
    }
    agg["checks"] = checks
    agg["pass"] = bool(n > 0) and all(c["pass"] for c in checks.values())
    agg["complete"] = bool(n > 0) and all(c["status"] != "insufficient_evidence" for c in checks.values())
    agg["incomplete_reason"] = None if agg["complete"] else ("no_episodes" if n == 0 else "insufficient_evidence")
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
