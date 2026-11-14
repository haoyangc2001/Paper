"""NAV-064 step-0 X-ray: what is the training signal actually rewarding?

Read-only diagnostic over the frozen r11 rollout chunks.  It does not change any
code, contract, prereg or checkpoint.  Output: reports/training_round_r11_xray.json

Questions (each answered with numbers, not impressions):
  Q1  In the per-step score, how big is the distance term vs the electricity term?
  Q2  Do forward-moving actions receive a WORSE advantage than idle ones, after
      controlling for how far the goal is?  (Minimization PPO lowers the
      probability of actions with positive advantage.)
  Q3  How often is the common-stop point "right here" (zero look-ahead)?
  Q4  Does the arrival step carry any distinctive signal at all?
  Q5  Does the reach critic learn "closer is better" (slope vs goal distance)?
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
WORK = "/pub/data/caohy/work/camra_navigation"
RUNS = [
    ("main_uncal", "nav064_r11_uncalibrated_s0"),
    ("main_full", "nav064_r11_full_s0"),
    ("control_uncal", "nav064_r11_uncalibrated_s0_steeronly"),
]
N_CHUNKS = 120           # evenly spaced across the whole training timeline
ENVS = list(range(0, 18, 3))   # 6 of 18 envs, spread over the map list
FIELDS = ["action_raw", "goal_distance_m", "goal_distance_m_next", "cost_norm",
          "obj_g", "obj_g_train", "obj_adv_total", "obj_stop_now",
          "value_reach", "terminated", "z_norm", "obj_feasible"]


def chunk_paths(run_id: str) -> list[str]:
    out = []
    for e in ENVS:
        fs = sorted(glob.glob(os.path.join(WORK, run_id, f"rollout_env{e}",
                                          "rollout", "chunk_*.npz")))
        if not fs:
            continue
        idx = np.linspace(0, len(fs) - 1, N_CHUNKS).astype(int)
        out += [(e, i, len(fs) - 1, fs[i]) for i in idx]
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bucket_stats(rows: dict, label: str) -> dict:
    """Q2 within distance strata, so 'far away => move more' cannot fake it."""
    d = rows["goal_distance_m"]
    vx = rows["action_raw"][:, 0]
    adv = rows["obj_adv_total"]
    edges = [0.0, 3.0, 6.0, 9.0, 12.0, 100.0]
    strata = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 30:
            continue
        cut = np.median(vx[m])
        hi_v = m & (vx > cut)
        lo_v = m & (vx <= cut)
        strata.append({
            "dist_m": f"{lo:.0f}-{hi:.0f}", "n": int(m.sum()),
            "vx_high": round(float(vx[hi_v].mean()), 4),
            "vx_low": round(float(vx[lo_v].mean()), 4),
            "adv_when_moving": round(float(adv[hi_v].mean()), 5),
            "adv_when_idle": round(float(adv[lo_v].mean()), 5),
            "adv_penalty_for_moving": round(float(adv[hi_v].mean()
                                                  - adv[lo_v].mean()), 5),
        })
    return {"by_distance": strata,
            "corr_vx_adv": round(corr(vx, adv), 4),
            "corr_dist_adv": round(corr(d, adv), 4)}


def analyse(run_tag: str, run_id: str) -> dict:
    paths = chunk_paths(run_id)
    acc = defaultdict(list)
    phases = defaultdict(lambda: defaultdict(list))
    n_files = 0
    for env_i, idx, last_idx, p in paths:
        with np.load(p) as z:
            phase = min(3, int(4 * idx / max(1, last_idx + 1)))
            d = {f: np.asarray(z[f]) for f in FIELDS}
        for f in FIELDS:
            acc[f].append(d[f])
        for f in ("obj_g", "obj_g_train", "obj_adv_total", "obj_stop_now",
                  "value_reach", "goal_distance_m", "action_raw"):
            phases[phase][f].append(d[f])
        n_files += 1
    rows = {f: np.concatenate(v) for f, v in acc.items()}

    g = rows["obj_g"]
    gt = rows["obj_g_train"]
    cost = rows["cost_norm"]
    adv = rows["obj_adv_total"]
    dist = rows["goal_distance_m"]
    vr = rows["value_reach"]
    stop_now = rows["obj_stop_now"].astype(bool)
    arrived = rows["terminated"].astype(bool) & (rows["goal_distance_m_next"] <= 0.5)

    # Q4: arrival vs ordinary terminal / ordinary step
    non_arr = ~arrived
    q4 = {
        "n_arrival_steps": int(arrived.sum()),
        "g_at_arrival": round(float(g[arrived].mean()), 5) if arrived.any() else None,
        "g_elsewhere": round(float(g[non_arr].mean()), 5),
        "adv_at_arrival": round(float(adv[arrived].mean()), 5) if arrived.any() else None,
        "adv_elsewhere": round(float(adv[non_arr].mean()), 5),
        "obj_feasible_at_arrival": (round(float(rows["obj_feasible"][arrived].mean()), 4)
                                    if arrived.any() else None),
        "obj_feasible_elsewhere": round(float(rows["obj_feasible"][non_arr].mean()), 4),
    }

    by_phase = {}
    for ph in sorted(phases):
        pp = {f: np.concatenate(v) for f, v in phases[ph].items()}
        by_phase[f"P{ph}"] = {
            "n": int(len(pp["obj_g"])),
            "g_mean": round(float(pp["obj_g"].mean()), 5),
            "electricity_term_mean": round(float((pp["obj_g_train"]
                                                  - pp["obj_g"]).mean()), 5),
            "adv_mean": round(float(pp["obj_adv_total"].mean()), 5),
            "stop_now_frac": round(float(pp["obj_stop_now"].mean()), 4),
            "value_reach_mean": round(float(pp["value_reach"].mean()), 5),
            "corr_reach_dist": round(corr(pp["goal_distance_m"], pp["value_reach"]), 4),
            "vx_mean": round(float(pp["action_raw"][:, 0].mean()), 4),
        }

    return {
        "run_id": run_id, "tag": run_tag,
        "n_chunk_files": n_files, "n_samples": int(len(g)),
        "Q1_score_composition": {
            "distance_term_g_mean": round(float(g.mean()), 5),
            "distance_term_g_std": round(float(g.std()), 5),
            "cost_norm_mean": round(float(cost.mean()), 6),
            "electricity_term_mean": round(float((gt - g).mean()), 5),
            "electricity_term_max": round(float((gt - g).max()), 5),
            "frac_steps_with_electricity_term": round(float(np.mean(gt - g > 1e-9)), 4),
            "ratio_electricity_to_per_step_distance_gain": (
                round(float((gt - g).mean() / (0.0465 / 15.0)), 1)
                if (gt - g).mean() > 0 else None),
        },
        "Q2_action_vs_advantage": bucket_stats(rows, run_tag),
        "Q3_zero_lookahead": {
            "stop_now_frac_overall": round(float(stop_now.mean()), 4),
            "adv_mean_when_stop_now": (round(float(adv[stop_now].mean()), 5)
                                       if stop_now.any() else None),
            "adv_mean_when_not": (round(float(adv[~stop_now].mean()), 5)
                                  if (~stop_now).any() else None),
        },
        "Q4_arrival_signal": q4,
        "Q5_reach_critic": {
            "corr_value_reach_vs_distance_overall": round(corr(dist, vr), 4),
            "value_reach_std": round(float(vr.std()), 5),
            "g_std": round(float(g.std()), 5),
        },
        "by_training_phase": by_phase,
    }


def main() -> int:
    rep = {
        "schema": "nav064-r11-xray-v1",
        "note": "Read-only diagnostic over frozen r11 rollout chunks. "
                "Chunk index is a monotone (not linear) proxy for training progress; "
                "P0..P3 are quartiles of chunk index. Minimization-PPO convention: "
                "positive obj_adv_total means WORSE than the baseline, so the "
                "probability of that action is pushed DOWN.",
        "method": {"n_chunks_per_env": N_CHUNKS, "envs_sampled": ENVS,
                   "fields": FIELDS},
        "runs": [analyse(tag, rid) for tag, rid in RUNS],
    }
    out = os.path.join(NAV, "reports", "training_round_r11_xray.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=str)

    for r in rep["runs"]:
        print(f"\n=== {r['tag']}  chunks={r['n_chunk_files']} samples={r['n_samples']} ===")
        q1 = r["Q1_score_composition"]
        print(f"Q1 distance_term={q1['distance_term_g_mean']} "
              f"electricity_term={q1['electricity_term_mean']} "
              f"frac_with_elec={q1['frac_steps_with_electricity_term']} "
              f"ratio_to_one_step_gain={q1['ratio_electricity_to_per_step_distance_gain']}")
        q2 = r["Q2_action_vs_advantage"]
        print(f"Q2 corr(vx,adv)={q2['corr_vx_adv']} corr(dist,adv)={q2['corr_dist_adv']}")
        for s in q2["by_distance"]:
            print(f"   d={s['dist_m']:>6}m n={s['n']:>6} "
                  f"moving_adv={s['adv_when_moving']:+.5f} "
                  f"idle_adv={s['adv_when_idle']:+.5f} "
                  f"penalty={s['adv_penalty_for_moving']:+.5f}")
        print(f"Q3 stop_now_frac={r['Q3_zero_lookahead']['stop_now_frac_overall']}")
        print(f"Q4 {r['Q4_arrival_signal']}")
        print(f"Q5 {r['Q5_reach_critic']}")
        for ph, v in r["by_training_phase"].items():
            print(f"   {ph} n={v['n']:>7} g={v['g_mean']:.4f} "
                  f"elec={v['electricity_term_mean']:.4f} adv={v['adv_mean']:+.4f} "
                  f"stop0={v['stop_now_frac']:.3f} vx={v['vx_mean']:+.3f} "
                  f"corr(Vh,d)={v['corr_reach_dist']:+.3f}")
    print("\nOUT " + out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
