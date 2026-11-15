#!/usr/bin/env python3
"""T4 反例门禁 v2：深井语义是否修复了 min 型递推的到达方向。

第一次尝试（19:2x）用的 r5/r1c 数据里**没有任何一步进入目标集**（到达率 ~0.05%），
v1 与 v2 结果逐位相同 —— 深井无从生效，门禁无效。据此改用 r12/r13 数据
（到达回合 2.5%-7.1%），并在判定里加入三条：

  S1 活跃性：含目标集步的回合占比 > 0，且成功回合的 h_T 在两版语义下不同
  S2 方向：含到达回合上 v2 的 corr(h, adv) > 0（h 越大 = 越远 = 优势越大=更差）
  S3 净改善：v2 的 corr 严格优于 v1（深井把方向拉正，而非原地不动）

任一门失败 → 不开训，回退方案 A（终点超额一次性计费）。
产出：reports/gate_r14_min_recursion.json（v2）
"""
import dataclasses
import json
import os
import sys

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(NAV, "src"))
sys.path.insert(0, os.path.join(NAV, "..", "calibration", "src"))

import numpy as np
from camra_navigation.algorithm import storage as S

WORK = "/pub/data/caohy/work/camra_navigation"
VALUE_CONTRACT = os.path.join(NAV, "config", "value_objective_contract_v1.json")
RUNS = [
    "nav064_r12_uncalibrated_s0",
    "nav064_r12_full_s0",
    "nav064_r13_uncalibrated_s0",
    "nav064_r13_full_s0",
]
CHUNKS_PER_ENV = 50


def cfg_for(h_semantics: str) -> S.ObjectiveConfig:
    cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
    return dataclasses.replace(cfg, joint_cost_weight=0.0,
                               joint_cost_base_weight=0.0,
                               joint_cost_on_overspend_only=False,
                               joint_min_recursion=True,
                               h_semantics=h_semantics)


def pearson(a, b) -> float:
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def run_gates(run_id: str) -> dict:
    run_dir = os.path.join(WORK, run_id)
    cfg_v1, cfg_v2 = cfg_for("v1"), cfg_for("v2_deep_well")
    stats = {t: {"h": [], "adv": [], "cors": [], "arr_h": [], "arr_adv": [],
                 "arr_cors": [], "h_T_arrive": [], "last_adv": []} for t in ("v1", "v2")}
    n_ep = n_arr = n_success = 0
    envs = sorted(e for e in os.listdir(run_dir) if e.startswith("rollout_env"))
    for i, env_name in enumerate(envs):
        store = S.RolloutStore(run_dir, env_name, obs_dim=41)
        n = store.n_chunks
        if n == 0:
            continue
        idxs = (np.linspace(0, n - 1, CHUNKS_PER_ENV).round().astype(int)
                if n > CHUNKS_PER_ENV else np.arange(n))
        for idx in idxs:
            meta = store.entries[int(idx)]
            kind = str(meta.get("terminal_kind", "truncation"))
            arrays = store.load_chunk(int(idx))
            d_all = np.asarray(arrays["goal_distance_m"], dtype=np.float64)
            d_T = float(np.asarray(arrays["goal_distance_m_next"], dtype=np.float64)[-1])
            has_goal = bool(np.any(d_all <= 0.5) or d_T <= 0.5)
            n_ep += 1
            n_arr += int(has_goal)
            n_success += int(kind == "success")
            for tag, cfg in (("v1", cfg_v1), ("v2", cfg_v2)):
                out = S.compute_objectives(arrays, cfg, terminal_kind=kind)
                h = np.asarray(out["obj_h"], dtype=np.float64)
                adv = np.asarray(out["obj_adv_total"], dtype=np.float64)
                s = stats[tag]
                s["h"].append(h)
                s["adv"].append(adv)
                s["cors"].append(pearson(h, adv))
                if has_goal:
                    s["arr_h"].append(h)
                    s["arr_adv"].append(adv)
                    s["arr_cors"].append(pearson(h, adv))
                    s["h_T_arrive"].append(float(out["obj_h_terminal"][0]))
                    s["last_adv"].append(float(adv[-1]))
        print(f"  {run_id} env {i + 1}/{len(envs)} ({n} chunks)", flush=True)

    out = {"run_id": run_id, "n_episodes": n_ep, "n_with_goal_steps": n_arr,
           "n_success": n_success, "goal_step_share": round(n_arr / max(n_ep, 1), 4)}
    for tag in ("v1", "v2"):
        s = stats[tag]
        h_all = np.concatenate(s["h"]); adv_all = np.concatenate(s["adv"])
        cors = np.array([c for c in s["cors"] if np.isfinite(c)])
        out[tag] = {"corr_pooled_all": round(pearson(h_all, adv_all), 4),
                    "corr_episode_median_all": round(float(np.median(cors)), 4) if cors.size else None}
        if s["arr_h"]:
            ah = np.concatenate(s["arr_h"]); aa = np.concatenate(s["arr_adv"])
            ac = np.array([c for c in s["arr_cors"] if np.isfinite(c)])
            out[tag].update({
                "n_arr_episodes": len(s["arr_h"]),
                "n_arr_steps": int(ah.size),
                "corr_pooled_arrival": round(pearson(ah, aa), 4),
                "corr_episode_median_arrival": round(float(np.median(ac)), 4) if ac.size else None,
                "mean_h_T_arrival": round(float(np.mean(s["h_T_arrive"])), 4),
                "mean_last_adv_arrival": round(float(np.mean(s["last_adv"])), 4),
            })
        else:
            out[tag].update({"n_arr_episodes": 0, "corr_pooled_arrival": None})
    return out


def main() -> int:
    rows, gates = [], {"s1_active": True, "s2_direction": True, "s3_improved": True}
    for rid in RUNS:
        r = run_gates(rid)
        v1, v2 = r["v1"], r["v2"]
        s1 = (r["n_with_goal_steps"] > 0
              and (v1["n_arr_episodes"] > 0)
              and abs(v2.get("mean_h_T_arrival", 0.0) - v1.get("mean_h_T_arrival", 0.0)) > 1e-9)
        s2 = (v2.get("corr_pooled_arrival") is not None
              and v2["corr_pooled_arrival"] > 0.0)
        s3 = (v1.get("corr_pooled_arrival") is not None
              and v2.get("corr_pooled_arrival") is not None
              and v2["corr_pooled_arrival"] > v1["corr_pooled_arrival"])
        gates["s1_active"] &= s1
        gates["s2_direction"] &= s2
        gates["s3_improved"] &= s3
        rows.append({**r, "s1_active": s1, "s2_direction": s2, "s3_improved": s3})
        print(f"{rid}: goal-episodes={r['n_with_goal_steps']}/{r['n_episodes']} "
              f"| corr(arrival) v1={v1.get('corr_pooled_arrival')} "
              f"v2={v2.get('corr_pooled_arrival')} "
              f"| S1={s1} S2={s2} S3={s3}", flush=True)

    verdict = "PASS" if all(gates.values()) else "FAIL"
    payload = {
        "schema": "camra-r14-min-recursion-gate-v2",
        "task": "NAV-064-r14",
        "verdict": verdict,
        "gates": gates,
        "rule": "S1 深井活跃（含目标集步的回合存在且 h_T 两版不同）；S2 到达回合上 v2 的 "
                "corr(h,adv)>0；S3 v2 严格优于 v1。全部 run 都要满足 S1-S3。",
        "invalidated_first_attempt": {
            "why": "首版门禁用 r5/r1c 数据，抽样中零个回合进入目标集，v1 与 v2 逐位相同，"
                   "深井无从生效 —— 是数据集选错，不是深井失败",
            "evidence": "reports/gate_r14_min_recursion_first_attempt.json",
        },
        "sampling": f"每个 rollout_env 均匀抽样至多 {CHUNKS_PER_ENV} 个 chunk",
        "value_contract_version": "1.3.0", "G_INSIDE": -3.0,
        "runs": rows,
        "note": "chunk 为历史 rollout（只读）；w=0 隔离语义效应；corr 口径 h 越大越差、adv>0 更差",
    }
    with open(os.path.join(NAV, "reports", "gate_r14_min_recursion.json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print("verdict:", verdict, "| gates:", gates, flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
