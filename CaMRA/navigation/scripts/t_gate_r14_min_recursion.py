#!/usr/bin/env python3
"""T4 反例门禁：深井语义下重验 min 型递推的到达梯度方向。

amend3 的反例：v1 语义（h 被 max(·,0) 抹平）下开 min 递推，corr(h, adv) 在
3 个历史 run 里 2 个翻负。r14 的假设：根因是符号体系（到达无负值锚点），
深井（G_INSIDE=-3）修复后方向应恢复。本门禁在同一批历史 chunk 上各算一遍：

  v1  + min 递推（w=0）：预期复现翻负（负对照，证明反例仍在）
  v2  + min 递推（w=0）：门禁主体，3 个 run 必须 corr(h, adv) > 0

性能约束（v2，2026-11-15 19:20 修订）：服务器 load 40+，全量 chunk 要 40 min。
改为每个环境按步长均匀抽样 chunks_per_env=60（几千个回合统计上足够），
且每个 chunk 只加载一次、两套配置各算一次。逐环境打进度到 stdout。

任一 v2 run ≤ 0 → 不开训，按预登记回退方案 A。
产出：reports/gate_r14_min_recursion.json
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
    "nav064_r5_uncalibrated_s0",
    "nav064_r1c_uncalibrated_s0",
    "nav064_r5_full_s0",
]
CHUNKS_PER_ENV = 60


def cfg_for(h_semantics: str) -> S.ObjectiveConfig:
    cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
    return dataclasses.replace(cfg, joint_cost_weight=0.0,
                               joint_cost_base_weight=0.0,
                               joint_cost_on_overspend_only=False,
                               joint_min_recursion=True,
                               h_semantics=h_semantics)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def run_gates(run_id: str) -> dict:
    run_dir = os.path.join(WORK, run_id)
    cfg_v1, cfg_v2 = cfg_for("v1"), cfg_for("v2_deep_well")
    acc = {"v1": ([], [], []), "v2": ([], [], [])}   # (h, adv, per-ep corr)
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
            for tag, cfg in (("v1", cfg_v1), ("v2", cfg_v2)):
                out = S.compute_objectives(arrays, cfg, terminal_kind=kind)
                h = np.asarray(out["obj_h"], dtype=np.float64)
                adv = np.asarray(out["obj_adv_total"], dtype=np.float64)
                hs, ads, cors = acc[tag]
                hs.append(h)
                ads.append(adv)
                cors.append(pearson(h, adv))
        print(f"  {run_id} env {i + 1}/{len(envs)} ({n} chunks) sampled "
              f"{len(idxs)}", flush=True)
    res = {}
    for tag in ("v1", "v2"):
        hs, ads, cors = acc[tag]
        h_all = np.concatenate(hs)
        adv_all = np.concatenate(ads)
        finite = np.array([np.isfinite(c) for c in cors])
        res[tag] = {
            "n_episodes": len(cors),
            "n_steps": int(h_all.size),
            "corr_pooled": round(pearson(h_all, adv_all), 4),
            "corr_episode_median": (round(float(np.median(np.array(cors)[finite])), 4)
                                    if finite.any() else None),
        }
    return res


def main() -> int:
    rows = []
    ok_v2, ok_v1_flip = True, False
    for rid in RUNS:
        g = run_gates(rid)
        r1, r2 = g["v1"], g["v2"]
        flipped = (r1["corr_pooled"] <= 0) and (r2["corr_pooled"] > 0)
        ok_v1_flip = ok_v1_flip or (r1["corr_pooled"] < 0)
        ok_v2 = ok_v2 and (r2["corr_pooled"] > 0)
        rows.append({"run_id": rid, "v1_min_recursion": r1,
                     "v2_deep_well_min_recursion": r2,
                     "direction_restored": flipped})
        print(f"{rid}: v1 corr={r1['corr_pooled']:+.4f} -> "
              f"v2 corr={r2['corr_pooled']:+.4f} "
              f"(episodes={r2['n_episodes']}, steps={r2['n_steps']})", flush=True)

    verdict = "PASS" if (ok_v2 and ok_v1_flip) else "FAIL"
    payload = {
        "schema": "camra-r14-min-recursion-gate-v1",
        "task": "NAV-064-r14",
        "verdict": verdict,
        "rule": "3 个历史 run 的 v2+min 递推 corr(h, adv) 全部 > 0；"
                "且 v1+min 递推复现翻负（负对照成立）",
        "sampling": f"每个 rollout_env 均匀抽样至多 {CHUNKS_PER_ENV} 个 chunk",
        "value_contract_version": "1.3.0",
        "G_INSIDE": -3.0,
        "runs": rows,
        "note": "chunk 为历史 rollout（只读），adv 用当前代码重算；w=0 隔离语义效应",
    }
    out = os.path.join(NAV, "reports", "gate_r14_min_recursion.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print("verdict:", verdict, "| wrote", out, flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
