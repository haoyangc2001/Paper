"""NAV-064 根因确证 ②：盖章段（y=1）在**梯度**里到底造成了什么。

只读诊断：读 r11 冻结 rollout chunk（每步的目标重算量都在），量化三件事：

  A. 盖章段是否真的「零信号」：|adv_total| 在 y=0 / y=1 两组的质量占比。
  B. 零信号样本是否通过 `normalize_advantage` 反向污染：标准化用的 mean/std
     由**全部**样本算出，若 y=1 组恒为 0，它们会被整体推成 (0−mean)/std 的
     常数偏置（非零！），等于对「靠近终点的那些动作」统一判贵。
  C. 屏蔽盖章段后剩下多少有效样本、优势尺度放大多少倍。

不写任何项目文件，不训练。输出 JSON 到 reports/。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

WORK = "/pub/data/caohy/work/camra_navigation"
OUT = os.path.join(
    "/home/caohy/repositories/Paper/CaMRA/navigation", "reports",
    "training_round_r11_deadsteps.json")
RUNS = [("uncalibrated", "nav064_r11_uncalibrated_s0"),
        ("full", "nav064_r11_full_s0")]
N_CHUNKS_PER_ENV = 60

FIELDS = ("safety_y", "obj_adv_total", "obj_target_total", "obj_total_value",
          "obj_stop_now", "obj_h", "obj_target_reach", "obj_feasible",
          "action_raw", "goal_distance_m", "obj_g_train")


def pick(run_id: str, n_per_env: int) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for env_dir in sorted(glob.glob(os.path.join(WORK, run_id, "rollout_env*"))):
        env_i = int(os.path.basename(env_dir).replace("rollout_env", ""))
        fs = sorted(glob.glob(os.path.join(env_dir, "rollout", "chunk_*.npz")))
        if not fs:
            continue
        idx = np.linspace(0, len(fs) - 1, min(n_per_env, len(fs))).astype(int)
        out += [(env_i, i, fs[i]) for i in idx]
    return out


def summarize(name: str, run_id: str) -> dict:
    paths = pick(run_id, N_CHUNKS_PER_ENV)
    n_chunks_total = sum(
        len(glob.glob(os.path.join(WORK, run_id, f"rollout_env{e}", "rollout",
                                   "chunk_*.npz"))) for e in {p[0] for p in paths})
    acc = {f: [] for f in FIELDS}
    phases = [[], [], [], []]          # 按 run 内进度四等分
    for env_i, ci, p in paths:
        with np.load(p) as z:
            n_env = len(glob.glob(os.path.join(
                WORK, run_id, f"rollout_env{env_i}", "rollout", "chunk_*.npz")))
            phase = min(3, int(4 * ci / max(1, n_env)))
            d = {f: np.asarray(z[f]) for f in FIELDS}
            for f in FIELDS:
                acc[f].append(d[f])
            phases[phase].append(d)

    cat = {f: np.concatenate(v, axis=0) for f, v in acc.items()}
    y = cat["safety_y"].astype(np.float64)
    adv = cat["obj_adv_total"].astype(np.float64)
    tgt = cat["obj_target_total"].astype(np.float64)
    val = cat["obj_total_value"].astype(np.float64)
    stop = cat["obj_stop_now"].astype(bool)
    vx = cat["action_raw"][:, 0].astype(np.float64)
    gd = cat["goal_distance_m"].astype(np.float64)

    dead = y > 0.5
    live = ~dead
    n = int(y.shape[0])
    n_dead = int(dead.sum())

    def stats(a: np.ndarray) -> dict:
        if a.size == 0:
            return {"n": 0}
        return {"n": int(a.size), "mean": round(float(a.mean()), 6),
                "std": round(float(a.std()), 6), "p05": round(float(np.percentile(a, 5)), 6),
                "p95": round(float(np.percentile(a, 95)), 6),
                "abs_mean": round(float(np.abs(a).mean()), 6),
                "abs_sum": round(float(np.abs(a).sum()), 3)}

    # ---- A. 信号质量占比 -------------------------------------------------
    abs_sum_all = float(np.abs(adv).sum())
    abs_sum_live = float(np.abs(adv[live]).sum()) if live.any() else 0.0
    zero_frac_dead = float((np.abs(adv[dead]) < 1e-9).mean()) if dead.any() else None
    zero_frac_live = float((np.abs(adv[live]) < 1e-9).mean()) if live.any() else None

    # ---- B. 标准化的常数偏置（按真实 minibatch 量级估算） ----------------
    m_all, s_all = float(adv.mean()), float(adv.std())
    m_live = float(adv[live].mean()) if live.any() else 0.0
    s_live = float(adv[live].std()) if live.any() else 0.0
    m_dead = float(adv[dead].mean()) if dead.any() else 0.0
    # 盖章段在「全样本标准化」下拿到的常数优势，用它**实测的**均值算，不假设为 0
    bias_all = round((m_dead - m_all) / (s_all + 1e-8), 4)
    bias_live_only = round((m_dead - m_live) / (s_live + 1e-8), 4)
    implied_live_scale = round(s_all / (s_live + 1e-8), 4) if s_live else None

    # ---- C. 屏蔽后的样本量与尺度 ----------------------------------------
    kept = float(live.sum()) / max(1, n)

    per_phase = []
    for ph in range(4):
        if not phases[ph]:
            per_phase.append({"phase": ph})
            continue
        dd = {f: np.concatenate([x[f] for x in phases[ph]], axis=0) for f in FIELDS}
        py = dd["safety_y"].astype(np.float64) > 0.5
        pa = dd["obj_adv_total"].astype(np.float64)
        pvx = dd["action_raw"][:, 0].astype(np.float64)
        per_phase.append({
            "phase": ph, "n": int(py.size),
            "dead_frac": round(float(py.mean()), 4),
            "adv_live_abs_mean": round(float(np.abs(pa[~py]).mean()), 6) if (~py).any() else None,
            "adv_all_std": round(float(pa.std()), 6),
            "vx_live": round(float(pvx[~py].mean()), 4) if (~py).any() else None,
            "vx_dead": round(float(pvx[py].mean()), 4) if py.any() else None,
            "goal_dist_live": round(float(dd["goal_distance_m"][~py].mean()), 3) if (~py).any() else None,
        })

    return {
        "run": run_id, "chunks": len(paths), "chunks_total": n_chunks_total,
        "steps": n,
        "dead_steps": n_dead, "dead_frac": round(n_dead / max(1, n), 4),
        "A_signal": {
            "adv_all": stats(adv),
            "adv_live": stats(adv[live]) if live.any() else {"n": 0},
            "adv_dead": stats(adv[dead]) if dead.any() else {"n": 0},
            "abs_adv_share_live": round(abs_sum_live / abs_sum_all, 4) if abs_sum_all else None,
            "zero_adv_frac_dead": None if zero_frac_dead is None else round(zero_frac_dead, 4),
            "zero_adv_frac_live": None if zero_frac_live is None else round(zero_frac_live, 4),
            "target_minus_value_dead_mean": round(float((tgt - val)[dead].mean()), 6) if dead.any() else None,
            "target_minus_value_live_mean": round(float((tgt - val)[live].mean()), 6) if live.any() else None,
            "stop_now_dead": round(float(stop[dead].mean()), 4) if dead.any() else None,
            "stop_now_live": round(float(stop[live].mean()), 4) if live.any() else None,
        },
        "B_normalization_bias": {
            "adv_mean_all": round(m_all, 6), "adv_std_all": round(s_all, 6),
            "adv_mean_live": round(m_live, 6), "adv_std_live": round(s_live, 6),
            "adv_mean_dead": round(m_dead, 6),
            "bias_unmasked": bias_all,
            "bias_if_live_stats": bias_live_only,
            "live_adv_shrink_factor": implied_live_scale,
            "note": ("bias_unmasked=(dead_mean−all_mean)/all_std：盖章段在**全样本**标准化下"
                     "拿到的常数优势；bias_if_live_stats 是同一批样本若按有效样本统计量"
                     "标准化会拿到什么（屏蔽后它们不再进损失，数值只用于说明偏置来源）"),
        },
        "C_masking": {"kept_fraction": round(kept, 4)},
        "speed_by_group": {
            "vx_live": round(float(vx[live].mean()), 4) if live.any() else None,
            "vx_dead": round(float(vx[dead].mean()), 4) if dead.any() else None,
            "goal_dist_live": round(float(gd[live].mean()), 3) if live.any() else None,
            "goal_dist_dead": round(float(gd[dead].mean()), 3) if dead.any() else None,
        },
        "per_phase": per_phase,
    }


def main() -> int:
    out = {"task": "NAV-064", "kind": "dead_step_gradient_diagnosis",
           "n_chunks_per_env": N_CHUNKS_PER_ENV,
           "runs": [summarize(v, r) for v, r in RUNS]}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    for r in out["runs"]:
        print("=" * 74)
        print(f"{r['run']}  steps={r['steps']}  dead={r['dead_frac']:.2%}")
        a = r["A_signal"]
        print(f"  |adv| 质量占比 盖章段之外 = {a['abs_adv_share_live']}")
        print(f"  adv 均值/标准差 全样本 = {a['adv_all']['mean']}/{a['adv_all']['std']}")
        print(f"  y=0 组 |adv| 均值 = {a['adv_live']['abs_mean']:.6f}   "
              f"y=1 组 |adv| 均值 = {a['adv_dead']['abs_mean']:.6f}")
        print(f"  零优势步占比 y=0 {a['zero_adv_frac_live']} / y=1 {a['zero_adv_frac_dead']}")
        print(f"  stop_now y=0 {a['stop_now_live']} / y=1 {a['stop_now_dead']}")
        b = r["B_normalization_bias"]
        print(f"  标准化：全样本 mean={b['adv_mean_all']} std={b['adv_std_all']}  "
              f"仅 y=0 mean={b['adv_mean_live']} std={b['adv_std_live']}  "
              f"y=1 mean={b['adv_mean_dead']}")
        print(f"  → 盖章段在全样本标准化下的常数优势 = {b['bias_unmasked']}"
              f"（若只按有效样本统计 = {b['bias_if_live_stats']}）；"
              f"有效样本尺度被压缩 {b['live_adv_shrink_factor']} 倍")
        print(f"  屏蔽后保留样本 {r['C_masking']['kept_fraction']:.2%}")
        print(f"  速度 vx  y=0 {r['speed_by_group']['vx_live']} / y=1 {r['speed_by_group']['vx_dead']}")
        for p in r["per_phase"]:
            if "dead_frac" in p:
                print(f"    phase{p['phase']}: dead={p['dead_frac']:.2%} "
                      f"|adv|y0={p['adv_live_abs_mean']} vx_y0={p['vx_live']} "
                      f"d_y0={p['goal_dist_live']}")
    print(f"\nSAVED {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
