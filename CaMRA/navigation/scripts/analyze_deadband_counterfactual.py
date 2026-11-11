"""NAV-060 处置 C 的预测步：把低于死区的 wz 命令归零后，ε_d / ε_c 会变成多少。

为什么可以预测而不用重跑仿真：
  NAV-060 已证底层几乎不执行 |wz| 小于死区的命令（实测执行路径对弦线横向偏移
  0.15-0.18 m，而命令累计转角 0.73-1.20 rad）。把这些命令改成 0，物理输入不变，
  执行轨迹应当逐位不变；变化的只有「名义参考」与「J_nom」。

所以本脚本用存档的实测位置 + 改写后的命令重算名义参考，得到重跑后的预期指标。
真实复测仍要跑（命令日志与 J_nom 必须由正式运行产生），本脚本只用于
"先判断值不值得跑"，结果如实标注为预测。

用法：
    python analyze_deadband_counterfactual.py \
        --runs reports/applicability_L1_empty.json ... --threshold 0.12 \
        --json reports/deadband_counterfactual.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(CALIB, "src"))
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))

POWER_PATH = os.path.join(CALIB, "config", "power_model_v1.json")
MARGINS_PATH = os.path.join(CALIB, "config", "margins_v1.json")
NAV_DT = 0.1
SUBSTEPS_PER_NAV = 20
PREFIXES_S = [5, 10, 20, 30]
J_MIN = 1.0


def nearest_rank(values, q: float = 0.9):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    k = min(max(math.ceil(q * len(vals)), 1), len(vals))
    return vals[k - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import numpy as np
    from camra_calibration.nominal import PlanarState, integrate
    from camra_calibration.power import PowerModel

    model = PowerModel.load(POWER_PATH)
    margins = json.load(open(MARGINS_PATH, encoding="utf-8"))
    delta = float(margins["delta"])

    rows = []
    for run_path in args.runs:
        run = json.load(open(run_path, encoding="utf-8"))
        npz_path = os.path.splitext(run_path)[0] + ".npz"
        if not os.path.exists(npz_path):
            continue
        z = np.load(npz_path, allow_pickle=True)
        ids = [str(x) for x in z["trial_ids"]]
        for ep in run["episodes"]:
            tid = ep["trial_id"]
            if tid not in ids:
                continue
            i = ids.index(tid)
            vk = int(z["valid_k"][i])
            if vk < 10:
                continue
            ex = z["exec_pos"][i][:vk + 1].astype(float)
            cmds = z["cmd"][i][:vk].astype(float).copy()
            zeroed = int(np.sum(np.abs(cmds[:, 2]) < args.threshold))
            cmds[:, 2] = np.where(np.abs(cmds[:, 2]) < args.threshold, 0.0, cmds[:, 2])
            full = np.vstack([cmds, cmds[-1:]])

            ix, iy, iyaw = ep["init_pose"]
            nom = integrate(PlanarState(ix, iy, iyaw), [tuple(r) for r in full], NAV_DT)

            d = []
            for k in range(vk + 1):
                pe = ex[k]
                pn = np.array([nom[k].px, nom[k].py])
                d.append(float(np.linalg.norm(pe - pn)))

            rec = {"trial_id": tid, "kind": ep["kind"], "profile": ep["profile"],
                   "zeroed_wz_steps": zeroed, "n_steps": vk,
                   "eps_d_before": ep["prefixes"]["30"]["eps_d"],
                   "eps_d_after": max(d) if d else None, "prefixes": {}}
            for T in PREFIXES_S:
                k = int(round(T / NAV_DT))
                base = ep["prefixes"][str(T)]
                j_exec = base["j_exec"]
                j_nom = float(model.j_nom(full[:k + 1]))
                rec["prefixes"][str(T)] = {
                    "j_nom_before": base["j_nom"], "j_nom_after": j_nom,
                    "j_exec": j_exec,
                    "eps_c_before": base["eps_c"],
                    "eps_c_after": ((j_exec - j_nom) / j_nom) if j_nom >= J_MIN else None,
                    "eps_d_after": max(d[:k + 1]),
                }
            rows.append(rec)

    groups = {}
    for r in rows:
        groups.setdefault(f"{r['kind']}_{r['profile']}", []).append(r)
        groups.setdefault(f"{r['kind']}_all", []).append(r)

    summary = {}
    for key, items in sorted(groups.items()):
        ec = [it["prefixes"]["30"]["eps_c_after"] for it in items]
        ed = [it["prefixes"]["30"]["eps_d_after"] for it in items]
        summary[key] = {"n": len(items),
                        "q90_eps_c_after": nearest_rank(ec),
                        "q90_eps_d_after": nearest_rank(ed),
                        "max_eps_d_after": max(ed) if ed else None}

    nc = summary.get("new_clean", {})
    out = {
        "task": "NAV-060", "role": "prediction_only",
        "threshold_rad_s": args.threshold,
        "prediction_basis": "执行轨迹取自已落盘实测（假定把未被执行的 wz 命令置零不改变物理）；"
                            "J_nom 与名义参考按改写后的命令重算。真实结论以复测运行为准。",
        "frozen_thresholds": {"q90_eps_c_max": 0.15, "q90_eps_d_max_m": delta},
        "predicted": summary,
        "predicted_verdict_primary": (
            "PASS" if (nc.get("q90_eps_c_after") is not None
                       and nc["q90_eps_c_after"] <= 0.15
                       and nc.get("q90_eps_d_after") is not None
                       and nc["q90_eps_d_after"] <= delta) else "FAIL"),
        "rows": rows,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
            fh.write("\n")
    print(json.dumps({"threshold": args.threshold,
                      "new_clean": nc,
                      "predicted_verdict": out["predicted_verdict_primary"]},
                     ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
