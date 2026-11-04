#!/usr/bin/env python3
"""CAL-014 验收：比对 on/off 双跑，并做离线复算与完整度检查（不启动仿真）。

检查项：
  1. 测量开关不改变动作与状态：on/off 的 50 Hz 数组逐字段比 max abs diff；
  2. 离线复算：用 on 运行落盘的原始 tau/qdot 重算能量，与在线累计比对（协议容差 1e-5）；
  3. 完整度：200 Hz 步数 = 6000、50 Hz 条数 = 1500、warmup 200 步不入窗口；
  4. 单位与口径：summary 的单位是 J、metric_id 是 D-ENERGY-001。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402

from camra_calibration.energy import integrate_series  # noqa: E402

FIELDS = ("action_50hz", "cmd_50hz", "root_pos_50hz", "root_quat_50hz",
          "root_lin_vel_b_50hz", "root_ang_vel_b_50hz", "joint_pos_50hz", "joint_vel_50hz")


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "/pub/data/caohy/logs/camra/meter_check")
    on = torch.load(d / "meter_on.npz", map_location="cpu")
    off = torch.load(d / "meter_off.npz", map_location="cpu")
    meta_on = json.loads((d / "meter_on.json").read_text(encoding="utf-8"))
    meta_off = json.loads((d / "meter_off.json").read_text(encoding="utf-8"))

    report: dict = {"dir": str(d), "checks": {}, "ok": True}

    # 1) on/off 动作与状态
    diffs = {}
    for f in FIELDS:
        a, b = on[f], off[f]
        if a.shape != b.shape:
            report["ok"] = False
            diffs[f] = {"shape_mismatch": [list(a.shape), list(b.shape)]}
            continue
        md = float((a - b).abs().max()) if a.numel() else 0.0
        diffs[f] = {"max_abs_diff": md, "bitwise_equal": bool(torch.equal(a, b))}
    report["checks"]["on_off_state_diff"] = diffs
    max_state_diff = max(v.get("max_abs_diff", 0.0) for v in diffs.values())
    report["checks"]["on_off_max_abs_diff"] = max_state_diff
    report["checks"]["on_off_within_1e_6"] = max_state_diff <= 1e-6
    if not report["checks"]["on_off_within_1e_6"]:
        report["ok"] = False

    # 2) 离线复算
    tau, qdot = on["tau"], on["qdot"]
    recomputed = [integrate_series(tau[:, i, :].numpy(), qdot[:, i, :].numpy(),
                                   dt=meta_on["sim_dt"]) for i in range(tau.shape[1])]
    online = meta_on["energy_total_j"]
    rel = [abs(r - o) / max(o, 1e-12) for r, o in zip(recomputed, online)]
    report["checks"]["energy_recompute"] = {
        "online_j": online, "offline_j": recomputed, "rel_err": rel,
        "tol": 1e-5, "pass": all(x <= 1e-5 for x in rel),
    }
    if not report["checks"]["energy_recompute"]["pass"]:
        report["ok"] = False

    # 3) 完整度
    steps = int(tau.shape[0])
    ctrl = int(on["action_50hz"].shape[0])
    completeness = {
        "physics_steps": steps,
        "expected_physics_steps": meta_on["expected_window_steps"],
        "control_steps": ctrl,
        "expected_control_steps": meta_on["expected_control_steps"],
        "warmup_steps": meta_on["warmup_steps"],
        "expected_warmup_steps": 200,
        "window_s": meta_on["window_s"],
        "energy_complete": meta_on["energy_complete"],
        "problems": meta_on["problems"],
    }
    completeness["pass"] = (
        steps == 6000 and ctrl == 1500 and meta_on["warmup_steps"] == 200
        and meta_on["outcome"] == "completed" and not meta_on["problems"]
    )
    report["checks"]["completeness"] = completeness
    if not completeness["pass"]:
        report["ok"] = False

    # 4) 单位与口径
    units = {
        "meta_unit": meta_on.get("unit", "J"),
        "energy_total_j": online,
        "profile": meta_on["profile"],
        "trial_seed": meta_on["trial_seed"],
        "program_sha256": meta_on["program_sha256"],
        "decimation": meta_on["decimation"],
        "sim_dt": meta_on["sim_dt"],
        "episode_length_s": meta_on["episode_length_s"],
    }
    report["checks"]["units_and_meta"] = units

    out = d / "verify_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps(report["checks"], ensure_ascii=False, indent=1)[:3000])
    print("VERDICT:", "pass" if report["ok"] else "fail")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
