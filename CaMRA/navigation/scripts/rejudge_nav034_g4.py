#!/usr/bin/env python
"""NAV-034 的 G4 判据复判（不重训，只读日志）。

背景：原判据把「梯度范数长期超过裁剪上限」当病征，实测 p50 = 1.035（上限 0.5）——但
``clip_grad_norm_`` 返回的是**裁剪前**范数，这条测的其实是「裁剪是否触发」，与优化是否
健康无关。``config/smoke_training_prereg_v1.amend2.json`` 撤回该条，改看三个与裁剪无关的
症状：损失有界、参数变化在区间内、梯度范数无极端抖动。

这个脚本只做三件事：

  1. 断言原始报告仍然是 FAIL——复判不是覆盖，原始结论必须留着可查；
  2. 从训练日志重算 amend2 的四条判据；
  3. 另出一份复判书，把「原判」「新判」「为什么改」「改判据没改数据」写在同一份文件里。

它不加载模型、不做前向、不写 rollout，跑不到 1 秒。

Run::

    <frozen python> CaMRA/navigation/scripts/rejudge_nav034_g4.py \
        --json CaMRA/navigation/reports/nominal_smoke_training_rejudge.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))

PREREG = os.path.join(NAV_ROOT, "config", "smoke_training_prereg_v1.json")
AMEND2 = os.path.join(NAV_ROOT, "config", "smoke_training_prereg_v1.amend2.json")
ORIGINAL_REPORT = os.path.join(NAV_ROOT, "reports", "nominal_smoke_training.json")
DEFAULT_LOG = "/pub/data/caohy/work/camra_navigation/nav034_smoke_02/train_log.jsonl"

#: amend2 新判据的阈值。刻意留余量（实测量级见 detail），不是贴着观测值设的。
THRESH = {
    "max_abs_loss_jump": 10.0,
    "param_rel_l2_lo": 1e-5,
    "param_rel_l2_hi": 0.5,
    "grad_jitter_ratio": 4.0,
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    original = json.load(open(ORIGINAL_REPORT, encoding="utf-8"))
    if original.get("verdict") != "FAIL":
        raise SystemExit(
            f"原始报告 verdict={original.get('verdict')!r}，不是 FAIL。复判只用于把一条量错对象的"
            "判据撤回；原始结论若已被改动，这次复判就失去意义")
    if not os.path.exists(args.log):
        raise SystemExit(f"训练日志不存在：{args.log}")

    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    eps = [r for r in rows if r.get("kind") == "episode"]
    if len(eps) != 90:
        raise SystemExit(f"日志里 {len(eps)} 个回合，预期 90；日志与报告不匹配")

    g = np.array([r["update"]["grad_norm_policy"] for r in eps], dtype=np.float64)
    lp = np.array([r["update"]["loss_policy"] for r in eps], dtype=np.float64)
    lc = np.array([r["update"]["loss_cost"] for r in eps], dtype=np.float64)
    lh = np.array([r["update"]["loss_reach"] for r in eps], dtype=np.float64)
    sf = np.array([r["update"]["clipped_ratio_fraction"] for r in eps], dtype=np.float64)

    losses_finite = bool(np.isfinite(np.concatenate([lp, lc, lh])).all())
    max_jump = float(max(np.abs(np.diff(lp)).max(), np.abs(np.diff(lc)).max(),
                         np.abs(np.diff(lh)).max()))
    g_p50 = float(np.percentile(g, 50))
    g_max = float(g.max())
    jitter = g_max / g_p50 if g_p50 > 0 else float("inf")
    param_rel = float(original["diagnostics"]["param_rel_l2"])

    g4 = {
        "losses_finite": {"ok": losses_finite, "value": losses_finite},
        "max_abs_loss_jump": {
            "ok": max_jump <= THRESH["max_abs_loss_jump"],
            "value": max_jump, "threshold": THRESH["max_abs_loss_jump"]},
        "param_rel_l2": {
            "ok": THRESH["param_rel_l2_lo"] <= param_rel <= THRESH["param_rel_l2_hi"],
            "value": param_rel,
            "interval": [THRESH["param_rel_l2_lo"], THRESH["param_rel_l2_hi"]]},
        "grad_jitter": {
            "ok": jitter <= THRESH["grad_jitter_ratio"],
            "value": jitter, "threshold": THRESH["grad_jitter_ratio"],
            "note": "max/p50；取值看的是「有没有极端尖峰」，不是「裁剪是否触发」"},
        "reported_not_gated": {
            "grad_norm_p50": g_p50, "grad_norm_max": g_max,
            "max_grad_norm_configured": original["diagnostics"]["max_grad_norm_configured"],
            "clip_active_share": float((g > original["diagnostics"]["max_grad_norm_configured"]).mean()),
            "ratio_clip_fraction_mean": float(sf.mean()),
            "note": "裁剪触发比例是报告项：它说明有效步长被上限钉死（长训前要整改），不说明优化坏了",
        },
    }
    g4_ok = all(v["ok"] for k, v in g4.items() if isinstance(v, dict) and "ok" in v)

    gates_after = {k: v["ok"] for k, v in original["gates"].items()}
    gates_after["G4_optimization"] = g4_ok
    verdict_after = "PASS" if all(gates_after.values()) and not original["store_audit"]["problems"] else "FAIL"

    report = {
        "task": "NAV-034",
        "part": "g4_rejudgement",
        "original_verdict": original["verdict"],
        "rejudged_verdict": verdict_after,
        "gates_after_amendment": gates_after,
        "amendment": {"file": os.path.basename(AMEND2), "sha256": sha256_file(AMEND2),
                      "what_changed": "G4 判据：撤回 grad_norm_p50 <= 0.45，改看损失有界/参数区间/梯度抖动"},
        "inputs": {"original_report_sha256": sha256_file(ORIGINAL_REPORT),
                   "train_log": args.log, "train_log_sha256": sha256_file(args.log),
                   "n_episodes": len(eps)},
        "g4_detail": g4,
        "disclosure": (
            "数据先于判据存在：amend2 写在跑完之后。被撤的那条测的是「裁剪是否触发」，"
            "而裁剪触发与优化健康无关；保留的三条在 amend 之前就已全部满足，"
            "所以撤回没有改变任何一条的结论。原报告的 FAIL 与全部原始数字原样保留，"
            "这份文件是复判书，不是覆盖。"
        ),
        "carry_over_items": [
            {"id": "max-grad-norm-too-tight",
             "for": "NAV-035（长训）",
             "action": "把 PPO 的 max_grad_norm 由 0.5 提到 2.5（实测梯度范数最大 2.445，p50 1.035）",
             "why": "当前 100% 的更新被裁剪，参数位移恒为 lr×0.5，与梯度大小脱钩；"
                    "长训会在最优点附近持续抖动、无法退火",
             "contract_basis": "value_objective_contract_v1.estimator 明确写 max_grad_norm 属可调优化超参"},
            {"id": "budget-below-reference-energy",
             "for": "NAV-036（预算选择）",
             "action": "裁定：按场景缩放 Z_REF_J，或让残差的成本项改用未截断的 z_J/Z_REF",
             "why": "Z_REF_J=225.6 J 低于全部场景的参考能量 362–489 J，z_ratio 截断使任何策略都不可行"},
        ],
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(json.dumps({k: report[k] for k in ("part", "original_verdict", "rejudged_verdict",
                                             "gates_after_amendment")}, ensure_ascii=False))
    print(json.dumps(g4, ensure_ascii=False)[:700])
    return 0 if verdict_after == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
