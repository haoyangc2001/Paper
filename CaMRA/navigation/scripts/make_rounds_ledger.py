"""Rebuild reports/training_loop_rounds.json from the four round reports."""
from __future__ import annotations

import json
import os
import statistics

NAV = os.getcwd()

rounds = []
for n in (1, 2, 3, 4):
    p = os.path.join(NAV, "reports", f"training_round_r{n}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    rounds.append({
        "round": n,
        "prereg": d["prereg"],
        "config_delta": d.get("config_delta") or d.get("prereg_version") or None,
        "closing_tier": d["closing_tier"]["by_method"],
        "sr_within_budget_max": max(
            r["dev"]["sr_within_budget"] for r in d["runs"]),
        "closing_tier_met": d["closing_tier"]["closing_tier_met"],
        "report": f"reports/training_round_r{n}.json",
    })

ledger = {
    "schema": "camra-training-loop-ledger-v1",
    "task": "NAV-064",
    "protocol": "config/training_loop_protocol_v1_1.json",
    "status": "rounds_exhausted",
    "closing_tier_met": any(r["closing_tier_met"] and r["sr_within_budget_max"] > 0
                            for r in rounds),
    "rounds": rounds,
    "chain_finding_entropy": {
        "topic": "log_std 无界增长（跨轮热启动链）",
        "observed": {
            "r1": "4.7 -> 4.8（无热启动链，单轮内稳定）",
            "r2_uncal_s0": "8.4 -> 17.9", "r2_full_s0": "10.2 -> 21.8",
            "r3_uncal_s0": "17.9 -> 21.8", "r3_full_s0": "21.8 -> 26.1",
            "r4": "21.8-27.5 -> 25.9-32.2（6 个 run 在第 566-584 个更新处退出，未到 600）",
        },
        "mechanism": "损失 = max(ratio·A, clip(ratio)·A) − ent_coef·H：优势接近零的样本上熵项恒为 −ent_coef·H，最小化它单调推高 H=log_std；只有优势方差能把它压回来。full 的安全饱和区（63-64% 步 h≥1）优势≈0，热启动链上的 uncal 也进入低优势区 → std 无界增长 → 动作近似纯噪声 → 训练在 ~570/600 更新处死亡",
        "why_not_fixed_in_loop": "动作概率契约（对角高斯、可学习 log_std、无界）在协议 frozen_across_rounds 里；钳位/衰减 = 契约变更，须用户授权",
    },
    "excluded_hypotheses": [
        "H(r2) 熵 0.02 导致 uncalibrated 退化：部分成立——r3 熵回退 0.01 后仍崩（0.667/0/0），说明主因是链上的 std 增长而非本轮熵系数",
        "H(r3) full 需要语义级处理才能离开停滞盆地：被否——full 在 r3/r4 保持 0.333-0.667，盆地可以靠常规训练离开",
        "H(r4) 学习率减半能稳定 uncalibrated：被否——r4 未达标且 run 在 566-584 更新处死亡",
    ],
    "remaining_options_for_user": [
        "A. 授权动作概率契约 v2（log_std 上限钳位或熵系数随训练衰减）并授权第 5 轮（超出协议 4 轮）",
        "B. 停止循环，用现有最好候选（r2/r3 的 full 与 r1 的 uncalibrated）走 NAV-065/066 正式预算判定，把『预算内到达不可行』按负结果记录",
        "C. 语义层裁定：C·y 扩展在危险饱和区造成平坦代价（协议允许的刻意偏离 M4），δ=2.61 m 下 full 的可行地图有限——回到 NAV-061 契约修订",
    ],
}
out = os.path.join(NAV, "reports", "training_loop_rounds.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(ledger, fh, ensure_ascii=False, indent=2, default=str)
print(json.dumps({"rounds": [(r["round"], r["closing_tier"]["uncalibrated"]["sr_median"],
                              r["closing_tier"]["full"]["sr_median"],
                              r["sr_within_budget_max"]) for r in rounds],
                  "written": out}, ensure_ascii=False, default=str))
