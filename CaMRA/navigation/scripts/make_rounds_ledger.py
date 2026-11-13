"""Rebuild reports/training_loop_rounds.json from all round reports (r1-r5)."""
from __future__ import annotations

import json
import os

NAV = os.getcwd()

rounds = []
for n in (1, 2, 3, 4, 5):
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
        "checkpoint": d["runs"][0].get("checkpoint", "state.pt"),
        "report": f"reports/training_round_r{n}.json",
    })

ledger = {
    "schema": "camra-training-loop-ledger-v1",
    "task": "NAV-064",
    "protocol": "config/training_loop_protocol_v1_1.json",
    "status": "rounds_exhausted_r5_user_authorized",
    "closing_tier_met": any(r["closing_tier_met"] and r["sr_within_budget_max"] > 0
                            for r in rounds),
    "rounds": rounds,
    "chain_finding_entropy": {
        "topic": "log_std 无界增长（跨轮热启动链）与 amend2 钳位",
        "observed": {
            "r1": "4.7 -> 4.8（无热启动链，单轮内稳定）",
            "r2_uncal_s0": "8.4 -> 17.9", "r2_full_s0": "10.2 -> 21.8",
            "r3_uncal_s0": "17.9 -> 21.8", "r3_full_s0": "21.8 -> 26.1",
            "r4": "21.8-27.5 -> 25.9-32.2（6 个 run 在第 566-584 个更新处退出，未到 600）",
            "r5": "4.3（重置）-> 6.4-7.3 稳定：钳位生效，6/6 run 完整跑完 600 更新，无死亡",
        },
        "mechanism": "损失 = max(ratio·A, clip(ratio)·A) − ent_coef·H：优势接近零的样本上熵项恒为 −ent_coef·H，最小化它单调推高 H=log_std；只有优势方差能把它压回来。full 的安全饱和区（63-64% 步 h≥1）优势≈0，热启动链上的 uncal 也进入低优势区 → std 无界增长 → 动作近似纯噪声 → 训练在 ~570/600 更新处死亡",
        "fix": "rcppo_alignment_contract_v1.amend2（v1.1.0，用户 2026-11-13 批准）：log_std 可学习范围 [-5,1]，forward 防御钳位 + 每梯度步参数钳位；warm start 时重置回契约初值 0",
    },
    "excluded_hypotheses": [
        "H(r2) 熵 0.02 导致 uncalibrated 退化：部分成立——r3 熵回退 0.01 后仍崩（0.667/0/0），说明主因是链上的 std 增长而非本轮熵系数",
        "H(r3) full 需要语义级处理才能离开停滞盆地：被否——full 在 r3/r4/r5 保持 0.333-0.667，盆地可以靠常规训练离开",
        "H(r4) 学习率减半能稳定 uncalibrated：被否——r4 未达标且 run 在 566-584 更新处死亡",
        "H(r5) 仅钳位即可稳定 uncalibrated：部分成立——不再发散、run 完整、s0 达 1.0，但种子间仍 1.0/0/0（中位 0.0），链外的种子级方差是残余因素",
    ],
    "user_decision_r5": "A（2026-11-13）：修订动作概率契约（amend2 log_std ∈ [-5,1]）+ 授权第 5 轮；选项 B（停止循环走负结果）与 C（C·y 语义层修订）未被选择",
    "objective_signal_fix": {
        "date": "2026-11-13",
        "user_decision": "批准诊断建议 ①+②：把逐步成本信号接回策略梯度；联合估计器补登记参考 min 型递推",
        "prereg": "config/objective_signal_fix_prereg_v1.json (sha acfc34263329)",
        "amendment": "config/rcppo_alignment_contract_v1.amend3.json",
        "result": "① 已启用（w=50，由冻结预登记驱动）：corr(每步成本,优势) 中位数 r5_full 0.000→+0.487、r5_uncal +0.080→+0.182、r1c +0.033→+0.318；到达信号 0.942→0.946。② 实装但默认关闭：严格参考形态会把到达梯度翻负（r5_uncal +0.946→−0.935、r1c +0.386→−0.744），登记为 D4 并用反例钉住",
        "verification": "reports/objective_signal_fix.json 7/7 PASS；G3_alignment_v2 PASS；16 个既有测试脚本全绿；冒烟训练确认开关生效",
        "not_claimed": "未跑新训练轮次，成功率/预算内到达均未验证；第 6 轮需新授权",
    },
    "remaining_options_for_user": [
        "A 已执行：钳位 + 第 5 轮完成，关闭档仍未达成（uncal 中位 0.0，full 0.333）",
        "B 仍可选：停止循环，用现有最好候选（r1 uncal s0-s2、r5 full s1）走 NAV-065/066 正式预算判定",
        "C 仍可选：C·y/δ 语义层修订（NAV-061），处理危险饱和区的平坦代价根因",
        "D 新选项：再加轮（每次均需授权）；r5 显示 uncal 的种子级不稳定需要更深的训练量或种子级干预（如按种子早停保留最优）",
        "E（2026-11-13 已完成实现层，尚未验证行为）：第 6 轮用 w=50 的新目标训练，验证『成本信号进梯度』能否带来预算内到达；需用户单独授权加轮",
    ],
}
out = os.path.join(NAV, "reports", "training_loop_rounds.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(ledger, fh, ensure_ascii=False, indent=2, default=str)
print(json.dumps({"rounds": [(r["round"], r["closing_tier"]["uncalibrated"]["sr_median"],
                              r["closing_tier"]["full"]["sr_median"],
                              r["sr_within_budget_max"]) for r in rounds],
                  "written": out}, ensure_ascii=False, default=str))
