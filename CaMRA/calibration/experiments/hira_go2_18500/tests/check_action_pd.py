"""动作/PD 适配器自测（HIRA-014）。

    python tests/check_action_pd.py --out reports/action_pd_tests.json

oracle 用 ``action_pd.source_q_target / source_pd_torque`` 两份**独立公式**（不是适配器方法），
逐关节按名字核对，避免"两边同一个 bug 互相抵消"。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
sys.path.insert(0, str(ADAPTER))

import torch  # noqa: E402

from hira_go2 import asset_cfg  # noqa: E402
from hira_go2.action_pd import (  # noqa: E402
    ACTION_SCALE,
    CLIP_ACTIONS,
    DAMPING,
    STIFFNESS,
    ActionTorqueAdapter,
    source_pd_torque,
    source_q_target,
)
from hira_go2.observation import JointOrder  # noqa: E402

LAB_DOF_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
SOURCE_ORDER = list(asset_cfg.HIRA_DEFAULT_ANGLES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(EXP / "reports/action_pd_tests.json"))
    args = parser.parse_args()

    torch.manual_seed(20261105)
    checks: list[dict] = []

    def add(check_id: str, name: str, ok: bool, detail: str) -> None:
        checks.append({"id": check_id, "name": name, "ok": bool(ok), "detail": detail})

    order = JointOrder(SOURCE_ORDER, LAB_DOF_ORDER)
    default_source = torch.tensor([asset_cfg.HIRA_DEFAULT_ANGLES[name] for name in SOURCE_ORDER])
    effort_table = asset_cfg.load_joint_table(REPO / asset_cfg.HIRA_URDF_RELATIVE)
    effort_source = torch.tensor([effort_table[name]["effort"] for name in SOURCE_ORDER])
    default_lab = order.source_to_target(default_source)
    effort_lab = order.source_to_target(effort_source)

    def make_adapter() -> ActionTorqueAdapter:
        return ActionTorqueAdapter(order, default_source, effort_source)

    # ---- A1 q_target 与独立公式一致 ----
    adapter = make_adapter()
    action = torch.linspace(-3.0, 3.0, 12).reshape(1, 12)
    q_target = adapter.set_policy_action(action)
    expected_q_target = source_q_target(action, default_source.reshape(1, 12))
    per_joint_q_ok = True
    for index, name in enumerate(SOURCE_ORDER):
        expected = asset_cfg.HIRA_DEFAULT_ANGLES[name] + ACTION_SCALE * float(action[0, index])
        per_joint_q_ok = per_joint_q_ok and abs(float(q_target[0, index]) - expected) < 1e-6
    add(
        "A1-q-target",
        "q_target = q_default + 0.25*a，逐关节与公式一致",
        bool(torch.allclose(q_target, expected_q_target, atol=1e-6)) and per_joint_q_ok,
        f"max_abs_diff={float((q_target - expected_q_target).abs().max()):.3e}，逐关节一致={per_joint_q_ok}",
    )

    # ---- A2 零动作 → 默认姿态 ----
    adapter = make_adapter()
    q_zero = adapter.set_policy_action(torch.zeros(1, 12))
    add(
        "A2-zero-action",
        "零动作时关节目标等于默认角",
        bool(torch.allclose(q_zero, default_source.reshape(1, 12), atol=1e-9)),
        f"max_abs_diff={float((q_zero - default_source.reshape(1, 12)).abs().max()):.3e}",
    )

    # ---- A3 正负饱和 ----
    adapter = make_adapter()
    adapter.set_policy_action(torch.full((1, 12), CLIP_ACTIONS))
    tau_pos = adapter.compute_substep(default_lab.reshape(1, 12), torch.zeros(1, 12))
    adapter.set_policy_action(torch.full((1, 12), -CLIP_ACTIONS))
    tau_neg = adapter.compute_substep(default_lab.reshape(1, 12), torch.zeros(1, 12))
    expected_pd = STIFFNESS * ACTION_SCALE * CLIP_ACTIONS
    add(
        "A3-saturation",
        "±100 动作把力矩顶到每关节 ±effort（先算 PD 再裁一次）",
        bool(torch.allclose(tau_pos["tau_clipped_source"].reshape(-1), effort_source, atol=1e-5))
        and bool(torch.allclose(tau_neg["tau_clipped_source"].reshape(-1), -effort_source, atol=1e-5))
        and int(tau_pos["n_saturated_joints"]) == 12,
        f"τ_PD = kp*25 = {float(STIFFNESS * ACTION_SCALE * CLIP_ACTIONS):.1f} N·m，"
        f"裁到 {[round(float(value), 2) for value in tau_pos['tau_clipped_source'].reshape(-1)[:3]]}…；"
        f"饱和关节数={int(tau_pos['n_saturated_joints'])}（PD 原始值 {expected_pd:.0f} 已被裁）",
    )

    # ---- A4 单关节扰动：符号与幅值 ----
    adapter = make_adapter()
    adapter.set_policy_action(torch.zeros(1, 12))
    target_name = "RL_calf_joint"
    target_index = SOURCE_ORDER.index(target_name)
    perturbed = default_source.reshape(1, 12).clone()
    perturbed[0, target_index] += 0.1
    trace_plus = adapter.compute_substep(order.source_to_target(perturbed), torch.zeros(1, 12))
    perturbed[0, target_index] -= 0.2
    trace_minus = adapter.compute_substep(order.source_to_target(perturbed), torch.zeros(1, 12))
    others_zero = sum(
        abs(float(trace_plus["tau_pd_source"].reshape(-1)[index]))
        for index in range(12)
        if index != target_index
    )
    add(
        "A4-single-joint",
        "单关节偏离 0.1 rad → τ = -kp*0.1（符号反向、其他关节为零）",
        abs(float(trace_plus["tau_pd_source"].reshape(-1)[target_index]) + STIFFNESS * 0.1) < 1e-5
        and abs(float(trace_minus["tau_pd_source"].reshape(-1)[target_index]) - STIFFNESS * 0.1) < 1e-5
        and others_zero < 1e-5,
        f"{target_name}: +0.1 rad → τ={float(trace_plus['tau_pd_source'].reshape(-1)[target_index]):.3f}，"
        f"-0.1 rad → τ={float(trace_minus['tau_pd_source'].reshape(-1)[target_index]):.3f}，其余关节和={others_zero:.3e}",
    )

    # ---- A5 q_target 恒定、τ 随子步变化 ----
    adapter = make_adapter()
    adapter.set_policy_action(torch.full((1, 12), 1.0))
    targets, taus = [], []
    for step in range(4):
        q = default_lab.reshape(1, 12) + 0.01 * step
        qd = torch.full((1, 12), 0.2 * step)
        trace = adapter.compute_substep(q, qd)
        targets.append(trace["q_target_source"].clone())
        taus.append(trace["tau_pd_source"].clone())
    same_target = all(bool(torch.equal(targets[0], item)) for item in targets[1:])
    tau_changed = all(not bool(torch.equal(taus[0], item)) for item in taus[1:])
    add(
        "A5-substep-pd",
        "4 个物理子步共用一个 q_target，但每步用新鲜 q/qdot 重算 τ",
        same_target and tau_changed and adapter.substep_index == 4,
        f"q_target 四步相同={same_target}，τ 每步变化={tau_changed}，子步计数={adapter.substep_index}",
    )

    # ---- A6 顺序映射与逆变换 ----
    adapter = make_adapter()
    adapter.set_policy_action(torch.linspace(-2.0, 2.0, 12).reshape(1, 12))
    trace = adapter.compute_substep(default_lab.reshape(1, 12), torch.zeros(1, 12))
    mapped_ok = True
    for index, name in enumerate(SOURCE_ORDER):
        lab_index = LAB_DOF_ORDER.index(name)
        mapped_ok = mapped_ok and abs(
            float(trace["effort_written_lab"].reshape(-1)[lab_index])
            - float(trace["tau_clipped_source"].reshape(-1)[index])
        ) < 1e-9
    round_trip = order.source_to_target(order.target_to_source(default_source.reshape(1, 12)))
    add(
        "A6-order-map",
        "源顺序的 τ 按名字写到 Lab 位置；置换可逆",
        mapped_ok and bool(torch.allclose(round_trip, default_source.reshape(1, 12), atol=1e-9)),
        f"逐关节按名字对齐={mapped_ok}，置换往返最大差={float((round_trip - default_source.reshape(1, 12)).abs().max()):.3e}",
    )

    # ---- A7 拒绝重复裁剪 ----
    adapter = make_adapter()
    rejected_over = rejected_under = False
    try:
        adapter.set_policy_action(torch.full((1, 12), CLIP_ACTIONS + 1.0))
    except ValueError:
        rejected_over = True
    try:
        adapter.set_policy_action(torch.full((1, 12), -CLIP_ACTIONS - 0.5))
    except ValueError:
        rejected_under = True
    adapter.set_policy_action(torch.zeros(1, 12))
    big = adapter.compute_substep(default_lab.reshape(1, 12) + 100.0, torch.zeros(1, 12))
    single_clip = bool(torch.allclose(big["tau_clipped_source"], torch.clamp(big["tau_pd_source"], -effort_source, effort_source)))
    add(
        "A7-no-double-clip",
        "超界动作被拒；力矩只裁一次（clipped == clamp(PD)）",
        rejected_over and rejected_under and single_clip,
        f"超上界被拒={rejected_over}，超下界被拒={rejected_under}，单次裁剪成立={single_clip}",
    )

    # ---- A8 增益来自源 ----
    add(
        "A8-gains",
        "kp=20、kd=0.5（不是 v2 的 25）",
        STIFFNESS == 20.0 and DAMPING == 0.5 and make_adapter().stiffness == 20.0,
        f"stiffness={STIFFNESS}, damping={DAMPING}, action_scale={ACTION_SCALE}",
    )

    # ---- A9 日志字段齐全 ----
    required = {
        "clipped_action_source", "q_target_source", "q_target_lab",
        "tau_pd_source", "tau_clipped_source", "effort_written_lab",
    }
    add(
        "A9-trace",
        "每子步记录 raw/clipped 动作、q_target、τ_PD、τ_clipped、写入 effort",
        required.issubset(trace.keys()),
        f"缺字段={sorted(required - set(trace.keys()))}",
    )

    # ---- A10 没有隐藏限速 ----
    adapter = make_adapter()
    adapter.set_policy_action(torch.zeros(1, 12))
    trace_fast = adapter.compute_substep(default_lab.reshape(1, 12), torch.full((1, 12), -40.0))
    expected_tau = DAMPING * 40.0
    velocity_limit_min = min(row["velocity_limit"] for row in effort_table.values())
    add(
        "A10-no-velocity-limit",
        "qdot=40 rad/s(>30.1 上限) 时 τ=-kd*qdot，未被速度限幅",
        abs(float(trace_fast["tau_pd_source"].reshape(-1)[0]) - expected_tau) < 1e-5,
        f"τ={float(trace_fast['tau_pd_source'].reshape(-1)[0]):.3f}（=0.5*40），最小速度上限={velocity_limit_min}",
    )

    failed = [item["id"] for item in checks if not item["ok"]]
    report = {
        "report": "action_pd_tests",
        "task": "HIRA-014",
        "constants": {"stiffness": STIFFNESS, "damping": DAMPING, "action_scale": ACTION_SCALE,
                      "clip_actions": CLIP_ACTIONS, "decimation": 4},
        "effort_limits_source": {name: float(value) for name, value in zip(SOURCE_ORDER, effort_source)},
        "checks": checks,
        "verdict": {"all_passed": not failed, "failed": failed, "n_checks": len(checks)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for item in checks:
        print(f"[{'ok ' if item['ok'] else 'FAIL'}] {item['id']:22s} {item['name']} :: {item['detail']}")
    print("ACTION_PD_OK" if not failed else f"ACTION_PD_FAIL {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
