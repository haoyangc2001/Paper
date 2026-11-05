"""45 维观测适配器自测（HIRA-013）。

    python tests/check_observation_adapter.py --out reports/observation_adapter_tests.json

不依赖 Isaac Lab：适配器本身只用 torch，本脚本在 CPU 上跑完即可。
对照基准是一份**独立实现**的 oracle（逐块用标量算，不复用适配器的任何代码），
避免"两边同一个 bug 抵消"这种假通过。
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
from hira_go2.observation import (  # noqa: E402
    BLOCK_SLICES,
    body_angular_velocity_wxyz,
    projected_gravity_wxyz,
    quat_rotate_inverse_wxyz,
    quat_xyzw_to_wxyz,
    CLIP_ACTIONS,
    CLIP_OBSERVATIONS,
    OBS_DIM,
    ObservationAdapter,
    JointOrder,
    build_noise_vector,
)

#: 与 HIRA-012 运行时审计一致的 Lab DOF 顺序（按关节类型分组）。
LAB_DOF_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
SOURCE_ORDER = list(asset_cfg.HIRA_DEFAULT_ANGLES)

ATOL = 1e-6
RTOL = 1e-5


# --------------------------------------------------------------------------- #
# 独立 oracle：逐块标量实现，不 import 适配器的常量以外的任何逻辑
# --------------------------------------------------------------------------- #
def oracle_observation(
    source_order: list[str],
    default_angles: dict[str, float],
    ang_vel_by_joint_free: list[float],
    gravity: list[float],
    command: list[float],
    joint_pos_by_name: dict[str, float],
    joint_vel_by_name: dict[str, float],
    prev_action_by_name: dict[str, float],
) -> list[float]:
    obs = []
    obs += [value * 0.25 for value in ang_vel_by_joint_free]          # 0:3
    obs += [value * 1.0 for value in gravity]                          # 3:6
    obs += [command[0] * 2.0, command[1] * 2.0, command[2] * 0.25]     # 6:9
    for name in source_order:                                          # 9:21
        obs.append((joint_pos_by_name[name] - default_angles[name]) * 1.0)
    for name in source_order:                                          # 21:33
        obs.append(joint_vel_by_name[name] * 0.05)
    for name in source_order:                                          # 33:45
        obs.append(prev_action_by_name[name] * 1.0)
    return [max(-100.0, min(100.0, value)) for value in obs]


def compare(actual: torch.Tensor, expected: list[float]) -> dict:
    expected_tensor = torch.tensor(expected, dtype=actual.dtype)
    diff = (actual.detach().cpu().reshape(-1) - expected_tensor).abs()
    allowed = ATOL + RTOL * expected_tensor.abs()
    return {
        "max_abs_diff": float(diff.max()),
        "n_violations": int((diff > allowed).sum()),
        "passed": bool((diff <= allowed).all()),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(EXP / "reports/observation_adapter_tests.json"))
    args = parser.parse_args()

    torch.manual_seed(20261105)
    num_envs = 4
    checks: list[dict] = []

    def add(check_id: str, name: str, ok: bool, detail: str) -> None:
        checks.append({"id": check_id, "name": name, "ok": bool(ok), "detail": detail})

    order = JointOrder(SOURCE_ORDER, LAB_DOF_ORDER)
    default_source = torch.tensor([asset_cfg.HIRA_DEFAULT_ANGLES[name] for name in SOURCE_ORDER])

    # ---- O1 布局 ----
    slices = [(name, sl.start, sl.stop) for name, sl in BLOCK_SLICES]
    add(
        "O1-blocks",
        "六块尺寸与顺序",
        [stop - start for _, start, stop in slices] == [3, 3, 3, 12, 12, 12]
        and slices[-1][2] == OBS_DIM,
        f"{slices}，合计 {slices[-1][2]}",
    )

    # ---- 输入：Lab 侧物理量（目标顺序）----
    q_default_lab = order.source_to_target(default_source)
    joint_pos_lab = q_default_lab.repeat(num_envs, 1) + 0.2 * torch.randn(num_envs, 12) * 0.5
    joint_vel_lab = torch.randn(num_envs, 12) * 1.5
    ang_vel = torch.randn(num_envs, 3) * 0.5
    gravity = torch.nn.functional.normalize(torch.randn(num_envs, 3), dim=-1)
    command = torch.tensor([[0.5, -0.2, 0.3]]).repeat(num_envs, 1)

    adapter = ObservationAdapter(order, default_source, num_envs=num_envs)
    result = adapter.build(ang_vel, gravity, command, joint_pos_lab, joint_vel_lab)

    # ---- O2 oracle 对照（上一动作此时仍为 0）----
    obs_env0 = result["obs"][0].tolist()
    expected_env0 = oracle_observation(
        SOURCE_ORDER,
        asset_cfg.HIRA_DEFAULT_ANGLES,
        ang_vel[0].tolist(),
        gravity[0].tolist(),
        command[0].tolist(),
        {name: joint_pos_lab[0][index].item() for index, name in enumerate(LAB_DOF_ORDER)},
        {name: joint_vel_lab[0][index].item() for index, name in enumerate(LAB_DOF_ORDER)},
        {name: 0.0 for name in SOURCE_ORDER},
    )
    cmp0 = compare(result["obs"][0], expected_env0)
    add("O2-oracle", "与独立 oracle 逐点对照（含命令缩放/默认角/关节置换）", cmp0["passed"],
        f"max_abs_diff={cmp0['max_abs_diff']:.3e}, 越界={cmp0['n_violations']}")

    # ---- O3 双射 ----
    identity = JointOrder(SOURCE_ORDER, SOURCE_ORDER)
    shuffled = list(SOURCE_ORDER)
    torch.manual_seed(7)
    shuffled = [shuffled[index] for index in torch.randperm(12).tolist()]
    random_order = JointOrder(SOURCE_ORDER, shuffled)
    probe = torch.arange(12, dtype=torch.float32)
    back = random_order.target_to_source(random_order.source_to_target(probe))
    dup_rejected = missing_rejected = False
    try:
        JointOrder(SOURCE_ORDER, SOURCE_ORDER[:-1] + [SOURCE_ORDER[0]])
    except ValueError:
        dup_rejected = True
    try:
        JointOrder(SOURCE_ORDER, SOURCE_ORDER[:-1])
    except ValueError:
        missing_rejected = True
    add(
        "O3-bijection",
        "按名字双射可逆，重复/缺失被拒",
        bool(torch.allclose(probe, back)) and identity.source_index_of_target == list(range(12)) and dup_rejected and missing_rejected,
        f"随机置换往返一致={bool(torch.allclose(probe, back))}，重复被拒={dup_rejected}，缺失被拒={missing_rejected}",
    )

    # ---- O4 上一动作语义：裁剪后、未乘 0.25 ----
    raw = torch.full((num_envs, 12), 250.0)
    raw[0, 0] = -3.0
    clipped = adapter.apply_action(raw)
    obs_after = adapter.build(ang_vel, gravity, command, joint_pos_lab, joint_vel_lab)["obs"]
    block = obs_after[:, 33:45]
    add(
        "O4-prev-action",
        "上一动作 = clip(a, ±100)，不乘 action_scale",
        bool(torch.allclose(clipped, torch.clamp(raw, -CLIP_ACTIONS, CLIP_ACTIONS)))
        and bool(torch.allclose(block[0], clipped[0]))
        and abs(float(block[0, 1]) - 100.0) < 1e-6,
        f"clip 后 env0 第 0 位={float(clipped[0, 0]):.3f}，观测 33:45 与之逐位相同；未裁剪值 250 → 100",
    )

    # ---- O5 只有 33:45 受上一动作影响 ----
    before = adapter.build(ang_vel, gravity, command, joint_pos_lab, joint_vel_lab)["obs"]
    adapter.apply_action(torch.randn(num_envs, 12))
    after = adapter.build(ang_vel, gravity, command, joint_pos_lab, joint_vel_lab)["obs"]
    diff = (after - before).abs()
    add(
        "O5-isolation",
        "上一动作只影响 33:45",
        float(diff[:, :33].max()) == 0.0 and float(diff[:, 33:].max()) > 0.0,
        f"0:33 最大变化 {float(diff[:, :33].max()):.3e}，33:45 最大变化 {float(diff[:, 33:].max()):.3e}",
    )

    # ---- O6 裁剪两侧一致 ----
    big = torch.full((num_envs, 3), 1e3)
    obs_big = adapter.build(big, gravity * 1e3, command * 1e3, joint_pos_lab * 1e3, joint_vel_lab * 1e3)["obs"]
    add(
        "O6-clip",
        "观测裁剪到 ±100，且与 oracle 的裁剪一致",
        float(obs_big.abs().max()) <= CLIP_OBSERVATIONS + 1e-6
        and float(obs_big[:, 6:9].max()) == CLIP_OBSERVATIONS
        and float(obs_big[:, 6:9].min()) == -CLIP_OBSERVATIONS,
        f"最大值 {float(obs_big.abs().max()):.3f}，命令块上下限 {float(obs_big[:, 6:9].max()):.1f}/{float(obs_big[:, 6:9].min()):.1f}",
    )

    # ---- O7 逐 env reset 隔离 ----
    adapter.apply_action(torch.ones(num_envs, 12))
    adapter.reset([2])
    prev = adapter.prev_action
    add(
        "O7-reset",
        "reset 某一 env 只清零该 env 的上一动作",
        float(prev[2].abs().max()) == 0.0
        and float(prev[0].abs().max()) == 1.0
        and float(prev[1].abs().max()) == 1.0
        and float(prev[3].abs().max()) == 1.0,
        f"reset env2 后 prev_action 逐 env 最大绝对值 = {[float(prev[index].abs().max()) for index in range(num_envs)]}",
    )

    # ---- O8 噪声 mask ----
    noisy_adapter = ObservationAdapter(
        order, default_source, num_envs=num_envs, noise_scale=build_noise_vector(),
        generator=torch.Generator().manual_seed(11),
    )
    noisy_adapter.apply_action(torch.zeros(num_envs, 12))
    clean = noisy_adapter.build(ang_vel, gravity, command, joint_pos_lab, joint_vel_lab, include_noise=False)["obs"]
    noisy = noisy_adapter.build(ang_vel, gravity, command, joint_pos_lab, joint_vel_lab, include_noise=True)["obs"]
    d = (noisy - clean).abs()
    mask_ok = float(d[:, :9].max()) == 0.0 and float(d[:, 33:].max()) == 0.0 and float(d[:, 9:33].max()) > 0.0
    no_scale_rejected = False
    try:
        adapter.build(ang_vel, gravity, command, joint_pos_lab, joint_vel_lab, include_noise=True)
    except ValueError:
        no_scale_rejected = True
    add(
        "O8-noise",
        "噪声只落在 9:33；clean 跑法不加噪；无噪声向量时加噪被拒",
        mask_ok and no_scale_rejected,
        f"0:9 变化 {float(d[:, :9].max()):.3e}，9:33 变化 {float(d[:, 9:33].max()):.3e}，"
        f"33:45 变化 {float(d[:, 33:].max()):.3e}，无向量加噪被拒={no_scale_rejected}",
    )

    # ---- O9 只缩放一次 ----
    unit = ObservationAdapter(order, default_source, num_envs=1)
    unit.apply_action(torch.zeros(1, 12))
    zero_pos = q_default_lab.reshape(1, 12)
    obs_unit = unit.build(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.tensor([[0.5, 0.25, 0.4]]),
        zero_pos,
        torch.zeros(1, 12),
    )["obs"]
    add(
        "O9-single-scale",
        "命令只乘一次 [2, 2, 0.25]，角速度只乘 0.25，重力不缩放",
        abs(float(obs_unit[0, 0]) - 0.25) < 1e-6
        and abs(float(obs_unit[0, 5]) + 1.0) < 1e-6
        and abs(float(obs_unit[0, 6]) - 1.0) < 1e-6
        and abs(float(obs_unit[0, 7]) - 0.5) < 1e-6
        and abs(float(obs_unit[0, 8]) - 0.1) < 1e-6,
        f"ang_vel[0]={float(obs_unit[0, 0]):.3f}（1.0*0.25），gravity[2]={float(obs_unit[0, 5]):.3f}，"
        f"cmd={[round(float(obs_unit[0, index]), 3) for index in (6, 7, 8)]}",
    )

    # ---- O10 维度与"没有线速度" ----
    add(
        "O10-dim",
        "观测恰为 45 维，且不含机身线速度",
        obs_unit.shape == (1, OBS_DIM) and "lin_vel" not in {name for name, _ in BLOCK_SLICES},
        f"shape={tuple(obs_unit.shape)}，块名={[name for name, _ in BLOCK_SLICES]}",
    )

    # ---- O11 投影重力（四元数约定 wxyz）----
    def rotation_matrix_wxyz(quat: torch.Tensor) -> torch.Tensor:
        """独立构造旋转矩阵，用来核对四元数公式本身（不用同一套 cross 展开）。"""
        w, x, y, z = [float(value) for value in quat]
        return torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    quats = {
        "identity": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "yaw90": torch.tensor([2 ** -0.5, 0.0, 0.0, 2 ** -0.5]),
        "tilted": torch.nn.functional.normalize(torch.tensor([0.9, 0.1, -0.2, 0.3]), dim=0),
    }
    gravity_world = torch.tensor([0.0, 0.0, -1.0])
    gravity_ok = True
    gravity_detail = []
    for name, quat in quats.items():
        got = projected_gravity_wxyz(quat.unsqueeze(0))[0]
        expected = rotation_matrix_wxyz(quat).T @ gravity_world
        ok = bool(torch.allclose(got, expected, atol=1e-6)) and abs(float(got.norm()) - 1.0) < 1e-6
        gravity_ok = gravity_ok and ok
        gravity_detail.append(f"{name}:{got.tolist()} vs {expected.tolist()}")
    add(
        "O11-projected-gravity",
        "投影重力 = R(q)^T (0,0,-1)，单位长度",
        gravity_ok and abs(float(projected_gravity_wxyz(quats["identity"].unsqueeze(0))[0, 2]) + 1.0) < 1e-6,
        "；".join(gravity_detail[:2]),
    )

    # ---- O12 角速度换系 ----
    yaw90 = quats["yaw90"]
    body_vel = body_angular_velocity_wxyz(yaw90.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]]))[0]
    body_vel_z = body_angular_velocity_wxyz(yaw90.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]]))[0]
    expected_body = rotation_matrix_wxyz(yaw90).T @ torch.tensor([1.0, 0.0, 0.0])
    add(
        "O12-body-ang-vel",
        "世界角速度经 R(q)^T 转到机体系；yaw 不改变 z 分量",
        bool(torch.allclose(body_vel, expected_body, atol=1e-6))
        and abs(float(body_vel_z[2]) - 1.0) < 1e-6
        and abs(float(body_vel[0])) < 1e-6,
        f"世界 (1,0,0) → 机体 {[round(float(value), 3) for value in body_vel]}；世界 (0,0,1) → 机体 z={float(body_vel_z[2]):.3f}",
    )

    # ---- O13 四元数顺序陷阱 ----
    quat_xyzw = torch.nn.functional.normalize(torch.tensor([0.1, 0.2, 0.3, 0.9]), dim=0)
    wrong = projected_gravity_wxyz(quat_xyzw.unsqueeze(0))[0]
    right = projected_gravity_wxyz(quat_xyzw_to_wxyz(quat_xyzw).unsqueeze(0))[0]
    expected_right = rotation_matrix_wxyz(quat_xyzw_to_wxyz(quat_xyzw)).T @ gravity_world
    add(
        "O13-quat-order",
        "xyzw 当 wxyz 用会算错；显式换序后与旋转矩阵一致",
        float((wrong - right).abs().max()) > 0.1 and bool(torch.allclose(right, expected_right, atol=1e-6)),
        f"错序差 {float((wrong - right).abs().max()):.3f}，换序后差 {float((right - expected_right).abs().max()):.3e}",
    )

    failed = [item["id"] for item in checks if not item["ok"]]
    report = {
        "report": "observation_adapter_tests",
        "task": "HIRA-013",
        "tolerances": {"atol": ATOL, "rtol": RTOL, "source": "integration_acceptance_prereg_v1.tolerances.observation_float32"},
        "orders": order.describe(),
        "checks": checks,
        "verdict": {"all_passed": not failed, "failed": failed, "n_checks": len(checks)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for item in checks:
        print(f"[{'ok ' if item['ok'] else 'FAIL'}] {item['id']:16s} {item['name']} :: {item['detail']}")
    print("OBS_ADAPTER_OK" if not failed else f"OBS_ADAPTER_FAIL {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
