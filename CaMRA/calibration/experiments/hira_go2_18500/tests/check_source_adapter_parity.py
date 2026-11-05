"""独立来源 oracle 与适配器的同状态对照（HIRA-015）。

    python tests/check_source_adapter_parity.py \
        --fixtures tests/reference_fixtures/fixtures_v1.json \
        --manifest tests/reference_fixtures/MANIFEST.json \
        --policy ../exported/policy.pt \
        --out ../reports/source_adapter_parity.json

三条纪律：

1. **oracle 不复用被测逻辑**：`source_oracle.py` 不 import ``hira_go2``，用旋转矩阵而非叉乘展开，
   关节一律按名字取字典；适配器用索引置换，两边实现路径不同。
2. **同一物理状态同时进两条实现**：fixture 里只有一份数字。
3. **负测试必须被检出**：故意错关节顺序 / 错缩放 / 错四元数 / 错默认角 / 错增益，
   每一类都要让对照报失败，否则说明检验本身是摆设。
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
sys.path.insert(0, str(EXP / "tests"))

import torch  # noqa: E402

import source_oracle  # noqa: E402
from hira_go2 import observation as observation_module  # noqa: E402
from hira_go2.action_pd import ActionTorqueAdapter  # noqa: E402
from hira_go2.observation import (  # noqa: E402
    JointOrder,
    ObservationAdapter,
    body_angular_velocity_wxyz,
    projected_gravity_wxyz,
)

TOLERANCES = {
    "observation_float32": {"atol": 1e-6, "rtol": 1e-5},
    "actor_cpu": {"atol": 1e-5, "rtol": 1e-4},
    "actor_gpu": {"atol": 1e-4, "rtol": 1e-4},
    "q_target": {"atol": 1e-6, "rtol": 1e-5},
    "torque": {"atol": 1e-4, "rtol": 1e-5},
}


def compare(actual, expected, atol: float, rtol: float) -> dict:
    """逐元素比 |a-e| <= atol + rtol*|e|；返回最差点，便于报告定位。"""
    if torch.is_tensor(actual):
        actual = actual.detach().reshape(-1).tolist()
    if torch.is_tensor(expected):
        expected = expected.detach().reshape(-1).tolist()
    if len(actual) != len(expected):
        return {"passed": False, "reason": f"长度 {len(actual)} != {len(expected)}"}
    worst_abs = 0.0
    worst_rel = 0.0
    worst_index = -1
    violations = 0
    for index, (got, ref) in enumerate(zip(actual, expected)):
        diff = abs(got - ref)
        allowed = atol + rtol * abs(ref)
        if diff > allowed:
            violations += 1
        if diff > worst_abs:
            worst_abs, worst_index = diff, index
        if abs(ref) > 1e-12:
            worst_rel = max(worst_rel, diff / abs(ref))
    return {
        "max_abs_diff": worst_abs,
        "max_rel_diff": worst_rel,
        "n_violations": violations,
        "worst_index": worst_index,
        "atol": atol,
        "rtol": rtol,
        "passed": violations == 0,
    }


def target_pipeline(
    fixture: dict,
    spec: dict,
    device: str,
    policy=None,
    default_override=None,
    stiffness=None,
    array_order=None,
    dtype=torch.float64,
):
    """把一条 fixture 喂进被测适配器，返回与 oracle 同形状的结果。

    ``array_order`` 是"数组实际按哪种顺序排列"，正常情况等于 ``fixture['target_order']``；
    负测试用它模拟"按位置对齐"的实现错误（数组按源顺序排，适配器却以为是目标顺序）。
    """
    source_order = spec["source_order"]
    target_order = fixture["target_order"]
    feed_order = target_order if array_order is None else array_order
    order = JointOrder(source_order, target_order)

    defaults = dict(spec["joint_defaults"]) if default_override is None else default_override
    default_tensor = torch.tensor([defaults[name] for name in source_order], dtype=dtype)
    effort_tensor = torch.tensor([spec["joint_effort"][name] for name in source_order], dtype=dtype)

    obs_adapter = ObservationAdapter(order, default_tensor, num_envs=1, device=device, dtype=dtype)
    # 上一动作走真实路径：策略步 N-1 的 apply_action 结果就是观测 33:45 的来源
    obs_adapter.apply_action(
        torch.tensor([fixture["prev_action"][name] for name in source_order], dtype=dtype).reshape(1, 12)
    )

    quat = torch.tensor([fixture["quat_wxyz"]], dtype=dtype, device=device)
    world_ang_vel = torch.tensor([fixture["world_ang_vel"]], dtype=dtype, device=device)
    ang_vel_body = body_angular_velocity_wxyz(quat, world_ang_vel)
    gravity_body = projected_gravity_wxyz(quat)
    command = torch.tensor([fixture["command"]], dtype=dtype, device=device)
    joint_pos = torch.tensor([[fixture["joint_pos"][name] for name in feed_order]], dtype=dtype, device=device)
    joint_vel = torch.tensor([[fixture["joint_vel"][name] for name in feed_order]], dtype=dtype, device=device)

    built = obs_adapter.build(ang_vel_body, gravity_body, command, joint_pos, joint_vel)

    actor_output = None
    if policy is not None:
        with torch.no_grad():
            actor_output = policy(built["obs"].to(torch.float32)).reshape(-1).double().cpu()

    raw_action = torch.tensor(
        [fixture["raw_action"][name] for name in source_order], dtype=dtype, device=device
    ).reshape(1, 12)
    clipped = obs_adapter.apply_action(raw_action)

    torque_adapter = ActionTorqueAdapter(
        order,
        default_tensor,
        effort_tensor,
        stiffness=20.0 if stiffness is None else stiffness,
        damping=0.5,
        device=device,
        dtype=dtype,
    )
    torque_adapter.set_policy_action(clipped)
    trace = torque_adapter.compute_substep(joint_pos, joint_vel)

    return {
        "obs": built["obs"].reshape(-1).double().cpu().tolist(),
        "actor_output": None if actor_output is None else actor_output.tolist(),
        "q_target": trace["q_target_source"].reshape(-1).double().cpu().tolist(),
        "tau_pd": trace["tau_pd_source"].reshape(-1).double().cpu().tolist(),
        "tau_clipped": trace["tau_clipped_source"].reshape(-1).double().cpu().tolist(),
        "effort_written": {
            name: float(trace["effort_written_lab"].reshape(-1)[index])
            for index, name in enumerate(target_order)
        },
        "clipped_action": clipped.reshape(-1).double().cpu().tolist(),
    }


def oracle_view(fixture: dict, spec: dict) -> dict:
    source_order = spec["source_order"]
    result = source_oracle.run(fixture, spec)
    return {
        "obs": result["obs"],
        "blocks": result["blocks"],
        "q_target": [result["q_target"][name] for name in source_order],
        "tau_pd": [result["tau_pd"][name] for name in source_order],
        "tau_clipped": [result["tau_clipped"][name] for name in source_order],
        "effort_written": {name: result["tau_clipped"][name] for name in fixture["target_order"]},
    }


def actor_output(policy, obs_list, device: str) -> list[float] | None:
    """用同一份导出策略分别吃两边的观测，比较输出差（观测若有偏差，这里会被放大）。

    设备显式传入：TorchScript 的 ``.to()`` 返回值语义在不同版本不一致，
    靠 ``next(parameters()).device`` 推断容易拿到旧设备。
    """
    if policy is None:
        return None
    with torch.no_grad():
        tensor = torch.tensor([obs_list], dtype=torch.float32, device=device)
        return policy(tensor).reshape(-1).double().cpu().tolist()


def evaluate_fixture(
    fixture: dict, spec: dict, device: str, policy, tolerances: dict, dtype=torch.float64
) -> dict:
    oracle = oracle_view(fixture, spec)
    oracle["actor_output"] = actor_output(policy, oracle["obs"], device)
    target = target_pipeline(fixture, spec, device, policy=policy, dtype=dtype)

    obs_cmp = compare(target["obs"], oracle["obs"], **tolerances["observation_float32"])
    block_cmp = {}
    for name, (start, stop) in source_oracle.BLOCKS.items():
        block_cmp[name] = compare(
            target["obs"][start:stop], oracle["obs"][start:stop], **tolerances["observation_float32"]
        )
    actor_cmp = None
    if target["actor_output"] is not None:
        actor_cmp = compare(target["actor_output"], oracle.get("actor_output", []), **tolerances["actor_cpu"])
    q_target_cmp = compare(target["q_target"], oracle["q_target"], **tolerances["q_target"])
    tau_pd_cmp = compare(target["tau_pd"], oracle["tau_pd"], **tolerances["torque"])
    tau_clip_cmp = compare(target["tau_clipped"], oracle["tau_clipped"], **tolerances["torque"])
    effort_cmp = compare(
        [target["effort_written"][name] for name in fixture["target_order"]],
        [oracle["effort_written"][name] for name in fixture["target_order"]],
        **tolerances["torque"],
    )

    return {
        "id": fixture["id"],
        "name": fixture["name"],
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "obs": obs_cmp,
        "obs_blocks": block_cmp,
        "actor": actor_cmp,
        "q_target": q_target_cmp,
        "tau_pd": tau_pd_cmp,
        "tau_clipped": tau_clip_cmp,
        "effort_written": effort_cmp,
        "passed": all(
            item["passed"]
            for item in [obs_cmp, q_target_cmp, tau_pd_cmp, tau_clip_cmp, effort_cmp]
        )
        and (actor_cmp is None or actor_cmp["passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default=str(EXP / "tests/reference_fixtures/fixtures_v1.json"))
    parser.add_argument("--manifest", default=str(EXP / "tests/reference_fixtures/MANIFEST.json"))
    parser.add_argument("--policy", default=str(EXP / "exported/policy.pt"))
    parser.add_argument("--out", default=str(EXP / "reports/source_adapter_parity.json"))
    args = parser.parse_args()

    import hashlib

    fixtures_path = Path(args.fixtures)
    manifest_path = Path(args.manifest)
    fixtures_body = fixtures_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(fixtures_body).hexdigest()
    spec = json.loads(fixtures_body.decode("utf-8"))

    policy_path = Path(args.policy)
    has_policy = policy_path.is_file()

    def load_policy(device: str):
        """每个设备单独 load 一次：同一个 ScriptModule 连续 ``.to()`` 后会留下
        参数设备元数据与实际权重不一致的状态（本机 torch 2.7 实测），
        表现为"weight 在 cuda 而 input 在 cpu"。重新 load 最省事也最难出错。"""
        if not has_policy:
            return None
        return torch.jit.load(str(policy_path)).to(device).eval()

    policy = load_policy("cpu")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")

    report_devices: dict[str, dict] = {}
    all_failures: list[dict] = []
    dtypes = [("float64", torch.float64), ("float32", torch.float32)]
    for device in devices:
        policy_on_device = load_policy(device)
        for dtype_name, dtype in dtypes:
            key = f"{device}:{dtype_name}"
            rows = []
            for fixture in spec["fixtures"]:
                row = evaluate_fixture(fixture, spec, device, policy_on_device, TOLERANCES, dtype=dtype)
                rows.append(row)
                if not row["passed"]:
                    all_failures.append({"device": key, "fixture": row["id"], "block": "core"})
            report_devices[key] = {
                "fixtures": rows,
                "summary": {
                    "n_fixtures": len(rows),
                    "n_passed": sum(1 for row in rows if row["passed"]),
                    "max_obs_abs": max(row["obs"]["max_abs_diff"] for row in rows),
                    "max_actor_abs": max(
                        (row["actor"]["max_abs_diff"] for row in rows if row["actor"]), default=0.0
                    ),
                    "max_q_target_abs": max(row["q_target"]["max_abs_diff"] for row in rows),
                    "max_tau_abs": max(
                        max(row["tau_pd"]["max_abs_diff"], row["tau_clipped"]["max_abs_diff"]) for row in rows
                    ),
                    "max_effort_abs": max(row["effort_written"]["max_abs_diff"] for row in rows),
                },
            }

    # ---------------- 负测试：故意做错，检验必须报错 ----------------
    base_fixture = next(item for item in spec["fixtures"] if item["id"] == "F03")
    baseline = evaluate_fixture(base_fixture, spec, "cpu", policy, TOLERANCES)
    negative: list[dict] = []

    # N1 关节顺序错位：数组按源顺序排，适配器却按目标顺序解释（"按位置对齐"的实现）
    broken_row = target_pipeline(base_fixture, spec, "cpu", policy=None, array_order=list(spec["source_order"]))
    oracle_base = oracle_view(base_fixture, spec)
    broken_dof = compare(broken_row["obs"][9:21], oracle_base["obs"][9:21], **TOLERANCES["observation_float32"])
    negative.append({
        "id": "N1-joint-order",
        "bug": "把源顺序的关节量当成目标顺序（按位置对齐）",
        "detected": not broken_dof["passed"],
        "evidence": {"max_abs": broken_dof["max_abs_diff"], "n_violations": broken_dof["n_violations"]},
    })

    # N2 缩放错：把命令缩放改成 [2,2,0.5]
    original_scale = observation_module.COMMAND_SCALE
    try:
        observation_module.COMMAND_SCALE = (2.0, 2.0, 0.5)
        row = evaluate_fixture(base_fixture, spec, "cpu", policy, TOLERANCES)
        negative.append({
            "id": "N2-command-scale",
            "bug": "命令缩放第三位 0.25 写成 0.5",
            "detected": not row["passed"],
            "evidence": {"command_block_passed": row["obs_blocks"]["command"]["passed"],
                         "max_abs": row["obs_blocks"]["command"]["max_abs_diff"]},
        })
    finally:
        observation_module.COMMAND_SCALE = original_scale

    # N3 四元数顺序错：xyzw 直接当 wxyz 用
    quat = torch.tensor([base_fixture["quat_wxyz"]], dtype=torch.float64)
    wrong_quat = quat[:, [1, 2, 3, 0]]
    oracle = oracle_view(base_fixture, spec)
    wrong_gravity = projected_gravity_wxyz(wrong_quat).reshape(-1).tolist()
    gravity_cmp = compare(wrong_gravity, oracle["blocks"]["projected_gravity"], **TOLERANCES["observation_float32"])
    negative.append({
        "id": "N3-quat-order",
        "bug": "四元数 xyzw 当 wxyz",
        "detected": not gravity_cmp["passed"],
        "evidence": {"max_abs": gravity_cmp["max_abs_diff"], "n_violations": gravity_cmp["n_violations"]},
    })

    # N4 默认角错：单个关节默认角 +0.05
    wrong_defaults = dict(spec["joint_defaults"])
    wrong_defaults["RL_calf_joint"] += 0.05
    row = target_pipeline(base_fixture, spec, "cpu", policy=None, default_override=wrong_defaults)
    dof_cmp = compare(row["obs"][9:21], oracle["obs"][9:21], **TOLERANCES["observation_float32"])
    negative.append({
        "id": "N4-default-angle",
        "bug": "RL_calf 默认角 0.05 rad 偏差",
        "detected": not dof_cmp["passed"],
        "evidence": {"max_abs": dof_cmp["max_abs_diff"], "n_violations": dof_cmp["n_violations"]},
    })

    # N5 增益错：把 kp 换成 v2 的 25
    row = target_pipeline(base_fixture, spec, "cpu", policy=None, stiffness=25.0)
    tau_cmp = compare(row["tau_pd"], oracle["tau_pd"], **TOLERANCES["torque"])
    negative.append({
        "id": "N5-stiffness",
        "bug": "kp 用 v2 的 25 而非源的 20",
        "detected": not tau_cmp["passed"],
        "evidence": {"max_abs": tau_cmp["max_abs_diff"], "n_violations": tau_cmp["n_violations"]},
    })

    negative_ok = all(item["detected"] for item in negative)
    baseline_ok = baseline["passed"]
    verdict = {
        "fixtures_sha_match_manifest": actual_sha == manifest["sha256"],
        "fixtures_sha256": actual_sha,
        "baseline_fixture_passes": baseline_ok,
        "negative_tests_all_detected": negative_ok,
        "all_devices_passed": all(
            report_devices[key]["summary"]["n_passed"] == report_devices[key]["summary"]["n_fixtures"]
            for key in report_devices
        ),
        "devices": devices,
        "policy_used": policy is not None,
    }
    verdict["all_passed"] = (
        verdict["fixtures_sha_match_manifest"]
        and baseline_ok
        and negative_ok
        and verdict["all_devices_passed"]
    )

    report = {
        "report": "source_adapter_parity",
        "task": "HIRA-015",
        "fixtures": {
            "path": str(fixtures_path),
            "sha256": actual_sha,
            "manifest": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "n_fixtures": len(spec["fixtures"]),
            "ids": [item["id"] for item in spec["fixtures"]],
        },
        "tolerances": TOLERANCES,
        "devices": report_devices,
        "negative_tests": negative,
        "failures": all_failures,
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for device in report_devices:
        summary = report_devices[device]["summary"]
        print(f"[{device}] fixtures {summary['n_passed']}/{summary['n_fixtures']} "
              f"obs_abs={summary['max_obs_abs']:.3e} q_target_abs={summary['max_q_target_abs']:.3e} "
              f"tau_abs={summary['max_tau_abs']:.3e}")
    for item in negative:
        print(f"[neg] {item['id']:18s} detected={item['detected']} max_abs={item['evidence'].get('max_abs'):.3e}")
    print(f"[verdict] {verdict}")
    print("PARITY_OK" if verdict["all_passed"] else "PARITY_FAIL")
    return 0 if verdict["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
