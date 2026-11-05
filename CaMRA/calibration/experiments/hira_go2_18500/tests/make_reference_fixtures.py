"""生成 HIRA-015 的固定样本（fixtures）。

    python tests/make_reference_fixtures.py

样本是**手写数值**，不依赖随机数：同一条命令重复执行，产物逐字节相同。
每条 fixture 自带默认角、effort 上限、源关节顺序与目标关节顺序，
因此独立 oracle 不需要读 URDF、也不需要 import 适配器任何模块——
"来源实现"和"目标实现"唯一的共同输入就是这份 JSON。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
URDF = REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"

SOURCE_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
LAB_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
#: 第三种顺序：打乱后再用于"关节置换"样本，确认实现不靠位置猜。
SHUFFLED_ORDER = [
    "RR_calf_joint", "FL_hip_joint", "RL_thigh_joint", "FR_calf_joint",
    "RR_hip_joint", "FL_calf_joint", "RL_hip_joint", "FR_thigh_joint",
    "RR_thigh_joint", "FL_thigh_joint", "RL_calf_joint", "FR_hip_joint",
]

DEFAULTS = {
    "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
}


def parse_effort(urdf_path: Path) -> dict[str, float]:
    text = urdf_path.read_text(encoding="utf-8")
    effort = {}
    for match in re.finditer(r'<joint\s+name="([^"]+)"\s+type="revolute">(.*?)</joint>', text, re.S):
        name, body = match.group(1), match.group(2)
        limit = re.search(r'effort="([0-9.]+)"', body)
        if name in DEFAULTS and limit:
            effort[name] = float(limit.group(1))
    missing = [name for name in SOURCE_ORDER if name not in effort]
    if missing:
        raise SystemExit(f"URDF 里缺 effort：{missing}")
    return effort


def scaled(pattern: dict[str, float], order: list[str]) -> dict[str, float]:
    """把"按腿类型给的值"展开成逐关节 dict。"""
    return {name: pattern[name.split("_", 1)[1]] for name in order}


def build_fixtures(effort: dict[str, float]) -> dict:
    def angles(hip: float, thigh_front: float, thigh_rear: float, calf: float) -> dict[str, float]:
        base = dict(DEFAULTS)
        for name in base:
            if name.endswith("_hip_joint"):
                base[name] = hip if name.startswith("FL") or name.startswith("RL") else -hip
            elif name.endswith("_thigh_joint"):
                base[name] = thigh_rear if name.startswith(("RL", "RR")) else thigh_front
            elif name.endswith("_calf_joint"):
                base[name] = calf
        return base

    def zeros(order=SOURCE_ORDER) -> dict[str, float]:
        return {name: 0.0 for name in order}

    fixtures = [
        {
            "id": "F01",
            "name": "静止",
            "notes": "单位四元数、零速度、默认姿态、零命令、零上一动作、零动作；最基本的等价点",
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "world_ang_vel": [0.0, 0.0, 0.0],
            "command": [0.0, 0.0, 0.0],
            "joint_pos": dict(DEFAULTS),
            "joint_vel": zeros(),
            "prev_action": zeros(),
            "raw_action": zeros(),
            "target_order": LAB_ORDER,
        },
        {
            "id": "F02",
            "name": "倾斜",
            "notes": "roll 12° + pitch -18°，重力投影落在 x/z 两轴；角速度非零但很小",
            "quat_wxyz": [0.97517, 0.10453, -0.15643, -0.01675],
            "world_ang_vel": [0.12, -0.25, 0.08],
            "command": [0.3, 0.0, 0.0],
            "joint_pos": angles(0.1, 0.85, 1.05, -1.45),
            "joint_vel": {name: 0.4 for name in SOURCE_ORDER},
            "prev_action": scaled({"hip_joint": 0.5, "thigh_joint": -0.5, "calf_joint": 0.25}, SOURCE_ORDER),
            "raw_action": scaled({"hip_joint": 0.2, "thigh_joint": 0.6, "calf_joint": -0.4}, SOURCE_ORDER),
            "target_order": LAB_ORDER,
        },
        {
            "id": "F03",
            "name": "转向",
            "notes": "yaw 60°，同时有世界系 z 角速度；命令带 wz；检验世界 z 角速度与机体系分量的区别",
            "quat_wxyz": [0.86603, 0.0, 0.0, 0.5],
            "world_ang_vel": [0.0, 0.0, 0.9],
            "command": [0.2, 0.1, 0.5],
            "joint_pos": angles(0.0, 0.8, 1.0, -1.5),
            "joint_vel": {name: (-1.0 if name.endswith("_hip_joint") else 1.0) for name in SOURCE_ORDER},
            "prev_action": scaled({"hip_joint": -1.5, "thigh_joint": 2.0, "calf_joint": -2.5}, SOURCE_ORDER),
            "raw_action": scaled({"hip_joint": 1.0, "thigh_joint": -1.0, "calf_joint": 3.0}, SOURCE_ORDER),
            "target_order": LAB_ORDER,
        },
        {
            "id": "F04",
            "name": "非零关节速度",
            "notes": "关节速度 ±5 rad/s，位置偏离默认角较大；观测 21:33 块非零且不裁",
            "quat_wxyz": [0.99500, 0.0, 0.09983, 0.0],
            "world_ang_vel": [-0.4, 0.3, -0.2],
            "command": [-0.4, 0.2, -0.3],
            "joint_pos": angles(0.2, 0.6, 0.9, -1.2),
            "joint_vel": {name: (5.0 if index % 2 == 0 else -5.0) for index, name in enumerate(SOURCE_ORDER)},
            "prev_action": zeros(),
            "raw_action": scaled({"hip_joint": -0.8, "thigh_joint": 1.2, "calf_joint": -1.6}, SOURCE_ORDER),
            "target_order": LAB_ORDER,
        },
        {
            "id": "F05",
            "name": "非零上一动作",
            "notes": "上一动作含 ±100 与 0，覆盖 33:45 块的边界与零值混合",
            "quat_wxyz": [0.92388, 0.0, 0.38268, 0.0],
            "world_ang_vel": [0.5, 0.5, 0.5],
            "command": [0.5, 0.0, 0.0],
            "joint_pos": dict(DEFAULTS),
            "joint_vel": zeros(),
            "prev_action": {
                name: (100.0 if index % 4 == 0 else (-100.0 if index % 4 == 2 else 0.0))
                for index, name in enumerate(SOURCE_ORDER)
            },
            "raw_action": scaled({"hip_joint": 0.0, "thigh_joint": 0.0, "calf_joint": 0.0}, SOURCE_ORDER),
            "target_order": LAB_ORDER,
        },
        {
            "id": "F06",
            "name": "裁剪边界",
            "notes": "命令 60/-60/300 → 缩放后 120/-120/75；位置偏离 200 rad；速度 3000 rad/s；动作 ±100 触发力矩饱和",
            "quat_wxyz": [0.70711, 0.70711, 0.0, 0.0],
            "world_ang_vel": [1000.0, -1000.0, 500.0],
            "command": [60.0, -60.0, 300.0],
            "joint_pos": {name: DEFAULTS[name] + (200.0 if index % 2 == 0 else -200.0) for index, name in enumerate(SOURCE_ORDER)},
            "joint_vel": {name: (3000.0 if index % 2 == 0 else -3000.0) for index, name in enumerate(SOURCE_ORDER)},
            "prev_action": {name: 100.0 for name in SOURCE_ORDER},
            "raw_action": {name: (100.0 if index % 3 == 0 else -100.0) for index, name in enumerate(SOURCE_ORDER)},
            "target_order": LAB_ORDER,
        },
        {
            "id": "F07",
            "name": "关节置换",
            "notes": "物理状态与 F03 相同，但按第三种（打乱）顺序表达；检验实现只按名字对齐",
            "quat_wxyz": [0.86603, 0.0, 0.0, 0.5],
            "world_ang_vel": [0.0, 0.0, 0.9],
            "command": [0.2, 0.1, 0.5],
            "joint_pos": angles(0.0, 0.8, 1.0, -1.5),
            "joint_vel": {name: (-1.0 if name.endswith("_hip_joint") else 1.0) for name in SOURCE_ORDER},
            "prev_action": scaled({"hip_joint": -1.5, "thigh_joint": 2.0, "calf_joint": -2.5}, SOURCE_ORDER),
            "raw_action": scaled({"hip_joint": 1.0, "thigh_joint": -1.0, "calf_joint": 3.0}, SOURCE_ORDER),
            "target_order": SHUFFLED_ORDER,
        },
        {
            "id": "F08",
            "name": "侧倾 90°",
            "notes": "重力完全落到机体 y 轴；最容易暴露四元数顺序或转置错误",
            "quat_wxyz": [0.70711, 0.70711, 0.0, 0.0],
            "world_ang_vel": [0.0, 0.0, 0.0],
            "command": [0.0, 0.0, 0.0],
            "joint_pos": dict(DEFAULTS),
            "joint_vel": zeros(),
            "prev_action": zeros(),
            "raw_action": zeros(),
            "target_order": LAB_ORDER,
        },
        {
            "id": "F09",
            "name": "低速巡航混合态",
            "notes": "接近真实运行的量级：0.3 m/s 前进 + 0.2 rad/s 转向，姿态小幅倾斜，关节速度 ±1",
            "quat_wxyz": [0.99144, 0.04362, -0.08716, 0.08716],
            "world_ang_vel": [0.05, -0.02, 0.2],
            "command": [0.3, 0.0, 0.2],
            "joint_pos": angles(0.08, 0.78, 0.98, -1.48),
            "joint_vel": {name: (1.0 if index < 6 else -1.0) for index, name in enumerate(SOURCE_ORDER)},
            "prev_action": scaled({"hip_joint": 0.1, "thigh_joint": -0.2, "calf_joint": 0.3}, SOURCE_ORDER),
            "raw_action": scaled({"hip_joint": -0.3, "thigh_joint": 0.5, "calf_joint": -0.7}, SOURCE_ORDER),
            "target_order": LAB_ORDER,
        },
    ]

    # 四元数统一归一化并保留 8 位小数：手写的 0.97517 之类不可能天然满足单位长度，
    # 而 oracle 会拒绝未归一化的输入——归一化放在生成端，避免两边各自容错。
    for fixture in fixtures:
        norm = math.sqrt(sum(value * value for value in fixture["quat_wxyz"]))
        fixture["quat_wxyz"] = [round(value / norm, 8) for value in fixture["quat_wxyz"]]
        fixture["quat_norm_after_rounding"] = round(
            math.sqrt(sum(value * value for value in fixture["quat_wxyz"])), 12
        )

    for fixture in fixtures:
        for key in ("joint_pos", "joint_vel", "prev_action", "raw_action"):
            missing = [name for name in SOURCE_ORDER if name not in fixture[key]]
            if missing:
                raise SystemExit(f"{fixture['id']} 的 {key} 缺 {missing}")
        if sorted(fixture["target_order"]) != sorted(SOURCE_ORDER):
            raise SystemExit(f"{fixture['id']} 的 target_order 不是同一组关节")
        if any(abs(value) > 100.0 for value in fixture["prev_action"].values()):
            raise SystemExit(f"{fixture['id']} 的上一动作超过 ±100（应已裁剪）")

    return {
        "version": "1.0.0",
        "kind": "hira_reference_fixtures",
        "task": "HIRA-015",
        "notes": (
            "手写数值样本，无随机数，重复生成逐字节一致。每条含完整物理状态、默认角、effort 上限、"
            "源顺序与目标顺序；独立 oracle 与目标实现只共享这一份 JSON。"
        ),
        "source_order": SOURCE_ORDER,
        "joint_defaults": DEFAULTS,
        "joint_effort": {name: effort[name] for name in SOURCE_ORDER},
        "joint_velocity_limit": {
            name: (30.1 if name.endswith(("_hip_joint", "_thigh_joint")) else 20.07) for name in SOURCE_ORDER
        },
        "fixtures": fixtures,
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(EXP / "tests/reference_fixtures"))
    args = parser.parse_args()

    payload = build_fixtures(parse_effort(URDF))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    data_path = out_dir / "fixtures_v1.json"
    data_path.write_text(body, encoding="utf-8")
    data_sha = sha256_bytes(body.encode("utf-8"))

    manifest = {
        "version": "1.0.0",
        "kind": "hira_reference_fixtures_manifest",
        "file": data_path.name,
        "sha256": data_sha,
        "n_fixtures": len(payload["fixtures"]),
        "fixture_ids": [fixture["id"] for fixture in payload["fixtures"]],
        "hash_rule": "sha256 over fixtures_v1.json bytes exactly as written (UTF-8, LF, indent=2)",
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_body = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_path.write_text(manifest_body, encoding="utf-8")

    print(f"[fixtures] {data_path} sha256={data_sha}")
    print(f"[fixtures] {manifest_path} sha256={sha256_bytes(manifest_body.encode('utf-8'))}")
    print(f"[fixtures] count={len(payload['fixtures'])} ids={manifest['fixture_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
