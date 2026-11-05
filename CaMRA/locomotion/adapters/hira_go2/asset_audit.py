"""资产与执行器的静态审计（HIRA-012）。

静态 = 只读 URDF 与契约文本，不启动仿真。目的是先把"参数是什么"钉死，
再让 ``check_asset_runtime.py`` 去证明"导入之后还是这个"。

报告分三块：
- ``source``：源 URDF / 源配置的原始事实（路径、散列、数值）
- ``contract_crosscheck``：与 HIRA-008 契约逐项比对（顺序/限位/effort/velocity/轴）
- ``import_plan``：打算怎么导入（这是**声明**，不是结果；结果在 runtime 段）
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import asset_cfg
from .urdf_model import check_structure, default_angles_in_limits, parse_urdf

#: 记录"这些数从哪来"。审计报告里原样输出，便于别人复核而不是相信我们。
SOURCE_FACTS = {
    "urdf": asset_cfg.HIRA_URDF_RELATIVE,
    "default_joint_angles": "HiRA/legged_gym_go2/legged_gym/envs/go2/go2_config.py :: GO2RoughCfg.init_state.default_joint_angles",
    "control": "HiRA/legged_gym_go2/legged_gym/envs/go2/go2_config.py :: GO2RoughCfg.control (stiffness 20, damping 0.5, action_scale 0.25, decimation 4)",
    "asset_options": "HiRA/legged_gym_go2/legged_gym/envs/base/legged_robot_config.py :: LeggedRobotCfg.asset "
    "(collapse_fixed_joints True, default_dof_drive_mode 3 = effort only, replace_cylinder_with_capsule True, thickness 0.01)",
    "self_collisions": "GO2RoughCfg.asset.self_collisions = 1 → Isaac Gym 碰撞组自过滤，等价于关闭自碰撞",
    "physx": "HiRA/legged_gym_go2/legged_gym/envs/base/legged_robot_config.py :: LeggedRobotCfg.sim.physx "
    "(dt 0.005, num_position_iterations 4, num_velocity_iterations 0)",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def static_audit(urdf_path: str | Path, contract_path: str | Path) -> dict:
    """返回报告里的 ``source`` + ``contract_crosscheck`` + ``import_plan`` 三段。"""
    urdf_path = Path(urdf_path)
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    urdf = parse_urdf(urdf_path)

    expected_order = list(contract["joints"]["order_source"])
    expected_limits = contract["joints"]["records"]

    structure_checks = check_structure(
        urdf,
        expected_order,
        expected_limits,
        replace_cylinders_with_capsules=asset_cfg.HIRA_REPLACE_CYLINDERS_WITH_CAPSULES,
    )
    defaults = default_angles_in_limits(urdf, asset_cfg.HIRA_DEFAULT_ANGLES)

    table = asset_cfg.load_joint_table(urdf_path)
    effort_values = sorted({row["effort"] for row in table.values()})
    velocity_values = sorted({row["velocity_limit"] for row in table.values()})

    # 源契约声明"effort 未从静态资产确认"，这里补齐：URDF 里的 <limit effort> 就是源引擎的裁剪值。
    return {
        "source": {
            "urdf": {
                "path": str(urdf_path),
                "sha256": urdf["sha256"],
                "size_bytes": urdf["size_bytes"],
                "robot_name": urdf["robot_name"],
                "n_links": len(urdf["links"]),
                "n_joints": len(urdf["joints"]),
                "n_dof": urdf["n_dof"],
                "dof_joint_names": urdf["dof_joint_names"],
                "fixed_joint_names": urdf["fixed_joint_names"],
                "dont_collapse_fixed": [
                    record["name"]
                    for record in urdf["joints"]
                    if record["type"] == "fixed" and record["dont_collapse"]
                ],
                "total_mass": urdf["total_mass"],
                "mass_by_link": urdf["mass_by_link"],
                "collision_kinds": sorted(
                    {kind for link in urdf["links"] for kind in link["geometry"]["collision"]}
                ),
            },
            "joint_limits": table,
            "distinct_effort_values": effort_values,
            "distinct_velocity_values": velocity_values,
            "source_facts": SOURCE_FACTS,
            "source_config_values": {
                "stiffness": asset_cfg.HIRA_STIFFNESS,
                "damping": asset_cfg.HIRA_DAMPING,
                "action_scale": asset_cfg.HIRA_ACTION_SCALE,
                "decimation": asset_cfg.HIRA_DECIMATION,
                "sim_dt": asset_cfg.HIRA_SIM_DT,
                "policy_dt": asset_cfg.HIRA_POLICY_DT,
                "init_base_pos": list(asset_cfg.HIRA_INIT_BASE_POS),
                "default_joint_angles": asset_cfg.HIRA_DEFAULT_ANGLES,
            },
        },
        "contract_crosscheck": {
            "contract_path": str(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "checks": structure_checks,
            "all_passed": all(item["ok"] for item in structure_checks),
            "default_angles": defaults,
            "effort_gap_closed": {
                "contract_said": contract["joints"]["evidence"].get("usd_effort"),
                "urdf_values": {name: row["effort"] for name, row in table.items()},
                "note": "契约把 effort 标为 unknown_from_static_asset；本条用 URDF <limit effort> 补齐，"
                "并由 runtime 段的 PhysX dof_max_forces 复核",
            },
        },
        "import_plan": {
            "spawn": "sim_utils.UrdfFileCfg(asset_path=<源 urdf>, merge_fixed_joints=True, "
            "replace_cylinders_with_capsules=True, joint_drive.target_type='none')",
            "actuator": f"IdealPDActuatorCfg(group={asset_cfg.HIRA_ACTUATOR_GROUP}, stiffness=0, damping=0, "
            "effort_limit=<URDF effort>, effort_limit_sim=<同值>)",
            "why_not_implicit": "ImplicitActuatorCfg 会把 PD 交给 PhysX 求解器；源是纯 effort 驱动 + 显式 PD，"
            "用隐式会出现双重 PD，且无法逐子步用最新 q/qdot 重算",
            "why_not_v2_usd": "v2 的 Go2/usd/go2.usd 由 unitree_ros go2_description 转换而来，与本 URDF 不是同一份文件；"
            "复用前必须逐参数证明一致，本轮选择不复用",
            "self_collisions": "关闭（与源 self_collisions=1 等价）",
            "soft_joint_pos_limit_factor": 1.0,
            "declared_engine_differences": [
                "源 num_velocity_iterations = 0，PhysX 要求 ≥1，导入时取 1",
                "源 collapse_fixed_joints 保留 dont_collapse 关节；Isaac Lab 的 URDF 导入无等价开关，"
                "fixed 关节会被合并（不产生 DOF，总质量不变，但 link 划分不同）",
                "源 isaacgym 与 PhysX 5.1 的接触/摩擦模型不同，接触力数值不可假定相等",
            ],
        },
    }
