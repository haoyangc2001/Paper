"""HiRA 专属的 Isaac Lab 机器人/执行器配置（HIRA-012）。

三条设计决定，都是为了"只在源里生效一次的机制，在 Lab 里也只生效一次"：

1. **纯力矩执行器**：``IdealPDActuatorCfg(stiffness=0, damping=0)`` 时
   ``computed_effort = 0*(q_des-q) + 0*(qd_des-qd) + τ_ff``，只剩下前馈力矩，
   随后按 ``effort_limit`` 裁剪一次。源 legged_gym 用的是
   ``default_dof_drive_mode = 3``（纯 effort）+ 每个物理子步自己算 PD 并裁一次，
   两者语义相同。**不用 ImplicitActuatorCfg**：那会把 PD 塞进 PhysX 求解器，
   与本轮"显式 PD"的要求冲突，也会出现"加一次、隐式再算一次"的双重 PD。
2. **资产从源 URDF 导入**，不复用 v2 的 ``Go2/usd/go2.usd``。v2 的 USD 来自
   unitree_ros 的 ``go2_description``，与本 URDF 不是同一份文件；形状像不等于
   质量惯性、碰撞盒、限位一样。
3. **不改源 URDF、不改 v2 配置**：URDF 只读，导入产物落在 HiRA 自己的缓存目录。

数值来源：
- 关节顺序/限位/effort/velocity：直接解析 ``go2.urdf``
- 默认角、PD 增益、action_scale、decimation：``GO2RoughCfg``（记录在常量里并带出处）
"""

from __future__ import annotations

from pathlib import Path

from .urdf_model import parse_urdf

#: 源 URDF 相对仓库根的位置（HiRA 仓库，只读）。
HIRA_URDF_RELATIVE = "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"

#: 源 ``GO2RoughCfg.init_state.default_joint_angles``，保持源里的书写顺序。
HIRA_DEFAULT_ANGLES: dict[str, float] = {
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
}

#: 源 ``GO2RoughCfg.control``。
HIRA_STIFFNESS = 20.0
HIRA_DAMPING = 0.5
HIRA_ACTION_SCALE = 0.25
HIRA_DECIMATION = 4
HIRA_SIM_DT = 0.005
HIRA_POLICY_DT = HIRA_SIM_DT * HIRA_DECIMATION

#: 源 ``GO2RoughCfg.init_state.pos``（base 初始高度）。
HIRA_INIT_BASE_POS = (0.0, 0.0, 0.42)

#: 源 ``LeggedRobotCfg.asset`` 的资产选项（逐条对应，不是"凭印象"）。
HIRA_COLLAPSE_FIXED_JOINTS = True          # collapse_fixed_joints = True
HIRA_SELF_COLLISIONS_ENABLED = False       # self_collisions = 1 → 碰撞组自过滤，等价于关闭自碰撞
HIRA_REPLACE_CYLINDERS_WITH_CAPSULES = True  # replace_cylinder_with_capsule = True
HIRA_CONTACT_OFFSET = 0.01                 # thickness = 0.01
HIRA_JOINT_FRICTION = 0.0                  # 源未设置关节摩擦
HIRA_ARMATURE = 0.0                        # 源 armature = 0

#: 执行器组名。审计报告里用它定位"PD 只生效一次"。
HIRA_ACTUATOR_GROUP = "hira_go2_effort"

#: 源 physx 求解器迭代（``sim.physx``），用于让引擎差异可解释。
HIRA_SOLVER_POSITION_ITERATIONS = 4
HIRA_SOLVER_VELOCITY_ITERATIONS = 0


def load_joint_table(urdf_path: str | Path) -> dict:
    """从 URDF 读取 12 个可驱动关节的限位表（名字 → 限位/effort/velocity/axis）。"""
    parsed = parse_urdf(urdf_path)
    table = {}
    for record in parsed["joints"]:
        if record["name"] in HIRA_DEFAULT_ANGLES and record["limit"] is not None:
            table[record["name"]] = {
                "lower": record["limit"]["lower"],
                "upper": record["limit"]["upper"],
                "effort": record["limit"]["effort"],
                "velocity_limit": record["limit"]["velocity"],
                "axis": record["axis"],
                "default_angle": HIRA_DEFAULT_ANGLES[record["name"]],
            }
    return table


def build_hira_go2_cfg(urdf_path: str | Path, usd_dir: str | Path | None = None):
    """构造 HiRA 专用的 ``ArticulationCfg``。

    这里才 import isaaclab，静态审计可以直接 import 本模块的常量而不需要 App。
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab.assets import ArticulationCfg

    urdf_path = Path(urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"源 URDF 不存在：{urdf_path}")

    table = load_joint_table(urdf_path)
    effort = {name: row["effort"] for name, row in table.items()}
    velocity = {name: row["velocity_limit"] for name, row in table.items()}

    spawn_kwargs = {
        "asset_path": str(urdf_path),
        "fix_base": False,
        # 与源 collapse_fixed_joints 一致；源里 dont_collapse="true" 的 fixed 关节
        # 在 Isaac Lab 这边没有等价开关，因此这条差异单独记录（只影响 link 划分，
        # 不产生 DOF，也不改变总质量）。
        "merge_fixed_joints": HIRA_COLLAPSE_FIXED_JOINTS,
        "replace_cylinders_with_capsules": HIRA_REPLACE_CYLINDERS_WITH_CAPSULES,
        # target_type="none" → USD 侧 stiffness/damping 写 0，PD 完全由适配器显式算。
        "joint_drive": sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        "articulation_props": sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=HIRA_SELF_COLLISIONS_ENABLED,
            solver_position_iteration_count=HIRA_SOLVER_POSITION_ITERATIONS,
            # 源 num_velocity_iterations = 0，PhysX 要求 ≥1，故记 1 并作为引擎差异登记
            solver_velocity_iteration_count=max(1, HIRA_SOLVER_VELOCITY_ITERATIONS),
        ),
        # 源 thickness = 0.01 是碰撞体的接触偏移；rest_offset 源为 0.0。
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=HIRA_CONTACT_OFFSET, rest_offset=0.0),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    }
    if usd_dir is not None:
        spawn_kwargs["usd_dir"] = str(usd_dir)

    spawn = sim_utils.UrdfFileCfg(**spawn_kwargs)

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HIRA_INIT_BASE_POS,
            joint_pos=dict(HIRA_DEFAULT_ANGLES),
            joint_vel={".*": 0.0},
        ),
        # 1.0 = 不额外收软限位；源的 PD 直接对着硬限位顶，没有软限位层。
        soft_joint_pos_limit_factor=1.0,
        actuators={
            HIRA_ACTUATOR_GROUP: IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=effort,
                velocity_limit=velocity,
                # 关键：显式执行器 + 零增益 = 只透传前馈力矩，且只裁一次
                effort_limit_sim=effort,
                stiffness=0.0,
                damping=0.0,
                friction=HIRA_JOINT_FRICTION,
                armature=HIRA_ARMATURE,
            ),
        },
    )


def explicit_pd_torque(joint_pos, joint_vel, q_target, stiffness=HIRA_STIFFNESS, damping=HIRA_DAMPING):
    """源 ``_compute_torques`` 的逐字复现（不含裁剪）。

    ``τ = kp * (q_target - q) - kd * qdot``；裁剪与写入由调用方按 URDF effort 做一次。
    保留成独立函数，是为了让 HIRA-013/014 的力矩对照有一个可单测的实现。
    """
    return stiffness * (q_target - joint_pos) - damping * joint_vel
