"""HiRA 在 Isaac Lab 里的最小闭环 runner（HIRA-016 起）。

只做三件事，但每一件都按源的时钟语义来：

1. **三时钟**：物理 200 Hz（`sim.dt = 0.005`）、策略 50 Hz（每 4 个物理步一次）、
   外部命令 10 Hz（每 5 个策略步一次）。计数器是真实的步进计数，不是"配置里写了多少"。
2. **外部命令**：写进去什么，策略就看到什么——不做 0.2 m/s 死区、不做 heading 反算、
   不内部重采样。命令块只乘 `[2, 2, 0.25]`（由观测适配器负责）。
3. **逐 env 终止**：某个 env 摔倒时，先抓终止状态再单独重置它，
   其他 env 的上一动作与命令不受影响。

刻意不做的：地形、障碍、profile、记录器。这些属于后续任务，混进来会让时钟验证失真。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .action_pd import ACTION_SCALE, DECIMATION, DAMPING, STIFFNESS, ActionTorqueAdapter
from .asset_cfg import HIRA_DEFAULT_ANGLES, HIRA_INIT_BASE_POS, build_hira_go2_cfg
from .observation import JointOrder, ObservationAdapter, body_angular_velocity_wxyz, projected_gravity_wxyz

#: 源的关节顺序（逐腿）。
SOURCE_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

#: 终止判据（与预登记一致：root 高度 < 0.15 m 或倾角 > 60°）。
FALL_HEIGHT = 0.15
FALL_TILT_DEG = 60.0


@dataclass
class TimingConfig:
    sim_dt: float = 0.005
    decimation: int = DECIMATION
    command_policy_steps: int = 5      # 10 Hz ÷ 50 Hz
    window_s: float = 30.0
    warmup_physics_steps: int = 200

    @property
    def policy_dt(self) -> float:
        return self.sim_dt * self.decimation

    @property
    def command_dt(self) -> float:
        return self.policy_dt * self.command_policy_steps

    @property
    def window_physics_steps(self) -> int:
        return int(round(self.window_s / self.sim_dt))

    @property
    def window_policy_steps(self) -> int:
        return int(round(self.window_s / self.policy_dt))

    @property
    def window_command_intervals(self) -> int:
        return int(round(self.window_s / self.command_dt))

    @property
    def paired_points(self) -> int:
        # 每个命令区间两端各一个采样点，相邻区间共享端点，故为区间数 + 1
        return self.window_command_intervals + 1


@dataclass
class TerminationRecord:
    env_id: int
    physics_step: int
    policy_step: int
    command_index: int
    sim_time: float
    root_height: float
    tilt_deg: float
    joint_pos: list[float]
    joint_vel: list[float]
    last_action: list[float]
    command: list[float]


@dataclass
class RunnerTrace:
    policy_steps: list[dict] = field(default_factory=list)
    terminations: list[TerminationRecord] = field(default_factory=list)


class HiraLabRunner:
    """单场景、单机器人、多 env 的最小闭环。"""

    def __init__(
        self,
        urdf_path: str | Path,
        usd_dir: str | Path | None,
        num_envs: int = 1,
        device: str = "cuda:0",
        policy_path: str | Path | None = None,
        timing: TimingConfig | None = None,
        env_spacing: float = 2.0,
        dtype: torch.dtype = torch.float32,
        order_mode: str = "urdf",
        enhanced_determinism: bool = False,
    ):
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils import configclass

        self.timing = timing or TimingConfig()
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        # PhysX 的 "enhanced determinism" 让求解结果与线程/批内顺序无关。
        # 默认关闭（有性能代价），但批量隔离（T-08）这类"逐位比较"的验证必须打开，
        # 否则 env 之间会因 ULP 级差异在几百步内发散（HIRA-020 实测）。
        physx_cfg = (
            sim_utils.PhysxCfg(enable_enhanced_determinism=True) if enhanced_determinism else None
        )
        sim_cfg = SimulationCfg(
            dt=self.timing.sim_dt, device=device, gravity=(0.0, 0.0, -9.81)
        )
        if physx_cfg is not None:
            sim_cfg.physx = physx_cfg
        self.enhanced_determinism = bool(enhanced_determinism)
        self.sim = SimulationContext(sim_cfg)
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        light = sim_utils.DomeLightCfg(intensity=300.0)
        light.func("/World/Light", light)

        robot_cfg = build_hira_go2_cfg(urdf_path, usd_dir=usd_dir).replace(prim_path="{ENV_REGEX_NS}/Robot")

        @configclass
        class _SceneCfg(InteractiveSceneCfg):
            # 场景里只有机器人：地形、障碍、传感器属于后续任务，混进来会让时钟验证失真
            robot: ArticulationCfg = robot_cfg

        self.scene = InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=env_spacing))
        self.sim.reset()
        self.robot: Articulation = self.scene["robot"]
        self.env_origins = self.scene.env_origins.clone()

        self.joint_names_lab = list(self.robot.joint_names)
        # order_mode="urdf"：假定源（isaacgym）DOF 顺序 = URDF 声明顺序（契约当前写法）
        # order_mode="lab" ：假定源顺序 = 引擎 BFS 顺序（即与 Lab 的 DOF 顺序相同）
        # 这是一个**必须用实验判定**的问题：两种假定都自洽，但只有一种能复现源行为。
        self.order_mode = order_mode
        source_order = list(self.joint_names_lab) if order_mode == "lab" else list(SOURCE_ORDER)
        self.source_order = source_order
        self.order = JointOrder(source_order, self.joint_names_lab)
        default_source = torch.tensor(
            [HIRA_DEFAULT_ANGLES[name] for name in source_order], dtype=dtype, device=device
        )
        effort_table = self._effort_limits(urdf_path)
        effort_source = torch.tensor(
            [effort_table[name] for name in source_order], dtype=dtype, device=device
        )

        self.default_source = default_source
        self.default_lab = self.order.source_to_target(default_source)

        self.obs_adapter = ObservationAdapter(
            self.order, default_source, num_envs=num_envs, device=device, dtype=dtype
        )
        self.torque_adapter = ActionTorqueAdapter(
            self.order, default_source, effort_source, device=device, dtype=dtype
        )

        self.policy = None
        if policy_path is not None:
            self.policy = torch.jit.load(str(policy_path)).to(device).eval()

        self.command = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.physics_step = 0
        self.policy_step_count = 0
        self.command_index = 0
        self.trace = RunnerTrace()
        # HIRA-019：可选的 200 Hz 计量记录器（只读；None 时路径完全不变）
        self.recorder = None
        self._last_action_source = torch.zeros((num_envs, 12), device=device, dtype=dtype)
        self.reset_all()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _effort_limits(urdf_path: str | Path) -> dict[str, float]:
        import re

        text = Path(urdf_path).read_text(encoding="utf-8")
        limits = {}
        for match in re.finditer(r'<joint\s+name="([^"]+)"\s+type="revolute">(.*?)</joint>', text, re.S):
            name, body = match.group(1), match.group(2)
            effort = re.search(r'effort="([0-9.]+)"', body)
            if effort:
                limits[name] = float(effort.group(1))
        return limits

    def reset_all(self) -> None:
        self._reset_envs(torch.arange(self.num_envs, device=self.device))

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """把 root 与关节写回初值，并只清零这些 env 的上一动作。"""
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        root_pose = torch.zeros((n, 7), device=self.device, dtype=self.dtype)
        root_pose[:, 0:3] = torch.tensor(HIRA_INIT_BASE_POS, device=self.device, dtype=self.dtype)
        root_pose[:, 0:2] += self.env_origins[env_ids, 0:2]
        root_pose[:, 3] = 1.0  # wxyz 单位四元数
        root_vel = torch.zeros((n, 6), device=self.device, dtype=self.dtype)
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        joint_pos = self.default_lab.reshape(1, 12).repeat(n, 1)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        # 写完必须刷新 data 缓冲：否则下一步的观测/PD 用的还是**上一轮 trial 结束时的**状态，
        # 会让"同 seed 两次运行"出现微小但确定的差异（profiles 的 P5 就是这么发现的）。
        try:
            self.robot.update(self.timing.sim_dt)
        except Exception:  # noqa: BLE001 - 初始化阶段还没有可更新的物理状态
            pass
        self.obs_adapter.reset(env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids)
        self.torque_adapter.reset()

    # ------------------------------------------------------------------ #
    def set_command(self, command: torch.Tensor) -> None:
        """外部命令直接写入，不做任何变换（源里也只做 obs 缩放）。"""
        command = command.to(self.device, self.dtype)
        if command.dim() == 1:
            command = command.reshape(1, 3).repeat(self.num_envs, 1)
        if command.shape != (self.num_envs, 3):
            raise ValueError(f"命令形状应为 ({self.num_envs}, 3)，实际 {tuple(command.shape)}")
        self.command = command.clone()

    def observe(self) -> torch.Tensor:
        quat = self.robot.data.root_quat_w
        world_ang_vel = self.robot.data.root_ang_vel_w
        ang_vel_body = body_angular_velocity_wxyz(quat, world_ang_vel)
        gravity_body = projected_gravity_wxyz(quat)
        built = self.obs_adapter.build(
            ang_vel_body,
            gravity_body,
            self.command,
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
        )
        return built["obs"]

    def _root_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        height = self.robot.data.root_pos_w[:, 2]
        gravity_body = projected_gravity_wxyz(self.robot.data.root_quat_w)
        tilt = torch.acos(torch.clamp(-gravity_body[:, 2], -1.0, 1.0)) * 180.0 / math.pi
        return height, tilt

    def terminated_mask(self) -> torch.Tensor:
        height, tilt = self._root_state()
        return (height < FALL_HEIGHT) | (tilt > FALL_TILT_DEG)

    # ------------------------------------------------------------------ #
    def policy_step(self, record: dict | None = None) -> None:
        """一次策略步：观测 → 推理 → 裁剪 → 4 个 PD 子步。"""
        obs = self.observe()
        if self.policy is None:
            raw_action = torch.zeros((self.num_envs, 12), device=self.device, dtype=self.dtype)
        else:
            with torch.no_grad():
                raw_action = self.policy(obs.to(torch.float32)).to(self.dtype)
        clipped = self.obs_adapter.apply_action(raw_action)
        self.torque_adapter.set_policy_action(clipped)

        substeps = []
        saturated_total = 0
        for _ in range(self.timing.decimation):
            trace = self.substep()
            substeps.append(float(trace["tau_clipped_source"].abs().max()))
            saturated_total += int(trace["n_saturated_joints"])

        self._last_action_source = clipped
        if record is not None:
            height, tilt = self._root_state()
            record.update(
                {
                    "policy_step": self.policy_step_count,
                    "sim_time": self.physics_step * self.timing.sim_dt,
                    "physics_step": self.physics_step,
                    "command_index": self.command_index,
                    "command_row0": self.command[0].detach().cpu().tolist(),
                    "raw_action_row0": raw_action[0].detach().cpu().tolist(),
                    "clipped_action_row0": clipped[0].detach().cpu().tolist(),
                    "q_target_row0": trace["q_target_source"][0].detach().cpu().tolist(),
                    "tau_max_abs": max(substeps),
                    "n_saturated_joints": saturated_total,
                    "height_row0": float(height[0]),
                    "tilt_deg_row0": float(tilt[0]),
                    "obs_row0": obs[0].detach().cpu().tolist(),
                }
            )
        self.policy_step_count += 1

    def handle_terminations(self) -> torch.Tensor:
        """抓取终止状态（reset 之前）并单独重置这些 env。"""
        done = self.terminated_mask()
        env_ids = torch.nonzero(done, as_tuple=False).reshape(-1)
        if env_ids.numel() == 0:
            return env_ids
        height, tilt = self._root_state()
        for env_id in env_ids.tolist():
            self.trace.terminations.append(
                TerminationRecord(
                    env_id=int(env_id),
                    physics_step=self.physics_step,
                    policy_step=self.policy_step_count,
                    command_index=self.command_index,
                    sim_time=self.physics_step * self.timing.sim_dt,
                    root_height=float(height[env_id]),
                    tilt_deg=float(tilt[env_id]),
                    joint_pos=self.robot.data.joint_pos[env_id].detach().cpu().tolist(),
                    joint_vel=self.robot.data.joint_vel[env_id].detach().cpu().tolist(),
                    last_action=self._last_action_source[env_id].detach().cpu().tolist(),
                    command=self.command[env_id].detach().cpu().tolist(),
                )
            )
        self._reset_envs(env_ids)
        return env_ids

    # ------------------------------------------------------------------ #
    def warmup(self, steps: int | None = None) -> None:
        """warmup：保持默认姿态但只推进物理，不计入策略步与命令区间。

        用零动作走 PD 通路（q_target = 默认角）而不是直接写零力矩：
        零力矩会让机器人在 1 s 内塌下去，窗口一开始就已经躺在地上，
        时钟也许没问题，但终止记录会被灌满噪声。
        """
        steps = self.timing.warmup_physics_steps if steps is None else steps
        self.torque_adapter.set_policy_action(
            torch.zeros((self.num_envs, 12), device=self.device, dtype=self.dtype)
        )
        for _ in range(steps):
            trace = self.torque_adapter.compute_substep(
                self.robot.data.joint_pos, self.robot.data.joint_vel
            )
            self.write_effort_lab(trace["effort_written_lab"])
            self.advance()
            self._record_substep(trace)

    # ------------------------------------------------------------------ #
    # 单子步动作与记录钩子（HIRA-019）
    def write_effort_lab(self, tau_lab: torch.Tensor) -> None:
        """把 Lab 顺序的力矩写进仿真缓冲（触发执行器模型 → PhysX）。"""
        self.robot.set_joint_effort_target(tau_lab.to(self.device, self.dtype))
        self.robot.write_data_to_sim()

    def advance(self) -> None:
        """推进一个物理子步并刷新 data 缓冲。

        ``sim.step()`` 不会刷新 articulation 的 data 缓冲；不 update 的话
        下一次 PD 用的是**上一步之前的** q/qdot，观测也会冻在初始状态（HIRA-016 实测）。
        """
        self.sim.step(render=False)
        self.robot.update(self.timing.sim_dt)
        self.physics_step += 1

    def substep(self) -> dict:
        """按当前 q_target 推进一个物理子步（不做策略推理、不推进策略计数）。

        采集/验证需要"窗口停在策略步中间"时必须用这个入口：``policy_step`` 一次走 4 步，
        无法表达非整策略步结束。
        """
        trace = self.torque_adapter.compute_substep(
            self.robot.data.joint_pos, self.robot.data.joint_vel
        )
        self.write_effort_lab(trace["effort_written_lab"])
        self.advance()
        self._record_substep(trace)
        return trace

    def step_known_torque(self, tau_source: torch.Tensor, saturated: int = 0) -> dict:
        """用一个**外部给定**的源顺序力矩推进一步（已知功率/符号测试用）。

        走的是同一条 ``write_effort_lab → advance → _record_substep`` 通路，
        所以测的是真实执行链，而不是另写一份绕过 PhysX 的模拟。
        """
        tau_source = tau_source.to(self.device, self.dtype).reshape(self.num_envs, 12)
        trace = {
            "tau_clipped_source": tau_source,
            "joint_vel_source": self.order.target_to_source(self.robot.data.joint_vel),
            "n_saturated_joints": int(saturated),
        }
        self.write_effort_lab(self.order.source_to_target(tau_source))
        self.advance()
        self._record_substep(trace)
        return trace

    def _record_substep(self, trace: dict) -> None:
        """把"实际施加的力矩 + 本子步结束时刻的速度"交给记录器（只读）。"""
        if self.recorder is None:
            return
        self.recorder.record(
            tau_applied_lab=self.robot.data.applied_torque,
            tau_written_source=trace["tau_clipped_source"],
            tau_request_source=trace.get("tau_pd_source"),
            qdot_pre_source=trace["joint_vel_source"],
            qdot_post_lab=self.robot.data.joint_vel,
            tau_computed_lab=self.robot.data.computed_torque,
            joint_acc_lab=self.robot.data.joint_acc,
            root_pos_lab=self.robot.data.root_pos_w,
            saturated_joints=int(trace["n_saturated_joints"]),
        )

    def run_window(self, commands: list[list[float]], window_s: float | None = None) -> RunnerTrace:
        """按 10 Hz 施加外部命令跑满一个窗口；命令列表循环使用。"""
        window_s = self.timing.window_s if window_s is None else window_s
        n_intervals = int(round(window_s / self.timing.command_dt))
        for index in range(n_intervals):
            self.command_index = index
            self.set_command(torch.tensor(commands[index % len(commands)], dtype=self.dtype))
            for _ in range(self.timing.command_policy_steps):
                record: dict = {}
                self.policy_step(record)
                self.trace.policy_steps.append(record)
                self.handle_terminations()
        self.command_index = n_intervals
        return self.trace
