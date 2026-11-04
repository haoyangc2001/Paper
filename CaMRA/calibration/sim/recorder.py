"""200 Hz 物理记录 + 50 Hz 状态记录（CAL-014/015）。

只观察、不写仿真：不调用任何 set_*/write_*，打开或关闭记录不改变动力学
（CAL-014 的 on/off 双跑实测：8 个 50 Hz 字段逐位相同）。

性能设计（CAL-015 实测后改进）：所有数据预分配为 **GPU 缓冲区**，逐子步只做
一次元素级 GPU 运算 + 原地写入；能量用 float64 在设备端累加（避免 float32
求和误差）；只在 ``finish()`` 时一次性搬回 CPU。首版逐子步 ``.tolist()`` 的写法
实测 480 s/8 条 30 s trial，比仿真本身慢一个量级。

时钟分离：200 Hz（物理子步）与 50 Hz（控制步）分属两个计数器，warmup 只计数
不进缓冲区，因此 1 s warmup 的 200 步不可能混入窗口。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from camra_calibration.energy import DEFAULT_DT


@dataclass
class TrialRecorder:
    robot: object
    num_envs: int
    dt: float = DEFAULT_DT
    expected_steps: int = 6000
    n_joints: int = 12
    record_physics: bool = True
    expected_control_steps: int = 1500
    label: str = ""
    warmup_steps: int = 0
    _in_window: bool = False
    outcome: str = "running"

    def __post_init__(self) -> None:
        dev = self.robot.device
        self.device = dev
        T, C, N, J = self.expected_steps, self.expected_control_steps, self.num_envs, self.n_joints
        f32 = torch.float32
        # 200 Hz
        self.tau_buf = torch.zeros((T, N, J), dtype=f32, device=dev)
        self.qdot_buf = torch.zeros((T, N, J), dtype=f32, device=dev)
        self.root_pos_buf = torch.zeros((T, N, 3), dtype=f32, device=dev)
        self.root_quat_buf = torch.zeros((T, N, 4), dtype=f32, device=dev)
        self.root_lin_buf = torch.zeros((T, N, 3), dtype=f32, device=dev)
        self.root_ang_buf = torch.zeros((T, N, 3), dtype=f32, device=dev)
        self.energy_step = torch.zeros((T, N), dtype=torch.float64, device=dev)
        self.total_j = torch.zeros(N, dtype=torch.float64, device=dev)
        self.per_joint_j = torch.zeros((N, J), dtype=torch.float64, device=dev)
        # 50 Hz
        self.action_buf = torch.zeros((C, N, J), dtype=f32, device=dev)
        self.cmd_buf = torch.zeros((C, N, 3), dtype=f32, device=dev)
        self.pos50_buf = torch.zeros((C, N, 3), dtype=f32, device=dev)
        self.quat50_buf = torch.zeros((C, N, 4), dtype=f32, device=dev)
        self.lin50_buf = torch.zeros((C, N, 3), dtype=f32, device=dev)
        self.ang50_buf = torch.zeros((C, N, 3), dtype=f32, device=dev)
        self.jpos50_buf = torch.zeros((C, N, J), dtype=f32, device=dev)
        self.jvel50_buf = torch.zeros((C, N, J), dtype=f32, device=dev)
        self.k = 0      # 物理步计数
        self.c = 0      # 控制步计数
        self._problems: list[str] = []

    # ---- 窗口控制 ----
    def begin_warmup(self) -> None:
        self._in_window = False
        self.warmup_steps = 0

    def begin_window(self) -> None:
        if self.k:
            raise RuntimeError("窗口已有数据，不能重复 begin_window")
        self._in_window = True

    # ---- 每物理步 ----
    def after_physics_step(self) -> None:
        if not self._in_window:
            self.warmup_steps += 1
            return
        if self.k >= self.expected_steps:
            raise RuntimeError(f"物理步数超过缓冲 {self.expected_steps}（trial 时限/窗口配置错误）")
        data = self.robot.data
        tau = data.applied_torque
        self.robot.update(self.dt)   # joint_vel 时间戳门控：必须每子步推进
        qdot = data.joint_vel
        power = (tau * qdot).clamp_min(0).to(torch.float64)
        self.energy_step[self.k] = power.sum(-1) * self.dt
        self.total_j += power.sum(-1) * self.dt
        self.per_joint_j += power * self.dt
        if self.record_physics:
            self.tau_buf[self.k].copy_(tau)
            self.qdot_buf[self.k].copy_(qdot)
            self.root_pos_buf[self.k].copy_(data.root_pos_w)
            self.root_quat_buf[self.k].copy_(data.root_quat_w)
            self.root_lin_buf[self.k].copy_(data.root_lin_vel_w)
            self.root_ang_buf[self.k].copy_(data.root_ang_vel_w)
        self.k += 1

    # ---- 每控制步 ----
    def after_policy_step(self, action: torch.Tensor, cmd: torch.Tensor) -> None:
        if self.c >= self.expected_control_steps:
            raise RuntimeError(f"控制步数超过缓冲 {self.expected_control_steps}")
        data = self.robot.data
        self.action_buf[self.c].copy_(action)
        self.cmd_buf[self.c].copy_(cmd)
        self.pos50_buf[self.c].copy_(data.root_pos_w)
        self.quat50_buf[self.c].copy_(data.root_quat_w)
        self.lin50_buf[self.c].copy_(data.root_lin_vel_b)
        self.ang50_buf[self.c].copy_(data.root_ang_vel_b)
        self.jpos50_buf[self.c].copy_(data.joint_pos)
        self.jvel50_buf[self.c].copy_(data.joint_vel)
        self.c += 1

    # ---- 结果 ----
    @property
    def window_steps(self) -> int:
        return self.k

    def finish(self, outcome: str = "completed") -> dict:
        self.outcome = outcome
        self._problems = []
        tau, qdot = self.tau_buf[:self.k], self.qdot_buf[:self.k]
        if self.record_physics:
            if not torch.isfinite(tau).all():
                self._problems.append("tau 含非有限值")
            if not torch.isfinite(qdot).all():
                self._problems.append("qdot 含非有限值")
        if self.k != self.expected_steps:
            self._problems.append(f"物理步数 {self.k} ≠ {self.expected_steps}")
        if self.record_physics and self.k:
            recomputed = (tau.double() * qdot.double()).clamp_min(0).sum(-1).sum(0) * self.dt
            dev = (recomputed - self.total_j).abs() / self.total_j.clamp_min(1e-9)
            if float(dev.max()) > 1e-9:
                self._problems.append(f"设备端逐步累加与总量不一致（rel {float(dev.max()):.2e}）")
            latest = torch.isfinite(self.energy_step[:self.k]).all()
            if not bool(latest):
                self._problems.append("逐步能量含非有限值")
        if outcome == "completed" and self.c != self.expected_control_steps:
            self._problems.append(f"控制步数 {self.c} ≠ {self.expected_control_steps}")
        if self.c * 4 != self.k:
            self._problems.append(f"50 Hz {self.c} 条 × 4 ≠ 200 Hz {self.k} 步")
        return self.summary()

    def summary(self) -> dict:
        return {
            "label": self.label,
            "outcome": self.outcome,
            "num_envs": self.num_envs,
            "dt_s": self.dt,
            "warmup_steps": self.warmup_steps,
            "window_steps": self.k,
            "window_s": self.k * self.dt,
            "control_steps_50hz": self.c,
            "record_physics": self.record_physics,
            "energy_total_j": [float(x) for x in self.total_j.tolist()],
            "energy_per_joint_j": self.per_joint_j.tolist(),
            "problems": self._problems,
            "ok": not self._problems,
            "unit": "J",
        }

    # ---- 导出（一次性搬回 CPU） ----
    def stack(self) -> dict[str, torch.Tensor]:
        k, c = self.k, self.c
        out = {
            "tau": self.tau_buf[:k].cpu(), "qdot": self.qdot_buf[:k].cpu(),
            "root_pos_200hz": self.root_pos_buf[:k].cpu(),
            "root_quat_200hz": self.root_quat_buf[:k].cpu(),
            "root_lin_vel_200hz": self.root_lin_buf[:k].cpu(),
            "root_ang_vel_200hz": self.root_ang_buf[:k].cpu(),
            "action_50hz": self.action_buf[:c].cpu(), "cmd_50hz": self.cmd_buf[:c].cpu(),
            "root_pos_50hz": self.pos50_buf[:c].cpu(), "root_quat_50hz": self.quat50_buf[:c].cpu(),
            "root_lin_vel_b_50hz": self.lin50_buf[:c].cpu(),
            "root_ang_vel_b_50hz": self.ang50_buf[:c].cpu(),
            "joint_pos_50hz": self.jpos50_buf[:c].cpu(), "joint_vel_50hz": self.jvel50_buf[:c].cpu(),
            "energy_step_200hz": self.energy_step[:k].cpu(),
        }
        return out

    def save(self, out_dir: Path, stem: str) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz, meta = out_dir / f"{stem}.npz", out_dir / f"{stem}.json"
        torch.save(self.stack(), npz)
        meta.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        return npz, meta
