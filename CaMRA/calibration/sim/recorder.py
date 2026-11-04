"""200 Hz 物理记录 + 50 Hz 状态记录（CAL-014）。

只观察、不写仿真：本模块不调用任何 set_* / write_*，因此打开或关闭记录
不改变动力学（是否改变动作由 CAL-014 的 on/off 双跑对比实测）。

两个时钟严格分离：

* **200 Hz（物理子步）**：每次 ``sim.step()`` 之后读一次
  ``applied_torque``（该步实际施加力矩）、``robot.update(dt)`` 后的
  ``joint_vel``（该步末关节速度），以及根位姿/速度；能量按 D-ENERGY-001
  逐子步累计（每 env 一个 :class:`EnergyMeter`）。
* **50 Hz（控制步）**：每个 ``env.step()`` 之后记录动作、命令、根线/角速度、
  姿态、关节位置/速度——与 200 Hz 数据同属一条 trial，靠窗口时间对齐。

warmup 与窗口分离：``begin_warmup()`` 后的物理步只计数、不进任何数组；
``begin_window()`` 之后才开始累计，因此 warmup 的 200 步不会被算进能量。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from camra_calibration.energy import DEFAULT_DT, EnergyMeter


@dataclass
class TrialRecorder:
    robot: object
    num_envs: int
    dt: float = DEFAULT_DT
    expected_steps: int = 6000
    n_joints: int = 12
    record_physics: bool = True
    label: str = ""
    warmup_steps: int = 0
    meters: list[EnergyMeter] = field(default_factory=list)
    _in_window: bool = False

    def __post_init__(self) -> None:
        self.meters = [EnergyMeter(dt=self.dt, n_joints=self.n_joints,
                                   expected_steps=self.expected_steps,
                                   label=f"{self.label}/env{i}")
                       for i in range(self.num_envs)]
        self._tau: list[torch.Tensor] = []
        self._qdot: list[torch.Tensor] = []
        self._root_pos: list[torch.Tensor] = []
        self._root_quat: list[torch.Tensor] = []
        self._root_lin: list[torch.Tensor] = []
        self._root_ang: list[torch.Tensor] = []
        # 50 Hz
        self._action: list[torch.Tensor] = []
        self._cmd: list[torch.Tensor] = []
        self._pos50: list[torch.Tensor] = []
        self._quat50: list[torch.Tensor] = []
        self._lin50: list[torch.Tensor] = []
        self._ang50: list[torch.Tensor] = []
        self._jpos50: list[torch.Tensor] = []
        self._jvel50: list[torch.Tensor] = []
        self.outcome: str = "running"

    # ---- 窗口控制 ----
    def begin_warmup(self) -> None:
        self._in_window = False
        self.warmup_steps = 0

    def count_warmup_step(self) -> None:
        self.warmup_steps += 1

    def begin_window(self) -> None:
        if self._tau:
            raise RuntimeError("窗口已有数据，不能重复 begin_window（新 trial 请新建 recorder）")
        self._in_window = True
        for m in self.meters:
            m.start()

    # ---- 每物理步 ----
    def after_physics_step(self) -> None:
        if not self._in_window:
            self.count_warmup_step()
            return
        tau = self.robot.data.applied_torque.detach().to("cpu", torch.float32).clone()
        # joint_vel 是时间戳门控的惰性读取：必须推进时间戳才会刷新（§9.3）
        self.robot.update(self.dt)
        qdot = self.robot.data.joint_vel.detach().to("cpu", torch.float32).clone()
        data = self.robot.data
        pos = data.root_pos_w.detach().to("cpu", torch.float32).clone()
        quat = data.root_quat_w.detach().to("cpu", torch.float32).clone()
        lin = data.root_lin_vel_w.detach().to("cpu", torch.float32).clone()
        ang = data.root_ang_vel_w.detach().to("cpu", torch.float32).clone()
        if not torch.isfinite(tau).all() or not torch.isfinite(qdot).all():
            raise FloatingPointError("力矩或关节速度出现非有限值")
        for i, m in enumerate(self.meters):
            m.add_step(tau[i].tolist(), qdot[i].tolist())
        if self.record_physics:
            self._tau.append(tau)
            self._qdot.append(qdot)
            self._root_pos.append(pos)
            self._root_quat.append(quat)
            self._root_lin.append(lin)
            self._root_ang.append(ang)

    # ---- 每控制步 ----
    def after_policy_step(self, action: torch.Tensor, cmd: torch.Tensor) -> None:
        data = self.robot.data
        self._action.append(action.detach().to("cpu", torch.float32).clone())
        self._cmd.append(cmd.detach().to("cpu", torch.float32).clone())
        self._pos50.append(data.root_pos_w.detach().to("cpu", torch.float32).clone())
        self._quat50.append(data.root_quat_w.detach().to("cpu", torch.float32).clone())
        self._lin50.append(data.root_lin_vel_b.detach().to("cpu", torch.float32).clone())
        self._ang50.append(data.root_ang_vel_b.detach().to("cpu", torch.float32).clone())
        self._jpos50.append(data.joint_pos.detach().to("cpu", torch.float32).clone())
        self._jvel50.append(data.joint_vel.detach().to("cpu", torch.float32).clone())

    # ---- 收尾 ----
    def finish(self, outcome: str = "completed") -> dict:
        self.outcome = outcome
        for m in self.meters:
            m.stop()
        return self.summary()

    @property
    def window_steps(self) -> int:
        return len(self.meters[0].per_step_j) if self.meters else 0

    def stack(self) -> dict[str, torch.Tensor]:
        def cat(items):
            return torch.stack(items, dim=0) if items else torch.zeros(0)
        return {
            "tau": cat(self._tau), "qdot": cat(self._qdot),
            "root_pos_200hz": cat(self._root_pos), "root_quat_200hz": cat(self._root_quat),
            "root_lin_vel_200hz": cat(self._root_lin), "root_ang_vel_200hz": cat(self._root_ang),
            "action_50hz": cat(self._action), "cmd_50hz": cat(self._cmd),
            "root_pos_50hz": cat(self._pos50), "root_quat_50hz": cat(self._quat50),
            "root_lin_vel_b_50hz": cat(self._lin50), "root_ang_vel_b_50hz": cat(self._ang50),
            "joint_pos_50hz": cat(self._jpos50), "joint_vel_50hz": cat(self._jvel50),
        }

    def summary(self) -> dict:
        meters = [m.summary() for m in self.meters]
        problems = []
        for i, m in enumerate(self.meters):
            for p in m.verify_complete():
                problems.append(f"env{i}: {p}")
        if self._action and len(self._action) * 4 != self.window_steps:
            problems.append(
                f"50 Hz 记录 {len(self._action)} 条 × 4 ≠ 200 Hz {self.window_steps} 步")
        return {
            "label": self.label,
            "outcome": self.outcome,
            "num_envs": self.num_envs,
            "dt_s": self.dt,
            "warmup_steps": self.warmup_steps,
            "window_steps": self.window_steps,
            "window_s": self.window_steps * self.dt,
            "control_steps_50hz": len(self._action),
            "record_physics": self.record_physics,
            "energy_total_j": [m["total_j"] for m in meters],
            "energy_per_joint_j": [m["per_joint_j"] for m in meters],
            "energy_complete": [m["complete"] for m in meters],
            "problems": problems,
            "ok": not problems,
            "unit": "J",
        }

    def save(self, out_dir: Path, stem: str) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz = out_dir / f"{stem}.npz"
        meta = out_dir / f"{stem}.json"
        torch.save(self.stack(), npz)          # torch.save 保留 dtype/shape，numpy 直接可读
        meta.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        return npz, meta
