"""HiRA 采集器（HIRA-021）。

唯一的入口是**已通过 G1 的冻结包**：构造时必须 `load_bundle()` 成功，并且

* controller_id 必须等于包目录名且 ≠ v2 的 ``camra_go2_velocity_it2999_v2``；
* 关节映射必须与运行时审计的参照一致（防"散列全对的自洽伪造包"）；
* 本轮的预登记、采集配置都会被散列后写进每条 trial 的 meta。

身份是显式的、可审计的，不靠目录名推断：

    experiment_id / controller_id / bundle_id / run_id /
    split / trial_id / program(program_sha256) / profile / seed /
    attempt / attempt_kind

attempt 规则（两条线，绝不能混）：

* **中断补跑（resume）**：进程被杀、掉电、输出不完整 → 新的 attempt 目录，
  旧 attempt 原样保留；
* **业务失败（business_failure）**：轨迹无效 → 如实记录 outcome，**不自动重试**；
  重试是人的决定（D-007 / R-06 的教训）。

所有写入先落 ``*.tmp`` 再 ``os.replace``；DONE 里带逐文件散列；
``plan_attempts()`` 是纯函数，恢复测试直接对伪造的目录状态跑。

文件布局与 v2 同构（meta.json / energy.json / state_50hz.npz / pairing_10hz.npz /
physics_200hz.npz / DONE / status.json），便于 HIRA-037 的跨控制器比较。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

V2_CONTROLLER_ID = "camra_go2_velocity_it2999_v2"
PROFILE_IDS = ("clean", "friction_only", "mass_only", "push_only", "combined")
PLANE_POSE_RANGE = {"px_m": (-1.0, 1.0), "py_m": (-1.0, 1.0), "yaw_rad": (-math.pi, math.pi)}
FRICTION_RANGE = (0.3, 1.25)
MASS_RANGE = (-1.0, 3.0)
PUSH_VEL_RANGE = (-0.5, 0.5)
PUSH_INTERVAL_RANGE = (5.0, 10.0)


class CollectorError(RuntimeError):
    """包不合法、身份缺失或运行状态冲突。"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _yaw_from_quat_wxyz(quat) -> float:
    w, x, y, z = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --------------------------------------------------------------------------- #
@dataclass
class TrialSpec:
    """一条 trial 的完整定义。命令序列由程序生成后**显式**存进来，不靠重放推断。"""

    trial_id: str
    split: str
    program: str
    profile: str
    seed: int
    commands: list[list[float]]
    window_s: float = 30.0
    warmup_s: float = 1.0
    initial_pose: dict | None = None          # {"px_m":..,"py_m":..,"yaw_rad":..}
    push_events: list[dict] | None = None     # [{"t_s":..,"dv_xy":[..]}]
    friction: float = 1.0
    added_mass_kg: float = 0.0
    obs_noise: bool = False

    def __post_init__(self) -> None:
        if self.profile not in PROFILE_IDS:
            raise CollectorError(f"未知 profile：{self.profile}")
        if not self.commands:
            raise CollectorError(f"{self.trial_id}: 命令序列为空")
        # HiRA 语料统一以 "hira-" 开头；v2 的命名（split-family-profile-index）没有前缀，
        # 一律拒绝——这是比黑名单更强的保证（v2 的 trial_id 千变万化，前缀法不会漏）
        if not self.trial_id.startswith("hira-"):
            raise CollectorError(
                f"{self.trial_id!r}: HiRA trial_id 必须以 'hira-' 开头（v2 语料的命名没有该前缀，直接拒）"
            )

    @property
    def program_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.commands, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def sample_spec(trial_id: str, split: str, program: str, profile: str, seed: int,
                commands: list[list[float]], window_s: float = 30.0) -> TrialSpec:
    """用 trial 自己的种子确定性地采样初态与推扰表（范围与 v2 协议一致）。"""
    rng = random.Random(seed)
    pose = None
    pushes = None
    friction, mass, noise = 1.0, 0.0, False
    if profile in ("friction_only", "combined"):
        friction = round(rng.uniform(*FRICTION_RANGE), 4)
    if profile in ("mass_only", "combined"):
        mass = round(rng.uniform(*MASS_RANGE), 4)
    if profile in ("push_only", "combined"):
        pushes = []
        t = rng.uniform(*PUSH_INTERVAL_RANGE)
        while t < window_s:
            pushes.append(
                {
                    "t_s": round(t, 3),
                    "dv_xy": [round(rng.uniform(*PUSH_VEL_RANGE), 4),
                              round(rng.uniform(*PUSH_VEL_RANGE), 4)],
                }
            )
            t += rng.uniform(*PUSH_INTERVAL_RANGE)
    if profile in ("mass_only", "push_only", "combined"):
        pose = {
            "px_m": round(rng.uniform(*PLANE_POSE_RANGE["px_m"]), 4),
            "py_m": round(rng.uniform(*PLANE_POSE_RANGE["py_m"]), 4),
            "yaw_rad": round(rng.uniform(*PLANE_POSE_RANGE["yaw_rad"]), 4),
        }
    if profile == "combined":
        noise = True
    return TrialSpec(
        trial_id=trial_id, split=split, program=program, profile=profile, seed=seed,
        commands=commands, window_s=window_s, warmup_s=1.0,
        initial_pose=pose, push_events=pushes,
        friction=friction, added_mass_kg=mass, obs_noise=noise,
    )


# --------------------------------------------------------------------------- #
# 恢复计划（纯函数）
def _attempt_dirs(trial_dir: Path) -> list[Path]:
    if not trial_dir.is_dir():
        return []
    return sorted(p for p in trial_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_"))


def attempt_number(path: Path) -> int:
    return int(path.name.split("_")[1])


def done_is_valid(attempt_dir: Path) -> bool:
    """DONE 必须存在，且它登记的每个文件都在、散列都对。"""
    done = attempt_dir / "DONE"
    if not done.is_file():
        return False
    try:
        record = json.loads(done.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    for name, expected in record.get("files", {}).items():
        path = attempt_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def plan_attempts(run_dir: Path, specs: list[TrialSpec]) -> list[dict]:
    """对每条 trial 给出动作；这是恢复语义的**唯一**裁决处。

    action ∈ skip / run_initial / run_resume / blocked_business_failure。
    """
    plan = []
    for spec in specs:
        trial_dir = Path(run_dir) / spec.trial_id
        attempts = _attempt_dirs(trial_dir)
        if attempts:
            latest = attempts[-1]
            if done_is_valid(latest):
                plan.append({"trial_id": spec.trial_id, "action": "skip",
                             "attempt": attempt_number(latest), "reason": "DONE 完整且散列一致"})
                continue
            outcome = None
            status_path = latest / "status.json"
            if status_path.is_file():
                try:
                    outcome = json.loads(status_path.read_text(encoding="utf-8")).get("outcome")
                except json.JSONDecodeError:
                    outcome = None
            if outcome == "business_failure":
                plan.append({"trial_id": spec.trial_id, "action": "blocked_business_failure",
                             "attempt": attempt_number(latest),
                             "reason": "业务失败只记录、不自动重试；重试须显式指令"})
                continue
            plan.append({"trial_id": spec.trial_id, "action": "run_resume",
                         "attempt": attempt_number(latest) + 1,
                         "reason": f"上次 attempt 未完成（outcome={outcome!r}），旧目录保留、写入新目录"})
        else:
            plan.append({"trial_id": spec.trial_id, "action": "run_initial",
                         "attempt": 1, "reason": "尚无任何 attempt"})
    return plan


# --------------------------------------------------------------------------- #
class HiraCollector:
    """一批（num_envs 条）trial 的执行器。每条 trial 独占一个 env 槽位。"""

    def __init__(
        self,
        bundle_dir: str | Path,
        run_dir: str | Path,
        prereg_path: str | Path,
        collect_config_path: str | Path,
        num_envs: int = 1,
        device: str = "cuda:0",
        expected_joints: dict | None = None,
    ):
        from camra_calibration.energy import DEFAULT_DT  # noqa: F401  契约常量的来源
        from hira_go2.bundle import load_bundle
        from hira_go2.env_runner import HiraLabRunner, TimingConfig

        self.bundle = load_bundle(
            Path(bundle_dir), expected_controller_id=Path(bundle_dir).name,
            expected_joints=expected_joints,
        )
        controller = self.bundle["controller"]
        if controller["controller_id"] == V2_CONTROLLER_ID:
            raise CollectorError("拒绝 v2 控制器包（新旧语料不得混用）")
        self.controller_id = controller["controller_id"]
        self.bundle_id = self.bundle["manifest"]["bundle_id"]

        self.prereg_path = Path(prereg_path)
        self.prereg = json.loads(self.prereg_path.read_text(encoding="utf-8"))
        self.experiment_id = self.prereg["experiment_id"]
        self.collect_config_path = Path(collect_config_path)
        self.collect_config = json.loads(self.collect_config_path.read_text(encoding="utf-8"))

        self.run_dir = Path(run_dir)
        self.num_envs = int(num_envs)
        self.device = device
        timing = TimingConfig(
            sim_dt=float(controller["timing"]["sim_dt"]),
            window_s=float(self.collect_config.get("window_s", 30.0)),
        )
        self.runner = HiraLabRunner(
            controller["asset"]["urdf"],
            self.collect_config.get("usd_dir"),
            num_envs=self.num_envs,
            device=device,
            policy_path=str(Path(bundle_dir) / controller["policy"]["file"]),
            timing=timing,
        )
        # clean 摩擦按源值显式写入（导入默认 0.5 ≠ 源 1.0，HIRA-017 实测）
        self.base_friction = float(controller["asset"]["friction"])
        self._set_friction(list(range(self.num_envs)), self.base_friction)
        self.timing = timing

    # ------------------------------------------------------------------ #
    # 物理旋钮（与 HIRA-018 的验证同一套写入方式）
    def _set_friction(self, env_ids: list[int], friction: float) -> dict:
        view = self.runner.robot.root_physx_view
        material = view.get_material_properties().clone()
        before = float(material[env_ids[0], 0, 0])
        for env in env_ids:
            material[env, :, 0] = friction
            material[env, :, 1] = friction
            material[env, :, 2] = 0.0
        view.set_material_properties(material, torch.tensor(env_ids, dtype=torch.int64))
        readback = float(view.get_material_properties()[env_ids[0], 0, 0])
        return {"before": before, "set": friction, "readback": readback}

    def _apply_mass(self, env_ids: list[int], delta_kg: float) -> dict:
        view = self.runner.robot.root_physx_view
        masses = view.get_masses().clone()
        inertias = view.get_inertias().clone()
        base_index = 0  # base 是 body 0（HIRA-012 审计确认）
        before = float(masses[env_ids[0], base_index])
        for env in env_ids:
            ratio = (masses[env, base_index] + delta_kg) / masses[env, base_index]
            masses[env, base_index] += delta_kg
            inertias[env, base_index, :] *= ratio
        ids = torch.tensor(env_ids, dtype=torch.int64)
        view.set_masses(masses, ids)
        view.set_inertias(inertias, ids)
        return {"before": before, "set": before + delta_kg,
                "readback": float(view.get_masses()[env_ids[0], base_index])}

    def _write_initial_pose(self, env_ids: list[int], pose: dict | None) -> None:
        runner = self.runner
        n = len(env_ids)
        yaw = float((pose or {}).get("yaw_rad", 0.0))
        px = float((pose or {}).get("px_m", 0.0))
        py = float((pose or {}).get("py_m", 0.0))
        root_pose = torch.zeros((n, 7), device=runner.device, dtype=runner.dtype)
        root_pose[:, 0] = runner.env_origins[env_ids, 0] + px
        root_pose[:, 1] = runner.env_origins[env_ids, 1] + py
        root_pose[:, 2] = float(runner.robot.data.default_root_state[0, 2])
        root_pose[:, 3] = math.cos(yaw / 2.0)
        root_pose[:, 6] = math.sin(yaw / 2.0)
        runner.robot.write_root_pose_to_sim(root_pose, env_ids=torch.tensor(env_ids, device=runner.device))
        root_vel = torch.zeros((n, 6), device=runner.device, dtype=runner.dtype)
        runner.robot.write_root_velocity_to_sim(root_vel, env_ids=torch.tensor(env_ids, device=runner.device))
        runner.robot.update(runner.timing.sim_dt)

    def _apply_spec_env(self, env_ids: list[int], spec: TrialSpec) -> dict:
        """把一条 trial 的物理旋钮写进指定 env，返回读回证据。"""
        evidence: dict = {"friction": None, "mass": None}
        if spec.profile in ("friction_only", "combined"):
            evidence["friction"] = self._set_friction(list(env_ids), spec.friction)
        if spec.profile in ("mass_only", "combined"):
            evidence["mass"] = self._apply_mass(list(env_ids), spec.added_mass_kg)
        if spec.profile in ("clean", "push_only"):
            # clean/push_only 也要把可能残留的旋钮拉回基准（串批/重试时尤其重要）
            evidence["friction"] = self._set_friction(list(env_ids), self.base_friction)
        return evidence

    def _set_noise(self, spec: TrialSpec) -> None:
        from hira_go2.observation import build_noise_vector

        if spec.obs_noise:
            scale = build_noise_vector().to(self.runner.device)
            self.runner.obs_adapter.noise_scale = scale
            self.runner.obs_noise_scale = scale
        else:
            self.runner.obs_adapter.noise_scale = None
            self.runner.obs_noise_scale = None

    # ------------------------------------------------------------------ #
    def execute_batch(self, specs: list[TrialSpec], attempts: list[int],
                      attempt_kinds: list[str] | None = None) -> list[dict]:
        """同批执行 len(specs) 条 trial（第 k 条占 env k），逐条落盘。

        中断安全：每个 attempt 先写 status.json(running)，跑完再写输出与 DONE；
        任何时刻被杀，目录里只剩 running 状态，plan_attempts 会安排 resume。
        """
        if len(specs) != self.num_envs:
            raise CollectorError(f"一批需要 {self.num_envs} 条 spec，得到 {len(specs)}")
        if len({s.window_s for s in specs}) > 1 or len({s.warmup_s for s in specs}) > 1:
            raise CollectorError("同一批的 window_s / warmup_s 必须一致")
        if len({s.obs_noise for s in specs}) > 1:
            raise CollectorError("同一批的 obs_noise 必须一致（噪声是 obs 级开关，无法逐 env）")
        attempt_kinds = attempt_kinds or ["initial" if a == 1 else "resume" for a in attempts]

        attempt_dirs = []
        for spec, attempt, kind in zip(specs, attempts, attempt_kinds):
            attempt_dir = self.run_dir / spec.trial_id / f"attempt_{attempt:02d}"
            if attempt_dir.exists():
                raise CollectorError(f"attempt 目录已存在，绝不覆盖：{attempt_dir}")
            attempt_dir.mkdir(parents=True)
            atomic_write_json(
                attempt_dir / "status.json",
                {"outcome": "running", "trial_id": spec.trial_id, "attempt": attempt,
                 "attempt_kind": kind, "bundle_id": self.bundle_id, "started_at": _now()},
            )
            attempt_dirs.append((spec, attempt, attempt_dir))

        profile_readback = {
            spec.trial_id: self._apply_spec_env([index], spec)
            for index, (spec, _a, _d) in enumerate(attempt_dirs)
        }
        self._set_noise(specs[0])
        self._write_initial_pose(list(range(self.num_envs)), specs[0].initial_pose)

        warmup_steps = int(round(specs[0].warmup_s / self.timing.sim_dt))
        self.runner.warmup(warmup_steps)

        capacity = int(round(specs[0].window_s / self.timing.sim_dt))
        from hira_go2.energy_recorder import HiRAEnergyRecorder

        recorder = HiRAEnergyRecorder(
            self.runner.order, capacity_steps=capacity, dt=self.timing.sim_dt,
            num_envs=self.num_envs, n_joints=12, device=self.runner.device,
            label="collector", expected_steps=capacity,
        )
        self.runner.recorder = recorder
        recorder.begin_window()
        window_start_step = self.runner.physics_step
        # 测试钩子：每个策略步后回调（终止注入等）；生产为 None
        hook = getattr(self, "policy_hook", None)

        state = self._new_state(specs)
        terminations: list[dict] = []
        push_trace: list[dict] = []
        n_intervals = int(round(specs[0].window_s / self.timing.command_dt))
        # 配对点 = 区间数 + 1（相邻区间共享端点）：每个区间起点记一次，窗口末再记一次
        for _interval in range(n_intervals):
            for k, spec in enumerate(specs):
                self.runner.set_command_row(k, spec.commands[_interval % len(spec.commands)])
            self._record_pairing(state, specs)
            for _ in range(self.timing.command_policy_steps):
                self._maybe_push(specs, state, window_start_step, push_trace)
                self.runner.policy_step()
                self._record_state50(state, specs)
                if hook is not None:
                    hook(self)
                done_ids = self.runner.handle_terminations()
                for env in done_ids.tolist():
                    spec = specs[env]
                    # 终止已由 runner 在 reset 前抓取；这里恢复该 env 的
                    # profile 旋钮与初态——否则 reset 会把摩擦/质量拉回默认
                    self._apply_spec_env([env], spec)
                    self._write_initial_pose([env], spec.initial_pose)
                    terminations.append(
                        {"env": env, "trial_id": spec.trial_id,
                         "physics_step": self.runner.physics_step,
                         "sim_time_s": round(
                             (self.runner.physics_step - window_start_step) * self.timing.sim_dt, 4
                         )}
                    )

        self._record_pairing(state, specs)  # 窗口终点（最后一个区间共享的端点）
        self.runner.recorder = None
        energy_result = recorder.finish()
        self._write_outputs(attempt_dirs, specs, attempts, attempt_kinds, energy_result,
                            profile_readback, terminations, push_trace, state, warmup_steps,
                            window_start_step)
        return [
            {"trial_id": spec.trial_id, "attempt": attempt, "dir": str(attempt_dir)}
            for spec, attempt, attempt_dir in attempt_dirs
        ]

    # ------------------------------------------------------------------ #
    def _new_state(self, specs: list[TrialSpec]) -> dict:
        return {
            "pairing": {spec.trial_id: [] for spec in specs},
            "state50": {spec.trial_id: [] for spec in specs},
            "next_push": {spec.trial_id: 0 for spec in specs},
        }

    def _record_pairing(self, state: dict, specs: list[TrialSpec]) -> None:
        """10 Hz 配对点：每个命令区间起点的（时间、命令、平面位姿）。

        位置是**相对 env origin** 的（名义轨迹也在同一坐标系里，初值 = initial_pose），
        GPU 上 clone、落盘时一次性搬回——逐点 .tolist() 会把 1500 步变成纯 CPU 同步（v2 的教训）。
        """
        runner = self.runner
        t = runner.physics_step * runner.timing.sim_dt
        for k, spec in enumerate(specs):
            pos = runner.robot.data.root_pos_w[k] - runner.env_origins[k]
            state["pairing"][spec.trial_id].append(
                {"t_s": t,
                 "command": runner.command[k].detach().clone(),
                 "pos_xy": pos[0:2].detach().clone(),
                 "yaw": _yaw_from_quat_wxyz(runner.robot.data.root_quat_w[k])}
            )

    def _record_state50(self, state: dict, specs: list[TrialSpec]) -> None:
        runner = self.runner
        t = runner.physics_step * runner.timing.sim_dt
        for k, spec in enumerate(specs):
            data = runner.robot.data
            state["state50"][spec.trial_id].append(
                {"t_s": t,
                 "command": runner.command[k].detach().clone(),
                 "action": runner._last_action_source[k].detach().clone(),
                 "root_pos": (data.root_pos_w[k] - runner.env_origins[k]).detach().clone(),
                 "root_quat": data.root_quat_w[k].detach().clone(),
                 "root_lin_b": data.root_lin_vel_b[k].detach().clone(),
                 "root_ang_b": data.root_ang_vel_b[k].detach().clone(),
                 "joint_pos": runner.order.target_to_source(data.joint_pos)[k].detach().clone(),
                 "joint_vel": runner.order.target_to_source(data.joint_vel)[k].detach().clone()}
            )

    def _maybe_push(self, specs: list[TrialSpec], state: dict,
                    window_start_step: int, push_trace: list[dict]) -> None:
        """按各 trial 的推扰表，在到达时刻给对应 env 写一次速度冲击。"""
        runner = self.runner
        t_window = (runner.physics_step - window_start_step) * runner.timing.sim_dt
        for k, spec in enumerate(specs):
            events = spec.push_events or []
            index = state["next_push"][spec.trial_id]
            while index < len(events) and events[index]["t_s"] <= t_window:
                before = runner.robot.data.root_lin_vel_w[k].clone()
                dv = torch.tensor(
                    [[events[index]["dv_xy"][0], events[index]["dv_xy"][1], 0.0]],
                    device=runner.device, dtype=runner.dtype,
                )
                runner.robot.write_root_velocity_to_sim(
                    torch.cat([before + dv, runner.robot.data.root_ang_vel_w[k:k + 1]], dim=-1),
                    env_ids=torch.tensor([k], device=runner.device),
                )
                runner.robot.update(runner.timing.sim_dt)
                after = runner.robot.data.root_lin_vel_w[k].clone()
                push_trace.append(
                    {"trial_id": spec.trial_id, "env": k,
                     "t_s": round(float(t_window), 4),
                     "dv_xy": events[index]["dv_xy"],
                     "measured_dv_xy": [float(after[0] - before[0]), float(after[1] - before[1])]}
                )
                index += 1
            state["next_push"][spec.trial_id] = index

    # ------------------------------------------------------------------ #
    def _write_outputs(self, attempt_dirs, specs, attempts, attempt_kinds, energy_result,
                       profile_readback, terminations, push_trace, state,
                       warmup_steps: int, window_start_step: int) -> None:
        import numpy as np

        def stack(rows, key, dtype=None):
            """把 GPU clone 的记录一次性搬回 CPU（避免逐点同步）。"""
            tensor = torch.stack([row[key] for row in rows]).detach().cpu()
            if dtype:
                tensor = tensor.to(dtype)
            return tensor.numpy()

        for index, (spec, attempt, attempt_dir) in enumerate(attempt_dirs):
            env = index
            state_rows = state["state50"][spec.trial_id]
            pairing = state["pairing"][spec.trial_id]
            np.savez_compressed(
                attempt_dir / "state_50hz.npz",
                t_s=np.array([row["t_s"] for row in state_rows], dtype=np.float64),
                command=stack(state_rows, "command", torch.float32),
                action_source=stack(state_rows, "action", torch.float32),
                root_pos_w=stack(state_rows, "root_pos", torch.float32),
                root_quat_w=stack(state_rows, "root_quat", torch.float32),
                root_lin_vel_b=stack(state_rows, "root_lin_b", torch.float32),
                root_ang_vel_b=stack(state_rows, "root_ang_b", torch.float32),
                joint_pos_source=stack(state_rows, "joint_pos", torch.float32),
                joint_vel_source=stack(state_rows, "joint_vel", torch.float32),
            )
            np.savez_compressed(
                attempt_dir / "pairing_10hz.npz",
                t_s=np.array([row["t_s"] for row in pairing], dtype=np.float64),
                command=stack(pairing, "command", torch.float32),
                pos_xy=stack(pairing, "pos_xy", torch.float32),
                yaw=np.array([row["yaw"] for row in pairing], dtype=np.float64),
            )
            np.savez_compressed(
                attempt_dir / "physics_200hz.npz",
                tau_applied_source=energy_result["tau_applied_source"][:, env, :],
                qdot_post_source=energy_result["qdot_post_source"][:, env, :],
                joint_acc_source=energy_result["joint_acc_source"][:, env, :],
                root_pos_w=energy_result["root_pos_w"][:, env, :],
            )
            total_j = float(energy_result["total_j"][env])
            atomic_write_json(
                attempt_dir / "energy.json",
                {
                    "unit": "J", "metric_id": "D-ENERGY-001",
                    "j_exec": total_j,
                    "per_joint_j": [float(v) for v in energy_result["per_joint_j"][env]],
                    "steps_executed": int(energy_result["steps"]),
                    "tau_source": "robot.data.applied_torque",
                    "qdot_phase": "post_substep",
                },
            )
            terminated_env = [t for t in terminations if t["env"] == env]
            meta = {
                "experiment_id": self.experiment_id,
                "controller_id": self.controller_id,
                "bundle_id": self.bundle_id,
                "policy_sha256": self.bundle["controller"]["policy"]["sha256"],
                "run_id": self.run_dir.name,
                "split": spec.split,
                "trial_id": spec.trial_id,
                "program": spec.program,
                "program_sha256": spec.program_sha256,
                "profile": spec.profile,
                "seed": spec.seed,
                "attempt": attempt,
                "attempt_kind": attempt_kinds[index],
                "prereg_sha256": sha256_file(self.prereg_path),
                "collect_config_sha256": sha256_file(self.collect_config_path),
                "timing": {
                    "sim_dt": self.timing.sim_dt, "policy_dt": self.timing.policy_dt,
                    "command_dt": self.timing.command_dt, "decimation": self.timing.decimation,
                    "warmup_physics_steps": warmup_steps,
                    "steps_expected": int(round(spec.window_s / self.timing.sim_dt)),
                    "steps_executed": int(energy_result["steps"]),
                    "window_s": spec.window_s,
                    "pairing_points": len(pairing),
                    "state50_points": len(state_rows),
                    "window_start_physics_step": window_start_step,
                },
                "env_origin": [float(v) for v in self.runner.env_origins[index]],
                "note_positions": "state50/pairing 的位置都是相对 env_origin 的；名义轨迹同坐标系",
                "profile_applied": {
                    "profile_id": spec.profile,
                    **(profile_readback[spec.trial_id] or {}),
                    "obs_noise": spec.obs_noise,
                    "initial_pose": spec.initial_pose,
                },
                "push_trace": [p for p in push_trace if p["trial_id"] == spec.trial_id],
                "terminations": terminated_env,
                "outcome": "completed" if not terminated_env else "terminated_early",
                "energy": {"j_exec": total_j, "unit": "J", "metric_id": "D-ENERGY-001"},
                "notes": "同 v2 布局；energy 由 applied_torque × 子步结束速度逐关节截断累计",
            }
            atomic_write_json(attempt_dir / "meta.json", meta)
            files = {
                path.name: sha256_file(path)
                for path in sorted(attempt_dir.iterdir())
                if path.is_file() and path.name not in ("DONE", "status.json")
            }
            atomic_write_json(
                attempt_dir / "DONE",
                {"trial_id": spec.trial_id, "attempt": attempt, "files": files, "sealed_at": _now()},
            )
            atomic_write_json(
                attempt_dir / "status.json",
                {"outcome": meta["outcome"], "trial_id": spec.trial_id, "attempt": attempt,
                 "attempt_kind": attempt_kinds[index], "bundle_id": self.bundle_id,
                 "finished_at": _now()},
            )
