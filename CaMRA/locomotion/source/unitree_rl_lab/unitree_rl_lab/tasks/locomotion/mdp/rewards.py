from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .stop_events import advance_stop_window, relatch_on_episode_start

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


"""
Command-transient rewards (TASK-023 tuning round 1).

Why these exist
---------------
The baseline reward set has no term that is sensitive to a *command step*:
``track_lin_vel_xy_exp`` averages the squared error over the whole episode, so a
0.55 s transient on a 20 s episode is ~2.5 % of the return. Measured on the
formal run (``dev_it2999``): doubling the start-up speed would raise
``track_lin_vel_xy`` by only +0.009 per step, while ``action_rate_l2`` would cost
more than that -- under the baseline reward, a slow start is the *optimal*
policy, which is why all six checkpoints failed ``start_stop``.

The terms below are active only where the baseline is blind: in the 0.6 s after
a command step, and in the 0.5 s after a command falls to zero. Round 4 drops
the always-on standstill velocity penalty (:func:`lin_vel_xy_at_standstill_l2`,
kept below for reference and rollback) in favour of
:func:`stop_drift_l2`, which charges the displacement itself over the falling
edge only.
"""


def _command_step_age(
    env: ManagerBasedRLEnv, command: torch.Tensor, window_steps: int, tol: float = 0.02
) -> torch.Tensor:
    """Steps elapsed since the command last changed, clamped to ``window_steps``.

    Command terms resample on a 4-8 s schedule, so a change is a jump rather than
    a drift; a per-environment counter is enough here and, unlike
    ``time_left_buf``, it does not require recovering the sampled interval.
    """
    prev = getattr(env, "_camra_prev_cmd", None)
    age = getattr(env, "_camra_step_age", None)
    if prev is None or age is None or prev.shape != command.shape or age.numel() != env.num_envs:
        prev = command.detach().clone()
        age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    changed = torch.sum(torch.abs(command - prev), dim=1) > tol
    age = torch.where(changed, torch.full_like(age, window_steps), torch.clamp(age - 1, min=0))
    env._camra_prev_cmd = command.detach().clone()
    env._camra_step_age = age
    return age


def track_lin_vel_xy_step_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    step_window_s: float = 0.6,
) -> torch.Tensor:
    """``track_lin_vel_xy_exp`` restricted to the transient after a command step.

    Same exponential kernel as the baseline term, with a narrower ``std`` so the
    gradient concentrates on the part of the response the baseline ignores.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)
    window_steps = max(1, int(round(step_window_s / env.step_dt)))
    age = _command_step_age(env, command, window_steps)
    return reward * (age > 0).float()


def lin_vel_xy_at_standstill_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal base speed while the command asks for (near) zero velocity.

    The baseline ``base_linear_velocity`` penalty is ``lin_vel_z_l2``: it only
    looks at the vertical axis, so nothing in the baseline reward discourages the
    ~0.13 m of coasting that made ``stop_drift_m`` fail on all six checkpoints.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = (torch.norm(command[:, :2], dim=1) < threshold).float()
    return standing * torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def stop_drift_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_m: float = 0.10,
    window_s: float = 0.5,
    onset_tol: float = 0.02,
    max_ratio: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the planar displacement travelled inside the window after a stop command.

    Tuning round 4 replaces :func:`lin_vel_xy_at_standstill_l2` with this term.
    Frame-level analysis of ``dev_it2500_r3`` shows the policy reacts in 2-4
    control steps and brakes at a 4.1-4.5 m/s^2 peak, yet still drifts 0.120 m:
    85 % of it (0.111 m) is travelled in the first 0.20 s, because the braking
    ramps up slowly, and the remaining backward rebound (-0.07 m/s) then pulls
    the 2 s displacement back down to 0.120 m. The always-on velocity penalty was
    worth 0.0036 against ``action_rate_l2``'s 0.1796, and it also fired *before*
    a standing start, which is the most plausible cause of the round 3 rise-time
    regression (0.225 -> 0.533 s).

    Charging the displacement itself -- normalised by the protocol threshold, and
    only inside ``window_s`` after a falling edge -- points the gradient straight
    at the graded metric and leaves the start transient alone.

    Two safeguards keep the term bounded, because an unbounded quadratic penalty
    on a handful of envs is enough to destroy the critic while leaving the logged
    mean untouched (measured on the first round 4 smoke run: value loss
    0.05 -> 16.9 in 40 iterations, see :func:`~.stop_events.relatch_on_episode_start`):

    * the latch is cleared for envs respawned on the previous step (rewards are
      computed before ``_reset_idx`` here, so the stale step is the one after);
    * the normalised displacement is clamped at ``max_ratio``, i.e. the penalty
      saturates at ``max_ratio * target_m`` of drift instead of growing without
      bound when a stop is commanded while the robot is being pushed.

    Returns:
        ``(N,)`` non-negative penalty, ``min((drift / target_m) ** 2, max_ratio)``
        while the window is active and zero outside it.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    pos = asset.data.root_pos_w[:, :2]
    window_steps = max(1, int(round(window_s / env.step_dt)))

    state = getattr(env, "_camra_stop_state", None)
    if state is None or state["age"].numel() != env.num_envs:
        state = {
            "prev": command.detach().clone(),
            "age": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "anchor": pos.detach().clone(),
        }

    age, anchor, drift2, active = advance_stop_window(
        command=command,
        prev_command=state["prev"],
        age=state["age"],
        anchor=state["anchor"],
        pos=pos.detach(),
        window_steps=window_steps,
        onset_tol=onset_tol,
    )
    # a window still open when the episode ends would compare the new spawn
    # position against the previous episode's anchor -- drop it instead
    age, anchor, drift2, active = relatch_on_episode_start(
        age=age,
        anchor=anchor,
        pos=pos.detach(),
        episode_length_buf=env.episode_length_buf,
    )
    state["prev"] = command.detach().clone()
    state["age"] = age
    state["anchor"] = anchor.detach()
    env._camra_stop_state = state

    ratio = torch.clamp(drift2 / target_m**2, max=max_ratio)
    return active.float() * ratio
