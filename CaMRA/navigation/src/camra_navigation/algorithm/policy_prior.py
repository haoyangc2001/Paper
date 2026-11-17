"""Observation-only exploration bias inside the actor distribution.

This is a reactive tangent field, not a planner or action filter. It has no
map identity, route, hidden state, parameters, or access to future observations.
PPO can cancel its latent mean offset through the learned actor residual.
"""
from __future__ import annotations

import math

import torch
from torch import nn


PRIOR_CONFIG = {
    "version": "clearance-tangent-v1",
    "enabled": True,
    "lookahead_m": 2.5,
    "buffer_m": 0.25,
    "repulsion_m": 0.75,
    "tangent_weight": 2.0,
    "yaw_gain": 0.8,
    "max_lateral_action": 0.8,
}


class ClearanceTangentPrior(nn.Module):
    def __init__(self, config, layout, forward_speed, steering_gain):
        super().__init__()
        if config != PRIOR_CONFIG:
            raise ValueError("unsupported clearance tangent prior configuration")
        if layout is None or layout.total_dim != 61:
            raise ValueError("clearance tangent prior requires obs-v7/61 layout")
        self.slices = {name: layout.slice_of(name) for name in (
            "obstacle_bins", "safety_margin_bins", "wall_margin_n",
            "goal_dir_x_body", "goal_dir_y_body", "goal_dist_n",
            "sin_yaw", "cos_yaw")}
        if not 0.0 < forward_speed < 1.0:
            raise ValueError("clearance tangent prior requires bounded forward initialization")
        self.forward_speed = float(forward_speed)
        self.steering_gain = float(steering_gain)

    def forward(self, obs):
        def field(name):
            return obs[..., self.slices[name]]

        goal = torch.cat((field("goal_dir_x_body"), field("goal_dir_y_body")), -1)
        # Each occupied bin describes the nearest surface, at an angular
        # resolution of pi/8. Raw sensing and signed margins share a 6m scale.
        theta = (torch.arange(16, device=obs.device, dtype=obs.dtype) + .5) * (math.pi / 8) - math.pi
        normal = torch.stack((theta.cos(), theta.sin()), -1)
        raw = (1.0 - field("obstacle_bins")) * 6.0
        margin = field("safety_margin_bins") * 6.0
        radius = (raw - margin).clamp_min(0.0)
        dot = (goal.unsqueeze(-2) * normal).sum(-1)
        cross = goal[..., :1] * normal[:, 1] - goal[..., 1:] * normal[:, 0]
        along = raw * dot
        visible = field("safety_margin_bins") < 1.0
        threat = (visible & (dot > 0.0)
                  & (along < field("goal_dist_n") * 15.0 + PRIOR_CONFIG["buffer_m"])
                  & (raw * cross.abs() < radius + PRIOR_CONFIG["buffer_m"]))
        weight = ((PRIOR_CONFIG["lookahead_m"] - margin) / PRIOR_CONFIG["lookahead_m"]).clamp(0., 1.)
        weight = torch.where(threat, weight, torch.zeros_like(weight))
        strength, index = weight.max(-1)
        n = normal[index]
        side = torch.where(cross.gather(-1, index.unsqueeze(-1)).squeeze(-1) >= 0., -1., 1.)
        tangent = side.unsqueeze(-1) * torch.stack((-n[..., 1], n[..., 0]), -1)
        clearance = margin.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        repel = ((PRIOR_CONFIG["repulsion_m"] - clearance) / PRIOR_CONFIG["repulsion_m"]).clamp(0., 2.)
        desired = goal + strength.unsqueeze(-1) * (PRIOR_CONFIG["tangent_weight"] * tangent - goal - repel.unsqueeze(-1) * n)

        # Walls are world-axis clearances; rotate their outward normals into
        # the body frame with the same observed yaw as the actor receives.
        sy, cy = field("sin_yaw").squeeze(-1), field("cos_yaw").squeeze(-1)
        wx = torch.stack((cy, -cy, sy, -sy), -1)
        wy = torch.stack((-sy, sy, -cy, cy), -1)
        wall = ((PRIOR_CONFIG["repulsion_m"] - field("wall_margin_n") * 6.) / PRIOR_CONFIG["repulsion_m"]).clamp(0., 2.)
        desired = desired + 2. * torch.stack(((wall * wx).sum(-1), (wall * wy).sum(-1)), -1)
        strength = torch.maximum(strength, wall.max(-1).values.clamp_max(1.))
        direction = desired / desired.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target = torch.stack((self.forward_speed * direction[..., 0],
                              (self.forward_speed * direction[..., 1] / .4).clamp(-PRIOR_CONFIG["max_lateral_action"], PRIOR_CONFIG["max_lateral_action"]),
                              PRIOR_CONFIG["yaw_gain"] * direction[..., 1]), -1)
        baseline = torch.stack((torch.full_like(sy, math.atanh(self.forward_speed)),
                                torch.zeros_like(sy), self.steering_gain * goal[..., 1]), -1)
        return strength.unsqueeze(-1) * (torch.atanh(target.clamp(-.95, .95)) - baseline)
