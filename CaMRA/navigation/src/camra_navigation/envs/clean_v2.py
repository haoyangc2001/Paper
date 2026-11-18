"""Isolated NAV-067 clean configuration and direct 50/200 Hz Isaac adapter.

The fixed v2 action manager still computes PD every physical step. Navigation
owns termination; bypassing gym.step prevents its implicit reset and command
resampling from mixing terminal evidence into a new episode.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .physical_v2 import DT, PhysicalState, require_command_evidence


def configure_clean(env_cfg, configure_profile, applied):
    report = configure_profile(env_cfg, applied).to_dict()
    rb = env_cfg.events.reset_base.params
    rb['pose_range'] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    rb['velocity_range'] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    env_cfg.events.reset_robot_joints.params['position_range'] = (1.,1.)
    env_cfg.events.reset_robot_joints.params['velocity_range'] = (0.,0.)
    allowed = {'reset_base','reset_robot_joints'}
    present = {k for k,v in vars(env_cfg.events).items()
               if not k.startswith('_') and v is not None}
    if present-allowed:
        raise ValueError('unreviewed clean event terms: '+str(sorted(present-allowed)))
    env_cfg.commands.base_velocity.resampling_time_range = (1.e9,1.e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.
    env_cfg.commands.base_velocity.rel_heading_envs = 0.
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.curriculum = {}
    env_cfg.terminations = {}
    if env_cfg.decimation != 4 or abs(env_cfg.sim.dt-DT)>1e-12:
        raise ValueError('fixed v2 must run 50/200 Hz')
    report.update(reset_base_pose_range=rb['pose_range'],
                  reset_base_velocity_range=rb['velocity_range'],
                  reset_joint_scale_range=[1.,1.],reset_joint_velocity_range=[0.,0.],
                  command_resampling_disabled=True,automatic_reset_disabled=True)
    return report


class IsaacSubstepAdapter:
    def __init__(self, env, policy, *, map_center):
        self.env, self.u = env,env.unwrapped
        self.robot,self.sim = self.u.scene['robot'],self.u.sim
        self.policy = policy
        self.center = np.asarray(map_center,dtype=float)
        self.num_envs = self.u.num_envs
        self.tick = 0
        self.policy_calls = 0
        self.evidence = []
        self.origins = self.u.scene.env_origins.detach().cpu().numpy().copy()
        if abs(self.sim.get_physics_dt()-DT)>1e-12:
            raise ValueError('runtime physics dt mismatch')

    def states(self):
        r = self.robot.data
        p = r.root_pos_w.detach().cpu().numpy()
        q = r.root_quat_w.detach().cpu().numpy()
        v = r.root_lin_vel_b.detach().cpu().numpy()
        w = r.root_ang_vel_b.detach().cpu().numpy()
        result = []
        for i in range(self.num_envs):
            qw,qx,qy,qz = (float(x) for x in q[i])
            yaw = math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
            xy = p[i,:2]-self.origins[i,:2]+self.center
            result.append(PhysicalState(float(xy[0]),float(xy[1]),yaw,
                float(v[i,0]),float(v[i,1]),float(w[i,2]),float(p[i,2]),
                (qw,qx,qy,qz)))
        return result

    def reset(self, *, start, yaw):
        self.env.reset()
        ids = torch.arange(self.num_envs,device=self.u.device)
        pose = self.robot.data.default_root_state[:,:7].clone()
        pose[:,:2] = torch.as_tensor(self.origins[:,:2]+np.asarray(start)-self.center,
                                    dtype=pose.dtype,device=pose.device)
        pose[:,3:7] = torch.tensor([math.cos(yaw/2),0.,0.,math.sin(yaw/2)],device=pose.device)
        self.robot.write_root_pose_to_sim(pose,env_ids=ids)
        self.robot.write_root_velocity_to_sim(torch.zeros((self.num_envs,6),device=pose.device),env_ids=ids)
        pos = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(pos,torch.zeros_like(pos),env_ids=ids)
        self.u.action_manager.reset(ids)
        self.u.observation_manager.reset(ids)
        self.u.episode_length_buf.zero_()
        self.u.reset_buf.zero_()
        self.env.bridge.set_command_batch(torch.zeros((self.num_envs,3),device=pose.device))
        self.u.scene.write_data_to_sim()
        # Invalidate cached states after writes; no physics/warmup is hidden.
        self.robot.update(0.)
        self.tick,self.policy_calls = 0,0
        self.evidence=[]
        reset_obs = self.u.observation_manager.compute(update_history=False)['policy']
        if not torch.isfinite(reset_obs).all() or reset_obs.shape!=(self.num_envs,45):
            raise ValueError('fixed v2 observation shape/finite check failed')
        if float(reset_obs[:,33:45].abs().max())>0:
            raise ValueError('previous action leaked across reset')
        return {'previous_action_max':float(reset_obs[:,33:45].abs().max()),
                'commands_seen':reset_obs[:,6:9].detach().cpu().tolist(),
                'root_pose_w':pose.detach().cpu().tolist(),
                'joint_pos':pos.detach().cpu().tolist()}

    def policy_step(self, commands, *, monitor=None, recorder=None):
        command = torch.as_tensor(commands,dtype=torch.float32,device=self.u.device)
        self.env.bridge.set_command_batch(command)
        observed = self.u.observation_manager.compute(update_history=False)['policy']
        seen = observed[:,6:9].detach().cpu().numpy().copy()
        if observed.shape!=(self.num_envs,45) or not torch.isfinite(observed).all():
            raise ValueError('invalid actual low-level observation')
        # Reject stale command before inference or physical execution.
        if not np.allclose(command.detach().cpu().numpy(),seen,rtol=0.,atol=1e-6):
            raise ValueError('actual observation command was stale')
        with torch.no_grad():
            action = self.policy(observed)
        if monitor is not None:
            mask = torch.tensor(monitor.active,device=action.device)
            action = torch.where(mask[:,None],action,torch.zeros_like(action))
        self.u.action_manager.process_action(action)
        begin = self.tick
        sim_start = float(self.sim.current_time)
        for _ in range(4):
            before = list(monitor.active) if monitor is not None else [True]*self.num_envs
            self.u._sim_step_counter += 1
            self.u.action_manager.apply_action()  # v2 PD is recomputed here at 200 Hz
            self.u.scene.write_data_to_sim()
            tau = self.robot.data.applied_torque.detach().clone()
            requested = self.robot.data.computed_torque.detach().clone()
            pre = self.robot.data.joint_vel.detach().clone()
            self.sim.step(render=False)
            self.u.scene.update(dt=DT)
            self.tick += 1
            if recorder is not None:
                recorder.record(tau_applied_lab=tau,qdot_pre_lab=pre,
                    qdot_post_lab=self.robot.data.joint_vel.detach(),
                    tau_request_lab=requested,root_pos_lab=self.robot.data.root_pos_w.detach(),
                    active_mask=before)
            if monitor is not None:
                events = monitor.advance(self.states(),physics_tick=monitor.tick+1)
                if recorder is not None:
                    for i in events['ended']:
                        kind = 'timeout' if monitor.labels[i]=='timeout' else 'early_termination:'+monitor.labels[i]
                        recorder.mark_event([i],kind)
                if not any(monitor.active):
                    break
        self.policy_calls+=1
        actual_duration = float(self.sim.current_time)-sim_start
        if abs(actual_duration-(self.tick-begin)*DT)>1e-7:
            raise ValueError('actual simulator clock differs from substep count')
        proof = require_command_evidence(command.detach().cpu().numpy(),seen,
            tick_before=begin,tick_after=self.tick,substeps=self.tick-begin)
        proof.update(sim_time_before=sim_start,sim_time_after=float(self.sim.current_time),
                     policy_index=self.policy_calls,observation_shape=list(observed.shape))
        self.evidence.append(proof)
        # The direct loop never invokes gym.step or its auto-reset/commands.
        return proof
