"""Timing and isolation checks for the velocity bridge (NAV-025).

The bridge must prove, per navigation step, that every new command reached the
low-level observations before the next policy inference, that nothing re-sampled
the command behind the bridge's back without being recorded, that parallel
environments stay independent, and that the joint action is refreshed five times
per 0.1 s rather than held.

Run:

    <frozen python> CaMRA/navigation/tests/test_control_timing.py \
        --json CaMRA/navigation/reports/control_timing_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_navigation.control.v2_policy import FrozenV2Policy  # noqa: E402
from camra_navigation.control.velocity_bridge import (  # noqa: E402
    V2VelocityController,
    VelocityBridge,
)

BUNDLE = "/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2"

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


class MockLowLevel:
    """A low-level environment that echoes its command store into the obs.

    ``clobber_every`` simulates a legacy command term that resamples behind the
    bridge's back during the observation recomputation.  A well-behaved level
    sets it to 0.
    """

    def __init__(self, num_envs: int, obs_dim: int = 45, clobber_every: int = 0) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.clobber_every = clobber_every
        self.commands = np.zeros((num_envs, 3))
        self._tick = 0
        self.n_writes = 0
        self.n_obs_recomputes = 0
        self.n_action_applies = 0
        self.n_physics_steps = 0
        self.clobber_ticks: list[int] = []

    def write_commands(self, cmds) -> None:
        self.commands = np.array(cmds, dtype=np.float64)
        self.n_writes += 1

    def recompute_observations(self) -> None:
        self._tick += 1
        if self.clobber_every and self._tick % self.clobber_every == 0:
            self.commands[:] = 9.9
            self.clobber_ticks.append(self._tick)
        self.n_obs_recomputes += 1

    def get_observations(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.obs_dim))
        obs[:, 6:9] = self.commands  # the velocity_commands slot of obs-v2
        return obs

    def commands_view(self):
        return [tuple(float(v) for v in row) for row in self.commands]

    def apply_action(self, q) -> None:
        self.n_action_applies += 1

    def physics_step(self) -> None:
        self.n_physics_steps += 1


class CountingController:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return np.full((obs.shape[0], 12), 0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # --- 1. exactly 5 policy steps x 4 substeps, 0.1 s, command current ----
    env = MockLowLevel(4)
    ctrl = CountingController()
    bridge = VelocityBridge(4, ctrl)
    tr = bridge.step_nav(env, [(0.5, 0.0, 0.0)] * 4)
    check(
        "nav_step_structure_exact",
        tr.n_policy_steps == 5
        and tr.n_physics_substeps == 20
        and abs(tr.sim_time_advanced_s - 0.1) < 1e-12
        and tr.command_current_everywhere
        and ctrl.calls == 5
        and env.n_action_applies == 5,
        {
            "policy_steps": tr.n_policy_steps,
            "substeps": tr.n_physics_substeps,
            "sim_time_s": tr.sim_time_advanced_s,
            "controller_calls": ctrl.calls,
            "action_applies": env.n_action_applies,
            "command_current": tr.command_current_everywhere,
            "note": "0.1 s = 5 x 0.02 s = 20 x 0.005 s；每步都重算观测并重新推理，未把关节动作保持 0.1 s",
        },
    )

    # --- 2. a new command enters at the very next inference ----------------
    env = MockLowLevel(1)
    bridge = VelocityBridge(1, ctrl)
    bridge.step_nav(env, [(0.5, 0.0, 0.0)])
    tr2 = bridge.step_nav(env, [(0.8, 0.1, -0.2)])
    stale = [
        p.policy_index
        for p in tr2.policy_steps
        if any(abs(a - b) > 1e-12 for row in p.command_seen for a, b in zip(row, (0.8, 0.1, -0.2)))
    ]
    check(
        "new_command_reaches_observations_immediately",
        tr2.applied == [(0.8, 0.1, -0.2)] and tr2.command_current_everywhere and not stale,
        {
            "applied": [list(v) for v in tr2.applied],
            "stale_policy_steps": stale,
            "first_seen": list(tr2.policy_steps[0].command_seen[0]),
            "note": "新命令在下一个 0.1 s 的第一次推理就已进入观测，无一拍陈旧",
        },
    )

    # --- 3. an internal resampler is detected and cannot persist ----------
    env = MockLowLevel(2, clobber_every=2)
    bridge = VelocityBridge(2, ctrl)
    tr = bridge.step_nav(env, [(0.4, 0.0, 0.0), (-0.3, 0.1, 0.2)])
    detected = [p.policy_index for p in tr.policy_steps if not p.command_seen == p.command_written]
    recovered = [
        p.policy_index for p in tr.policy_steps if p.command_seen == p.command_written
    ]
    check(
        "internal_resampling_detected_and_bounded",
        bool(detected) and tr.command_current_everywhere is False
        and bool(recovered) and max(recovered) > max(detected),
        {
            "clobber_ticks": env.clobber_ticks,
            "policy_steps_with_foreign_command": detected,
            "policy_steps_recovered": recovered,
            "note": "桥以 50 Hz 重写指令：外部覆盖只能污染当前一拍，下一拍即被写回并被 trace 记录",
        },
    )

    # --- 4. parallel environments are independent --------------------------
    env = MockLowLevel(4)
    bridge = VelocityBridge(4, ctrl)
    cmds = [(0.5, 0.0, 0.0), (-1.0, 0.4, 1.0), (0.0, -0.4, -1.0), (1.0, 0.0, 0.0)]
    tr = bridge.step_nav(env, cmds)
    per_env = [p.command_seen[i] for p in tr.policy_steps for i in range(4)]
    unique_seen = len({tuple(round(v, 9) for v in row) for row in per_env}) >= 4
    check(
        "multi_env_independent_commands",
        tr.applied == [tuple(c) for c in cmds]
        and tr.command_current_everywhere
        and unique_seen,
        {
            "applied": [list(v) for v in tr.applied],
            "distinct_commands_seen": len({tuple(round(v, 9) for v in row) for row in per_env}),
            "note": "四个环境同时各持各的 applied 指令，观测回显互不串扰",
        },
    )

    # --- 5. clip once at the boundary --------------------------------------
    env = MockLowLevel(1)
    bridge = VelocityBridge(1, ctrl)
    tr = bridge.step_nav(env, [(2.0, -5.0, 0.5)])
    check(
        "clip_applied_once",
        tr.requested == [(2.0, -5.0, 0.5)]
        and tr.applied == [(1.0, -0.4, 0.5)]
        and tr.clipped == [True]
        and tr.policy_steps[0].command_seen[0] == (1.0, -0.4, 0.5),
        {
            "requested": [list(v) for v in tr.requested],
            "applied": [list(v) for v in tr.applied],
            "clipped": tr.clipped,
            "note": "限幅只在 10 Hz 应用一次；之后 5 次推理用的都是同一个 applied 值",
        },
    )

    # --- 6. non-finite commands are rejected, not propagated ---------------
    env = MockLowLevel(2)
    bridge = VelocityBridge(2, ctrl)
    tr = bridge.step_nav(env, [(float("nan"), 0.0, 0.0), (0.3, float("inf"), 0.0)])
    check(
        "nonfinite_commands_rejected",
        tr.invalid == [True, True]
        and tr.applied == [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0)]
        and tr.command_current_everywhere,
        {
            "invalid": tr.invalid,
            "applied": [list(v) for v in tr.applied],
            "note": "非有限分量按 cmd-v1 归零并置 invalid；同批次其它环境的合法分量不受影响",
        },
    )

    # --- 7. reset clears everything -----------------------------------------
    env = MockLowLevel(2)
    bridge = VelocityBridge(2, ctrl)
    bridge.step_nav(env, [(0.5, 0.0, 0.0), (0.2, 0.0, 0.0)])
    bridge.reset()
    tr = bridge.step_nav(env, [(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)])
    check(
        "reset_clears_state",
        bridge.bridge.nav_index == 1
        and abs(bridge.t_s - 0.1) < 1e-12
        and len(bridge.traces) == 1
        and tr.nav_index == 0
        and bridge.bridge.records[0].t_s == 0.0
        and bridge.bridge.records[0].nav_index == 0,
        {
            "nav_index_after_reset_and_step": bridge.bridge.nav_index,
            "trace_nav_index_zero_based": tr.nav_index,
            "first_record_t_s": bridge.bridge.records[0].t_s,
            "t_s": bridge.t_s,
            "traces": len(bridge.traces),
            "note": "trace.nav_index 是本步的 0 基索引；bridge.nav_index 是步进后的计数",
            "note": "reset 后时钟、索引、trace 与指令记录全部清零，不残留上一回合",
        },
    )

    # --- 8. clock does not drift over a second of simulated time ------------
    env = MockLowLevel(1)
    bridge = VelocityBridge(1, ctrl)
    for _ in range(10):
        bridge.step_nav(env, [(0.3, 0.0, 0.0)])
    check(
        "clock_exact_over_ten_nav_steps",
        abs(bridge.t_s - 1.0) < 1e-9 and bridge.bridge.nav_index == 10,
        {
            "t_s_after_10_steps": bridge.t_s,
            "expected": 1.0,
            "nav_index": bridge.bridge.nav_index,
        },
    )

    # --- 9. the real frozen policy works through the bridge -----------------
    policy = FrozenV2Policy.load(BUNDLE, device="cpu", backend="jit")
    ctrl = V2VelocityController(policy)
    env = MockLowLevel(2)
    bridge = VelocityBridge(2, ctrl)
    tr = bridge.step_nav(env, [(0.5, 0.0, 0.2), (0.0, 0.0, 0.0)])
    obs0 = env.get_observations()
    targets_ref, _ = policy.joint_targets(obs0)
    ref_head = [float(v) for v in targets_ref.numpy()[0][:4]]
    check(
        "frozen_v2_runs_through_bridge",
        ctrl.calls == 5
        and all(math.isfinite(v) for v in tr.policy_steps[0].q_targets_head)
        and all(abs(a - b) < 1e-12 for a, b in zip(tr.policy_steps[0].q_targets_head, ref_head)),
        {
            "controller_calls": ctrl.calls,
            "q_head_first_step": tr.policy_steps[0].q_targets_head,
            "reference_q_head": ref_head,
            "note": "真策略经桥推理；关节目标与直接调用 FrozenV2Policy 逐位一致",
        },
    )

    # --- 10. the bridge refuses an incompatible low level -------------------
    class Broken:
        pass

    try:
        VelocityBridge(1, ctrl).step_nav(Broken(), [(0.0, 0.0, 0.0)])
        refused = False
        why = ""
    except Exception as exc:  # BridgeError expected
        refused = True
        why = str(exc)[:80]
    check(
        "incompatible_low_level_refused",
        refused,
        {"raised": why, "note": "缺少 commands_view 等接口就无法证明命令及时进入观测，桥拒绝运行"},
    )

    verdict = all(r["ok"] for r in RESULTS)
    report = {
        "task": "NAV-025",
        "verdict": "PASS" if verdict else "FAIL",
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "structure": {
            "nav_step_s": 0.1,
            "policy_step_s": 0.02,
            "physics_substep_s": 0.005,
            "policy_steps_per_nav": 5,
            "substeps_per_nav": 20,
        },
        "checks": RESULTS,
    }
    print(json.dumps({k: report[k] for k in ("task", "verdict", "n_checks", "structure")},
                     ensure_ascii=False))
    for r in RESULTS:
        print(("  OK   " if r["ok"] else "  FAIL ") + r["check"])
        if not r["ok"]:
            print("        " + json.dumps(r["detail"], ensure_ascii=False)[:350])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
