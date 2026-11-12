"""NAV-042 checks for the online loop wiring (no simulator, no GPU).

The low level here is a **stub**: it integrates the frozen nominal model so the
bookkeeping can be inspected deterministically.  Nothing in this file is
evidence about physical execution -- it is evidence that the navigation layer

* runs 5 policy steps of 4 physics substeps per 0.1 s navigation step,
* writes the applied command before every re-observation,
* charges the internal budget with ``(1+alpha)*c_nom`` while keeping the measured
  work in a separate field,
* keeps the nominal reference on its own track (never re-anchored to the actual
  pose),
* terminates in the frozen priority order, refuses to step a finished episode,
  and cannot stitch two episodes into one trace.

    <frozen python> CaMRA/navigation/tests/test_physical_loop.py \
        --json CaMRA/navigation/reports/physical_loop_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

import numpy as np  # noqa: E402

from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.nominal import DT_NAV_S  # noqa: E402
from camra_navigation.envs.physical import (  # noqa: E402
    PHYSICAL_LABELS,
    POLICY_STEPS_PER_NAV,
    SUBSTEPS_PER_POLICY,
    EpisodeOverError,
    PhysicalConfig,
    PhysicalEnvError,
    PhysicalNavEnv,
)
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")

POLICY_DT = DT_NAV_S / POLICY_STEPS_PER_NAV
SUB_DT = POLICY_DT / SUBSTEPS_PER_POLICY

CHECKS: list[dict] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail) -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        FAILED.append(name)


class StubLowLevel:
    """Deterministic planar plant + the bridge methods the env needs.

    ``drift_rate_mps`` makes the *actual* state disagree with the model so the
    reference twin can be shown to stay on its own track; it is a deliberately
    artificial plant, not a model of anything.
    """

    def __init__(self, spec, geometry, *, num_envs: int = 1, drift_rate_mps: float = 0.0,
                 time_jump_s: float = 0.0, fall_at_policy_step: int | None = None) -> None:
        from camra_calibration.nominal import PlanarState, step_exact

        self._step_exact = step_exact
        self._PlanarState = PlanarState
        self.spec = spec
        self.geometry = geometry
        self.num_envs = int(num_envs)
        self.drift_rate_mps = float(drift_rate_mps)
        self.time_jump_s = float(time_jump_s)
        self.fall_at_policy_step = fall_at_policy_step
        self._policy_step_count = 0
        self.rows: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(self.num_envs)]
        self.reset()

    def reset(self) -> None:
        goal = (float(self.spec.goal[0]), float(self.spec.goal[1]))
        start = (float(self.spec.start[0]), float(self.spec.start[1]))
        yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        self._pose = [(start[0], start[1], yaw) for _ in range(self.num_envs)]
        self._t = 0.0
        self._measured = [0.0] * self.num_envs
        self._written = [[0.0, 0.0, 0.0] for _ in range(self.num_envs)]
        self._seen: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(self.num_envs)]

    def write_commands(self, rows) -> None:
        if len(rows) != self.num_envs:
            raise AssertionError("command batch size mismatch")
        self._written = [list(map(float, r)) for r in rows]

    def step_policy(self) -> dict:
        for i in range(self.num_envs):
            self._seen[i] = list(self._written[i])
            state = self._PlanarState(self._pose[i][0], self._pose[i][1], self._pose[i][2])
            for _ in range(SUBSTEPS_PER_POLICY):
                state = self._step_exact(state, tuple(self._written[i]), SUB_DT)
            px, py = float(state.px), float(state.py)
            yaw = float(state.yaw)
            if self.drift_rate_mps:
                # artificial sideways drift: the plant is NOT the nominal model
                px += self.drift_rate_mps * POLICY_DT
            self._pose[i] = (px, py, yaw)
            v = self._written[i]
            self._measured[i] += (abs(v[0]) + abs(v[1]) + 0.5 * abs(v[2])) * POLICY_DT
        self._t += POLICY_DT + self.time_jump_s
        self._policy_step_count += 1
        events = {}
        if (self.fall_at_policy_step is not None
                and self._policy_step_count == self.fall_at_policy_step):
            events = {i: "fall" for i in range(self.num_envs)}
        return {
            "obs_command": [list(c) for c in self._seen],
            "substeps": SUBSTEPS_PER_POLICY,
            "sim_time": self._t,
            "events": events,
            "obs_fingerprint": [float(c) for c in self._seen[0]],
        }

    def step_time(self) -> float:
        return self._t

    def true_state(self, env_id: int):
        px, py, yaw = self._pose[env_id]
        v = self._written[env_id]
        return (px, py, yaw, float(v[0]), float(v[1]), float(v[2]))

    def measured_J(self) -> list[float]:
        return list(self._measured)


def build(variant: str = "full"):
    design = M.load_design(DESIGN)
    geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    if variant == "uncalibrated":
        import dataclasses
        geometry = dataclasses.replace(geometry, delta_m=0.0)
    cm = JCostModel.from_frozen(POWER, MARGINS, variant=variant)
    return design, geometry, cm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    design, geometry, cm = build("full")
    spec = M.build("L1_empty", 3001, design)
    forward = lambda vec: [0.8, 0.0, 0.0]  # noqa: E731

    # ---- 1. 10/50/200 structure, command freshness, clock ------------------
    low = StubLowLevel(spec, geometry)
    env = PhysicalNavEnv(spec, geometry, cm, low, forward,
                         candidate_id="stub", cfg=PhysicalConfig())
    env.reset(z0_J=400.0)
    recs = env.step()
    r = recs[0]
    check("five_policy_steps_four_substeps_each",
          len(r.policy_steps) == POLICY_STEPS_PER_NAV
          and all(p.substeps == SUBSTEPS_PER_POLICY for p in r.policy_steps),
          {"policy_steps": len(r.policy_steps),
           "substeps": [p.substeps for p in r.policy_steps]})
    tiles = all(abs(p.t_start_s - (r.t_start_s + k * POLICY_DT)) < 1e-9
                and abs(p.t_end_s - (r.t_start_s + (k + 1) * POLICY_DT)) < 1e-9
                for k, p in enumerate(r.policy_steps))
    check("policy_step_windows_tile_the_nav_interval", tiles,
          {"nav": [r.t_start_s, r.t_end_s],
           "policy": [[p.t_start_s, p.t_end_s] for p in r.policy_steps]})
    check("command_written_equals_seen_every_policy_step",
          all(p.as_dict()["command_current"] for p in r.policy_steps),
          {"written": r.policy_steps[0].command_written,
           "seen": r.policy_steps[0].command_seen})
    check("nav_clock_advances_exactly_one_dt",
          abs(r.t_end_s - r.t_start_s - DT_NAV_S) < 1e-12
          and abs(low.step_time() - DT_NAV_S) < 1e-9,
          {"t_start": r.t_start_s, "t_end": r.t_end_s, "sim": low.step_time()})
    check("applied_equals_clipped_requested",
          r.applied == r.requested and not r.clipped and not r.invalid
          and r.applied == (0.8, 0.0, 0.0),
          {"requested": r.requested, "applied": r.applied, "clipped": r.clipped})

    # ---- 2. budget vs measured energy stay separate ------------------------
    c_nom = r.cost_nom_J
    ratio = 1.0 + cm.alpha
    check("budget_charged_with_one_plus_alpha_times_c_nom",
          abs(r.cost_model_J - ratio * c_nom) < 1e-9
          and abs((400.0 - r.budget_z_J) - ratio * c_nom) < 1e-9,
          {"c_nom": c_nom, "c_model": r.cost_model_J, "alpha": cm.alpha,
           "z": r.budget_z_J})
    check("measured_energy_is_a_separate_field",
          r.measured_cum_J > 0.0 and abs(r.measured_cum_J - r.cost_nom_J) > 1e-9,
          {"measured_J": r.measured_cum_J, "c_nom": r.cost_nom_J,
           "note": "measured work comes from the 200 Hz recorder; c_nom comes "
                   "from the frozen power model - different sources on purpose"})
    check("z_ratio_uses_the_scene_constant", 
          abs(r.budget_z_ratio - r.budget_z_J / 550.0) < 1e-9,
          {"z_ratio": r.budget_z_ratio, "z_J": r.budget_z_J})

    # ---- 3. reference twin is NOT re-anchored ------------------------------
    low2 = StubLowLevel(spec, geometry, drift_rate_mps=0.05)
    env2 = PhysicalNavEnv(spec, geometry, cm, low2, lambda v: [0.0, 0.0, 0.0],
                          candidate_id="stub_drift", cfg=PhysicalConfig())
    env2.reset(z0_J=400.0)
    rr = env2.step()[0]
    start = spec.start
    ref_pose = rr.ref_pose
    check("reference_stays_put_when_command_is_zero",
          abs(ref_pose[0] - float(start[0])) < 1e-9
          and abs(ref_pose[1] - float(start[1])) < 1e-9,
          {"ref_pose": ref_pose, "start": list(start),
           "note": "zero command -> the twin must not move, even though the plant drifted"})
    check("execution_deviation_is_visible",
          rr.deviation_m > 1e-3,
          {"deviation_m": rr.deviation_m,
           "note": "re-anchoring the reference would make this structurally 0"})

    # ---- 4. episode lifecycle: terminal priority, no stitching -------------
    spec3 = M.build("L1_empty", 3002, design)
    env3 = PhysicalNavEnv(spec3, geometry, cm,
                          StubLowLevel(spec3, geometry),
                          forward, candidate_id="stub2", cfg=PhysicalConfig())
    trace = env3.reset(z0_J=2000.0)
    labels = []
    for _ in range(400):
        if all(env3.done):
            break
        env3.step()
    labels = list(env3.label)
    check("episode_terminates_with_a_frozen_label",
          all(lb in PHYSICAL_LABELS for lb in labels) and all(env3.done),
          {"labels": labels, "n_steps": len(trace.records)})
    try:
        env3.step()
        stepped_after_done = True
    except EpisodeOverError:
        stepped_after_done = False
    check("finished_episode_refuses_further_steps", not stepped_after_done, {})

    t_before_new = trace
    env3.reset(z0_J=2000.0)
    new_rec = env3.step()[0]
    check("new_episode_gets_a_new_id_and_no_mixing",
          new_rec.episode_id == t_before_new.episode_id + 1
          and all(r.episode_id == new_rec.episode_id for r in env3.traces[-1].records)
          and all(r.episode_id == t_before_new.episode_id for r in t_before_new.records),
          {"old_episode": t_before_new.episode_id, "new_episode": new_rec.episode_id,
           "old_steps": len(t_before_new.records)})

    # ---- 5. safety history is irreversible --------------------------------
    y_trace = [r.safety_y for r in trace.records]
    monotone = all(y_trace[i] <= y_trace[i + 1] + 1e-12 for i in range(len(y_trace) - 1))
    check("safety_y_is_irreversible", monotone, {"y_first_last": [y_trace[0], y_trace[-1]]})

    # ---- 6. no feasible budget is a result, not a fallback ----------------
    low4 = StubLowLevel(spec, geometry)
    env4 = PhysicalNavEnv(spec, geometry, cm, low4, forward, candidate_id="stub4")
    tr4 = env4.reset(z0_J=None)
    try:
        env4.step()
        refused = False
    except EpisodeOverError:
        refused = True
    check("no_feasible_budget_is_recorded_not_invented",
          tr4.status == "no_feasible_budget" and tr4.label == "no_feasible_budget"
          and tr4.z0_J is None and len(tr4.records) == 0 and refused,
          {"status": tr4.status, "z0_J": tr4.z0_J, "steps": len(tr4.records),
           "notes": tr4.notes})

    # ---- 7. a hidden reset cannot pass silently ---------------------------
    low5 = StubLowLevel(spec, geometry, time_jump_s=0.2)
    env5 = PhysicalNavEnv(spec, geometry, cm, low5, forward, candidate_id="stub5")
    env5.reset(z0_J=400.0)
    try:
        env5.step()
        caught = False
        msg = ""
    except PhysicalEnvError as exc:
        caught = True
        msg = str(exc)
    check("simulator_clock_jump_is_refused", caught, {"error": msg})

    # ---- 8. a substep event terminates the episode ------------------------
    spec8 = M.build("L1_empty", 3003, design)
    low8 = StubLowLevel(spec8, geometry, fall_at_policy_step=3)
    env8 = PhysicalNavEnv(spec8, geometry, cm, low8, forward, candidate_id="stub8")
    tr8 = env8.reset(z0_J=2000.0)
    r8 = env8.step()[0]
    check("substep_event_terminates_the_episode",
          env8.label[0] == "fall" and env8.done[0] and r8.label == "fall"
          and r8.failure_mask["fall"] is True,
          {"label": env8.label[0], "failure_mask": r8.failure_mask,
           "note": "the event arrives from a physics substep, before the 0.1 s boundary"})

    # ---- 9. failure masks are consistent with the terminal label ----------
    masks = [r.failure_mask for r in trace.records]
    keys_ok = all(set(m) == {"danger", "internal_violation", "collision",
                             "out_of_bounds", "budget_exhausted", "fall"}
                  for m in masks)
    label = trace.records[-1].label
    consistent = (label not in ("collision", "out_of_bounds", "fall")
                  or masks[-1][label] is True)
    never_both = all(not (m["collision"] and m["fall"]) for m in masks)
    check("failure_masks_are_complete_and_consistent",
          keys_ok and consistent and never_both,
          {"n_steps": len(masks), "label": label, "last_mask": masks[-1] if masks else None,
           "note": "budget_exhausted/fall flags are derived from the label; the geometric "
                   "collision and out_of_bounds flags are independent per step"})

    payload = {"task": "NAV-042", "schema": "camra-physical-loop-check-v1",
               "verdict": "PASS" if not FAILED else "FAIL",
               "n_checks": len(CHECKS), "failed": FAILED, "checks": CHECKS,
               "scope_note": "stub low level integrating the frozen nominal model: "
                             "wiring evidence only, not physical-execution evidence"}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
    print(json.dumps({"verdict": payload["verdict"], "n_checks": len(CHECKS),
                      "failed": FAILED}, ensure_ascii=False))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
