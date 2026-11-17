"""Verify the exact implementation and numerical release before sampling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_training_bindings(prereg: dict, nav_root: str | Path) -> None:
    root = Path(nav_root)
    bindings = prereg.get("binding", {})
    for name, entry in bindings.items():
        path = root / entry["path"]
        if not path.is_file() or file_sha(path) != entry["sha256"]:
            raise ValueError(f"binding mismatch: {name}: {path}")
    runtime = prereg.get("runtime_contract")
    if runtime is not None:
        if runtime.get("reach_schedule") != "reference_gamma_2_half_horizon":
            raise ValueError("unsupported reach discount schedule")
        rows = prereg.get("run_registry", [])
        expected = {(m, s) for m in prereg["method"]["variants"] for s in prereg["method"]["seeds"]}
        pairs = [(r["variant"], r["seed"]) for r in rows]
        if len(rows) != 6 or len(set(r["run_id"] for r in rows)) != 6 or set(pairs) != expected or len(set(pairs)) != 6:
            raise ValueError("run registry is incomplete or contains duplicates")
        if not all(r["run_id"].startswith(runtime["run_prefix"] + "_") for r in rows):
            raise ValueError("run IDs use a previous attempt prefix")
        for field in ("store_auditor", "development_evaluator", "runner", "reference_energy_report"):
            if field not in bindings:
                raise ValueError(f"required runtime binding missing: {field}")
    versions = prereg.get("implementation_versions", {})
    if int(prereg.get("round", 0)) >= 14 and not versions:
        raise ValueError("r14 requires the repaired numerical gate and explicit versions")
    if not versions:
        return
    feedback = (versions.get("observation") in ("obs-v5", "obs-v6", "obs-v7")
                and versions.get("estimator") == "rcppo-feedback-v4"
                and set(versions) == {"observation", "estimator"})
    if not feedback and versions != {"observation": "obs-v5", "estimator": "rcppo-r14-v3"}:
        raise ValueError("unsupported observation/estimator pairing")
    if feedback:
        if prereg.get("dead_step_mask", {}).get("enabled", False):
            raise ValueError("feedback-v4 requires all policy/reach samples")
        if prereg.get("feedback_policy") != "finite_horizon_all_states":
            raise ValueError("feedback-v4 requires explicit task-ending policy")
        if prereg.get("authorization", {}).get("training_approved") is not True:
            raise ValueError("feedback repair is not authorization for additional training")
    for field in ("navigation_contract", "value_contract", "numerical_gate"):
        if field not in bindings:
            raise ValueError(f"required binding missing: {field}")
    nav = json.loads((root / bindings["navigation_contract"]["path"]).read_bytes())
    value = json.loads((root / bindings["value_contract"]["path"]).read_bytes())
    if nav["observation_spec"]["version"] != versions["observation"]:
        raise ValueError("navigation observation version mismatch")
    if nav.get("observation_version") != versions["observation"]:
        raise ValueError("navigation top-level observation version mismatch")
    if value["estimator"].get("version") != versions["estimator"]:
        raise ValueError("value estimator version mismatch")
    if feedback:
        if (nav.get("value_objective_reference", {}).get("contract")
                != bindings["value_contract"]["path"]
                or nav.get("termination_semantics", {}).get("training_boundary", {}).get("version")
                != prereg["feedback_policy"]):
            raise ValueError("feedback navigation/value boundary contracts disagree")
        if float(nav["task_definition"]["horizon_s"]) != float(prereg["environment"]["t_max_s"]):
            raise ValueError("feedback task horizon differs from navigation contract")
        if versions["observation"] in ("obs-v6", "obs-v7"):
            from camra_navigation.observation import TIME_REF_S_BY_VERSION
            scale = TIME_REF_S_BY_VERSION[versions["observation"]]
            if (float(nav["observation_spec"]["constants"]["t_max_s"]) != scale
                    or float(prereg["environment"]["t_max_s"]) != scale):
                raise ValueError("time scale and task deadline must both be 45 seconds")
        if versions["observation"] == "obs-v7":
            from camra_navigation.observation import layout_for_version, observation_dim
            spec = nav["observation_spec"]
            layout = tuple((row["name"], row["dim"]) for row in spec["layout"])
            if layout != layout_for_version("obs-v7") or spec["total_dim"] != observation_dim("obs-v7"):
                raise ValueError("obs-v7 clearance layout mismatch")
            if spec.get("safety_clearance") != {
                    "version": "signed-clearance-v1", "radius_source": "environment.geometry.inflation_radius_m",
                    "normalizer_m": 6.0, "raw_surface_sensing_range_m": 6.0,
                    "empty_bin": 1.0, "clip": [-1.0, 1.0], "wall_order": ["x_min", "x_max", "y_min", "y_max"]}:
                raise ValueError("obs-v7 clearance geometry contract mismatch")
            prefix = prereg["runtime_contract"]["run_prefix"]
            if prefix not in ("nav064_clearance_feedback_v1", "nav064_episode_feedback_v1", "nav064_avoidance_prior_v1"):
                raise ValueError("obs-v7 requires an independent attempt namespace")
            if prefix in ("nav064_episode_feedback_v1", "nav064_avoidance_prior_v1"):
                window = prereg["window_semantics"]
                if (window.get("collection") != "complete_episodes_v1"
                        or window["calls_per_update"] != 450
                        or window["env_steps_per_update"] != 8100
                        or window["max_updates"] != 300
                        or float(value["estimator"]["gae_lambda"]) != .999):
                    raise ValueError("episode feedback collection/value/allocation mismatch")
    prior = prereg.get("avoidance_prior")
    prior_candidate = runtime is not None and runtime.get("run_prefix") == "nav064_avoidance_prior_v1"
    if prior_candidate:
        from .policy_prior import PRIOR_CONFIG
        if (prior != PRIOR_CONFIG or versions.get("observation") != "obs-v7"
                or prereg.get("action_bounded") is not True):
            raise ValueError("avoidance candidate policy configuration mismatch")
        if prereg.get('exploration_gate') != {
                'after_updates': 50, 'variant': 'full',
                'minimum_safe_l2_arrivals_per_seed': 1, 'failure_action': 'HOLD_NO_RETRY'}:
            raise ValueError('avoidance candidate requires registered exploration prerequisite')
    elif prior is not None:
        raise ValueError("avoidance prior requires its independent candidate namespace")
    fix = prereg.get("objective_signal_fix", {})
    if (fix.get("h_semantics") != "v2_deep_well"
            or fix.get("joint_min_recursion") is not True
            or float(fix.get("joint_cost_weight", -1)) != 0.0
            or float(fix.get("joint_cost_base_weight", -1)) != 0.0):
        raise ValueError("repaired estimator requires registered min/deep-well/no-additive-cost settings")
    if (float(value["estimator"]["gamma_cost"]) != 1.0
            or float(value["estimator"]["gamma_joint"]) != 1.0):
        raise ValueError("repaired cost/joint discount mismatch")
    gate = json.loads((root / bindings["numerical_gate"]["path"]).read_bytes())
    if gate.get("verdict") != "PASS" or gate.get("implementation_versions") != versions:
        raise ValueError("numerical gate is not PASS for these versions")
    if not gate.get("checks") or not all(c["ok"] for c in gate["checks"]):
        raise ValueError("numerical gate has missing or failed checks")
    if prior_candidate:
        passed = {c.get("check") for c in gate["checks"] if c["ok"]}
        required = {"prior_default_bitwise", "prior_direction_and_walls", "prior_probability_and_residual",
                    "prior_actual_producer", "prior_actual_auditor", "prior_actual_development_build",
                    "prior_corrupt_replay_rejected", "prior_observation_only", "prior_exploration_prefix_gate"}
        if gate.get("avoidance_prior") != prior or not required.issubset(passed):
            raise ValueError("avoidance gate omitted policy/production evidence")
    if feedback:
        passed = {c.get("check") for c in gate["checks"] if c["ok"]}
        required = {"task_deadline_no_learned_future", "all_graded_states_in_loss",
                    "crossing_action_bad_probability_decreases", "train_replay_feedback_bounds"}
        if versions["observation"] in ("obs-v6", "obs-v7"):
            required |= {"time_450_steps_strictly_decreasing", "only_time_field_changes",
                         "production_build18_train_time_v6", "production_step_time_decreases",
                         "production_development_constructor_time_v6",
                         "production_coordinator_time_v6"}
        if versions["observation"] == "obs-v7":
            required |= {"clearance_independent_signed_geometry", "clearance_actual_method_radius",
                         "clearance_legacy_bitwise_replay", "production_train_dev_clearance_v7",
                         "production_three_networks_clearance_v7", "production_auditor_clearance_v7",
                         "production_coordinator_clearance_v7", "clearance_actual_pre_crossing_action",
                         "crossing_v7_probability_decreases"}
        if not required.issubset(passed):
            raise ValueError("feedback gate omitted ending/safety/update/replay evidence")
        if prereg.get("window_semantics", {}).get("collection") == "complete_episodes_v1":
            required = {"episode_production_complete_batches", "episode_old_log_prob_fixed",
                        "episode_all_terminal_kinds_retained", "episode_horizon_feedback_18_of_18",
                        "episode_auditor_variable_samples", "episode_coordinator_allocation"}
            if not required.issubset(passed):
                raise ValueError("complete-episode production gate missing")
    if not gate.get("code_files") or not gate.get("protocol"):
        raise ValueError("numerical gate has no source/protocol bindings")
    for rel, digest in gate["code_files"].items():
        if file_sha(root / rel) != digest:
            raise ValueError(f"gate source mismatch: {rel}")
    if runtime is not None:
        passed = {c.get("check") for c in gate["checks"] if c["ok"]}
        required = {"runtime_schedule_reaches_final_halfway", "runtime_writer_snapshot_before_reset",
                    "runtime_auditor_rejects_stale_identity", "runtime_registry_selects_current_attempt"}
        if not required.issubset(passed):
            raise ValueError("gate omitted required production runtime checks")
    protocol = gate["protocol"]
    if file_sha(root / protocol["path"]) != protocol["sha256"]:
        raise ValueError("gate protocol mismatch")
