"""Hand-calculated deployment cases, independent of the training estimators."""

import json
from pathlib import Path

import numpy as np
import pytest

from camra_navigation.algorithm.deployment_oracle import joint_residual, reconstruct_truth

NAV = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((NAV / "config/value_objective_contract_terminal_tail_v1.json").read_bytes())


def episode(label, budget_J, distance_m, safety):
    # Fixed obs-v7 positions for the fields under test. These are pre-action
    # values, while the trace columns beside them are the post-action values.
    trace = []
    for t in range(len(budget_J) - 1):
        observation = [0.0] * 61
        observation[9] = distance_m[t] / 15.0
        observation[-3:] = [safety[t], np.clip(budget_J[t] / 550.0, -3, 3),
                            (3.0 - t) / 3.0]
        trace.append(dict(step=t, observation=observation, z_J=budget_J[t + 1],
                          goal_distance_m=distance_m[t + 1], safety_y=safety[t + 1],
                          danger_violation=safety[t + 1] > safety[t]))
    return dict(label=label, trace=trace, n_steps=len(trace), t_s=float(len(trace)),
                z0_J=budget_J[0], z_J_end=budget_J[-1],
                cost_model_J=budget_J[0] - budget_J[-1],
                danger_steps=sum(r["danger_violation"] for r in trace),
                feasible=label == "goal_reached" and not safety[-1] and budget_J[-1] >= 0)


def truth(row):
    return reconstruct_truth(row, initial_goal_distance_m=4.0, value_contract=CONTRACT,
                             horizon_s=3.0, goal_radius_m=1.0, step_s=1.0)


def test_success_at_exact_budget_boundary():
    result = truth(episode("goal_reached", [165, 55, 0], [4, 3, 1], [0, 0, 0]))
    assert result.feasible
    assert result.stop_index.tolist() == [2, 2]
    assert result.budget_J.tolist() == [165, 55]
    assert result.cost_target == pytest.approx([0.3, 0.1])
    assert result.reach_target == pytest.approx([-3, -3])
    assert result.path_residual == pytest.approx([0, 0])


def test_unsafe_history_survives_arrival_after_danger_clears():
    result = truth(episode("goal_reached", [165, 110, 55], [4, 3, 1], [0, .4, .4]))
    assert not result.feasible
    assert result.stop_index.tolist() == [0, 1]
    assert result.cost_target == pytest.approx([0, 0])
    assert result.reach_target == pytest.approx([.2, .4])


def test_overspend_changes_the_shared_stop_without_ending_the_episode():
    result = truth(episode("goal_reached", [110, 55, -110, -165],
                           [4, 2.5, 2, .8], [0] * 4))
    assert not result.feasible
    assert result.stop_index.tolist() == [1, 1, 2]
    assert result.cost_target == pytest.approx([.1, 0, 0])
    assert result.reach_target == pytest.approx([.1, .1, 1 / 15])


@pytest.mark.parametrize("label", ["timeout", "stagnation"])
def test_finite_task_end_uses_measured_reach_without_bootstrap(label):
    result = truth(episode(label, [220, 165, 110, 55], [4, 3, 2.5, 1.75], [0] * 4))
    assert not result.feasible
    assert result.stop_index.tolist() == [3, 3, 3]
    assert result.cost_target == pytest.approx([.3, .2, .1])
    assert result.reach_target == pytest.approx([.05] * 3)


@pytest.mark.parametrize("label", ["collision", "out_of_bounds"])
def test_hard_failure_forces_terminal_and_replaces_final_cost(label):
    # Nominal earlier stops have a much lower residual. They cannot hide a
    # hard failure, which takes precedence even if the endpoint is in goal.
    result = truth(episode(label, [220, 165, 55], [4, 2.5, 1], [0, 0, .4]))
    assert not result.feasible
    assert result.stop_index.tolist() == [2, 2]
    assert result.cost_target == pytest.approx([1.1, 1.0])
    assert result.reach_target == pytest.approx([1.0, 1.0])


def test_shared_stop_ties_take_the_earliest_state():
    result = truth(episode("stagnation", [220, 165, 110], [4, 2.5, 2.5], [0] * 3))
    assert result.stop_index.tolist() == [1, 1]


def test_budget_feature_is_checked_but_residual_uses_unclipped_joules():
    row = episode("goal_reached", [2200, 1650, 550], [4, 2.5, 1], [0, 0, 0])
    result = truth(row)
    assert result.budget_feature_clipped.tolist() == [True, False]
    assert joint_residual([3.2, .3, .1], [-3, -3, -3], [2200, 110, -55],
                          550).tolist() == pytest.approx([-.8, .1, .2])
    row["trace"][1]["observation"][-2] = 1650  # raw J in the normalized slot
    with pytest.raises(ValueError, match="pre-action z_ratio"):
        truth(row)


def test_rejects_shifted_trace_and_incomplete_attempt():
    row = episode("goal_reached", [110, 55, 0], [4, 2.5, 1], [0, 0, 0])
    row["trace"][0]["observation"][-2] = 55 / 550  # z_(t+1) leaked into obs_t
    with pytest.raises(ValueError, match="pre-action z_ratio"):
        truth(row)
    row["label"] = "cap_step"
    with pytest.raises(ValueError, match="not a complete"):
        truth(row)
