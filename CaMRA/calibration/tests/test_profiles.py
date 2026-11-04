import pytest

from camra_calibration.profiles import (
    CLEAN_FRICTION,
    CLEAN_MASS,
    FRICTION_RANGE,
    MASS_RANGE,
    PROFILE_IDS,
    PROFILE_KNOBS,
    PUSH_INTERVAL_RANGE,
    PUSH_VEL_RANGE,
    TRAINING_OBS_NOISE,
    AppliedProfile,
    knob_diff,
    sample_applied,
)


def test_sample_is_deterministic_and_salt_separated():
    a = sample_applied("friction_only", 6000)
    b = sample_applied("friction_only", 6000)
    assert a == b and a.sha256() == b.sha256()
    # 与其他 profile / 其他 seed 不同
    assert sample_applied("friction_only", 6001) != a
    assert sample_applied("mass_only", 6000).friction == CLEAN_FRICTION


def test_all_profiles_build_and_pass_invariants():
    for pid in PROFILE_IDS:
        for seed in (5000, 6000, 8000, 12345):
            assert sample_applied(pid, seed).check_invariants() == [], (pid, seed)


def test_clean_is_exactly_the_reference():
    a = sample_applied("clean", 6000)
    assert a.friction == 1.0 and a.added_mass_kg == 0.0
    assert a.push_events == () and a.obs_noise is None
    assert a.knobs_changed() == frozenset()


def test_single_knob_profiles_change_only_their_knob():
    assert sample_applied("friction_only", 1).knobs_changed() == frozenset({"friction"})
    assert sample_applied("mass_only", 1).knobs_changed() == frozenset({"added_mass"})
    assert sample_applied("push_only", 1).knobs_changed() == frozenset({"push_events"})
    assert sample_applied("combined", 1).knobs_changed() == \
        frozenset({"friction", "added_mass", "push_events", "obs_noise"})


def test_friction_and_mass_inside_protocol_ranges():
    for seed in range(50):
        f = sample_applied("friction_only", seed).friction
        assert FRICTION_RANGE[0] <= f <= FRICTION_RANGE[1]
        m = sample_applied("mass_only", seed).added_mass_kg
        assert MASS_RANGE[0] <= m <= MASS_RANGE[1]


def test_push_events_within_window_and_bounds():
    for seed in range(50):
        a = sample_applied("push_only", seed, duration_s=30.0)
        assert len(a.push_events) >= 1
        for i in range(len(a.push_events) - 1):
            dt = a.push_events[i + 1][0] - a.push_events[i][0]
            assert PUSH_INTERVAL_RANGE[0] <= dt <= PUSH_INTERVAL_RANGE[1]
        for t, vx, vy in a.push_events:
            assert 0.0 < t <= 30.0
            assert PUSH_VEL_RANGE[0] <= vx <= PUSH_VEL_RANGE[1]
            assert PUSH_VEL_RANGE[0] <= vy <= PUSH_VEL_RANGE[1]


def test_obs_noise_equals_training_values_for_combined_only():
    assert sample_applied("combined", 1).obs_noise == TRAINING_OBS_NOISE
    for pid in ("clean", "friction_only", "mass_only", "push_only"):
        assert sample_applied(pid, 1).obs_noise is None


def test_manifest_record_shape():
    a = sample_applied("combined", 7000)
    rec = a.to_manifest_record()
    assert set(rec) == {"friction", "added_mass_kg", "push_events", "obs_noise"}
    assert rec["obs_noise"] == {k: list(v) for k, v in TRAINING_OBS_NOISE.items()}
    clean = sample_applied("clean", 7000).to_manifest_record()
    assert clean["obs_noise"] == "disabled" and clean["push_events"] == []


def test_knob_diff_matrix():
    d = knob_diff()
    assert set(d) == set(PROFILE_IDS)
    assert d["clean"] == {k: "unchanged" for k in d["clean"]}
    assert d["friction_only"]["friction"] == "varied"
    assert d["friction_only"]["added_mass"] == "unchanged"
    assert d["combined"]["obs_noise"] == "varied"


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        sample_applied("nope", 1)


def test_bad_duration_rejected():
    with pytest.raises(ValueError):
        sample_applied("clean", 1, duration_s=0.0)


def test_restitution_stays_clean_for_all_profiles():
    for pid in PROFILE_IDS:
        assert sample_applied(pid, 42).restitution == 0.0


def test_initial_pose_deterministic_within_protocol_range():
    from camra_calibration.profiles import PLANE_POSE_RANGE, sample_initial_pose
    a = sample_initial_pose(6000)
    b = sample_initial_pose(6000)
    assert a == b
    assert -1.0 <= a["px"] <= 1.0 and -1.0 <= a["py"] <= 1.0
    assert -3.141592653589793 <= a["yaw"] <= 3.141592653589793
    assert sample_initial_pose(6001) != a
