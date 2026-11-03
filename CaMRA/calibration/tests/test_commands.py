import pytest

from camra_calibration.commands import FAMILIES, LIMITS, build_program


def prog(family="straight_shift", seed=6000):
    return build_program(family, seed)


def test_all_families_build_with_301_points():
    for fam in FAMILIES:
        p = prog(fam)
        assert len(p.commands) == 301, fam
        assert p.n_intervals == 300
        assert len(p.targets) == 30


def test_commands_stay_inside_domain():
    for fam in FAMILIES:
        p = prog(fam)
        for vx, vy, wz in p.commands:
            assert -1.0 <= vx <= 1.0
            assert -0.4 <= vy <= 0.4
            assert -1.0 <= wz <= 1.0


def test_straight_shift_has_no_yaw_rate():
    assert all(wz == 0.0 for _vx, _vy, wz in prog("straight_shift").commands)


def test_start_stop_turn_contains_stops_and_pure_turns():
    p = prog("start_stop_turn")
    assert any(vx == 0 and vy == 0 and wz == 0 for vx, vy, wz in p.commands)
    assert any(vx == 0 and vy == 0 and wz != 0 for vx, vy, wz in p.commands)


def test_arc_combined_has_yaw_rate():
    assert any(wz != 0 for _vx, _vy, wz in prog("arc_combined").commands)


def test_deterministic_for_same_seed():
    a, b = prog("arc_combined", 6000), prog("arc_combined", 6000)
    assert a.commands == b.commands
    assert a.sha256() == b.sha256()


def test_different_seed_changes_program():
    assert prog("arc_combined", 6000).sha256() != prog("arc_combined", 6001).sha256()


def test_family_changes_program():
    assert prog("arc_combined", 6000).sha256() != prog("straight_shift", 6000).sha256()


def test_ramp_starts_from_previous_target():
    p = prog("straight_shift", 6000)
    # 段边界（t=1.0 s，第 10 点）仍是上一段目标；0.2 s 后（第 12 点）到达新目标
    assert p.commands[10] == p.targets[0]
    assert p.commands[12] == p.targets[1]
    # 中间点处于两者之间
    mid = p.commands[11]
    for i in range(3):
        lo, hi = sorted((p.targets[0][i], p.targets[1][i]))
        assert lo - 1e-12 <= mid[i] <= hi + 1e-12


def test_low_speed_strata_are_forced():
    for fam in FAMILIES:
        p = prog(fam)
        assert p.low_speed_share(strict=True) >= 0.25, (fam, p.low_speed_share(True))


def test_unknown_family_rejected():
    with pytest.raises(ValueError):
        build_program("nope", 1)


def test_as_dict_reports_both_low_speed_readings():
    d = prog("straight_shift").as_dict()
    assert d["n_points"] == 301
    assert 0.0 <= d["low_speed_share_strict"] <= 1.0
    assert d["low_speed_share_literal"] >= d["low_speed_share_strict"]
    assert len(d["sha256"]) == 64
