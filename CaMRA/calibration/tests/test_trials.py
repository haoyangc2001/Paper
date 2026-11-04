import pytest

from camra_calibration.commands import FAMILIES
from camra_calibration.protocol import Protocol
from camra_calibration.trials import (
    SPLIT_ORDER,
    build_all,
    build_trial_list,
    check_lists,
    hash_list,
    per_cell,
    profiles_for,
)

PROTO = Protocol.load()


def test_per_cell_matches_protocol():
    # 五 profile 的 split 与协议 per_cell 一致
    assert per_cell("calibration", PROTO) == 40
    assert per_cell("heldout", PROTO) == 20
    assert per_cell("near_obstacle", PROTO) == 10
    # pilot 自 v1.0.1 起覆盖五 profile：20 格 × 1 条
    assert per_cell("pilot", PROTO) == 1
    assert per_cell("baseline_fit", PROTO) == 60
    assert per_cell("baseline_dev", PROTO) == 20
    assert per_cell("baseline_validate", PROTO) == 20


def test_clean_only_splits_use_only_clean():
    for split in ("baseline_fit", "baseline_dev", "baseline_validate"):
        assert profiles_for(split, PROTO) == ("clean",)
    # pilot 覆盖五个 profile（协议 v1.0.1）
    assert profiles_for("pilot", PROTO) == tuple(PROTO.profile_ids())
    assert len(profiles_for("calibration", PROTO)) == 5


def test_quotas_and_cell_counts():
    assert len(build_trial_list("pilot", PROTO)) == 20
    assert len(build_trial_list("calibration", PROTO)) == 800
    assert len(build_trial_list("heldout", PROTO)) == 400
    assert len(build_trial_list("near_obstacle", PROTO)) == 200
    assert len(build_trial_list("baseline_fit", PROTO)) == 240


def test_calibration_covers_every_cell_40_times():
    trials = build_trial_list("calibration", PROTO)
    counts = {}
    for t in trials:
        counts[(t.family, t.profile)] = counts.get((t.family, t.profile), 0) + 1
    assert len(counts) == 20
    assert set(counts.values()) == {40}


def test_trial_ids_unique_and_well_formed():
    trials = build_trial_list("heldout", PROTO)
    ids = [t.trial_id for t in trials]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "heldout-straight_shift-clean-0000"
    for t in trials:
        split, family, profile, idx = t.trial_id.split("-")
        assert split == t.split and family == t.family and profile == t.profile
        assert len(idx) == 4 and idx.isdigit()


def test_seeds_follow_the_split_block():
    trials = build_trial_list("calibration", PROTO)
    start, count = PROTO.seed_block("calibration")
    assert [t.seed for t in trials] == list(range(start, start + count))


def test_list_hash_is_stable_and_order_sensitive():
    a = build_trial_list("pilot", PROTO)
    b = build_trial_list("pilot", PROTO)
    assert hash_list(a) == hash_list(b)
    assert hash_list(a) != hash_list(list(reversed(a)))


def test_all_lists_pass_cross_split_checks():
    lists = build_all(PROTO)
    assert set(lists) == set(SPLIT_ORDER)
    assert check_lists(lists, PROTO) == []
    total = sum(len(v) for v in lists.values())
    assert total == 1820


def test_no_seed_leaks_between_splits():
    lists = build_all(PROTO)
    seen = {}
    for split, trials in lists.items():
        for t in trials:
            assert t.seed not in seen, (t.seed, seen.get(t.seed), split)
            seen[t.seed] = split


def test_unknown_split_rejected():
    with pytest.raises(ValueError):
        build_trial_list("training", PROTO)


def test_program_hash_recorded_per_trial():
    trials = build_trial_list("pilot", PROTO)
    assert all(len(t.program_sha256) == 64 for t in trials)
    assert len({t.program_sha256 for t in trials}) == len(trials)


def test_strata_report_covers_every_cell_and_low_speed_floor():
    from camra_calibration.trials import strata_report
    lists = build_all(PROTO)
    rep = strata_report(lists, PROTO)
    assert set(rep["splits"]) == set(SPLIT_ORDER)
    for split, v in rep["splits"].items():
        assert v["all_cells_low_speed_ok"], split
        assert v["cells"] == (4 if split in ("baseline_fit", "baseline_dev", "baseline_validate") else 20)
        for cell in v["detail"].values():
            # 格级约束按平均占比判定；逐条最小值只作观测
            assert cell["low_speed_share_strict"] >= 0.25, (split, cell)
            assert cell["low_speed_share_literal"] >= cell["low_speed_share_strict"]
