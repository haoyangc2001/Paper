"""CAL-017 批次规划、断点续采与派生统计不变性测试（test_fixture）。"""
import json

import pytest

from camra_calibration.store import (
    TrialStore, flatten_batches, plan_batches, resume_filter,
)


def trials(n=20):
    return [{"trial_id": f"t{i:03d}", "seed": 6000 + i, "profile": "clean"} for i in range(n)]


def fake_npz(path, payload):
    path.write_text(json.dumps(sorted(payload)), encoding="utf-8")


# ---------------- plan_batches ----------------
def test_plan_batches_preserves_order_and_size():
    ts = trials(20)
    for n_env in (1, 7, 8, 32):
        batches = plan_batches(ts, n_env)
        assert flatten_batches(batches) == ts
        assert all(len(b) <= n_env for b in batches)
        assert sum(len(b) for b in batches) == len(ts)


def test_plan_batches_empty_and_invalid():
    assert plan_batches([], 8) == []
    with pytest.raises(ValueError):
        plan_batches(trials(3), 0)


def test_batch_size_does_not_change_derived_records(tmp_path):
    """不同批大小写出的 records 与派生统计必须完全一致。"""
    ts = trials(20)

    def run(n_env, root):
        store = TrialStore(root, "split", writer=fake_npz)
        todo, _ = resume_filter(ts, store)
        for batch in plan_batches(todo, n_env):
            for tr in batch:
                d, attempt = store.prepare(tr["trial_id"])
                fake_npz(d / "payload", {"x": tr["trial_id"]})
                store.append_record({"trial_id": tr["trial_id"], "seed": tr["seed"],
                                     "attempt": attempt, "j_exec": 100.0 + tr["seed"] % 7})
                store.mark_done(d)
        return store

    s8 = run(8, tmp_path / "a")
    s32 = run(32, tmp_path / "b")
    assert s8.read_records() == s32.read_records()
    assert s8.done_ids() == s32.done_ids() == {t["trial_id"] for t in ts}


# ---------------- resume_filter 幂等 ----------------
def test_resume_filter_skips_done_and_is_idempotent(tmp_path):
    ts = trials(10)
    store = TrialStore(tmp_path, "s", writer=fake_npz)
    for tr in ts[:4]:
        d, _ = store.prepare(tr["trial_id"])
        store.mark_done(d)
    todo1, skipped1 = resume_filter(ts, store)
    todo2, skipped2 = resume_filter(ts, store)
    assert [t["trial_id"] for t in todo1] == [t["trial_id"] for t in ts[4:]]
    assert todo1 == todo2 and skipped1 == skipped2 == 4


def test_resume_filter_allow_overwrite_returns_all(tmp_path):
    ts = trials(3)
    store = TrialStore(tmp_path, "s", writer=fake_npz)
    for tr in ts:
        d, _ = store.prepare(tr["trial_id"])
        store.mark_done(d)
    todo, skipped = resume_filter(ts, store, allow_overwrite=True)
    assert len(todo) == 3 and skipped == 0


# ---------------- 不覆盖旧分片 ----------------
def test_prepare_first_attempt_then_suffix(tmp_path):
    store = TrialStore(tmp_path, "s", writer=fake_npz)
    d1, a1 = store.prepare("t000")
    assert a1 == 1 and d1.name == "t000"
    d2, a2 = store.prepare("t000")          # 已存在且无 DONE
    assert a2 == 2 and d2.name == "t000__a2"
    d3, a3 = store.prepare("t000")
    assert a3 == 3 and d3.name == "t000__a3"
    # 原目录仍在
    assert (store.base / "t000").is_dir()


def test_failed_attempt_then_resume_creates_new_attempt(tmp_path):
    store = TrialStore(tmp_path, "s", writer=fake_npz)
    d, attempt = store.prepare("t001")
    fake_npz(d / "partial", {"x": 1})       # 模拟中途崩溃：无 DONE
    assert not store.is_done("t001")
    todo, skipped = resume_filter([{"trial_id": "t001"}], store)
    assert len(todo) == 1 and skipped == 0
    d2, attempt2 = store.prepare("t001")
    assert attempt2 == 2 and d2.name == "t001__a2"
    store.mark_done(d2)
    assert store.is_done("t001")
    assert store.done_ids() == {"t001"}


def test_done_in_suffixed_dir_counts(tmp_path):
    store = TrialStore(tmp_path, "s", writer=fake_npz)
    d, _ = store.prepare("t002")
    d2, _ = store.prepare("t002")
    store.mark_done(d2)
    assert store.is_done("t002")
    assert store.trial_dir("t002").name == "t002"      # 主目录存在时优先返回它


# ---------------- 原子写与追加 ----------------
def test_write_json_leaves_no_tmp(tmp_path):
    store = TrialStore(tmp_path, "s", writer=fake_npz)
    d, _ = store.prepare("t003")
    store.write_json(d, "meta.json", {"a": 1})
    assert (d / "meta.json").is_file()
    assert not list(d.glob(".*.tmp"))


def test_append_record_never_truncates(tmp_path):
    store = TrialStore(tmp_path, "s", writer=fake_npz)
    store.append_record({"trial_id": "a"})
    store.append_record({"trial_id": "b"})
    again = TrialStore(tmp_path, "s", writer=fake_npz)   # 重开（模拟恢复）
    again.append_record({"trial_id": "c"})
    assert [r["trial_id"] for r in again.read_records()] == ["a", "b", "c"]


# ---------------- 配额与 seed 独立性（跨 split） ----------------
def test_split_seed_blocks_are_disjoint_and_quota_exact():
    from camra_calibration.protocol import Protocol
    from camra_calibration.trials import build_all

    proto = Protocol.load()
    lists = build_all(proto)
    seen = {}
    for split, ts in lists.items():
        assert len(ts) == proto.quota(split)
        start, count = proto.seed_block(split)
        for t in ts:
            assert start <= t.seed < start + count
            assert t.seed not in seen, (t.seed, seen.get(t.seed), split)
            seen[t.seed] = split
    assert len(seen) == 1820


def test_batch_size_does_not_change_trial_list():
    from camra_calibration.protocol import Protocol
    from camra_calibration.trials import build_trial_list, hash_list

    trials_list = build_trial_list("calibration", Protocol.load())
    for n_env in (8, 32):
        batches = plan_batches(trials_list, n_env)
        assert hash_list(flatten_batches(batches)) == hash_list(trials_list)
