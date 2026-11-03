"""Regression tests for the variant scheduler (TASK-031).

Pure Python: no Isaac Sim, no GPU. Run with::

    python -m pytest CaMRA/locomotion/tests/test_variant_scheduler.py -q

The last test reproduces the pre-fix behaviour (all environments start at
``ep_counter == 0``, the loop stops when the total episode count is reached),
which is why the delivered 300-episode runs only ever executed variant 0.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

LOCOMOTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCOMOTION / "source" / "unitree_rl_lab"))

from unitree_rl_lab.utils.variant_scheduler import (  # noqa: E402
    VariantScheduler,
    assign_variants,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_scenario(num_envs: int, num_variants: int, quota: int, *, lengths, max_steps: int | None = None):
    """Drive a fake simulation: every env runs episodes until the budget is met.

    ``lengths(rng)`` returns an episode length in steps; varying it makes the
    environments finish asynchronously, which is the realistic case.
    """
    rng = random.Random(0)
    sched = VariantScheduler(assign_variants(num_envs, num_variants), quota)
    remaining = [lengths(rng) for _ in range(num_envs)]
    steps = 0
    cap = max_steps if max_steps is not None else 10_000_000
    while not sched.complete() and steps < cap:
        steps += 1
        for env_id in range(num_envs):
            remaining[env_id] -= 1
            if remaining[env_id] <= 0:
                sched.record(env_id)
                remaining[env_id] = lengths(rng)
    return sched, steps


def fixed(n):
    return lambda rng: n


def jittered(rng):
    return rng.randint(5, 25)


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------


def test_assign_is_balanced_and_covers_every_variant():
    assert assign_variants(64, 4) == [i % 4 for i in range(64)]
    got = assign_variants(10, 4)
    sizes = [got.count(v) for v in range(4)]
    assert sizes == [3, 3, 2, 2]
    assert set(got) == {0, 1, 2, 3}


def test_assign_rejects_fewer_envs_than_variants():
    with pytest.raises(ValueError):
        assign_variants(3, 4)


def test_scheduler_rejects_variant_without_environment():
    # the protocol declares 3 variants but the assignment only populates 0 and 1
    with pytest.raises(ValueError):
        VariantScheduler([0, 0, 1], quota=5, num_variants=3)


def test_scheduler_rejects_bad_quota():
    with pytest.raises(ValueError):
        VariantScheduler([0, 1], quota=0)


# ---------------------------------------------------------------------------
# budgeting
# ---------------------------------------------------------------------------


def test_synchronous_environments_hit_exact_quota():
    sched, _ = run_scenario(64, 4, 30, lengths=fixed(10))
    assert sched.counts == [30, 30, 30, 30]
    assert sched.total_accepted == 120
    assert sched.complete()


def test_asynchronous_environments_hit_exact_quota():
    sched, _ = run_scenario(64, 4, 30, lengths=jittered)
    assert sched.counts == [30, 30, 30, 30]
    assert sched.complete()


def test_quota_not_divisible_by_envs_per_variant():
    # 64 envs / 4 variants = 16 slots per variant; quota 7 leaves surplus episodes
    sched, _ = run_scenario(64, 4, 7, lengths=jittered)
    assert sched.counts == [7, 7, 7, 7]
    cov = sched.coverage()
    assert cov["complete"] is True
    # 16 slots chasing 7 accepted episodes must produce surplus, all of it counted
    assert sum(cov["rejected_per_variant"]) >= 1


def test_fewer_envs_than_quota_recycles_until_quota():
    sched, _ = run_scenario(4, 4, 30, lengths=fixed(5))
    assert sched.counts == [30, 30, 30, 30]


def test_surplus_episodes_are_counted_not_dropped_silently():
    sched, _ = run_scenario(64, 4, 2, lengths=fixed(5))
    cov = sched.coverage()
    assert cov["accepted_per_variant"] == [2, 2, 2, 2]
    assert sum(cov["rejected_per_variant"]) > 0
    # every finished episode is accounted for: accepted + rejected == attempted
    for a, r, t in zip(cov["accepted_per_variant"], cov["rejected_per_variant"], cov["attempted_per_variant"]):
        assert a + r == t


def test_episode_ids_are_contiguous_per_variant():
    seen: dict[int, list[int]] = {v: [] for v in range(4)}
    original = VariantScheduler.record

    def spy(self, env_id):
        rec = original(self, env_id)
        seen[rec["variant"]].append(rec["episode_id"])
        return rec

    VariantScheduler.record = spy
    try:
        run_scenario(64, 4, 12, lengths=jittered)
    finally:
        VariantScheduler.record = original
    for v, ids in seen.items():
        assert ids == list(range(len(ids))), f"variant {v} episode ids not contiguous"


# ---------------------------------------------------------------------------
# incomplete / coverage reporting
# ---------------------------------------------------------------------------


def test_guard_stop_reports_missing_variants():
    # cap too low for the budget: the run must end "incomplete", not "pass"
    sched, steps = run_scenario(64, 4, 300, lengths=fixed(50), max_steps=100)
    assert not sched.complete()
    assert steps == 100
    assert sched.missing()  # at least one variant short of quota
    cov = sched.coverage()
    assert cov["complete"] is False
    assert cov["missing"]


def test_uncovered_variant_is_visible():
    # 6 envs / 3 variants -> env {0,3}=v0, {1,4}=v1, {2,5}=v2; variant 2 never finishes
    sched = VariantScheduler(assign_variants(6, 3), quota=5, num_variants=3)
    for _ in range(5):
        for env_id in (0, 3, 1, 4):
            sched.record(env_id)
    assert sched.uncovered_variants() == [2]
    assert sched.missing() == {2: 5}
    assert sched.complete() is False


# ---------------------------------------------------------------------------
# the bug this task fixes
# ---------------------------------------------------------------------------


def test_old_allocator_starves_variants_when_num_envs_exceeds_target():
    """Pre-fix behaviour: ``(ep_counter % n) == vi`` with every counter at 0.

    With 64 environments and a target of 30 episodes the loop stopped after the
    first cohort, so only variant 0 was ever executed - the exact defect found
    in TASK-029.
    """
    num_envs, num_variants, wanted = 64, 4, 30
    ep_counter = [0] * num_envs
    done = 0
    executed = {v: 0 for v in range(num_variants)}
    finished = [False] * num_envs
    while done < wanted:
        for env_id in range(num_envs):
            vi = ep_counter[env_id] % num_variants
            executed[vi] += 1
            if not finished[env_id]:
                finished[env_id] = True
                done += 1
                ep_counter[env_id] += 1
        if all(finished):
            break
    assert executed[0] == num_envs and sum(executed.values()) == num_envs
    assert [executed[v] for v in range(1, num_variants)] == [0, 0, 0]

    # the fixed scheduler covers all four variants with the same 30-episode target
    sched, _ = run_scenario(num_envs, num_variants, wanted, lengths=fixed(10))
    assert sched.counts == [wanted] * num_variants
