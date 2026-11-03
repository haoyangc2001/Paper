"""Variant-aware episode budgeting for the evaluator (TASK-031).

Why this module exists
----------------------
The original evaluator decided each environment's command variant from an
episode counter that starts at zero:

    sel = (ep_counter % len(variants)) == vi

At the first step every environment has ``ep_counter == 0``, so *all* of them
run variant 0. The loop then stopped as soon as the total number of episodes
reached the per-scenario target (``while len(done_rows) < wanted``). When
``num_envs`` (64) exceeded that target (30), the run finished before any
environment had completed a second episode, so variants 1..n were never
executed - while the summary still reported "300 episodes, PASS".

This module fixes the scheduling only. It is pure Python (no Isaac, no torch)
so the allocation rules can be regression-tested without a simulator:

* every variant is bound to a fixed, balanced subset of environments for the
  whole scenario, so no variant can be starved of parallel slots;
* the loop terminates when *each* variant has reached its own quota, not when
  the total does;
* quotas need not divide ``num_envs`` or the number of environments per
  variant, and environments that fail early simply recycle until the quota is
  met;
* surplus episodes beyond a variant's quota are counted (``rejected``) and
  dropped, so the recorded data is exactly ``quota`` per variant.
"""

from __future__ import annotations


def assign_variants(num_envs: int, num_variants: int) -> list[int]:
    """Assign environment ``i`` to variant ``i % num_variants``.

    The subsets differ in size by at most one, and every variant gets at least
    one environment when ``num_envs >= num_variants``. Fewer environments than
    variants is a configuration error: some variant could never run, which is
    exactly the silent-coverage bug this module exists to prevent.
    """
    num_envs = int(num_envs)
    num_variants = int(num_variants)
    if num_variants < 1:
        raise ValueError(f"num_variants must be >= 1, got {num_variants}")
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    if num_envs < num_variants:
        raise ValueError(
            f"num_envs ({num_envs}) < num_variants ({num_variants}): "
            "at least one variant would get no environment and could never be evaluated"
        )
    return [i % num_variants for i in range(num_envs)]


class VariantScheduler:
    """Per-variant episode budget.

    ``record`` is called once per finished episode (including episodes that
    ended by falling over). It returns a record describing whether the episode
    was accepted into the dataset; rejected episodes are still counted so the
    report can prove nothing was silently dropped.
    """

    def __init__(self, variant_of_env: list[int], quota: int, num_variants: int | None = None):
        if not variant_of_env:
            raise ValueError("variant_of_env must not be empty")
        quota = int(quota)
        if quota < 1:
            raise ValueError(f"quota must be >= 1, got {quota}")
        self.variant_of_env = [int(v) for v in variant_of_env]
        if min(self.variant_of_env) < 0:
            raise ValueError("variant ids must be non-negative")
        declared = max(self.variant_of_env) + 1 if num_variants is None else int(num_variants)
        if declared < max(self.variant_of_env) + 1:
            raise ValueError(
                f"num_variants ({declared}) is smaller than the largest variant id present "
                f"({max(self.variant_of_env)})"
            )
        self.num_variants = declared
        missing = set(range(self.num_variants)) - set(self.variant_of_env)
        if missing:
            raise ValueError(f"variants {sorted(missing)} have no environment")
        self.quota = quota
        self._accepted = [0] * self.num_variants
        self._rejected = [0] * self.num_variants
        self._attempted = [0] * self.num_variants
        self._envs = [0] * self.num_variants
        for v in self.variant_of_env:
            self._envs[v] += 1

    # -- queries ---------------------------------------------------------
    def variant_for(self, env_id: int) -> int:
        return self.variant_of_env[int(env_id)]

    def accepted(self, variant: int) -> int:
        return self._accepted[int(variant)]

    @property
    def counts(self) -> list[int]:
        return list(self._accepted)

    @property
    def total_accepted(self) -> int:
        return sum(self._accepted)

    @property
    def target_total(self) -> int:
        return self.quota * self.num_variants

    def complete(self) -> bool:
        return all(c >= self.quota for c in self._accepted)

    def missing(self) -> dict[int, int]:
        return {v: self.quota - c for v, c in enumerate(self._accepted) if c < self.quota}

    def uncovered_variants(self) -> list[int]:
        """Variants that produced no accepted episode at all."""
        return [v for v, c in enumerate(self._accepted) if c == 0]

    # -- mutation --------------------------------------------------------
    def record(self, env_id: int) -> dict:
        """Register one finished episode of ``env_id``; returns the decision."""
        env_id = int(env_id)
        v = self.variant_for(env_id)
        episode_id = self._attempted[v]
        self._attempted[v] += 1
        if self._accepted[v] < self.quota:
            self._accepted[v] += 1
            accepted = True
        else:
            self._rejected[v] += 1
            accepted = False
        return {
            "variant": v,
            "episode_id": episode_id,
            "accepted": accepted,
            "accepted_count": self._accepted[v],
        }

    # -- reporting -------------------------------------------------------
    def coverage(self) -> dict:
        return {
            "quota_per_variant": self.quota,
            "num_variants": self.num_variants,
            "envs_per_variant": list(self._envs),
            "accepted_per_variant": list(self._accepted),
            "rejected_per_variant": list(self._rejected),
            "attempted_per_variant": list(self._attempted),
            "complete": self.complete(),
            "missing": {str(k): v for k, v in self.missing().items()},
            "uncovered_variants": self.uncovered_variants(),
        }
