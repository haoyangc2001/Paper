"""trial 清单生成（CAL-012）。

按冻结协议把每个 split 展开成固定配额的 trial 序列：
  * 20 个 cell = 4 命令族 × 5 profile（baseline 与 pilot 只用 clean）；
  * 每 cell 条数 = 该 split 配额 / 20，整除校验不通过直接报错；
  * seed 从该 split 的种子块**顺序**分配，因此同一 split 的 seed 集合与
    并行度、断点恢复、分批顺序都无关；
  * trial_id 形如 ``<split>-<family>-<profile>-<NNNN>``，NNNN 为 cell 内序号；
  * 输出顺序固定：族 → profile → cell 内序号。

``build_trial_list`` 是 (split, protocol) 的纯函数；``hash_list`` 给出整表散列，
用于把正式采集与清单绑定。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Iterable

from .commands import FAMILIES, build_program
from .protocol import Protocol, ProtocolError

#: baseline 与 pilot 只用 clean profile（协议 split.baseline.profile = "clean only"）
CLEAN_ONLY_SPLITS = ("pilot", "baseline_fit", "baseline_dev", "baseline_validate")

SPLIT_ORDER = ("pilot", "baseline_fit", "baseline_dev", "baseline_validate",
               "calibration", "heldout", "near_obstacle")


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    split: str
    family: str
    profile: str
    seed: int
    attempt: int
    command_dt: float
    duration_s: float
    program_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)


def proto_split_key(split: str) -> str:
    """把 baseline_fit/dev/validate 归一到协议里的 baseline 键。"""
    return "baseline" if split.startswith("baseline_") else split


def profiles_for(split: str, protocol: Protocol) -> tuple[str, ...]:
    if split in CLEAN_ONLY_SPLITS:
        return ("clean",)
    return tuple(protocol.profile_ids())


def per_cell(split: str, protocol: Protocol) -> int:
    """该 split 每个 cell 的条数 = 配额 / 实际格数。

    格数按**实际 profile 组合**算：clean-only 的 split 只有 4 个族格，而不是协议
    ``split.cells = 20`` 的 20 格。协议里 ``baseline.per_cell = 20`` 与
    ``baseline.profile = "clean only"`` 互相矛盾（4 × 20 = 80 ≠ 400），此处以
    总额 400/240/80/80/20 为准反推格内条数，并把矛盾记录在案，详见
    docs/implementation_handoff.md §14.3。
    """
    quota = protocol.quota(split)
    cells = len(FAMILIES) * len(profiles_for(split, protocol))
    if cells <= 0 or quota % cells != 0:
        raise ProtocolError(f"{split}: 配额 {quota} 不能被实际格数 {cells} 整除")
    return quota // cells


def build_trial_list(split: str, protocol: Protocol | None = None) -> list[TrialSpec]:
    proto = protocol or Protocol.load()
    if split not in SPLIT_ORDER:
        raise ValueError(f"未知 split {split!r}（允许 {SPLIT_ORDER}）")
    if proto_split_key(split) not in proto.payload["split"]:
        raise ProtocolError(f"协议没有定义 split {split!r}")
    n = per_cell(split, proto)
    start, count = proto.seed_block(split)
    clock = proto.payload["clocks"]
    duration = float(clock["measured_window_s"])
    # 命令点 = duration/command_dt + 1，最后一点的命令不参与积分，因此程序时长取 duration
    command_dt = float(clock["command_dt_s"])

    out: list[TrialSpec] = []
    k = 0
    for family in FAMILIES:
        for profile in profiles_for(split, proto):
            for i in range(n):
                if k >= count:
                    raise ProtocolError(f"{split}: 生成的 trial 数超过种子块容量 {count}")
                seed = start + k
                prog = build_program(family, seed, duration_s=duration, command_dt=command_dt)
                out.append(TrialSpec(
                    trial_id=f"{split}-{family}-{profile}-{i:04d}",
                    split=split, family=family, profile=profile,
                    seed=seed, attempt=1,
                    command_dt=command_dt, duration_s=duration,
                    program_sha256=prog.sha256(),
                ))
                k += 1
    if len(out) != proto.quota(split):
        raise ProtocolError(f"{split}: 生成 {len(out)} 条，与配额 {proto.quota(split)} 不符")
    if k != count:
        raise ProtocolError(f"{split}: 用掉 {k} 个 seed，但种子块只分配了 {count} 个")
    return out


def build_all(protocol: Protocol | None = None) -> dict[str, list[TrialSpec]]:
    proto = protocol or Protocol.load()
    return {split: build_trial_list(split, proto) for split in SPLIT_ORDER}


def hash_list(trials: Iterable[TrialSpec]) -> str:
    payload = [t.as_dict() for t in trials]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def check_lists(lists: dict[str, list[TrialSpec]], protocol: Protocol | None = None) -> list[str]:
    """整体自检：重复、跨 split 泄漏、seed 归属、配额与格覆盖。"""
    proto = protocol or Protocol.load()
    problems: list[str] = []

    ids: dict[str, str] = {}
    seeds: dict[int, str] = {}
    cell_counts: dict[tuple[str, str, str], int] = {}

    for split, trials in lists.items():
        expect = proto.quota(split)
        if len(trials) != expect:
            problems.append(f"{split}: {len(trials)} 条，配额 {expect}")
        start, count = proto.seed_block(split)
        block = set(range(start, start + count))
        used = set()
        for t in trials:
            if t.trial_id in ids:
                problems.append(f"trial_id 重复: {t.trial_id}（{ids[t.trial_id]} 与 {split}）")
            ids[t.trial_id] = split
            if t.seed in seeds:
                problems.append(f"seed 跨 split 泄漏: {t.seed}（{seeds[t.seed]} 与 {split}）")
            seeds[t.seed] = split
            if t.seed not in block:
                problems.append(f"{split}: seed {t.seed} 不在 [{start}, {start + count})")
            used.add(t.seed)
            if t.split != split:
                problems.append(f"{t.trial_id}: split 字段与所在列表不一致")
            cell_counts[(split, t.family, t.profile)] = cell_counts.get((split, t.family, t.profile), 0) + 1
        if len(used) != len(trials):
            problems.append(f"{split}: seed 在列表内重复（{len(used)} 唯一 / {len(trials)} 条）")

    # 每格条数一致且等于 per_cell
    for split in lists:
        n = per_cell(split, proto)
        prots = profiles_for(split, proto)
        for family in FAMILIES:
            for profile in prots:
                got = cell_counts.get((split, family, profile), 0)
                if got != n:
                    problems.append(f"{split}/{family}/{profile}: {got} 条，应为 {n}")
    return problems


def strata_report(lists: dict[str, list[TrialSpec]], protocol: Protocol | None = None) -> dict:
    """分层统计：每个 split × cell 的条数、低速达标情况与方向覆盖。

    低速口径按更严格的"且"（|vx|<=0.3 且 |wz|<=0.3）判定，同时给出协议字面"或"的占比，
    两种读数并列，便于核对而不必猜测采用了哪一种。
    """
    proto = protocol or Protocol.load()
    out: dict = {"low_speed_definition": {"strict": "|vx|<=0.3 and |wz|<=0.3",
                                          "literal": "|vx|<=0.3 or |wz|<=0.3",
                                          "min_share_of_cell": 0.25},
                 "splits": {}}
    for split, trials in lists.items():
        cells: dict[str, dict] = {}
        for t in trials:
            prog = build_program(t.family, t.seed, duration_s=t.duration_s, command_dt=t.command_dt)
            key = f"{t.family}/{t.profile}"
            cell = cells.setdefault(key, {
                "n": 0, "low_speed_strict_sum": 0.0, "low_speed_literal_sum": 0.0,
                "low_speed_strict_min": 1.0, "low_speed_literal_min": 1.0,
                "has_forward": False, "has_backward": False, "has_lateral": False,
                "has_yaw": False, "has_stationary": False, "has_boundary": False,
            })
            cell["n"] += 1
            strict = prog.low_speed_share(True)
            literal = prog.low_speed_share(False)
            cell["low_speed_strict_sum"] += strict
            cell["low_speed_literal_sum"] += literal
            cell["low_speed_strict_min"] = min(cell["low_speed_strict_min"], strict)
            cell["low_speed_literal_min"] = min(cell["low_speed_literal_min"], literal)
            for vx, vy, wz in prog.commands:
                cell["has_forward"] |= vx > 0.05
                cell["has_backward"] |= vx < -0.05
                cell["has_lateral"] |= abs(vy) > 0.05
                cell["has_yaw"] |= abs(wz) > 0.05
                cell["has_stationary"] |= (abs(vx) <= 1e-9 and abs(vy) <= 1e-9 and abs(wz) <= 1e-9)
                cell["has_boundary"] |= abs(abs(vx) - 1.0) < 1e-9 or abs(abs(wz) - 1.0) < 1e-9
        for k, v in cells.items():
            n = v["n"]
            share_strict = v["low_speed_strict_sum"] / n
            share_literal = v["low_speed_literal_sum"] / n
            # min_share_of_cell 是格级约束：按格内平均占比判定，逐条最小值只作观测
            v["low_speed_share_strict"] = round(share_strict, 6)
            v["low_speed_share_literal"] = round(share_literal, 6)
            v["low_speed_strict_min"] = round(v["low_speed_strict_min"], 6)
            v["low_speed_literal_min"] = round(v["low_speed_literal_min"], 6)
            v["low_speed_ok"] = share_strict >= 0.25
            for k2 in ("low_speed_strict_sum", "low_speed_literal_sum"):
                del v[k2]
        cells = {k: v for k, v in sorted(cells.items())}
        low_ok = all(c["low_speed_ok"] for c in cells.values())
        out["splits"][split] = {
            "cells": len(cells),
            "trials": len(trials),
            "all_cells_low_speed_ok": low_ok,
            "detail": cells,
        }
    return out
