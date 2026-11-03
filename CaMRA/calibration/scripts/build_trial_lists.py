#!/usr/bin/env python3
"""写出各 split 的 trial 清单与锁文件（CAL-012）。

每行一个 trial（JSONL），另有 lock.json 记录各 split 的条数与整表散列。
清单是 (split, protocol) 的纯函数，重复运行必然得到相同文件。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from camra_calibration.protocol import Protocol          # noqa: E402
from camra_calibration.trials import (  # noqa: E402
    build_all, check_lists, hash_list, strata_report,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出目录")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    proto = Protocol.load()
    lists = build_all(proto)
    problems = check_lists(lists, proto)
    if problems:
        print("SELF-CHECK FAILED")
        for p in problems:
            print("  -", p)
        return 2

    lock = {"protocol_version": proto.version, "protocol_sha256": proto.sha256, "splits": {}}
    for split, trials in lists.items():
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for t in trials:
                fh.write(json.dumps(t.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        lock["splits"][split] = {"count": len(trials), "sha256": hash_list(trials)}
        print(f"  {split:<18} {len(trials):>4} trials  {lock['splits'][split]['sha256'][:16]}")

    strata = strata_report(lists, proto)
    (out / "strata.json").write_text(
        json.dumps(strata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bad = [s for s, v in strata["splits"].items() if not v["all_cells_low_speed_ok"]]
    if bad:
        print(f"STRATA WARNING: 低速下限未达标的 split: {bad}")

    total = sum(v["count"] for v in lock["splits"].values())
    lock["total_trials"] = total
    lock["self_check"] = "pass"
    (out / "lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"total {total} trials; lock written to {out / 'lock.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
