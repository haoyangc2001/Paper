#!/usr/bin/env python3
"""CAL-018 故障注入校验（仅在工具 sandbox 内进行，不触碰正式数据）。

从一条真实完成的 trial 复制出若干副本，逐一注入故障，确认分类器给出**预期的
错误类别与 reason**，而不是静默通过：

  1. missing_dir     目录不存在                → data_invalid / missing_shard
  2. corrupt_npz     截断 pairing npz          → data_invalid / missing_shard
  3. nan_exec        exec_pos 注入 NaN         → data_invalid / nan
  4. time_jump       时间轴整体 +0.01 s        → data_invalid / time_jump
  5. duplicate       同名目录出现第二个        → duplicate_dirs 被检出
  6. business        改写 outcome=terminated   → business_failure、official=False
  7. resume          DONE 缺失                → resume_filter 会重新入列

任何一条不符合预期即退出码 2。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402

from camra_calibration.inspect import classify_trial_dir  # noqa: E402
from camra_calibration.pairing import detect_duplicates  # noqa: E402
from camra_calibration.store import TrialStore, resume_filter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-trial-dir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    src = Path(args.source_trial_dir)
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    results = []

    def record(name, expect_class, expect_reason, got) -> None:
        reasons = [i.reason for i in got.issues]
        ok = (got.error_class == expect_class and
              (expect_reason is None or expect_reason in reasons))
        results.append({"scenario": name, "expect_class": expect_class,
                        "expect_reason": expect_reason, "actual_class": got.error_class,
                        "actual_reasons": reasons, "official": got.official, "ok": ok})

    # 0) 基线：真实 trial 应可分类（正式）
    base_cls = classify_trial_dir(src)
    record("baseline", base_cls.error_class, None, base_cls)

    # 1) 目录不存在
    record("missing_dir", "data_invalid", "missing_shard",
           classify_trial_dir(work / "does_not_exist"))

    # 2) npz 截断
    d = work / "corrupt_npz"
    shutil.copytree(src, d)
    p = d / "pairing_10hz.npz"
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 3])
    record("corrupt_npz", "data_invalid", "missing_shard", classify_trial_dir(d))

    # 3) NaN 注入
    d = work / "nan_exec"
    shutil.copytree(src, d)
    pair = torch.load(d / "pairing_10hz.npz", map_location="cpu")
    pos = pair["exec_pos"].clone()
    pos[7, 0] = float("nan")
    pair["exec_pos"] = pos
    torch.save(pair, d / "pairing_10hz.npz")
    record("nan_exec", "data_invalid", "nan", classify_trial_dir(d))

    # 4) 时间轴跳变
    d = work / "time_jump"
    shutil.copytree(src, d)
    pair = torch.load(d / "pairing_10hz.npz", map_location="cpu")
    pair["t"] = pair["t"] + 0.01
    torch.save(pair, d / "pairing_10hz.npz")
    record("time_jump", "data_invalid", "time_jump", classify_trial_dir(d))

    # 5) 重复目录
    dup_root = work / "dup"
    dup_root.mkdir()
    shutil.copytree(src, dup_root / "trialX")
    shutil.copytree(src, dup_root / "trialX__a2")
    dup = detect_duplicates([p.name.split("__a")[0] for p in dup_root.iterdir() if p.is_dir()])
    results.append({"scenario": "duplicate", "expect": {"trialX": 2}, "actual": dup,
                    "ok": dup == {"trialX": 2}})

    # 6) 业务失败改写
    d = work / "business"
    shutil.copytree(src, d)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta["outcome"] = "terminated"
    meta["steps_executed"] = 5176
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    r = classify_trial_dir(d)
    record("business_failure", "business_failure", None, r)
    results.append({"scenario": "business_not_official", "expect": False,
                    "actual": r.official, "ok": r.official is False})

    # 7) 续采：删掉 DONE 后应重新入列
    store = TrialStore(work / "resume_root", "pilot")
    d, attempt = store.prepare("trialY")
    store.mark_done(d)
    before = store.is_done("trialY")
    (d / "DONE").unlink()
    after = store.is_done("trialY")
    todo, skipped = resume_filter([{"trial_id": "trialY"}], store)
    results.append({"scenario": "resume_after_done_removed",
                    "expect": {"done_before": True, "done_after": False, "requeued": 1},
                    "actual": {"done_before": before, "done_after": after, "requeued": len(todo)},
                    "ok": before and not after and len(todo) == 1 and skipped == 0})

    ok = all(r["ok"] for r in results)
    report = {"source": str(src), "workdir": str(work), "scenarios": results, "ok": ok}
    out = Path(args.report) if args.report else work / "fault_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print("VERDICT:", "pass" if ok else "fail")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
