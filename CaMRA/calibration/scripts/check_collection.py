#!/usr/bin/env python3
"""CAL-015 采集结果检查（不启动仿真）。

对照清单核对：
  * 覆盖：清单里该 split（可选 profile 过滤）的每条 trial 是否都有 DONE 目录；
  * 重复/缺失：records.jsonl 与目录是否一致；多次 attempt（`__aN`）记为重试历史而**不是**重复，
    真正的重复只有"同一 trial 有多个 DONE 目录"或"同一 (trial, attempt) 写了两次 record"；
  * 形状与口径：200 Hz 6000 步、50 Hz 1500 条、配对 301 点（含端点）、command 表长度；
  * 能量：用落盘 tau/qdot 离线复算并与 energy.json 比对（容差 1e-5）；
  * 元数据：sim_dt/decimation/protocol_sha256/初始位姿/outcome/steps 是否齐全且自洽。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402

from camra_calibration.energy import integrate_series  # noqa: E402

REQUIRED_META = ("trial_id", "split", "family", "profile", "seed", "attempt", "controller",
                 "protocol_version", "protocol_sha256", "program_sha256", "outcome",
                 "steps_executed", "warmup_steps", "pairing_points", "profile_applied",
                 "initial_state", "sim_dt", "policy_dt", "decimation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--expect-steps", type=int, default=6000)
    ap.add_argument("--expect-control", type=int, default=1500)
    ap.add_argument("--expect-pairs", type=int, default=301)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    manifest = [json.loads(l) for l in
                (Path(args.manifest_dir) / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    if args.profiles:
        manifest = [t for t in manifest if t["profile"] in args.profiles]
    if args.limit:
        manifest = manifest[:args.limit]
    expected = [t["trial_id"] for t in manifest]

    base = Path(args.out_root) / args.split

    def _attempt(p: Path) -> int:
        """目录名 `<trial>` 记作 attempt 1，`<trial>__aN` 记作 N。"""
        return int(p.name.split("__a")[-1]) if "__a" in p.name else 1

    # 目录索引按 attempt 归并：多次 attempt 是"重试历史"，不是重复 trial。
    # 每个 trial 只取**含 DONE 的最大 attempt**作为正式产物（与 fit_power_model 的装载器一致）。
    attempts: dict[str, list[int]] = {}
    done_dirs: dict[str, list[Path]] = {}
    dirs: dict[str, Path] = {}
    if base.is_dir():
        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue
            tid = p.name.split("__a")[0]
            n = _attempt(p)
            attempts.setdefault(tid, []).append(n)
            if (p / "DONE").is_file():
                done_dirs.setdefault(tid, []).append(p)
                cur = dirs.get(tid)
                if cur is None or _attempt(cur) < n:
                    dirs[tid] = p
    done = set(dirs)
    records = []
    rec_path = base / "records.jsonl"
    if rec_path.is_file():
        records = [json.loads(l) for l in rec_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    problems: list[str] = []
    missing = [t for t in expected if t not in done]
    if missing:
        problems.append(f"缺失 {len(missing)} 条: {missing[:5]}")
    # 真正的重复只有两种：同一 trial 有多个 DONE 目录，或同一 (trial, attempt) 写了两次 record。
    dup = sorted(tid for tid, ps in done_dirs.items() if len(ps) > 1)
    if dup:
        problems.append(f"同一 trial 存在多个 DONE 目录: {dup[:5]}")
    dup_rec = sorted(k for k, v in Counter((r["trial_id"], r.get("attempt", 1))
                                          for r in records).items() if v > 1)
    if dup_rec:
        problems.append(f"records 中 (trial, attempt) 重复: {dup_rec[:5]}")
    retried = {tid: sorted(ns) for tid, ns in attempts.items() if len(ns) > 1}
    extra = [t for t in done if t not in set(expected)]
    if extra:
        problems.append(f"清单外多出 {len(extra)} 条: {extra[:5]}")

    energies = {}
    for tid in expected:
        if tid not in done:
            continue
        d = dirs[tid]
        meta_p = d / "meta.json"
        if not meta_p.is_file():
            problems.append(f"{tid}: 缺 meta.json")
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        miss = [k for k in REQUIRED_META if k not in meta]
        if miss:
            problems.append(f"{tid}: meta 缺字段 {miss}")
        if meta.get("protocol_sha256") != meta.get("protocol_sha256", "").lower():
            problems.append(f"{tid}: protocol_sha256 非十六进制小写")
        if abs(meta.get("sim_dt", 0) - 0.005) > 1e-12:
            problems.append(f"{tid}: sim_dt={meta.get('sim_dt')}")
        if meta.get("decimation") != 4:
            problems.append(f"{tid}: decimation={meta.get('decimation')}")
        if meta.get("warmup_steps") != 200:
            problems.append(f"{tid}: warmup_steps={meta.get('warmup_steps')}")
        phys = torch.load(d / "physics_200hz.npz", map_location="cpu")
        state = torch.load(d / "state_50hz.npz", map_location="cpu")
        pair = torch.load(d / "pairing_10hz.npz", map_location="cpu")
        steps = int(phys["tau"].shape[0])
        # completed 必须跑满窗口；terminated/interrupted/truncated 则必须与 meta 一致且 < 窗口
        if meta.get("outcome") == "completed":
            if steps != args.expect_steps:
                problems.append(f"{tid}: 200 Hz 步数 {steps} ≠ {args.expect_steps}")
        else:
            if not (0 < steps <= args.expect_steps):
                problems.append(f"{tid}: 非完整 trial 的步数异常 {steps}")
            if steps == args.expect_steps:
                problems.append(f"{tid}: outcome={meta.get('outcome')} 却跑满 {steps} 步（应为 completed）")
        if phys["tau"].shape[-1] != 12 or phys["qdot"].shape != phys["tau"].shape:
            problems.append(f"{tid}: tau/qdot 形状异常 {tuple(phys['tau'].shape)}")
        if meta.get("steps_executed") != steps:
            problems.append(f"{tid}: meta.steps_executed={meta.get('steps_executed')} 与 npz {steps} 不符")
        if meta.get("outcome") == "completed":
            if int(state["action"].shape[0]) != args.expect_control:
                problems.append(f"{tid}: 50 Hz 条数 {state['action'].shape[0]} ≠ {args.expect_control}")
            if int(pair["t"].shape[0]) != args.expect_pairs:
                problems.append(f"{tid}: 配对点数 {pair['t'].shape[0]} ≠ {args.expect_pairs}")
            if pair["t"].shape[0] != pair["exec_pos"].shape[0]:
                problems.append(f"{tid}: 配对 t 与 exec_pos 长度不一致")
            if pair["cmd"].shape[0] != pair["t"].shape[0]:
                problems.append(f"{tid}: 配对 cmd 长度不一致")
            if pair["t"].shape[0] >= 2:
                dt = (pair["t"][-1] - pair["t"][0]).item() / (pair["t"].shape[0] - 1)
                if abs(dt - 0.1) > 1e-5:
                    problems.append(f"{tid}: 配对间隔 {dt:.6f} ≠ 0.1")
        if (state["action"].shape[0] * 4) != steps:
            problems.append(f"{tid}: 50 Hz×4={state['action'].shape[0] * 4} ≠ 200 Hz {steps}")
        if meta.get("outcome") != "completed":
            # 非完整 trial 不检查 1500/301，但必须记录未外推
            if "未外推" not in str(meta.get("notes", "")):
                problems.append(f"{tid}: 非完整 trial 未注明未外推")
        energy = json.loads((d / "energy.json").read_text(encoding="utf-8"))
        if energy.get("unit") != "J":
            problems.append(f"{tid}: energy.unit={energy.get('unit')}")
        recomputed = integrate_series(phys["tau"].numpy(), phys["qdot"].numpy(), dt=meta["sim_dt"])
        j = energy.get("j_exec", float("nan"))
        rel = abs(recomputed - j) / max(abs(j), 1e-12)
        energies[tid] = {"j_exec": j, "recomputed": recomputed, "rel_err": rel,
                         "outcome": meta.get("outcome")}
        if rel > 1e-5:
            problems.append(f"{tid}: 能量复算相对误差 {rel:.3e} > 1e-5")

    report = {
        "split": args.split, "expected": len(expected), "done": len(done & set(expected)),
        "records": len(records), "missing": missing, "duplicates": dup,
        "duplicate_records": dup_rec, "retried_attempts": retried, "extra": extra,
        "outcomes": dict(Counter(r["outcome"] for r in records)),
        "energy_summary": {
            "n": len(energies),
            "j_exec_min": min((e["j_exec"] for e in energies.values()), default=None),
            "j_exec_max": max((e["j_exec"] for e in energies.values()), default=None),
            "max_rel_err": max((e["rel_err"] for e in energies.values()), default=None),
        },
        "problems": problems, "ok": not problems,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "energy_summary"}, ensure_ascii=False)[:1500])
    print("energy:", json.dumps(report["energy_summary"], ensure_ascii=False))
    print("VERDICT:", "pass" if report["ok"] else "fail")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
