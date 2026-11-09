#!/usr/bin/env python3
"""按交付索引逐 trial 核算迁移后的数据散列（HIRA 迁移验收）。

口径与 ``tests/build_delivery_manifest.py`` 完全一致：
  - 单文件：sha256(文件字节)
  - 目录/trial：对 ``相对路径\\0文件散列\\n`` 排序后整体 sha256，跳过 ``__pycache__``

只读：不修改任何数据，只比对 ``delivery/bundle_manifest_hira_v1.json`` 里冻结的
``sha256_tree`` / ``n_files`` / ``bytes`` 与目标机上实际重算的结果。

用法：
  python tests/verify_migration_hira.py \
      --manifest delivery/bundle_manifest_hira_v1.json \
      --data-root /pub/data/caohy \
      --out reports/migration_tiger11_20261109.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path, *, skip_names=("__pycache__",)) -> tuple[str, int, int]:
    """返回 (树散列, 文件数, 总字节)，与 build_delivery_manifest.sha256_tree 同口径。"""
    entries: list[tuple[str, str]] = []
    total = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in rel.split("/") for part in skip_names):
            continue
        entries.append((rel, sha256_file(p)))
        total += p.stat().st_size
    entries.sort(key=lambda kv: kv[0])
    h = hashlib.sha256()
    for rel, digest in entries:
        h.update(f"{rel}\0{digest}\n".encode("utf-8"))
    return h.hexdigest(), len(entries), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(EXP / "delivery/bundle_manifest_hira_v1.json"))
    ap.add_argument("--data-root", default="/pub/data/caohy")
    ap.add_argument("--out", default=str(EXP / "reports/migration_tiger11_20261109.json"))
    ap.add_argument("--max-detail", type=int, default=20)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    big = manifest["big_data"]
    data_root = Path(args.data_root)
    runs_root = data_root / "logs" / "camra_hira" / "runs"

    print(f"[mig] 索引: {Path(args.manifest).name}（hash_rule: {big['hash_rule']}）")
    print(f"[mig] 目标: {data_root}")

    ok_runs = 0
    bad_runs: list[dict] = []
    ok_trials = 0
    bad_trials: list[dict] = []
    missing_trials: list[str] = []
    bytes_checked = 0

    for run_name in sorted(big["runs"]):
        info = big["runs"][run_name]
        run_dir = runs_root / run_name
        if not run_dir.is_dir():
            bad_runs.append({"run": run_name, "problem": "run 目录不存在", "expected": info["logical_path"]})
            continue
        run_bad = 0

        for fname, finfo in (info.get("top_level_files") or {}).items():
            fpath = run_dir / fname
            if not fpath.is_file():
                bad_trials.append({"run": run_name, "file": fname, "problem": "顶层文件缺失"})
                run_bad += 1
                continue
            got = sha256_file(fpath)
            if got != finfo["sha256"]:
                bad_trials.append({"run": run_name, "file": fname, "problem": "顶层文件散列不符",
                                   "expected": finfo["sha256"][:16], "got": got[:16]})
                run_bad += 1

        for trial_id, tinfo in info["trials"].items():
            tdir = run_dir / trial_id
            if not tdir.is_dir():
                missing_trials.append(f"{run_name}/{trial_id}")
                run_bad += 1
                continue
            digest, nf, nb = sha256_tree(tdir)
            bytes_checked += nb
            if digest == tinfo["sha256_tree"] and nf == tinfo["n_files"]:
                ok_trials += 1
            else:
                bad_trials.append({
                    "run": run_name, "trial": trial_id,
                    "expected": tinfo["sha256_tree"][:16], "got": digest[:16],
                    "expected_files": tinfo["n_files"], "got_files": nf,
                    "expected_bytes": tinfo["bytes"], "got_bytes": nb,
                })
                run_bad += 1

        if run_bad == 0:
            ok_runs += 1
        else:
            bad_runs.append({"run": run_name, "problem": f"{run_bad} 处不符"})

    usd = big.get("usd_cache") or {}
    usd_res = {"checked": False}
    if usd.get("exists") and usd.get("sha256_tree"):
        usd_dir = Path(usd.get("logical_path") or (data_root / "logs/camra_hira/assets/usd_cache"))
        if usd_dir.is_dir():
            d, nf, nb = sha256_tree(usd_dir)
            usd_res = {"checked": True, "path": str(usd_dir), "match": d == usd["sha256_tree"],
                       "n_files": nf, "expected_files": usd.get("n_files")}
        else:
            usd_res = {"checked": True, "path": str(usd_dir), "match": False, "problem": "目录不存在"}

    n_trials_expected = sum(len(v["trials"]) for v in big["runs"].values())
    verdict = (ok_runs == len(big["runs"]) and not bad_trials and not missing_trials
               and usd_res.get("match") is not False)

    report = {
        "schema": "camra-hira-migration-verify-v1",
        "manifest": str(args.manifest),
        "data_root": str(data_root),
        "hash_rule": big["hash_rule"],
        "runs": {"expected": len(big["runs"]), "ok": ok_runs, "bad": bad_runs},
        "trials": {"expected": n_trials_expected, "ok": ok_trials,
                   "bad": bad_trials[: args.max_detail], "n_bad": len(bad_trials),
                   "missing": missing_trials[: args.max_detail], "n_missing": len(missing_trials)},
        "usd_cache": usd_res,
        "bytes_checked": bytes_checked,
        "verdict": {"all_matched": verdict, "failed": [] if verdict else [
            *( ["runs"] if ok_runs != len(big["runs"]) else [] ),
            *( ["trials"] if bad_trials or missing_trials else [] ),
            *( ["usd_cache"] if usd_res.get("match") is False else [] ),
        ]},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(out.suffix + ".sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")

    print(f"[mig] runs  {ok_runs}/{len(big['runs'])}")
    print(f"[mig] trials {ok_trials}/{n_trials_expected}"
          f"（不符 {len(bad_trials)}，缺失 {len(missing_trials)}）")
    print(f"[mig] usd_cache {usd_res}")
    print(f"[mig] 校验字节 {bytes_checked/1e9:.2f} GB")
    print(f"[mig] 结论 {'MIGRATION_OK' if verdict else 'MIGRATION_FAILED ' + str(report['verdict']['failed'])}")
    print(f"[mig] 报告 {out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
