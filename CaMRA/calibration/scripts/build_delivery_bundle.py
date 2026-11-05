#!/usr/bin/env python3
"""CAL-030：生成可复现交付包的索引（不改 v2 本体、不搬原始大数据）。

产物 ``delivery/bundle_manifest_v1.json``：

* ``artifacts``：协议 / 功率模型 / 足迹 / step-cost schema / margins / 数据锁 的
  路径 + 文件散列 + 内容散列；
* ``reports``：真实报告（功率拟合、margin、heldout 覆盖、近障评估、全数据审计）路径 + 散列；
* ``data_index``：每个 split 的清单散列、条数、产物根（**在仓库外**／pub）；
* ``code``：代码文件数 + 目录树散列；
* ``controller``：冻结 v2 包路径 + 目录散列（只读引用，明确"不改本体"）；
* ``access_boundary``：哪些在仓库内、哪些在仓库外、哪些只读。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CAL = Path(__file__).resolve().parent.parent
if str(CAL / "src") not in sys.path:
    sys.path.insert(0, str(CAL / "src"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: Path, *, content_key: str | None = None) -> dict:
    entry = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if content_key:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if content_key in payload:
                entry["content_sha256"] = payload[content_key]
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            pass
    return entry


def tree_sha256(root: Path, pattern: str = "*.py") -> dict:
    files = sorted(p for p in root.rglob(pattern) if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(sha256_file(p).encode())
    return {"n_files": len(files), "tree_sha256": h.hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CAL-030 交付包索引")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--logs-root", required=True, help="仓库外的产物根（/pub/...）")
    ap.add_argument("--bundle", default="/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = CAL / "config"
    artifacts = {
        "protocol": artifact(cfg / "calibration_protocol_v1.0.2.json"),
        "power_model": artifact(cfg / "power_model_v1.json", content_key="sha256"),
        "footprint": artifact(cfg / "footprint_v1.json", content_key="sha256"),
        "step_cost_schema": artifact(cfg / "step_cost_schema_v1.json", content_key="sha256"),
        "margins": artifact(cfg / "margins_v1.json", content_key="sha256"),
        "data_lock": artifact(cfg / "data_lock_v1.json", content_key="sha256"),
        "near_obstacle_maps": artifact(cfg / "near_obstacle_maps_v1.json", content_key="sha256"),
        "asset_geometry": artifact(cfg / "go2_asset_geometry_v1.json", content_key="sha256"),
    }

    logs = Path(args.logs_root)
    report_paths = {
        "power_fit": logs / "power_fit" / "power_fit_report.json",
        "margins": logs / "margins" / "margin_report.json",
        "heldout_coverage": logs / "heldout" / "heldout_coverage.json",
        "near_obstacle_eval": logs / "near_obstacle" / "near_obstacle_eval.json",
        "audit_all_splits": logs / "audit" / "audit_all_splits.json",
    }
    reports = {k: artifact(p) for k, p in report_paths.items() if p.is_file()}
    missing_reports = [k for k, p in report_paths.items() if not p.is_file()]

    lock = json.loads((cfg / "data_lock_v1.json").read_text(encoding="utf-8"))
    data_index = {}
    for split, entry in sorted(lock["trial_lists"]["files"].items()):
        man = Path(args.manifest_dir) / f"{split}.jsonl"
        n = sum(1 for line in man.read_text(encoding="utf-8").splitlines() if line.strip())
        data_index[split] = {"manifest": str(man), "sha256": entry["sha256"], "n_trials": n,
                             "artifacts_root": str(logs / "trials" / split)}

    bundle = Path(args.bundle)
    manifest = {
        "schema": "camra-delivery-bundle-v1",
        "controller": {
            "name": "camra_go2_velocity_it2999_v2", "path": str(bundle),
            "read_only": True,
            "note": "交付包只引用、不修改、不复制 v2 本体（政策：不改 v2 与参考代码）",
        },
        "artifacts": artifacts,
        "reports": reports,
        "missing_reports": missing_reports,
        "data_index": data_index,
        "code": {"calibration_root": str(CAL), **tree_sha256(CAL)},
        "access_boundary": {
            "in_repo": ["config/*.json", "src/camra_calibration/*.py", "scripts/*.py",
                        "sim/*.py", "tests/*.py", "docs/implementation_handoff.md"],
            "outside_repo": [str(logs), str(bundle)],
            "rule": "原始大文件（逐 trial 产物、冻结 bundle、日志）不复制进仓库；本索引记录其路径与散列",
        },
        "lock_sha256": (artifacts["data_lock"].get("content_sha256")
                        or artifacts["data_lock"]["sha256"]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "n_artifacts": len(artifacts),
                      "n_reports": len(reports), "missing_reports": missing_reports,
                      "n_splits": len(data_index), "code_files": manifest["code"]["n_files"],
                      "code_tree": manifest["code"]["tree_sha256"][:16]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
