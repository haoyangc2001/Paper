"""组装正式数据锁（CAL-023）：把依赖、配额、seed、统计口径与资源预算写成一份可机检文件。

用法：
    python scripts/build_data_lock.py \
        --manifest-dir /pub/data/caohy/logs/camra/trial_lists_v1.0.2 \
        --out config/data_lock_v1.json

锁里绑定的东西（缺一不可）：
  protocol(v1.0.2) / power_model / footprint / step_cost_schema / 资产几何与 USD /
  七个 split 的清单散列、条数与 seed 块 / 近障地图 / 代码文件散列 /
  配额与统计准入（从协议抄录，不改写）/ 资源预算 / 采集顺序与门禁。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from camra_calibration.protocol import Protocol  # noqa: E402

CODE_DIRS = ("src/camra_calibration", "scripts", "sim")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(cal_dir: Path, dirs) -> tuple[dict, str]:
    files: dict[str, str] = {}
    for d in dirs:
        for p in sorted((cal_dir / d).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            files[str(p.relative_to(cal_dir))] = sha256_file(p)
    blob = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return files, hashlib.sha256(blob).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-023 数据锁")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--power-model", default=None)
    ap.add_argument("--footprint", default=None)
    ap.add_argument("--step-cost-schema", default=None)
    ap.add_argument("--asset", default=None)
    ap.add_argument("--near-maps", default=None)
    ap.add_argument("--resource-budget", default=None,
                    help="可选：覆盖协议里的资源预算（JSON 文件）")
    args = ap.parse_args(argv)

    cal = Path(__file__).resolve().parent.parent
    proto_p = Path(args.protocol) if args.protocol else cal / "config" / "calibration_protocol_v1.0.2.json"
    proto = Protocol.load(proto_p)
    man = Path(args.manifest_dir)
    lists_lock = json.loads((man / "lock.json").read_text(encoding="utf-8"))

    def entry(path: Path, **extra) -> dict:
        """文件散列 + （若有）文件内自带的内容散列，两者都记，避免与文档里的
        "模型/轮廓内容散列"混淆。"""
        out = {"path": str(path), "sha256": sha256_file(path), **extra}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "sha256" in payload:
                out["content_sha256"] = payload["sha256"]
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            pass
        return out

    model_p = Path(args.power_model) if args.power_model else cal / "config" / "power_model_v1.json"
    fp_p = Path(args.footprint) if args.footprint else cal / "config" / "footprint_v1.json"
    sc_p = (Path(args.step_cost_schema) if args.step_cost_schema
            else cal / "config" / "step_cost_schema_v1.json")
    asset_p = Path(args.asset) if args.asset else cal / "config" / "go2_asset_geometry_v1.json"
    asset = json.loads(asset_p.read_text(encoding="utf-8"))
    maps_p = Path(args.near_maps) if args.near_maps else cal / "config" / "near_obstacle_maps_v1.json"
    if not maps_p.is_file():
        raise SystemExit(f"FATAL: 近障地图未冻结: {maps_p}")
    maps = json.loads(maps_p.read_text(encoding="utf-8"))

    files: dict[str, dict] = {}
    for split in proto.payload["seeds"]["blocks"]:
        p = man / f"{split}.jsonl"
        if not p.is_file():
            raise SystemExit(f"FATAL: 清单缺失 {p}")
        start, count = proto.seed_block(split)
        n = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
        files[split] = {"path": str(p), "sha256": sha256_file(p), "count": n,
                        "seed_block": [start, count],
                        "quota": proto.quota(split)}

    code_files, code_tree = tree_hashes(cal, CODE_DIRS)
    # 资源预算的唯一来源是协议里的实测记录（不另建文件，避免两份口径）
    budget = proto.payload.get("resource_budget", {})
    if args.resource_budget:
        budget = json.loads(Path(args.resource_budget).read_text(encoding="utf-8"))

    stat = proto.payload["statistics"]
    acc = proto.payload["statistical_acceptance"]
    payload = {
        "schema": "camra-data-lock-v1",
        "cal_task": "CAL-023",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": ("冻结正式采集的全部依赖与口径；任何一项变化都必须出新锁版本，"
                    "不得就地修改。"),
        "protocol": entry(proto_p, version=proto.version,
                          baseline_bound={"version": "1.0.1",
                                          "sha256": "16aa5ebeb24f4d0d2d7810d646b8e36f7d783d5e93efdedb5eef88432a71f2ac",
                                          "note": "baseline 400 条与 pilot 在该版本下采集"}),
        "power_model": entry(model_p),
        "footprint": entry(fp_p),
        "step_cost_schema": entry(sc_p),
        "asset": {
            "geometry": entry(asset_p),
            "usd": {"path": asset["source"]["usd"], "sha256": asset["source"]["usd_sha256"]},
            "physics_usd": {"path": asset["source"]["physics_usd"],
                            "sha256": asset["source"]["physics_usd_sha256"]},
        },
        "trial_lists": {
            "dir": str(man),
            "lock_sha256": sha256_file(man / "lock.json"),
            "self_check": lists_lock.get("self_check"),
            "protocol_version": lists_lock.get("protocol_version"),
            "total_trials": lists_lock.get("total_trials"),
            "files": files,
        },
        "near_obstacle_maps": entry(maps_p, n_cases=maps["totals"]["n_cases"],
                                    clearance_tiers=maps["clearance_tiers_m"],
                                    footprint_sha256=maps["footprint"]["sha256"]),
        "quotas": {split: proto.quota(split) for split in files},
        "statistics": {
            "alpha_quantile": stat.get("quantile"),
            "alpha_rule": stat.get("alpha"),
            "delta_rule": stat.get("delta"),
            "window_s": proto.payload["clocks"]["measured_window_s"],
            "j_min_J": proto.j_min,
        },
        "statistical_acceptance": acc,
        "resource_budget": budget,
        "code": {"files": code_files, "tree_sha256": code_tree},
        "collection_rules": {
            "order": ["calibration", "heldout", "near_obstacle"],
            "formal_splits": ["calibration", "heldout", "near_obstacle"],
            "margin_gated": ["heldout", "near_obstacle"],
            "map_gated": ["near_obstacle"],
            "no_overwrite_formal": True,
            "retry_policy": ("数据损坏/中断只允许改名为 DONE.interrupted_* 后以新 attempt 重采，"
                             "保留全部 attempt 与旧产物"),
            "tuning_ban": ("heldout / near_obstacle 在 margin 冻结前不得采集；"
                           "不得用其结果选模型、改协议或改 margin"),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    digest = sha256_file(out)
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")

    print(f"protocol {proto.version} {proto.sha256[:16]}")
    print(f"model {payload['power_model']['sha256'][:16]}  "
          f"footprint {payload['footprint']['sha256'][:16]}  "
          f"schema {payload['step_cost_schema']['sha256'][:16]}")
    print(f"lists total {payload['trial_lists']['total_trials']}  "
          f"near maps {payload['near_obstacle_maps']['n_cases']} "
          f"{payload['near_obstacle_maps']['sha256'][:16]}")
    print(f"code files {len(code_files)}  tree {code_tree[:16]}")
    print(f"wrote {out} sha {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
