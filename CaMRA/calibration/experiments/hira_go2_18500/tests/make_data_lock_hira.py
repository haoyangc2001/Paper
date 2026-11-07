"""HIRA-029：冻结正式采集的数据锁（G2 前置）。

锁把"这一次采集到底绑定了什么"写成不可变文件：控制器包、协议、功率模型、足迹轮廓、
步成本接口、七分片清单、近障地图、采集路径上的代码、环境版本，以及四份门禁报告。
采集入口（`run_split.py --lock`）在启动仿真**之前**调用
`camra_calibration.lock.enforce_collection`，任何一项散列不符即拒绝。

派生规则（预登记）：`margins` 在 G2 时尚未产出（HIRA-031 才冻结 α/δ），因此锁里记
`state = not_yet_produced`；HIRA-031 之后**追加**一把新的不可变 stage lock
（`data_lock_hira_margins_v1`），**不回写**本锁，已有校准数据仍归属本锁。

    python tests/make_data_lock_hira.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from camra_calibration.footprint import Footprint  # noqa: E402
from camra_calibration.power import PowerModel  # noqa: E402
from camra_calibration.step_cost import schema_sha256  # noqa: E402

SCHEMA = "camra-data-lock-v1"
SPLITS = ("pilot", "baseline_fit", "baseline_dev", "baseline_validate",
          "calibration", "heldout", "near_obstacle")
CODE_GLOBS = (
    (REPO / "CaMRA" / "locomotion" / "adapters" / "hira_go2", "*.py"),
    (REPO / "CaMRA" / "calibration" / "src" / "camra_calibration", "*.py"),
    (EXP / "tests", "run_split.py"),
)

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=str(EXP / "config/data_lock_hira_calibration_v1.json"))
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--stage", default="calibration_collection")
parser.add_argument("--note", default=None, help="修正/重采的原因（写进锁，便于追溯）")
parser.add_argument("--parent-lock", default=None, help="被取代的旧锁路径（原子归属用）")
args_cli = parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry(path: Path, **extra) -> dict:
    out = {"path": str(path), "sha256": sha256_file(path)}
    out.update(extra)
    return out


def code_tree() -> dict:
    files = {}
    for base, pattern in CODE_GLOBS:
        for path in sorted(base.glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            files[str(path)] = sha256_file(path)
    return {"files": files, "n_files": len(files)}


def environment() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch  # noqa: PLC0415

        versions["torch"] = torch.__version__
        versions["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        versions["torch"] = None
    audit = EXP / "reports/asset_actuator_audit.json"
    if audit.is_file():
        payload = json.loads(audit.read_text(encoding="utf-8"))
        versions["isaaclab"] = payload.get("isaaclab")
        versions["n_dof"] = payload.get("runtime", {}).get("n_dof")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        versions["nvidia_driver"] = out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        versions["nvidia_driver"] = None
    return versions


def main() -> int:
    bundle_dir = EXP / "frozen/hira_go2_18500"
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    controller = json.loads((bundle_dir / "controller.json").read_text(encoding="utf-8"))

    lock_manifest = json.loads((Path(args_cli.manifests) / "lock.json").read_text(encoding="utf-8"))
    lists = {}
    for split in SPLITS:
        path = Path(args_cli.manifests) / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"缺少清单 {path}")
        meta = lock_manifest["splits"][split]
        lists[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "count": int(meta["count"]),
            "quota": int(meta["count"]),
            "seed_block": meta.get("seed_block"),
            "n_matched_v2": meta.get("n_matched_v2"),
        }

    model_path = EXP / "config/power_model_hira_v1.json"
    model = PowerModel.load(model_path, verify_hash=True)
    fp_path = EXP / "config/footprint_hira_v1.json"
    fp = Footprint.load(fp_path, verify_hash=True)
    sc_path = EXP / "config/step_cost_schema_hira_v1.json"
    sc = json.loads(sc_path.read_text(encoding="utf-8"))

    maps_path = EXP / "config/near_obstacle_maps_v1.json"
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    if sha256_file(maps_path) != "78c0145f0ecddc71edaf2f64de7be05af3bd4d690e110b54ebaedfa6d9952ec3":
        raise SystemExit("近障地图副本与 v2 记录的原始散列不一致（78c0145f…）")

    gates = {}
    for name, rel in (
        ("G1_integration", "reports/integration_gate.json"),
        ("baseline_integrity", "reports/baseline_collection_integrity.json"),
        ("power_model", "reports/power_model_validate.json"),
        ("footprint", "reports/footprint_validation.json"),
        ("step_cost", "reports/step_cost_interface_tests.json"),
        ("config_isolation", "reports/config_isolation_tests.json"),
    ):
        path = EXP / rel
        if not path.is_file():
            raise SystemExit(f"缺门禁报告 {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        # 各报告的历史字段名不一致：G1 用 verdict.g1_passed，其余用 verdict.all_passed 或顶层 ok
        verdict = payload.get("verdict") or {}
        if "g1_passed" in verdict:
            passed, failed = verdict["g1_passed"], verdict.get("failed_rows")
        elif "all_passed" in verdict:
            passed, failed = verdict["all_passed"], verdict.get("failed")
        else:
            passed, failed = payload.get("all_passed", payload.get("ok")), payload.get("failed")
        if passed is None:
            raise SystemExit(f"{rel}: 无法从报告里解析出通过与否")
        gates[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "passed": bool(passed),
            "failed": failed,
        }

    payload = {
        "schema": SCHEMA,
        "cal_task": "HIRA-029",
        "stage": args_cli.stage,
        "purpose": "正式校准采集（calibration 800 条）的前置锁：绑定控制器/模型/轮廓/接口/清单/地图/代码/环境",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "amendment": (None if not args_cli.note else {
            "note": args_cli.note,
            "parent_lock": (None if not args_cli.parent_lock else
                            {"path": str(args_cli.parent_lock),
                             "sha256": sha256_file(Path(args_cli.parent_lock))}),
        }),
        "controller": {
            "controller_id": manifest.get("controller", {}).get("controller_id", "hira_go2_18500"),
            "bundle_id": manifest["bundle_id"],
            "manifest": entry(bundle_dir / "manifest.json"),
            "controller_json": entry(bundle_dir / "controller.json"),
            "policy": entry(bundle_dir / controller["policy"]["file"]),
            "urdf": entry(bundle_dir / "urdf_go2.urdf"),
        },
        "protocol": entry(EXP / "config/calibration_protocol_hira_v1.json",
                          version=json.loads((EXP / "config/calibration_protocol_hira_v1.json")
                                             .read_text(encoding="utf-8")).get("version")),
        "power_model": entry(model_path, content_sha256=model.sha256(),
                             history_k=model.history_k, n_params=len(model.weights)),
        "footprint": entry(fp_path, content_sha256=fp.sha256(),
                           bbox_m=fp.stats.get("bbox_m"), n_vertices=fp.polygon.n),
        "step_cost_schema": entry(sc_path, content_sha256=schema_sha256(
            {k: v for k, v in sc.items() if k != "sha256"})),
        "asset": {
            "geometry": entry(EXP / "config/go2_asset_geometry_hira_v1.json"),
            "urdf_sha256": json.loads((EXP / "config/go2_asset_geometry_hira_v1.json")
                                      .read_text(encoding="utf-8"))["source"]["urdf_sha256"],
            "usd_cache": str(Path("/pub/data/caohy/logs/camra_hira/assets/usd_cache")),
        },
        "trial_lists": {
            "dir": str(Path(args_cli.manifests)),
            "lock_sha256": sha256_file(Path(args_cli.manifests) / "lock.json"),
            "total_trials": sum(item["count"] for item in lists.values()),
            "files": lists,
        },
        "near_obstacle_maps": {
            "path": str(maps_path),
            "sha256": sha256_file(maps_path),
            "n_cases": int(maps.get("n_cases", len(maps.get("cases", [])))),
            "footprint_sha256": fp.sha256(),
            "origin": "v2 的原始物理地图（config/near_obstacle_maps_v1.json，sha 78c0145f…）的只读副本；"
            "footprint 与筛选定义已绑定本实验，未改动地图内容",
            "clearance_tiers": maps.get("clearance_tiers"),
        },
        "margins": {
            "state": "not_yet_produced",
            "path": None,
            "rule": "HIRA-031 冻结 α/δ 后**追加**新的不可变 stage lock（data_lock_hira_margins_v1.json），"
            "不回写本锁；本锁下的校准数据永远归属本锁",
            "gated_splits": ["heldout", "near_obstacle"],
        },
        "code": code_tree(),
        "environment": environment(),
        "gates": gates,
        "collection_rules": {
            "formal_splits": ["calibration", "heldout", "near_obstacle"],
            "margin_gated_splits": ["heldout", "near_obstacle"],
            "map_gated_splits": ["near_obstacle"],
            "no_overwrite": "正式 split 禁止 --allow-overwrite；重试只走「改名/删 DONE + 新 attempt」",
            "attempt_policy": "只认第一有效主 attempt；业务失败不补成功替换；基础设施恢复追加 attempt",
            "energy_partial_rule": "env 终止即该 trial 结束，J 只覆盖实际执行步数（v1.0.2 继承）",
        },
        "quotas": {split: item["count"] for split, item in lists.items()},
        "resource_budget": {
            "measured_s_per_trial": 2.07,
            "measured_mb_per_trial": 1.135,
            "calibration_800_estimate_min": round(800 * 2.07 / 60, 1),
            "calibration_800_estimate_mb": round(800 * 1.135, 1),
            "num_envs": 8,
        },
    }
    out = Path(args_cli.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    digest = sha256_file(out)
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(f"[lock] wrote {out.name} sha={digest[:16]} code_files={payload['code']['n_files']}")
    print(f"[lock] gates: " + ", ".join(f"{k}={v['passed']}" for k, v in gates.items()))
    print(f"[lock] lists total={payload['trial_lists']['total_trials']} margins={payload['margins']['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
