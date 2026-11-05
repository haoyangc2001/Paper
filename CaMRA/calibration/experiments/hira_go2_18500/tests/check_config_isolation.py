"""HIRA-010 配置隔离自测：拒混、路径隔离、无 fallback、旧产物不受影响。

只依赖标准库，可在 CPU 上跑：
    python tests/check_config_isolation.py --out ../reports/config_isolation_tests.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent          # .../Paper
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters" / "hira_go2"
sys.path.insert(0, str(ADAPTER))

from experiment import (  # noqa: E402
    CONTROLLER_ID, EXPERIMENT_ID, ConfigError, HiraExperiment,
)

V2_LOCK = REPO / "CaMRA" / "calibration" / "config" / "data_lock_v1.json"
V2_LOCK_SHA_PREFIX = "b214809bd287b170"


def base_env() -> dict[str, str]:
    """正常情况下的显式配置：HiRA 自己的路径，全部指向 hira_* 产物。"""
    return {
        "HIRA_EXPERIMENT_ROOT": str(EXP),
        "HIRA_DATA_ROOT": "/pub/data/caohy/logs/camra_hira",
        "HIRA_BUNDLE_ROOT": "/pub/data/caohy/camra_hira_bundles",
        "HIRA_CHECKPOINT": str(REPO / "HiRA/logs/rough_go2/Sep08_11-57-26_/model_18500.pt"),
        "HIRA_URDF": str(REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"),
        "HIRA_ASSET_ROOT": "/pub/data/caohy/unitree_model",
        "HIRA_PROTOCOL": str(EXP / "config/hira_protocol_v1.json"),
        "HIRA_MODEL": str(EXP / "config/hira_power_model_v1.json"),
        "HIRA_MARGINS": str(EXP / "config/hira_margins_v1.json"),
        "HIRA_MAPS": str(EXP / "config/hira_near_obstacle_maps_v1.json"),
        "HIRA_LOCK": str(EXP / "config/hira_data_lock_v1.json"),
        "HIRA_DEVICE": "cuda:0",
        "HIRA_VISIBLE_DEVICES": "2",
    }


def run_cases() -> list[dict]:
    cases: list[dict] = []

    def case(cid: str, name: str, fn) -> None:
        try:
            detail = fn() or ""
            cases.append({"id": cid, "name": name, "pass": True, "detail": detail})
        except AssertionError as exc:
            cases.append({"id": cid, "name": name, "pass": False, "detail": f"断言失败：{exc}"})
        except Exception as exc:                              # noqa: BLE001
            cases.append({"id": cid, "name": name, "pass": False,
                          "detail": f"{type(exc).__name__}: {exc}"})

    # U1 缺字段必须失败
    def u1():
        env = base_env()
        env.pop("HIRA_PROTOCOL")
        try:
            HiraExperiment.from_env(env)
        except ConfigError as exc:
            return f"按预期拒绝：{str(exc)[:90]}"
        raise AssertionError("缺 HIRA_PROTOCOL 时未报错")

    # U2 controller_id 被改成 v2
    def u2():
        exp = HiraExperiment.from_env(base_env())
        bad = HiraExperiment(**{**exp.__dict__, "controller_id": "camra_go2_velocity_it2999_v2"})
        p = bad.identity_problems()
        assert p, "controller_id 改成 v2 后未报错"
        return p[0]

    # U3 protocol 指向 v2 协议
    def u3():
        exp = HiraExperiment.from_env(base_env())
        bad = HiraExperiment(**{**exp.__dict__, "protocol": V2_LOCK.parent / "calibration_protocol_v1.0.2.json"})
        p = bad.v2_contamination_problems()
        assert p, "protocol 指向 v2 未报错"
        return p[0]

    # U4 margins 指向 v2 margins
    def u4():
        exp = HiraExperiment.from_env(base_env())
        bad = HiraExperiment(**{**exp.__dict__, "margins": V2_LOCK.parent / "margins_v1.json"})
        p = bad.v2_contamination_problems()
        assert p, "margins 指向 v2 未报错"
        return p[0]

    # U5 data_root 设成 v2 日志根
    def u5():
        exp = HiraExperiment.from_env(base_env())
        bad = HiraExperiment(**{**exp.__dict__, "data_root": Path("/pub/data/caohy/logs/camra")})
        p = bad.v2_contamination_problems()
        assert p, "data_root 指向 v2 日志根未报错"
        return p[0]

    # U6 正常配置（不要求文件已存在）应零问题
    def u6():
        exp = HiraExperiment.from_env(base_env())
        p = exp.validate(require_files=False)
        assert not p, f"正常配置仍报错：{p[:2]}"
        return "0 problems（身份与拒混均通过）"

    # U7 要求文件存在时，必须逐项报缺失（证明没有 fallback 到 v2）
    def u7():
        exp = HiraExperiment.from_env(base_env())
        p = exp.validate(require_files=True)
        assert p, "hira 产物尚未生成，却报告一切就绪"
        missing = [x for x in p if "不存在" in x or "不是文件" in x]
        assert len(missing) >= 5, f"缺失项过少，可能发生了 fallback：{p}"
        return f"{len(p)} 项问题，其中缺失文件 {len(missing)} 项（无 fallback）"

    # U8 run_id 路径穿越必须拒绝
    def u8():
        exp = HiraExperiment.from_env(base_env())
        for bad in ("../x", "a/b", ".."):
            try:
                exp.run_dir(bad)
            except ConfigError:
                continue
            raise AssertionError(f"run_id {bad!r} 未被拒绝")
        return "三种非法 run_id 全部被拒"

    # U9 旧 v2 产物未被本任务改动
    def u9():
        h = hashlib.sha256(V2_LOCK.read_bytes()).hexdigest()
        assert h.startswith(V2_LOCK_SHA_PREFIX), f"v2 锁散列变了：{h[:16]}"
        return f"v2 data_lock_v1.json sha {h[:16]}… 未变"

    # U10 指纹包含 checkpoint 的真实散列
    def u10():
        exp = HiraExperiment.from_env(base_env())
        fp = exp.fingerprint()
        sha = fp["files"]["checkpoint"]["sha256"]
        assert sha and sha.startswith("d30e5212"), f"checkpoint 散列异常：{sha}"
        assert fp["controller_id"] == CONTROLLER_ID and fp["experiment_id"] == EXPERIMENT_ID
        return f"fingerprint.ok，checkpoint {sha[:16]}…"

    for cid, name, fn in [
        ("U1", "缺必需环境变量必须失败", u1),
        ("U2", "controller_id 指向 v2 必须失败", u2),
        ("U3", "protocol 指向 v2 必须失败", u3),
        ("U4", "margins 指向 v2 必须失败", u4),
        ("U5", "data_root 指向 v2 日志根必须失败", u5),
        ("U6", "正常配置零问题", u6),
        ("U7", "缺 hira 产物时报缺失而非 fallback", u7),
        ("U8", "run_id 路径穿越必须拒绝", u8),
        ("U9", "v2 旧产物散列未变", u9),
        ("U10", "指纹含真实 checkpoint 散列", u10),
    ]:
        case(cid, name, fn)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXP / "reports" / "config_isolation_tests.json"))
    args = ap.parse_args()

    cases = run_cases()
    n_pass = sum(1 for c in cases if c["pass"])
    report = {
        "report": "config_isolation_tests",
        "task": "HIRA-010",
        "controller_id": CONTROLLER_ID,
        "experiment_id": EXPERIMENT_ID,
        "adapter_root": str(ADAPTER),
        "cases": cases,
        "n_cases": len(cases),
        "n_pass": n_pass,
        "ok": n_pass == len(cases),
        "note": "U7 的缺失项是 HIRA-023/026/029 才会生成的 hira_* 产物；此刻报缺失正是预期行为，证明没有 fallback 到 v2",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in cases:
        print(("PASS " if c["pass"] else "FAIL ") + c["id"] + " " + c["name"])
    print(f"\n{n_pass}/{len(cases)} passed -> {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
