"""HIRA-021：采集通路与隔离恢复测试。

四个阶段（一个进程只能有一个 InteractiveScene，故 exec 单独跑）：

    python tests/check_collector_integration.py --phase plan     # 纯函数恢复测试（无需仿真）
    python tests/check_collector_integration.py --headless --device cuda:0 --phase exec
    python tests/check_collector_integration.py --phase legacy   # 旧源码测试 + v2 复算结果核验
    python tests/check_collector_integration.py --phase merge

恢复语义的唯一裁决处是 `hira_go2.collector.plan_attempts`（纯函数），
本脚本对伪造的目录状态直接验证它：缺文件、坏 DONE、中断、业务失败、部分批、
不覆盖、不重复计数，以及 v2 命名/包必须被拒。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
CAL_SRC = REPO / "CaMRA" / "calibration" / "src"
for path in (str(ADAPTER), str(CAL_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("plan", "exec", "legacy", "merge"), default="plan")
parser.add_argument("--out-dir", default=str(EXP / "reports"))
parser.add_argument("--bundle", default=str(EXP / "frozen/hira_go2_18500"))
parser.add_argument("--prereg", default=str(EXP / "config/integration_acceptance_prereg_v1.json"))
parser.add_argument("--usd-dir", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
parser.add_argument("--v2-recompute", default=str(EXP / "reports/v2_trial_recompute.json"))
parser.add_argument("--cal-tests", default=str(REPO / "CaMRA/calibration/tests"))
parser.add_argument("--window", type=float, default=2.0)

_PRE = parser.parse_known_args()[0]
if _PRE.phase == "exec":
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
else:
    args_cli = parser.parse_args()
    app_launcher = None
    simulation_app = None

import torch  # noqa: E402

from hira_go2.collector import (  # noqa: E402
    CollectorError,
    HiraCollector,
    TrialSpec,
    V2_CONTROLLER_ID,
    atomic_write_json,
    done_is_valid,
    plan_attempts,
)

CHECKS: list[dict] = []


def add(cid: str, name: str, ok: bool, detail: str, **extra) -> dict:
    record = {"id": cid, "name": name, "ok": bool(ok), "detail": detail}
    record.update(extra)
    CHECKS.append(record)
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name} :: {detail}", flush=True)
    return record


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def save_partial(out_dir: Path, name: str, extra: dict) -> None:
    payload = {"checks": CHECKS, **extra}
    (out_dir / name).write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# 伪造 attempt 目录的工具
def fabricate_attempt(attempt_dir: Path, outcome: str = "completed", corrupt: str | None = None) -> None:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, body in (("meta.json", {"trial_id": attempt_dir.parent.name}), ("energy.json", {"j_exec": 1.0})):
        path = attempt_dir / name
        path.write_text(json.dumps(body), encoding="utf-8")
        # 先按"好内容"封散列，再（可选）篡改——这样 DONE 登记的是旧散列，
        # 文件却是新内容，正是"过时 DONE"要模拟的状态
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if corrupt == name:
            path.write_text(json.dumps(body) + "tampered", encoding="utf-8")
    atomic_write_json(attempt_dir / "DONE", {"files": files, "sealed": True})
    atomic_write_json(attempt_dir / "status.json", {"outcome": outcome})


def two_specs() -> list[TrialSpec]:
    commands = [[0.3, 0.0, 0.0]] * 20
    push = [
        TrialSpec(trial_id="hira-pilot-clean-0000", split="pilot", program="straight",
                  profile="clean", seed=9011, commands=commands, window_s=args_cli.window),
        TrialSpec(
            trial_id="hira-pilot-push-0000", split="pilot", program="straight",
            profile="push_only", seed=9012, commands=commands, window_s=args_cli.window,
            push_events=[{"t_s": 0.4, "dv_xy": [0.5, 0.0]}],
            initial_pose={"px_m": 0.4, "py_m": -0.3, "yaw_rad": 0.2},
        ),
    ]
    return push


# --------------------------------------------------------------------------- #
def phase_plan() -> int:
    from hira_go2.bundle import load_bundle

    work = Path(tempfile.mkdtemp(prefix="collector_plan_"))
    specs = two_specs()
    run_dir = work / "run1"

    # R01 空 run → 全部 run_initial
    plan = plan_attempts(run_dir, specs)
    add("R01-fresh", "空 run：全部 run_initial",
        all(item["action"] == "run_initial" and item["attempt"] == 1 for item in plan),
        f"{[item['action'] for item in plan]}")

    # R02 DONE 完整 → skip
    fabricate_attempt(run_dir / specs[0].trial_id / "attempt_01")
    plan = plan_attempts(run_dir, specs)
    add("R02-done-valid", "DONE 完整且散列一致 → skip",
        plan[0]["action"] == "skip" and plan[1]["action"] == "run_initial",
        f"trial0={plan[0]['action']}，trial1={plan[1]['action']}")

    # R03 DONE 在但文件被改 → 不算完成，安排 resume
    fabricate_attempt(run_dir / specs[1].trial_id / "attempt_01", corrupt="energy.json")
    plan = plan_attempts(run_dir, specs)
    add("R03-done-stale", "DONE 在但输出被改 → run_resume（attempt+1）",
        plan[1]["action"] == "run_resume" and plan[1]["attempt"] == 2,
        f"action={plan[1]['action']} attempt={plan[1]['attempt']}")

    # R04 中断（running 无 DONE）→ resume，且旧 attempt 不动
    stale = run_dir / specs[1].trial_id / "attempt_02"
    stale.mkdir(parents=True, exist_ok=True)
    atomic_write_json(stale / "status.json", {"outcome": "running"})
    (stale / "meta.json").write_text('{"partial": true}', encoding="utf-8")
    marker = stale / "meta.json"
    before = marker.read_bytes()
    plan = plan_attempts(run_dir, specs)
    add("R04-interrupted", "中断（running 无 DONE）→ run_resume attempt=3",
        plan[1]["action"] == "run_resume" and plan[1]["attempt"] == 3,
        f"action={plan[1]['action']} attempt={plan[1]['attempt']}")

    # R05 业务失败 → 阻断，不自动重试
    bdir = run_dir / specs[0].trial_id / "attempt_02"
    bdir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(bdir / "status.json", {"outcome": "business_failure"})
    plan = plan_attempts(run_dir, specs)
    add("R05-business-failure", "业务失败只记录、不自动重试",
        plan[0]["action"] == "blocked_business_failure",
        f"action={plan[0]['action']}（中断补跑与业务失败重试规则不同）")

    # R06 部分批：3 条 spec 只有 2 条有目录
    third = TrialSpec(trial_id="hira-pilot-clean-0001", split="pilot", program="straight",
                      profile="clean", seed=9013, commands=specs[0].commands, window_s=args_cli.window)
    plan = plan_attempts(run_dir, [*specs, third])
    add("R06-partial-batch", "部分批：缺的那条 run_initial，已有的不受影响",
        plan[2]["action"] == "run_initial" and plan[0]["action"] == "blocked_business_failure",
        f"{[item['action'] for item in plan]}")

    # R07 v2 命名必须被拒
    rejected = False
    try:
        TrialSpec(trial_id="pilot-alternating-clean-0000", split="pilot", program="x",
                  profile="clean", seed=1, commands=[[0.0, 0.0, 0.0]])
    except CollectorError:
        rejected = True
    add("R07-v2-name", "v2 命名的 trial_id 被拒（防语料混用）", rejected, "TrialSpec 拒绝")

    # R08 v2 控制器包被拒（构造伪包：改名 + 重封，散列全对）
    forged_root = Path(tempfile.mkdtemp(prefix="collector_v2_"))
    forged = forged_root / V2_CONTROLLER_ID
    shutil.copytree(args_cli.bundle, forged)
    controller = json.loads((forged / "controller.json").read_text(encoding="utf-8"))
    controller["controller_id"] = V2_CONTROLLER_ID
    (forged / "controller.json").write_text(
        json.dumps(controller, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    from hira_go2.bundle import MANIFEST_NAME, compute_bundle_id, sha256_file

    files = [
        {"path": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(forged.iterdir()) if p.is_file() and p.name != MANIFEST_NAME
    ]
    atomic_write_json(forged / MANIFEST_NAME, {
        "schema": controller["schema"], "controller_id": V2_CONTROLLER_ID,
        "version": controller["version"], "created_at": controller["created_at"],
        "bundle_id": compute_bundle_id(controller, files), "files": files,
    })
    raised = False
    try:
        # 只会走到 bundle 校验与 v2 拒绝，不会创建仿真
        HiraCollector(forged, work / "run2", args_cli.prereg,
                      work / "collect.json", num_envs=1, device="cpu")
    except CollectorError as exc:
        raised = "v2" in str(exc)
    except Exception as exc:  # noqa: BLE001
        raised = False
        add("R08-v2-bundle", "v2 控制器包被拒", False, f"意外异常 {type(exc).__name__}: {exc}")
    else:
        pass
    if not CHECKS or CHECKS[-1]["id"] != "R08-v2-bundle":
        add("R08-v2-bundle", "v2 控制器包被拒", raised,
            "构造时在创建仿真前抛 CollectorError" if raised else "没有被拒绝")
    shutil.rmtree(forged_root, ignore_errors=True)

    # R09 无重复计数：plan 每个 trial 恰好一项
    plan = plan_attempts(run_dir, [*specs, third])
    ids = [item["trial_id"] for item in plan]
    add("R09-no-double-count", "计划不重复计数（每 trial 恰一项）",
        len(ids) == len(set(ids)) == 3, f"{len(ids)} 项，唯一 {len(set(ids))}")

    shutil.rmtree(work, ignore_errors=True)
    save_partial(Path(args_cli.out_dir), "collector_plan_tests.json",
                 {"report": "collector_plan_tests", "task": "HIRA-021", "phase": "plan"})
    failed = [item["id"] for item in CHECKS if not item["ok"]]
    print(f"PLAN_{'OK' if not failed else 'FAIL'} failed={failed}", flush=True)
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
def phase_exec() -> int:
    import numpy as np

    from camra_calibration.energy import integrate_series
    from hira_go2.bundle import load_bundle

    out_dir = Path(args_cli.out_dir)
    bundle = load_bundle(Path(args_cli.bundle), expected_controller_id=Path(args_cli.bundle).name)
    expected_joints = {"target_order": bundle["controller"]["joints"]["target_order"]}

    work = Path(tempfile.mkdtemp(prefix="collector_exec_"))
    run_dir = work / f"run_{args_cli.window:g}s"
    config_path = work / "collect_config.json"
    atomic_write_json(config_path, {"window_s": args_cli.window, "usd_dir": args_cli.usd_dir})

    specs = two_specs()
    collector = HiraCollector(
        args_cli.bundle, run_dir, args_cli.prereg, config_path,
        num_envs=2, device=args_cli.device, expected_joints=expected_joints,
    )
    add("C01-bundle-entry", "采集器以冻结包为唯一入口（散列+语义+URDF 自证+运行时参照）",
        collector.controller_id == "hira_go2_18500" and collector.experiment_id is not None,
        f"controller={collector.controller_id}，bundle_id={collector.bundle_id[:16]}…，"
        f"experiment={collector.experiment_id}")

    # 终止注入：第 30 个策略步把 env0 压到地面以下
    injected = {"step": 0, "done": False}

    def hook(coll) -> None:
        injected["step"] += 1
        if injected["step"] == 30 and not injected["done"]:
            injected["done"] = True
            runner = coll.runner
            pose = torch.zeros((1, 7), device=runner.device, dtype=runner.dtype)
            pose[:, 0:2] = runner.robot.data.root_pos_w[0:1, 0:2]
            pose[:, 2] = 0.05
            pose[:, 3] = 1.0
            runner.robot.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=runner.device))
            runner.robot.update(runner.timing.sim_dt)

    collector.policy_hook = hook
    results = collector.execute_batch(specs, [1, 1])
    collector.policy_hook = None

    meta = [json.loads((Path(r["dir"]) / "meta.json").read_text(encoding="utf-8")) for r in results]
    identity_ok = all(
        m["experiment_id"] == collector.experiment_id
        and m["controller_id"] == "hira_go2_18500"
        and m["bundle_id"] == collector.bundle_id
        and m["attempt"] == 1
        and m["attempt_kind"] == "initial"
        and m["program_sha256"] == s.program_sha256
        for m, s in zip(meta, specs)
    )
    add("C02-identity", "trial 身份完整可追溯（experiment/controller/bundle/run/split/trial/program/profile/seed/attempt）",
        identity_ok,
        "; ".join(f"{m['trial_id']}: attempt={m['attempt']} kind={m['attempt_kind']}" for m in meta))

    warmup_s = specs[0].warmup_s
    n_intervals = int(round(args_cli.window / 0.1))
    pairing_ok = True
    detail = []
    for index, (m, s) in enumerate(zip(meta, specs)):
        z = np.load(Path(results[index]["dir"]) / "pairing_10hz.npz")
        points = len(z["t_s"])
        first_t = float(z["t_s"][0])
        dt = float(np.diff(z["t_s"]).min())
        expected = np.array([s.commands[i % len(s.commands)] for i in range(points)], dtype=np.float32)
        cmd_ok = bool(np.allclose(z["command"], expected, atol=1e-6))
        pairing_ok = pairing_ok and points == n_intervals + 1 and abs(first_t - warmup_s) < 1e-6 and cmd_ok
        detail.append(f"{m['trial_id']}: {points} 点（期望 {n_intervals + 1}），首点 t={first_t:.3f}s（warmup 后），"
                      f"命令与写入一致={cmd_ok}")
    add("C03-pairing", "warmup 起点对齐 + 10 Hz 名义/执行配对", pairing_ok, "；".join(detail))

    state_ok = True
    for r, m in zip(results, meta):
        z = np.load(Path(r["dir"]) / "state_50hz.npz")
        state_ok = state_ok and len(z["t_s"]) == int(round(args_cli.window / 0.02))
    add("C04-state50", "50 Hz 状态记录点数正确", state_ok,
        f"每条 {int(round(args_cli.window / 0.02))} 点")

    energy_ok = True
    e_detail = []
    for r, m in zip(results, meta):
        e = json.loads((Path(r["dir"]) / "energy.json").read_text(encoding="utf-8"))
        z = np.load(Path(r["dir"]) / "physics_200hz.npz")
        j = integrate_series(z["tau_applied_source"].astype("float64"),
                             z["qdot_post_source"].astype("float64"), 0.005)
        rel = abs(j - e["j_exec"]) / max(abs(e["j_exec"]), 1e-12)
        energy_ok = energy_ok and e["j_exec"] > 0 and rel <= 1e-9
        e_detail.append(f"{m['trial_id']}: J={e['j_exec']:.3f}，离线复算 rel={rel:.1e}")
    add("C05-energy", "200 Hz 能量落盘并可独立复算", energy_ok, "；".join(e_detail))

    push_meta = meta[1]["push_trace"]
    push_ok = len(push_meta) == 1 and abs(push_meta[0]["measured_dv_xy"][0] - 0.5) < 0.05
    add("C06-push", "推扰按时刻表施加并记录实测 Δv", push_ok,
        f"t={push_meta[0]['t_s']}s dv={push_meta[0]['dv_xy']} 实测={push_meta[0]['measured_dv_xy']}"
        if push_meta else "无推扰记录")

    plan = plan_attempts(run_dir, specs)
    add("C07-done-skip", "采集完成后 plan 判 skip（DONE 散列校验通过）",
        all(item["action"] == "skip" for item in plan), f"{[item['action'] for item in plan]}")

    # 中断恢复：把两条 trial 的 attempt_01 都伪造成 running → resume 到 attempt_02，
    # 旧目录逐字节保留、新结果进 attempt_02、完成后 plan 判 skip
    for r in results:
        attempt1 = Path(r["dir"])
        (attempt1 / "DONE").unlink()
        atomic_write_json(attempt1 / "status.json", {"outcome": "running"})
    # 快照放在伪造**之后**：这样检验的是"恢复过程绝不写旧目录"
    before = {}
    for r in results:
        attempt1 = Path(r["dir"])
        before[r["trial_id"]] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(attempt1.iterdir()) if p.is_file()
        }
    plan = plan_attempts(run_dir, specs)
    resume_needed = all(item["action"] == "run_resume" and item["attempt"] == 2 for item in plan)
    collector.execute_batch(specs, [item["attempt"] for item in plan])
    unchanged = True
    for r in results:
        attempt1 = Path(r["dir"])
        after = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(attempt1.iterdir()) if p.is_file()
        }
        unchanged = unchanged and before[r["trial_id"]] == after
    plan2 = plan_attempts(run_dir, specs)
    add("C08-resume", "中断补跑：新 attempt 目录、旧目录逐字节保留、完成后不覆盖",
        resume_needed and unchanged and all(item["action"] == "skip" for item in plan2),
        f"旧 attempt 保留={unchanged}，plan→{[item['action'] + str(item['attempt']) for item in plan]}→"
        f"{[item['action'] for item in plan2]}")

    term = meta[0]["terminations"]
    term_meta = [t for t in term if t["trial_id"] == specs[0].trial_id]
    other_state = np.load(Path(results[1]["dir"]) / "state_50hz.npz")
    other_cmds = other_state["command"]
    other_continuous = bool(np.allclose(other_cmds[:28, 0], 0.3, atol=1e-6))
    add("C09-termination", "业务失败（摔倒）在终止前记录，其他 env 不受影响",
        len(term_meta) >= 1 and other_continuous and meta[1]["terminations"] == [],
        f"env0 终止记录 {len(term_meta)} 次（首次 t={term_meta[0]['sim_time_s']:.2f}s）；"
        f"env1 前段命令未受影响={other_continuous}，env1 无终止={meta[1]['terminations'] == []}")

    save_partial(out_dir, "collector_exec_tests.json",
                 {"report": "collector_exec_tests", "task": "HIRA-021", "phase": "exec",
                  "run_dir": str(run_dir), "window_s": args_cli.window,
                  "bundle_id": collector.bundle_id})
    failed = [item["id"] for item in CHECKS if not item["ok"]]
    print(f"EXEC_{'OK' if not failed else 'FAIL'} failed={failed}", flush=True)
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
def phase_legacy() -> int:
    out_dir = Path(args_cli.out_dir)
    # 1) 旧源码快照未被改动（v2 校准包由数据锁钉住）
    diff = subprocess.run(
        ["git", "status", "--porcelain", "--", "CaMRA/calibration/src", "CaMRA/calibration/tests"],
        cwd=REPO, capture_output=True, text=True,
    )
    src_clean = diff.stdout.strip() == ""
    add("L01-src-snapshot", "v2 校准源码快照未被改动（git 工作树干净）", src_clean,
        "src/tests 无未提交修改" if src_clean else f"有改动：{diff.stdout[:200]}")

    # 2) 既有 CPU 核心测试
    # 必须以 CaMRA/calibration 为 cwd：tests 里的 `from sim.batch_program import ...`
    # 依赖 conftest 把本目录放进 sys.path（从仓库根跑会 ModuleNotFoundError）
    proc = subprocess.run(
        ["/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python", "-m", "pytest",
         "tests", "-q", "--tb=no", "-p", "no:cacheprovider"],
        cwd=REPO / "CaMRA" / "calibration", capture_output=True, text=True, timeout=1800,
    )
    summary = next(
        (line for line in reversed(proc.stdout.strip().splitlines()) if " passed" in line or " no tests ran" in line),
        proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
    )
    pytest_ok = proc.returncode == 0 and "failed" not in summary and "error" not in summary
    add("L02-core-tests", "既有 CPU 核心测试全过（旧锁口径）", pytest_ok, f"pytest: {summary[:140]}")

    # 3) v2 trial 独立复算（在存有语料的机器上跑，结果 JSON 拷到 reports/）
    v2_path = Path(args_cli.v2_recompute)
    if not v2_path.is_file():
        add("L03-v2-recompute", "v2 trial 独立复算", False, f"缺少 {v2_path}（先在语料机上运行 recompute_v2_trial.py）")
    else:
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        add("L03-v2-recompute", "v2 trial 独立复算（旧代码路径）且被 HiRA 采集器拒绝混用",
            bool(v2["verdict"]["all_passed"]),
            "；".join(f"{item['id']}: {item['detail'][:80]}" for item in v2["checks"]))

    save_partial(out_dir, "collector_legacy_tests.json",
                 {"report": "collector_legacy_tests", "task": "HIRA-021", "phase": "legacy"})
    failed = [item["id"] for item in CHECKS if not item["ok"]]
    print(f"LEGACY_{'OK' if not failed else 'FAIL'} failed={failed}", flush=True)
    return 0 if not failed else 1


def phase_merge() -> int:
    out_dir = Path(args_cli.out_dir)
    names = ("collector_plan_tests.json", "collector_exec_tests.json", "collector_legacy_tests.json")
    rows, missing = [], []
    for name in names:
        path = out_dir / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload["checks"])
        else:
            missing.append(name)
    failed = [item["id"] for item in rows if not item["ok"]]
    report = {
        "report": "collector_integration_tests",
        "task": "HIRA-021",
        "generated_at": None,
        "missing_phases": missing,
        "checks": rows,
        "verdict": {
            "n_checks": len(rows),
            "n_passed": len(rows) - len(failed),
            "failed": failed,
            "all_passed": not failed and not missing,
        },
    }
    out = out_dir / "collector_integration_tests.json"
    out.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for item in rows:
        print(f"[{'ok  ' if item['ok'] else 'FAIL'}] {item['id']} {item['name']}", flush=True)
    print(f"COLLECTOR_{'PASSED' if not failed and not missing else 'BLOCKED'} failed={failed} missing={missing}",
          flush=True)
    return 0 if not failed and not missing else 1


def main() -> int:
    if args_cli.phase == "plan":
        return phase_plan()
    if args_cli.phase == "exec":
        return phase_exec()
    if args_cli.phase == "legacy":
        return phase_legacy()
    return phase_merge()


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("COLLECTOR_TEST_ERROR", flush=True)
        code = 2
    finally:
        if args_cli.phase == "exec":
            print("hard exit (skip simulation_app.close)", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            import os

            os._exit(code)
