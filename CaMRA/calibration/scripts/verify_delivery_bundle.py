#!/usr/bin/env python3
"""CAL-030：从**干净目录**重载交付包并验证可复现。

做四件事：
1. 切到一个新建的空临时目录（证明不依赖仓库 cwd）；
2. 按索引逐个校验文件散列与内容散列；
3. 从索引里的路径导入 ``camra_calibration``、加载 margins 与功率模型，核对散列一致；
4. 抽 3 条已采集 trial 现场复算 ε_c/ε_d，确认与报告口径可比（只读，不写任何产物）。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CAL-030 交付包重载验证")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--sample-split", default="heldout")
    ap.add_argument("--n-sample", type=int, default=3)
    args = ap.parse_args(argv)

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    problems: list[str] = []
    checked = 0

    # 1) 干净 cwd
    with tempfile.TemporaryDirectory(prefix="camra_reload_") as tmp:
        import os
        os.chdir(tmp)
        cwd_now = Path.cwd()

        # 2) 散列校验
        for name, entry in man["artifacts"].items():
            p = Path(entry["path"])
            if not p.is_file():
                problems.append(f"artifact 缺失: {name} {p}")
                continue
            if sha256_file(p) != entry["sha256"]:
                problems.append(f"artifact 散列不符: {name}")
            checked += 1
        for name, entry in man["reports"].items():
            p = Path(entry["path"])
            if not p.is_file():
                problems.append(f"报告缺失: {name}")
            elif sha256_file(p) != entry["sha256"]:
                problems.append(f"报告散列不符: {name}")
            else:
                checked += 1
        for split, entry in man["data_index"].items():
            p = Path(entry["manifest"])
            if not p.is_file():
                problems.append(f"清单缺失: {split}")
            elif sha256_file(p) != entry["sha256"]:
                problems.append(f"清单散列不符: {split}")
            else:
                checked += 1

        # 3) 从索引导入代码并加载冻结对象
        sys.path.insert(0, str(Path(man["code"]["calibration_root"]) / "src"))
        from camra_calibration.margins import Margins
        from camra_calibration.power import PowerModel

        margins = Margins.load(Path(man["artifacts"]["margins"]["path"]))
        model = PowerModel.load(Path(man["artifacts"]["power_model"]["path"]))
        if margins.sha256() != man["artifacts"]["margins"]["content_sha256"]:
            problems.append("margins 内容散列与索引不符")
        if model.sha256() != man["artifacts"]["power_model"]["content_sha256"]:
            problems.append("power_model 内容散列与索引不符")

        # 4) 抽 3 条 trial 现场复算
        import numpy as np
        import torch
        from camra_calibration.coverage import cost_covered, position_covered
        from camra_calibration.pairing import compute_trial_errors

        split = args.sample_split
        root = Path(man["data_index"][split]["artifacts_root"])
        trials = [p for p in sorted(root.glob("*/DONE"))][: args.n_sample]
        sample = []
        for done in trials:
            d = done.parent
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            energy = json.loads((d / "energy.json").read_text(encoding="utf-8"))
            pair = torch.load(d / "pairing_10hz.npz", map_location="cpu")
            state = torch.load(d / "state_50hz.npz", map_location="cpu")
            cmd = np.asarray(pair["cmd"], dtype=np.float64)
            j_nom = model.j_nom(cmd)
            err = compute_trial_errors(
                trial_id=meta["trial_id"], outcome=meta["outcome"],
                t_pair=np.asarray(pair["t"], dtype=np.float64),
                exec_pos=np.asarray(pair["exec_pos"], dtype=np.float64),
                nom_pos=np.asarray(pair["nominal_pos"], dtype=np.float64),
                j_exec=float(energy["j_exec"]), j_nom=j_nom,
                steps_executed=meta.get("steps_executed"),
                t50=np.arange(int(state["root_pos"].shape[0]), dtype=np.float64) * 0.02,
                exec50_pos=np.asarray(state["root_pos"], dtype=np.float64)[:, :2],
                cmd=cmd, source=str(d))
            sample.append({"trial_id": meta["trial_id"], "outcome": meta["outcome"],
                           "eps_c": err.epsilon_c, "eps_d": err.epsilon_d,
                           "cost_covered": cost_covered(err.epsilon_c, margins.alpha),
                           "position_covered": position_covered(err.epsilon_d, margins.delta)})

        report = {
            "cwd_isolated": str(cwd_now), "n_checked": checked,
            "problems": problems, "ok": not problems,
            "margins_sha256": margins.sha256(), "alpha": margins.alpha, "delta": margins.delta,
            "model_sha256": model.sha256(),
            "lock_sha256": man.get("lock_sha256"),
            "sample": sample,
            "code": man["code"],
        }
    rep_p = Path(args.report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ok", "n_checked", "problems",
                                             "alpha", "delta", "margins_sha256")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
