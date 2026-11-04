"""逐 trial 读取与分类（CAL-016/018 共用，不含仿真依赖）。

``classify_trial_dir`` 是"目录 → TrialErrors"的唯一入口：采集后的复算脚本与
故障注入脚本都走它，保证分类口径只有一处。npz 用 torch 读取（延迟导入），
缺失/损坏一律归类为 ``data_invalid:missing_shard``，不抛异常。
"""
from __future__ import annotations

import json
from pathlib import Path

from .pairing import Issue, TrialErrors, compute_trial_errors

J_MIN = 1.0
EXPECTED_STEPS = 6000
FILES = ("pairing_10hz.npz", "state_50hz.npz", "energy.json", "meta.json")


def load_npz(path: Path):
    if not path.is_file():
        return None
    try:
        import torch  # 延迟导入
        return torch.load(path, map_location="cpu")
    except Exception:
        return "CORRUPT"


def _invalid(trial_id: str, outcome: str, reason: str, detail: str, source: str,
             complete: bool) -> TrialErrors:
    return TrialErrors(trial_id=trial_id, outcome=outcome, complete=complete,
                       official=False, issues=[Issue(reason, detail, source)])


def classify_trial_dir(trial_dir: Path, trial_id: str | None = None,
                       j_min: float = J_MIN, expected_steps: int = EXPECTED_STEPS,
                       meta: dict | None = None) -> TrialErrors:
    d = Path(trial_dir)
    tid = trial_id or d.name.split("__a")[0]
    if not d.is_dir():
        return _invalid(tid, "missing", "missing_shard", "trial 目录不存在", str(d), False)
    missing = [f for f in FILES if not (d / f).is_file()]
    if meta is None:
        if (d / "meta.json").is_file():
            try:
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return _invalid(tid, "unknown", "missing_shard", "meta.json 损坏", str(d), False)
        else:
            meta = {"outcome": "unknown"}
    outcome = meta.get("outcome", "unknown")
    complete = outcome == "completed"
    if missing:
        return _invalid(tid, outcome, "missing_shard", f"缺少 {missing}", str(d), complete)

    pair = load_npz(d / "pairing_10hz.npz")
    state = load_npz(d / "state_50hz.npz")
    if pair in (None, "CORRUPT") or state in (None, "CORRUPT"):
        return _invalid(tid, outcome, "missing_shard", "pairing_10hz/state_50hz 缺失或损坏",
                        str(d), complete)
    if not (d / "energy.json").is_file():
        return _invalid(tid, outcome, "missing_shard", "energy.json 缺失", str(d), complete)
    energy = json.loads((d / "energy.json").read_text(encoding="utf-8"))

    import numpy as np
    t50 = ((np.arange(int(state["root_pos"].shape[0])) + 1) * 0.02
           if "root_pos" in state else None)
    r = compute_trial_errors(
        trial_id=tid, outcome=outcome,
        t_pair=pair["t"].numpy(), exec_pos=pair["exec_pos"].numpy(),
        nom_pos=pair["nominal_pos"].numpy(),
        j_exec=energy.get("j_exec"), j_nom=energy.get("j_nom"), j_min=j_min,
        cmd=pair["cmd"].numpy() if "cmd" in pair else None,
        t50=t50,
        exec50_pos=state["root_pos"][:, :2].numpy() if "root_pos" in state else None,
        steps_executed=meta.get("steps_executed"),
        expected_steps=meta.get("steps_expected", expected_steps),
        source=str(d),
    )
    if not r.complete and r.epsilon_d is not None:
        r.notes.append(f"prefix 诊断：仅前 {int(np.asarray(pair['t']).size)} 个配对点")
    return r
