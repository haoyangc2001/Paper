"""结果区守卫（HIRA-024）：heldout / near_obstacle 在 α/δ 冻结前不采、不分析。

不是文档约定，而是**可执行**的检查：采集脚本与分析脚本在开工前调用
:func:`assert_allowed`，缺 `config/margins_hira_v1.json`（HIRA-031 的 α/δ 冻结文件）
就直接拒绝。

用法：

    from results_guard import assert_allowed
    assert_allowed("heldout", stage="collect", experiment_dir=EXP)

设计取舍：
* 受保护 split 的判定不接受环境变量或命令行覆盖——只能靠冻结文件存在；
* 冻结文件必须自带 ``state == "frozen"`` 与 ``alpha``/``delta`` 字段，
  防止"放一个空文件就算冻结"。
"""

from __future__ import annotations

import json
from pathlib import Path

PROTECTED_SPLITS = ("heldout", "near_obstacle")
FREEZE_FILE = "config/margins_hira_v1.json"
STAGES = ("collect", "analyze", "publish")


class GuardError(RuntimeError):
    """未满足冻结前置条件即试图采集/分析受保护 split。"""


def freeze_state(experiment_dir: Path) -> dict:
    path = Path(experiment_dir) / FREEZE_FILE
    if not path.is_file():
        return {"frozen": False, "reason": f"缺少 {FREEZE_FILE}（HIRA-031 尚未冻结 α/δ）"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"frozen": False, "reason": f"{FREEZE_FILE} 不是合法 JSON：{exc}"}
    if payload.get("state") != "frozen":
        return {"frozen": False, "reason": f"state={payload.get('state')!r}，不是 frozen"}
    for key in ("alpha", "delta"):
        if key not in payload:
            return {"frozen": False, "reason": f"冻结文件缺少 {key}"}
    return {"frozen": True, "path": str(path), "alpha": payload["alpha"], "delta": payload["delta"]}


def assert_allowed(split: str, stage: str, experiment_dir: Path) -> None:
    if stage not in STAGES:
        raise GuardError(f"未知 stage {stage!r}（允许 {STAGES}）")
    if split not in PROTECTED_SPLITS:
        return
    state = freeze_state(experiment_dir)
    if not state["frozen"]:
        raise GuardError(
            f"{split} 在 α/δ 冻结前不得 {stage}：{state['reason']}。"
            "协议规定 heldout/near_obstacle 不得提前采集、读取或据此调整任何参数。"
        )


def describe(experiment_dir: Path) -> dict:
    return {
        "protected_splits": list(PROTECTED_SPLITS),
        "freeze_file": FREEZE_FILE,
        "freeze_state": freeze_state(experiment_dir),
        "rule": "受保护 split 的 collect/analyze/publish 一律先过 assert_allowed；不可用环境变量绕过",
    }
