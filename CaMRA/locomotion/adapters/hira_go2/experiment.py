"""HiRA 实验的显式配置入口。

设计原则（HIRA-010）：
1. 所有资源必须显式给出，**不提供任何 v2 默认值**；
2. 任何指向 v2 冻结物的路径都视为配置错误，在启动仿真之前拒绝；
3. 不读取 ``Protocol.load()`` 的默认协议、不读默认 margins、不做输出路径 fallback；
4. 校验只做静态检查，不导入 isaaclab，因此可在 CPU 上单测。

本模块只定义「配置」，不定义「怎么跑」——运行入口在 runner 层，复用
``camra_calibration`` 里的纯函数（时间轴、命令、名义积分、能量、配对）。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

CONTROLLER_ID = "camra_hira_go2_18500_lab_v1"
EXPERIMENT_ID = "hira_go2_18500"

#: 一旦出现在任何路径里，说明配置误指向 v2 的冻结物（或 v2 数据树）。
#: 注意：配置文件类标记必须带 ``/config/`` 前缀。否则 ``hira_power_model_v1.json``
#: 这类合法名字会被 ``power_model_v1.json`` 子串误判成 v2（HIRA-010 自测 U6 抓到过一次）。
V2_MARKERS: tuple[str, ...] = (
    "camra_go2_velocity_it2999_v2",
    "/logs/camra/",
    "/config/calibration_protocol_v1.0.2.json",
    "/config/power_model_v1.json",
    "/config/footprint_v1.json",
    "/config/margins_v1.json",
    "/config/step_cost_schema_v1.json",
    "/config/data_lock_v1.json",
    "/config/near_obstacle_maps_v1.json",
)

#: 必须显式提供、没有默认值的字段。
REQUIRED_FIELDS: tuple[str, ...] = (
    "experiment_root",
    "data_root",
    "bundle_root",
    "checkpoint",
    "urdf",
    "asset_root",
    "protocol",
    "model",
    "margins",
    "maps",
    "lock",
)

#: 必须真实存在的文件（目录类字段单独检查）。
REQUIRED_FILES: tuple[str, ...] = ("checkpoint", "urdf", "protocol", "model", "margins", "maps", "lock")

#: 必须真实存在且可写的目录。
REQUIRED_DIRS: tuple[str, ...] = ("experiment_root", "data_root", "bundle_root")

#: 只读目录（存在即可）。
READONLY_DIRS: tuple[str, ...] = ("asset_root",)

#: 从环境变量读取时使用的名字前缀。
ENV_PREFIX = "HIRA_"


class ConfigError(RuntimeError):
    """配置缺失、误指向 v2，或与实验身份不一致。"""


@dataclass(frozen=True)
class HiraExperiment:
    """一次 HiRA 实验的全部显式绑定。字段无默认值，避免静默落到 v2。"""

    experiment_root: Path
    data_root: Path
    bundle_root: Path
    checkpoint: Path
    urdf: Path
    asset_root: Path
    protocol: Path
    model: Path
    margins: Path
    maps: Path
    lock: Path
    device: str = "cuda:0"
    controller_id: str = CONTROLLER_ID
    experiment_id: str = EXPERIMENT_ID
    visible_devices: str = "2"
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ---- 构造 ----

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "HiraExperiment":
        """从环境变量构造；**缺任何一项都失败**，不填默认值。"""
        src = os.environ if env is None else env
        values: dict[str, object] = {}
        missing: list[str] = []
        for name in REQUIRED_FIELDS:
            key = ENV_PREFIX + name.upper()
            raw = src.get(key)
            if not raw:
                missing.append(key)
            else:
                values[name] = Path(raw)
        if missing:
            raise ConfigError("缺少必需的环境变量（无默认值）：" + ", ".join(missing))
        values["device"] = src.get("HIRA_DEVICE", "cuda:0")
        values["visible_devices"] = src.get("HIRA_VISIBLE_DEVICES", "2")
        values["controller_id"] = src.get("HIRA_CONTROLLER_ID", CONTROLLER_ID)
        values["experiment_id"] = src.get("HIRA_EXPERIMENT_ID", EXPERIMENT_ID)
        return cls(**values)  # type: ignore[arg-type]

    # ---- 校验 ----

    def identity_problems(self) -> list[str]:
        problems: list[str] = []
        if self.controller_id != CONTROLLER_ID:
            problems.append(f"controller_id 必须是 {CONTROLLER_ID}，得到 {self.controller_id!r}")
        if self.experiment_id != EXPERIMENT_ID:
            problems.append(f"experiment_id 必须是 {EXPERIMENT_ID}，得到 {self.experiment_id!r}")
        return problems

    def v2_contamination_problems(self) -> list[str]:
        """任何字段指向 v2 冻结物 → 报错。这是「拒混」的核心检查。"""
        problems: list[str] = []
        for name in REQUIRED_FIELDS:
            text = str(getattr(self, name)).replace("\\", "/")
            for marker in V2_MARKERS:
                if marker in text:
                    problems.append(f"{name} 指向 v2 资源（命中 {marker!r}）：{text}")
        # 数据根必须位于 HiRA 自己的树下，不能在 v2 的 logs/camra 里
        data_root = str(self.data_root).replace("\\", "/")
        if data_root.rstrip("/").endswith("/logs/camra"):
            problems.append(f"data_root 不能是 v2 的日志根：{data_root}")
        return problems

    def existence_problems(self) -> list[str]:
        problems: list[str] = []
        for name in REQUIRED_FILES:
            p: Path = getattr(self, name)
            if not p.is_file():
                problems.append(f"{name} 不是文件或不存在：{p}")
        for name in REQUIRED_DIRS:
            p = getattr(self, name)
            if not p.is_dir():
                problems.append(f"{name} 不是目录或不存在：{p}")
            elif not os.access(p, os.W_OK):
                problems.append(f"{name} 不可写：{p}")
        for name in READONLY_DIRS:
            p = getattr(self, name)
            if not p.is_dir():
                problems.append(f"{name} 不是目录或不存在：{p}")
        return problems

    def validate(self, *, require_files: bool = True) -> list[str]:
        """返回全部问题；空列表表示可进入运行阶段。"""
        problems = self.identity_problems() + self.v2_contamination_problems()
        if require_files:
            problems += self.existence_problems()
        return problems

    def require_valid(self, *, require_files: bool = True) -> None:
        problems = self.validate(require_files=require_files)
        if problems:
            raise ConfigError("HiRA 实验配置不可用：\n  - " + "\n  - ".join(problems))

    # ---- 指纹 ----

    def fingerprint(self) -> dict:
        """记录身份与文件散列，供报告与后续锁使用。"""
        files = {}
        for name in REQUIRED_FILES:
            p: Path = getattr(self, name)
            files[name] = {"path": str(p), "sha256": _sha256(p) if p.is_file() else None}
        return {
            "controller_id": self.controller_id,
            "experiment_id": self.experiment_id,
            "device": self.device,
            "visible_devices": self.visible_devices,
            "roots": {
                "experiment_root": str(self.experiment_root),
                "data_root": str(self.data_root),
                "bundle_root": str(self.bundle_root),
                "asset_root": str(self.asset_root),
            },
            "files": files,
        }

    # ---- 输出路径（全部在 data_root 或 bundle_root 下，无 fallback）----

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or run_id in (".", ".."):
            raise ConfigError(f"run_id 非法：{run_id!r}")
        return self.data_root / "trials" / run_id

    def integration_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id:
            raise ConfigError(f"run_id 非法：{run_id!r}")
        return self.data_root / "integration" / run_id

    def output_paths(self, run_id: str) -> dict[str, Path]:
        return {
            "trials": self.run_dir(run_id),
            "integration": self.integration_dir(run_id),
            "bundle": self.bundle_root / self.controller_id,
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    """读 JSON；不做任何默认值填充。"""
    if not path.is_file():
        raise ConfigError(f"文件不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))
