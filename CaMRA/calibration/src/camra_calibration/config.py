"""机器配置解析：环境变量 > CaMRA/locomotion/config/repro.env > 内置默认。

约定与仓库一致（见 CaMRA/locomotion/config/repro.env 与 docs/reproduction.md）：
换机器只改 repro.env 或导出同名环境变量，不改源码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """配置无法解析或指向不可用资源。"""


#: 内置默认值，与本机（tiger10）实测布局一致。
DEFAULTS: dict[str, str] = {
    "CAMRA_PYTHON": "/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python",
    "CAMRA_LOG_ROOT": "/pub/data/caohy/logs/camra",
    "CAMRA_RSL_LOG_ROOT": "/pub/data/caohy/logs/rsl_rl",
    "CAMRA_RUN_ROOT": "/pub/data/caohy/run/camra_go2",
    "CAMRA_ASSET_ROOT": "/pub/data/caohy/unitree_model",
    "CAMRA_BUNDLE_ROOT": "/pub/data/caohy/camra_bundles",
    "CAMRA_XDG_DATA": "/pub/data/caohy/xdgdata",
    "CAMRA_XDG_CACHE": "/pub/data/caohy/xdgcache",
    "CAMRA_TMPDIR": "/pub/data/caohy/tmp",
    "CAMRA_DEVICE": "cuda:0",
    "CAMRA_TMUX_SESSION": "camra-locomotion",
}

#: 冻结控制器版本名，用于拼出默认包路径。
CONTROLLER_VERSION = "camra_go2_velocity_it2999_v2"


def find_repro_env(start: Path | None = None) -> Path | None:
    """向上查找 CaMRA/locomotion/config/repro.env；找不到返回 None。"""
    override = os.environ.get("CAMRA_REPRO_ENV")
    if override:
        p = Path(override)
        return p if p.is_file() else None
    cur = (start or Path(__file__).resolve()).parent
    for _ in range(8):
        cand = cur / "locomotion" / "config" / "repro.env"
        if cand.is_file():
            return cand
        cand2 = cur / "CaMRA" / "locomotion" / "config" / "repro.env"
        if cand2.is_file():
            return cand2
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def parse_env_file(path: Path) -> dict[str, str]:
    """解析 KEY=VALUE 形式的 repro.env（忽略注释与空行，不做变量展开）。"""
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


@dataclass(frozen=True)
class Config:
    python: Path
    log_root: Path
    rsl_log_root: Path
    run_root: Path
    asset_root: Path
    bundle_root: Path
    xdg_data: Path
    xdg_cache: Path
    tmpdir: Path
    device: str
    tmux_session: str
    sources: dict[str, str]
    repro_env: Path | None

    # ---- 由基础目录派生的常用路径 ----
    @property
    def frozen_bundle(self) -> Path:
        return self.bundle_root / "frozen" / CONTROLLER_VERSION

    @property
    def protocol_path(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent / "config" / "calibration_protocol_v1.0.0.json"

    @classmethod
    def load(cls, environ: dict[str, str] | None = None, repro_env: Path | None = None) -> "Config":
        env = dict(os.environ if environ is None else environ)
        re_path = repro_env if repro_env is not None else find_repro_env()
        file_vals: dict[str, str] = {}
        if re_path is not None:
            if not re_path.is_file():
                raise ConfigError(f"repro.env 不存在: {re_path}")
            file_vals = parse_env_file(re_path)
        vals: dict[str, str] = {}
        sources: dict[str, str] = {}
        for key, default in DEFAULTS.items():
            if key in env and env[key].strip():
                vals[key] = env[key].strip()
                sources[key] = "env"
            elif key in file_vals and file_vals[key]:
                vals[key] = file_vals[key]
                sources[key] = "repro.env"
            else:
                vals[key] = default
                sources[key] = "default"
        if not vals["CAMRA_DEVICE"].startswith(("cuda", "cpu")):
            raise ConfigError(f"CAMRA_DEVICE 非法: {vals['CAMRA_DEVICE']!r}（应形如 cuda:0 或 cpu）")
        return cls(
            python=Path(vals["CAMRA_PYTHON"]),
            log_root=Path(vals["CAMRA_LOG_ROOT"]),
            rsl_log_root=Path(vals["CAMRA_RSL_LOG_ROOT"]),
            run_root=Path(vals["CAMRA_RUN_ROOT"]),
            asset_root=Path(vals["CAMRA_ASSET_ROOT"]),
            bundle_root=Path(vals["CAMRA_BUNDLE_ROOT"]),
            xdg_data=Path(vals["CAMRA_XDG_DATA"]),
            xdg_cache=Path(vals["CAMRA_XDG_CACHE"]),
            tmpdir=Path(vals["CAMRA_TMPDIR"]),
            device=vals["CAMRA_DEVICE"],
            tmux_session=vals["CAMRA_TMUX_SESSION"],
            sources=sources,
            repro_env=re_path,
        )

    def preflight(self, require_python: bool = True) -> list[tuple[str, bool, str]]:
        """返回 [(检查名, 是否通过, 说明)]；调用方决定是否因失败退出。"""
        rows: list[tuple[str, bool, str]] = []

        def add(name: str, ok: bool, detail: str = "") -> None:
            rows.append((name, ok, detail))

        add("repro_env", self.repro_env is not None,
            str(self.repro_env) if self.repro_env else "未找到，使用内置默认值")
        if require_python:
            add("python", self.python.is_file(), str(self.python))
        add("log_root", self.log_root.is_dir(), str(self.log_root))
        add("bundle_root", self.bundle_root.is_dir(), str(self.bundle_root))
        add("frozen_bundle", self.frozen_bundle.is_dir(), str(self.frozen_bundle))
        add("asset_root", self.asset_root.is_dir(), str(self.asset_root))
        add("go2_usd", (self.asset_root / "Go2" / "usd" / "go2.usd").is_file(),
            str(self.asset_root / "Go2" / "usd" / "go2.usd"))
        add("protocol", self.protocol_path.is_file(), str(self.protocol_path))
        add("device", self.device.startswith(("cuda", "cpu")), self.device)
        return rows
