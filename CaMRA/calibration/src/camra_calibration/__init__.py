"""CaMRA 执行误差校准模块（CAL-010 骨架）。

本包只依赖标准库，负责：
  * 机器路径与运行配置解析（config）
  * 冻结采样协议的加载与散列校验（protocol）
  * trial 清单的加载与 schema 校验（manifest）

仿真、策略推理与能量计量在后续任务中实现，不在本包初始化时导入。
"""
from .config import Config, ConfigError
from .manifest import (
    ManifestError,
    ManifestReport,
    SPLITS,
    load_manifest,
    validate_manifest,
)
from .protocol import Protocol, ProtocolError

__all__ = [
    "Config",
    "ConfigError",
    "Protocol",
    "ProtocolError",
    "ManifestError",
    "ManifestReport",
    "SPLITS",
    "load_manifest",
    "validate_manifest",
]
__version__ = "0.1.0"
