"""命令行入口：camra-calib {preflight,protocol,validate-manifest}。

退出码约定：0 通过；2 校验/预检失败；3 散列不匹配；4 文件缺失。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config, ConfigError
from .manifest import ManifestError, validate_manifest
from .protocol import Protocol, ProtocolError


def _cmd_preflight(args: argparse.Namespace) -> int:
    try:
        cfg = Config.load()
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2
    rows = cfg.preflight(require_python=not args.no_python)
    width = max(len(n) for n, _, _ in rows)
    for name, ok, detail in rows:
        print(f"{name:<{width}}  {'ok  ' if ok else 'FAIL'}  {detail}")
    print()
    print(f"repro.env: {cfg.repro_env}  (来源: " +
          ", ".join(f"{k}={v}" for k, v in cfg.sources.items() if v != 'default') + ")")
    bad = [n for n, ok, _ in rows if not ok]
    if bad:
        print(f"VERDICT: blocked -> {bad}")
        return 2
    print("VERDICT: ready")
    return 0


def _cmd_protocol(args: argparse.Namespace) -> int:
    path = Path(args.path) if args.path else None
    if path is not None and not path.is_file():
        print(f"FATAL: protocol not found: {path}")
        return 4
    try:
        proto = Protocol.load(path, verify_hash=not args.no_verify)
    except ProtocolError as exc:
        msg = str(exc)
        print(f"PROTOCOL ERROR: {msg}")
        return 3 if "散列不匹配" in msg else 2
    print(f"version {proto.version}  frozen={proto.frozen}")
    print(f"path    {proto.path}")
    print(f"sha256  {proto.sha256}")
    print(f"unit    {proto.unit}   j_min {proto.j_min} J   steps {proto.measured_steps}   cells {proto.cells}")
    for split in ("pilot", "baseline_fit", "baseline_dev", "baseline_validate",
                  "calibration", "heldout", "near_obstacle"):
        start, count = proto.seed_block(split)
        print(f"  {split:<18} quota {proto.quota(split):>4}   seeds [{start}, {start + count})")
    return 0


def _cmd_validate_manifest(args: argparse.Namespace) -> int:
    try:
        proto = Protocol.load()
    except ProtocolError as exc:
        print(f"PROTOCOL ERROR: {exc}")
        return 2
    try:
        report = validate_manifest(
            Path(args.manifest),
            protocol=proto,
            root=Path(args.root) if args.root else None,
            require_files=args.require_files,
        )
    except ManifestError as exc:
        print(f"MANIFEST ERROR: {exc}")
        return 4
    print(report.summary())
    for w in report.warnings:
        print(f"  WARN  {w}")
    for e in report.errors:
        print(f"  ERR   {e}")
    if not report.ok:
        print("VERDICT: rejected")
        return 2
    print("VERDICT: accepted")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="camra-calib", description="CaMRA 校准工具入口")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("preflight", help="解析并检查机器配置")
    p1.add_argument("--no-python", action="store_true", help="不检查解释器是否存在（便于在别的机器上做静态检查）")
    p1.set_defaults(func=_cmd_preflight)

    p2 = sub.add_parser("protocol", help="加载并展示冻结协议")
    p2.add_argument("--path", default=None)
    p2.add_argument("--no-verify", action="store_true")
    p2.set_defaults(func=_cmd_protocol)

    p3 = sub.add_parser("validate-manifest", help="校验 trial 清单 JSONL")
    p3.add_argument("manifest")
    p3.add_argument("--root", default=None, help="产物根目录；给出时检查是否逃逸与是否存在")
    p3.add_argument("--require-files", action="store_true", help="要求 paths 指向的文件确实存在")
    p3.set_defaults(func=_cmd_validate_manifest)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
