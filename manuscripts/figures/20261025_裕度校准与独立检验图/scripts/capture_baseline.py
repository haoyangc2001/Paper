"""Capture read-only sources and the existing local plotting environment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    source_paths = [
        "CCDC/latex/1020/main.tex",
        "CCDC/references/calibration_review_20261025",
        "figures/20261021_轨迹与同步曲线/data",
        "figures/20261021_部署与校准总览图/data",
        "figures/20261023_多方法指标对比/data",
        "figures/20261025_预算扫描图/data",
        "figures/20261025_模块消融图/data",
    ]
    files = []
    for rel in source_paths:
        path = WORKSPACE / rel
        assert path.exists(), rel
        for f in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if f.is_file():
                files.append({"path": str(f.relative_to(WORKSPACE)), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size})
    manifest = {
        "scope": "current local paper workspace; no remote experimental logs searched",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip(),
        "initial_changes": subprocess.check_output(["git", "status", "--short"], cwd=WORKSPACE, text=True, encoding="utf-8"),
        "files": files,
        "batch_data_status": "No existing 800/400 paired calibration/test record set identified; existing trajectory and cost-pair data are illustrative and duplicated across outputs.",
    }
    packages = {}
    for name in ["numpy", "matplotlib", "Pillow", "scipy", "pandas", "PyMuPDF"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed; optional and not required"
    for name in ["numpy", "matplotlib", "Pillow", "PyMuPDF"]:
        assert not packages[name].startswith("not installed"), name
    environment = {"python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "packages": packages, "dependency_changes": False}
    out = ROOT / "reviews"
    out.mkdir(exist_ok=True)
    for name, obj in [("source_manifest.json", manifest), ("environment.json", environment)]:
        target = out / name
        if target.exists():
            raise FileExistsError(f"Baseline already recorded: {target}")
        target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source_file_count": len(files), "environment": environment}, ensure_ascii=False))


if __name__ == "__main__":
    main()
