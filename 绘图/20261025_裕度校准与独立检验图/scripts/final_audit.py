"""Check delivery, prior source integrity, documentation links and worklog state."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from compute_statistics import ROOT
from validate_data import validate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(complete=False):
    workspace = ROOT.parents[1]
    baseline = json.loads((ROOT / "reviews/source_manifest.json").read_text(encoding="utf-8"))
    altered = [f["path"] for f in baseline["files"] if not (workspace/f["path"]).is_file() or sha(workspace/f["path"])!=f["sha256"]]
    plan_path = workspace / "plan/margin_calibration_figure_20261025_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    task_done = [t["id"] for t in plan["tasks"] if t["status"]]
    record_errors = [t["id"] for t in plan["tasks"] if t["status"] and (not t["work_records"] or not t["verification_result"])]
    stages = list((ROOT / "reviews").glob("stage_*.md"))
    required = ["data/paired_rollouts.csv", "data/statistics.json", "data/metadata.json", "README.md", "data/README.md", "reviews/real_data_replacement.md", "scripts/plot_margin_calibration.py", "config/style.json"]
    required += [f"output/final/{n}" for n in ["margin_calibration.pdf", "margin_calibration.svg", "margin_calibration.png", "caption.txt", "single_column_preview.png", "render_manifest.json"]]
    missing = [p for p in required if not (ROOT/p).is_file()]
    broken_links = []
    for doc in [ROOT/"README.md", ROOT/"data/README.md", ROOT/"reviews/real_data_replacement.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if "://" not in target and not (doc.parent/target.split("#")[0]).exists():
                broken_links.append({"file":str(doc.relative_to(ROOT)),"target":target})
    stats = json.loads((ROOT/"data/statistics.json").read_text(encoding="utf-8"))
    cm = stats["groups"]["calibration"]["metrics"]
    anchors = {"median_cost":cm["cost_overrun"]["median"]==.09,"q90_cost":cm["cost_overrun"]["q90"]==.18,"median_geometry":cm["max_deviation_m"]["median"]==.06,"q90_geometry":cm["max_deviation_m"]["q90"]==.12}
    audits = {n:json.loads((ROOT/f"reviews/{n}.json").read_text(encoding="utf-8"))["valid"] for n in ["figure_numeric_audit","export_inspection","reproduction_audit","reproduction_data_audit"]}
    data = validate(ROOT/"data/paired_rollouts.csv", final=True)
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=workspace,text=True).strip()
    # Confirm the only modification to the tracked drawing index is an append.
    original_index = subprocess.check_output(["git","show","HEAD:绘图/README.md"],cwd=workspace).decode("utf-8").replace("\r\n","\n")
    current_index = (workspace/"绘图/README.md").read_text(encoding="utf-8").replace("\r\n","\n")
    index_append_only = current_index.startswith(original_index)
    expected_tasks = 38 if complete else 37
    errors = []
    if altered: errors.append("read-only source changed")
    if missing or broken_links: errors.append("missing files or documentation links")
    if record_errors or len(task_done)!=expected_tasks: errors.append("worklog status/evidence mismatch")
    if len(stages)!=12: errors.append("stage review count mismatch")
    if not all(anchors.values()) or not all(audits.values()) or not data["valid"]: errors.append("data/artifact/reproduction check failed")
    if current_head!=baseline["git_head"] or not index_append_only: errors.append("Git head or append-only index changed unexpectedly")
    manifest = {"valid":not errors,"errors":errors,"data_status":"hypothetical","complete":complete,"completed_tasks":len(task_done),"task_total":len(plan["tasks"]),"plan_sha256":sha(plan_path),"stage_reviews":len(stages),"source_files_checked":len(baseline["files"]),"changed_source_files":altered,"missing_files":missing,"broken_document_links":broken_links,"work_record_errors":record_errors,"calibration_anchors":anchors,"audit_results":audits,"data_validation":data,"git_head_unchanged":current_head==baseline["git_head"],"drawing_index_append_only":index_append_only,"files":[{"path":str(p.relative_to(ROOT)).replace("\\","/"),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(ROOT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name!="delivery_manifest.json"]}
    (ROOT/"reviews/delivery_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k!="files"},ensure_ascii=False))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser()
    p.add_argument("--complete",action="store_true")
    args=p.parse_args()
    run(args.complete)
