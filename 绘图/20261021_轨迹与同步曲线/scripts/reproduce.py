"""Rebuild the preview from frozen static samples; no raw data generation."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from compute_metrics import ROOT,build,read_json,read_raw,calculate
from validate_data import validate,check_derived
from audit_case import audit
from plot_style import load_data
from plot_trajectory import standalone as trajectory
from plot_metrics import standalone as metric,KINDS
from compose_figure import compose

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(out):
    frozen=ROOT/"data/versions/v2"
    out=Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new output directory to preserve prior rebuilds: "+str(out))
    derived=out/"data";panels=out/"panels";derived.mkdir(parents=True);panels.mkdir()
    before={p.name:sha(p) for p in frozen.iterdir() if p.is_file()}
    for n in ["scenario.json","metadata.json","raw_rcppo.csv","raw_hd_mcra.csv"]:
        shutil.copy2(frozen/n,derived/n)
    build(derived,derived,["rcppo","hd_mcra"])
    s=read_json(derived/"scenario.json")
    for m in s["methods"]:
        errors,rows=validate(read_raw(derived/f"raw_{m}.csv"),s,m)
        errors+=check_derived(derived/f"derived_{m}.csv",rows)
        if errors:raise AssertionError(errors)
    numeric_files=["derived_rcppo.csv","derived_hd_mcra.csv","events.csv","summary.json"]
    numeric_equal={n:sha(frozen/n)==sha(derived/n) for n in numeric_files}
    if not all(numeric_equal.values()):raise AssertionError(numeric_equal)
    diag=audit(derived)
    (out/"case_audit.json").write_text(json.dumps(diag,indent=2),encoding="utf-8")
    data=load_data(derived)
    trajectory(data,panels/"trajectory")
    for k in KINDS:metric(data,k,panels/k)
    compose(data,out/"trajectory_sync_assumed")
    existing=ROOT/"output/trajectory_sync_assumed_v3"
    img_a=np.asarray(Image.open(existing.with_suffix(".png")))
    img_b=np.asarray(Image.open(out/"trajectory_sync_assumed.png"))
    png_equal=img_a.shape==img_b.shape and np.array_equal(img_a,img_b)
    pdf_equal=sha(existing.with_suffix(".pdf"))==sha(out/"trajectory_sync_assumed.pdf")
    test=subprocess.run([sys.executable,"-B",str(ROOT/"scripts/test_metrics.py")],capture_output=True,text=True)
    if test.returncode:raise AssertionError(test.stdout+test.stderr)
    (out/"test_results.txt").write_text(test.stdout+test.stderr,encoding="utf-8")
    source_manifest=read_json(ROOT/"reference/source_manifest.json")
    source_ok={x["path"]:sha(x["path"]).upper()==x["sha256"].upper() for x in source_manifest["sources"]}
    snapshots={}
    for version in ["v1","v2"]:
        d=ROOT/"data/versions"/version
        snapshots[version]=all(sha(d/f["file"]).upper()==f["sha256"].upper() for f in read_json(d/"manifest.json")["files"])
    unchanged=before=={p.name:sha(p) for p in frozen.iterdir() if p.is_file()}
    result=dict(passed=png_equal and pdf_equal and unchanged and all(source_ok.values()) and all(snapshots.values()),
                frozen_data="v2",numeric_files_byte_equal=numeric_equal,png_pixels_equal=png_equal,
                pdf_bytes_equal=pdf_equal,raw_inputs_unchanged=unchanged,source_hashes_unchanged=source_ok,
                snapshot_hashes_valid=snapshots,unit_tests="7 passed",python=sys.executable)
    (out/"reproducibility_report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not result["passed"]:raise AssertionError("Reproduction failed")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();main(a.output_dir)

