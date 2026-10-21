"""Check raw and derived consistency; diagnostic hypothetical data, not physical validation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from compute_metrics import ROOT, read_json, read_raw, calculate, swept_clearance, summarize

def validate(raw,s,method,complete=True):
    errors=[]
    if not np.allclose(raw[:,0],np.arange(len(raw))*s["dt_s"],rtol=0,atol=1e-8):
        errors.append("raw timestamps: expected 0,dt,...; order/gaps invalid")
    for i,r in enumerate(raw):
        if np.any(np.abs(r[1:4])>np.array(s["command_limits"])+1e-8):
            errors.append(f"raw row {i+2}: command limit")
        if np.linalg.norm(r[4:6])>s["max_actual_linear_speed_mps"]+1e-8:
            errors.append(f"raw row {i+2}: actual speed")
        if abs(r[6])>s["command_limits"][2]+1e-8:
            errors.append(f"raw row {i+2}: actual yaw rate")
    for i,d in enumerate(np.diff(raw[:,4:7],axis=0)/s["dt_s"]):
        if np.linalg.norm(d[:2])>s["max_linear_acceleration_mps2"]+1e-8 or abs(d[2])>s["max_yaw_acceleration_radps2"]+1e-8:
            errors.append(f"raw row {i+3}: acceleration jump")
    rows=calculate(raw,s,method)
    for i,r in enumerate(rows):
        if i<len(raw) and r["terminal"]:
            errors.append(f"state row {i}: motion after {r['terminal']}")
        if r["clearance_nom_m"] < s["methods"][method]["delta_m"]-1e-8:
            errors.append(f"state row {i}: nominal margin violated")
        if i:
            a,b=rows[i-1],r
            if swept_clearance([a["x_exec_m"],a["y_exec_m"]],[b["x_exec_m"],b["y_exec_m"]],s)<=0 and b["terminal"]!="collision":
                errors.append(f"state row {i}: unreported swept collision")
        if r["t_s"]>s["horizon_s"]+1e-8:
            errors.append(f"state row {i}: horizon exceeded")
    if complete and not rows[-1]["terminal"]:
        errors.append("complete trial must end at first goal/collision/timeout")
    return errors,rows

def check_derived(path,expected):
    errors=[]
    with open(path,encoding="utf-8-sig") as f: observed=list(csv.DictReader(f))
    if len(observed)!=len(expected):return ["derived row count mismatch"]
    for i,(a,b) in enumerate(zip(observed,expected)):
        if set(a)!=set(b):
            return ["derived schema mismatch"]
        for k,v in b.items():
            if v is None:
                ok=a[k]==""
            elif isinstance(v,str):
                ok=a[k]==v
            else:
                try:ok=np.isclose(float(a[k]),v,rtol=0,atol=1e-8)
                except ValueError:ok=False
            if not ok:errors.append(f"derived row {i+2}: {k} mismatch")
    return errors

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,default=ROOT/"data")
    p.add_argument("--derived-dir",type=Path)
    p.add_argument("--methods",nargs="+",default=["rcppo","hd_mcra"])
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    s=read_json(a.data_dir/"scenario.json")
    report={"data_status":"assumed","methods":{}}
    for m in a.methods:
        try:
            raw=read_raw(a.data_dir/f"raw_{m}.csv")
            errors,rows=validate(raw,s,m,not a.pilot)
            d=(a.derived_dir or a.data_dir)/f"derived_{m}.csv"
            if not d.exists():errors.append("derived file missing")
            else:errors.extend(check_derived(d,rows))
            report["methods"][m]={"passed":not errors,"errors":errors,"summary":summarize(rows,raw,s)}
        except (ValueError,KeyError) as ex:
            report["methods"][m]={"passed":False,"errors":[str(ex)]}
    report["passed"]=all(v["passed"] for v in report["methods"].values())
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report["passed"] else 1)

