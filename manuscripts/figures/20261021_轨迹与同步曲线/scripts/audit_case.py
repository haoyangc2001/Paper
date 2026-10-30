import csv,json,hashlib,argparse
from pathlib import Path
import numpy as np
from compute_metrics import ROOT,read_json,read_raw,calculate,swept_clearance

def audit(data_dir):
    d=Path(data_dir);s=read_json(d/"scenario.json")
    rows={m:calculate(read_raw(d/f"raw_{m}.csv"),s,m) for m in s["methods"]}
    events=list(csv.DictReader(open(d/"events.csv",encoding="utf-8")))
    checks=[]
    for e in events:
        r=rows[e["method"]][int(e["sample_index"])]
        assert abs(float(e["t_s"])-r["t_s"])<1e-8
        assert abs(float(e["x_m"])-r["x_exec_m"])<1e-8
        assert abs(float(e["y_m"])-r["y_exec_m"])<1e-8
        threshold=next(v["x_m"] for v in s["events"] if v["id"]==e["event_id"])
        assert r["x_exec_m"]>=threshold
        assert rows[e["method"]][int(e["sample_index"])-1]["x_exec_m"]<threshold
        checks.append(f"{e['method']} {e['event_id']}: first crossing and coordinates verified")
    diag={}
    for m,r in rows.items():
        raw=read_raw(d/f"raw_{m}.csv")
        points=np.array([[x["x_exec_m"],x["y_exec_m"]] for x in r])
        # Independent cumulative displacement at fixed yaw=0 for this authored case.
        assert np.allclose(points[1:],np.cumsum(raw[:,4:6]*s["dt_s"],axis=0),rtol=0,atol=1e-8)
        assert np.all(raw[:,[3,6]]==0)
        assert sum(bool(x["terminal"]) for x in r)==1
        diag[m]=dict(min_swept_clearance_m=min(swept_clearance(a,b,s) for a,b in zip(points[:-1],points[1:])),
                     max_acceleration_mps2=float(np.linalg.norm(np.diff(raw[:,4:6],axis=0)/s["dt_s"],axis=1).max()),
                     terminal_count=sum(bool(x["terminal"]) for x in r),
                     raw_sha256=hashlib.sha256((d/f"raw_{m}.csv").read_bytes()).hexdigest())
    n=min(len(rows["rcppo"]),len(rows["hd_mcra"]))-1
    ec=np.array([rows["hd_mcra"][i]["tracking_error_mps"]-rows["rcppo"][i]["tracking_error_mps"] for i in range(n)])
    return dict(passed=True,data_status="assumed",events_checked=checks,diagnostics=diag,
                common_intervals=n,ours_error_higher=int((ec>1e-10).sum()),ours_error_lower=int((ec<-1e-10).sum()),
                caveat="No policy or physical simulation. No paired real disturbances. Values merely illustrate plausible shared-interface residuals.")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--output",type=Path,default=ROOT/"output/case_audit.json")
    args=parser.parse_args()
    result=audit(args.data_dir)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

