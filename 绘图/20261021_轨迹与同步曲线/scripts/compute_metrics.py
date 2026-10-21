"""Derived calculations only. Never generates or modifies raw motion samples."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["t_s", "v_cmd_x_mps", "v_cmd_y_mps", "yaw_rate_cmd_radps",
           "v_exec_x_mps", "v_exec_y_mps", "yaw_rate_exec_radps"]

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_raw(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Raw columns/units must be exactly " + ",".join(COLUMNS))
        result = []
        for i, row in enumerate(reader):
            try:
                result.append([float(row[k]) for k in COLUMNS])
            except (TypeError, ValueError) as e:
                raise ValueError(f"raw row {i+2}: invalid/missing number") from e
    a = np.asarray(result, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Raw data empty or contains nonfinite values")
    return a

def advance(pose, velocity, dt):
    x, y, yaw = pose
    vx, vy, wz = velocity
    return np.array([x + dt*(math.cos(yaw)*vx-math.sin(yaw)*vy),
                     y + dt*(math.sin(yaw)*vx+math.cos(yaw)*vy), yaw+dt*wz])

def clearance(pose, scenario):
    return min(math.hypot(pose[0]-o["center"][0], pose[1]-o["center"][1])
               - o["radius_m"] - scenario["robot_radius_m"] for o in scenario["obstacles"])

def swept_clearance(a, b, scenario):
    a, b = np.asarray(a[:2]), np.asarray(b[:2])
    d = b-a
    den = float(d@d)
    vals = []
    for o in scenario["obstacles"]:
        q = np.asarray(o["center"])
        u = float(np.clip((q-a)@d/den, 0, 1)) if den else 0
        vals.append(float(np.linalg.norm(a+u*d-q))-o["radius_m"]-scenario["robot_radius_m"])
    return min(vals)

def rate(pose, vel, cmd, scenario):
    c = scenario["cost"]
    return (c["base_rate"] + c["speed_squared_weight"]*float(vel[:2]@vel[:2])
            + c["tracking_squared_weight"]*float((vel[:2]-cmd[:2])@(vel[:2]-cmd[:2]))
            + c["clearance_penalty_weight"]*max(0, c["clearance_reference_m"]-clearance(pose,scenario)))

def calculate(raw, scenario, method):
    dt, budget = scenario["dt_s"], scenario["initial_budget"]
    alpha = scenario["methods"][method]["alpha"]
    pn, pe = np.array(scenario["initial_pose"],float), np.array(scenario["initial_pose"],float)
    jn, je = 0., 0.
    rows, terminal = [], ""
    for i in range(len(raw)+1):
        row = dict(t_s=float(i*dt), x_nom_m=pn[0], y_nom_m=pn[1], yaw_nom_rad=pn[2],
                   x_exec_m=pe[0], y_exec_m=pe[1], yaw_exec_rad=pe[2],
                   clearance_nom_m=clearance(pn,scenario), clearance_exec_m=clearance(pe,scenario),
                   path_deviation_m=float(np.linalg.norm(pe[:2]-pn[:2])),
                   J_nom=jn,J_exec=je,J_charged=(1+alpha)*jn,
                   z_internal=budget-(1+alpha)*jn,z_actual=budget-je,
                   budget_violation=int(je>budget+1e-8),terminal=terminal)
        if i == len(raw):
            row.update(tracking_error_mps=None,yaw_error_radps=None,
                       cost_nom_step=None,cost_exec_step=None,cost_charged_step=None)
            rows.append(row)
            break
        cmd, act = raw[i,1:4], raw[i,4:7]
        cn, ce = dt*rate(pn,cmd,cmd,scenario), dt*rate(pe,act,cmd,scenario)
        row.update(tracking_error_mps=float(np.linalg.norm(act[:2]-cmd[:2])),
                   yaw_error_radps=abs(float(act[2]-cmd[2])),
                   cost_nom_step=cn,cost_exec_step=ce,cost_charged_step=(1+alpha)*cn)
        rows.append(row)
        previous = pe.copy()
        pn,pe = advance(pn,cmd,dt),advance(pe,act,dt)
        jn,je=jn+cn,je+ce
        if swept_clearance(previous,pe,scenario)<=0:
            terminal="collision"
        elif np.linalg.norm(pe[:2]-scenario["goal_center"])<=scenario["goal_radius_m"]:
            terminal="goal"
        elif (i+1)*dt>=scenario["horizon_s"]-1e-8:
            terminal="timeout"
        else:
            terminal=""
    return rows

def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def summarize(rows, raw, scenario):
    last=rows[-1]
    path=np.array([[r["x_exec_m"],r["y_exec_m"]] for r in rows])
    return dict(duration_s=last["t_s"],intervals=len(raw),terminal=last["terminal"],
                path_length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                min_clearance_m=min(r["clearance_exec_m"] for r in rows),
                min_nominal_clearance_m=min(r["clearance_nom_m"] for r in rows),
                max_deviation_m=max(r["path_deviation_m"] for r in rows),
                tracking_rmse_mps=float(np.sqrt(np.mean(np.sum((raw[:,4:6]-raw[:,1:3])**2,axis=1)))),
                J_nom=last["J_nom"],J_exec=last["J_exec"],J_charged=last["J_charged"],
                relative_cost_overrun=last["J_exec"]/last["J_nom"]-1,
                budget=scenario["initial_budget"],budget_violation=bool(last["budget_violation"]))

def build(data_dir, output_dir, methods):
    data_dir,output_dir=Path(data_dir),Path(output_dir)
    s=read_json(data_dir/"scenario.json")
    summaries,events={},[]
    for method in methods:
        raw=read_raw(data_dir/f"raw_{method}.csv")
        rows=calculate(raw,s,method)
        write_csv(output_dir/f"derived_{method}.csv",rows)
        summaries[method]=summarize(rows,raw,s)
        for ev in s["events"]:
            matches=[(i,r) for i,r in enumerate(rows) if r["x_exec_m"]>=ev["x_m"]]
            if matches:
                i,r=matches[0]
                events.append(dict(event_id=ev["id"],method=method,trial_id=s["scenario_id"]+"_"+method,
                                   t_s=r["t_s"],sample_index=i,x_m=r["x_exec_m"],y_m=r["y_exec_m"],
                                   description=ev["description"]))
    if events:
        write_csv(output_dir/"events.csv",events)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"summary.json").write_text(json.dumps({"data_status":s.get("data_status","assumed"),"methods":summaries},indent=2),encoding="utf-8")
    return summaries

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,default=ROOT/"data")
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--methods",nargs="+",default=["rcppo","hd_mcra"])
    a=p.parse_args()
    print(json.dumps(build(a.data_dir,a.output_dir or a.data_dir,a.methods),indent=2))

