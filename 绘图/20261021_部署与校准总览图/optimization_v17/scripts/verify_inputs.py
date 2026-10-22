"""Verify immutable sources, numerical consistency, and independent projection anchors."""
from pathlib import Path
import sys,json,hashlib,argparse
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from draw_overview import project
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);a=ap.parse_args()
sources=read(ROOT/"source_manifest.json")["files"]
bad=[v["path"] for v in sources if not (ROOT/v["path"]).is_file() or sha(ROOT/v["path"])!=v["sha256"]]
sc=read(ROOT/"data/reference/scenario.json");s=read(ROOT/"config/style.json")
dt=sc["dt_s"];cost=sc["cost"];summary=read(ROOT/"data/cost_pairs/summary.json")["records"]
checks={};command=None
def advance(p,v):
    co,si=np.cos(p[2]),np.sin(p[2])
    return p+dt*np.array([co*v[0]-si*v[1],si*v[0]+co*v[1],v[2]])
def clearance(points):
    return np.min([np.linalg.norm(points[:,:2]-o["center"],axis=1)-o["radius_m"]-sc["robot_radius_m"] for o in sc["obstacles"]],axis=0)
for pid in ["R1","R2","R3"]:
    raw=pd.read_csv(ROOT/"data/cost_pairs"/("raw_"+pid+".csv")).to_numpy()
    d=pd.read_csv(ROOT/"data/cost_pairs"/("derived_"+pid+".csv"))
    assert len(raw)==74 and len(d)==75 and np.allclose(raw[:,0],np.arange(74)*dt)
    if command is None:command=raw[:,:4]
    assert np.array_equal(command,raw[:,:4])
    for kind,start in [("nom",1),("exec",4)]:
        vel=raw[:,start:start+3];poses=[np.array(sc["initial_pose"],float)]
        for v in vel:poses.append(advance(poses[-1],v))
        poses=np.array(poses);cl=clearance(poses)
        assert np.allclose(d[["x_"+kind+"_m","y_"+kind+"_m","yaw_"+kind+"_rad"]],poses,rtol=0,atol=1e-10)
        rates=cost["base_rate"]+cost["speed_squared_weight"]*np.sum(vel[:,:2]**2,axis=1)
        rates+=cost["tracking_squared_weight"]*np.sum((vel[:,:2]-raw[:,1:3])**2,axis=1)
        rates+=cost["clearance_penalty_weight"]*np.maximum(0,cost["clearance_reference_m"]-cl[:-1])
        totals=np.r_[0,np.cumsum(dt*rates)]
        assert np.allclose(d["J_"+kind],totals,rtol=0,atol=1e-10)
        assert np.allclose(d["cost_"+kind+"_step"][:-1],dt*rates,rtol=0,atol=1e-10)
        assert np.isclose(totals[-1],summary[pid]["summary"]["J_"+kind])
    assert np.allclose(d["J_charged"],(1+s["alpha"])*d["J_nom"])
    assert np.allclose(d["z_internal"],sc["initial_budget"]-d["J_charged"])
    overrun=float(d["J_exec"].iloc[-1]/d["J_nom"].iloc[-1]-1)
    assert np.isclose(overrun,summary[pid]["summary"]["relative_cost_overrun"])
    checks[pid]={"J_nom":float(d["J_nom"].iloc[-1]),"J_exec":float(d["J_exec"].iloc[-1]),"overrun_percent":100*overrun,"pass":True}
assert sha(ROOT/"data/cost_pairs/derived_R1.csv")==sha(ROOT/"data/reference/derived_hd_mcra.csv")
m=read(ROOT/"assets/renders/trajectory_scene_metadata.json");fid=read(ROOT/"assets/fiducials.json")
assert m["projection"]==fid["projection"] and m["shape"]==fid["shape"]
errors={}
for k,world in fid["anchors_world"].items():
    uv=project([world],m)[0];pix=np.array([uv[0]*1600,(1-uv[1])*1050])
    errors[k]=float(np.linalg.norm(pix-fid["measured_fiducials_px"][k]))
assert len(errors)==17 and max(errors.values())<2
assert s["alpha"]==.18==sc["methods"]["hd_mcra"]["alpha"]
assert s["delta_m"]==.12==sc["methods"]["hd_mcra"]["delta_m"]
assert sc["robot_radius_m"]==.2
input_scenes={}
cfg=read(ROOT/"config/input_scenes.json")
for name in ["observation","goal","budget"]:
    meta=read(ROOT/"assets/input_renders"/(name+"_metadata.json"))
    assert sha(ROOT/"assets/input_renders"/(name+".png"))==meta["image_sha256"]
    assert meta["config_sha256"]==sha(ROOT/"config/input_scenes.json")
    assert meta["poses_sha256"]==sha(ROOT/"data/reference/derived_hd_mcra.csv")
    assert meta["render_script_sha256"]==sha(ROOT/"scripts/render_input_scenes_remote.py")
    assert meta["model_sha256"]==cfg["model_sha256"]
    assert meta["source_frame"]==cfg["views"][name]["frame"]
    assert meta["physics_steps"]==0 and not meta["policy_executed"]
    assert meta["source_pose_is_assumed"] and meta["shape"]==[800,1280,3]
    input_scenes[name]={"image_sha256":meta["image_sha256"],"source_frame":meta["source_frame"],"physics_steps":0,"pass":True}
command_scenes={}
command_cfg=read(ROOT/"config/command_scenes.json")
for name in ['velocity','joint']:
    meta=read(ROOT/'assets/command_renders'/(name+'_metadata.json'))
    assert sha(ROOT/'assets/command_renders'/(name+'.png'))==meta['image_sha256']
    assert meta['config_sha256']==sha(ROOT/'config/command_scenes.json')
    assert meta['script_sha256']==sha(ROOT/'scripts/render_command_scenes_remote.py')
    assert meta['model_sha256']==command_cfg['model_sha256']
    assert meta['physics_steps']==0 and not meta['policy_executed']
    assert meta['shape']==[900,1280,3]
    for key,world in meta['overlay_world'].items():
        assert np.allclose(project(world,meta),meta['overlay_uv'][key],atol=1e-7)
    command_scenes[name]={'pass':True,'physics_steps':0,'image_sha256':meta['image_sha256'],'projected_groups':list(meta['overlay_world'])}
interaction_layout=read(ROOT/'config/interaction_layout.json')
interaction_dir=ROOT/interaction_layout['asset_dir']
interaction_meta=read(interaction_dir/'interaction_metadata.json')
assert interaction_meta['config_sha256']==sha(ROOT/interaction_layout['scene_config'])
assert interaction_meta['poses_sha256']==sha(ROOT/'data/reference/derived_hd_mcra.csv')
assert interaction_meta['render_script_sha256']==sha(ROOT/'scripts/render_interaction_remote.py')
assert interaction_meta['model_sha256']==command_cfg['model_sha256']
assert interaction_meta['physics_steps']==0 and not interaction_meta['policy_executed']
assert interaction_meta['obstacles']==cfg['obstacles'] and interaction_meta['goal']==sc['goal_center']
for name,digest in interaction_meta['files'].items():assert sha(interaction_dir/name)==digest
assert sha(interaction_dir/'interaction.png')==interaction_meta['image_sha256']
original=pd.read_csv(ROOT/'data/reference/derived_hd_mcra.csv')
for item in interaction_meta['records']:
    assert np.allclose(item['source_pose'],original.iloc[item['frame']][['x_exec_m','y_exec_m','yaw_exec_rad']].to_numpy(float),atol=1e-12)
report={"pass":not bad,"source_files":len(sources),"changed_sources":bad,"self_contained":True,
"costs":checks,"projection_anchors":len(errors),"max_projection_error_px":max(errors.values()),
"new_data_generated":False,"physics_steps":m["physics_steps"],"policy_executed":m["policy_executed"],"new_static_input_renders":input_scenes,"command_scenes":command_scenes,
"interaction_scene":{"pass":True,"frames":[r['frame'] for r in interaction_meta['records']],"one_robot_at_three_times":True,"new_data":False,"files_checked":len(interaction_meta['files'])}}
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=True));assert report["pass"]
