"""Render real Go2 mesh and scene off screen; no dynamics, learning, or new data."""
import argparse, csv, hashlib, json, os
from pathlib import Path
import xml.etree.ElementTree as ET

ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True)
ap.add_argument("--config",required=True)
ap.add_argument("--poses",required=True)
ap.add_argument("--out",required=True)
ap.add_argument("--device",default="1")
a=ap.parse_args()
os.environ["MUJOCO_GL"]="egl"
os.environ["PYOPENGL_PLATFORM"]="egl"
os.environ["MUJOCO_EGL_DEVICE_ID"]=a.device
import mujoco
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def nums(v):return " ".join(map(str,v))
src=Path(a.model).resolve();cfg=json.loads(Path(a.config).read_text())
assert sha(src)==cfg["model_sha256"]
with open(a.poses,newline="") as stream:rows=list(csv.DictReader(stream))
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)

for name,cc in cfg["views"].items():
    root=ET.parse(src).getroot()
    compiler=root.find("compiler")
    compiler.set("meshdir",str((src.parent/compiler.get("meshdir",".")).resolve()))
    for old in root.findall("visual"):root.remove(old)
    visual=ET.SubElement(root,"visual")
    ET.SubElement(visual,"global",offwidth=str(cfg["width"]),offheight=str(cfg["height"]))
    ET.SubElement(visual,"quality",shadowsize="4096",offsamples="4")
    ET.SubElement(visual,"headlight",diffuse=".65 .65 .65",ambient=".35 .35 .35",specular=".10 .10 .10")
    asset=root.find("asset")
    ET.SubElement(asset,"texture",name="input_floor_tex",type="2d",builtin="checker",width="512",height="512",
                  rgb1=nums(cfg["floor"]["rgb1"]),rgb2=nums(cfg["floor"]["rgb2"]))
    ET.SubElement(asset,"material",name="input_floor_mat",texture="input_floor_tex",texrepeat="2 2",texuniform="true",reflectance="0")
    world=root.find("worldbody")
    ET.SubElement(world,"geom",name="input_floor",type="plane",size="8 8 .05",material="input_floor_mat")
    ET.SubElement(world,"light",pos="-2 -3 6",dir=".3 .3 -1",directional="true",diffuse=".5 .5 .5",castshadow="true")
    for ob in cfg["obstacles"]:
        x,y=ob["center"];h=ob["height_m"]
        ET.SubElement(world,"geom",name=ob["id"],type="cylinder",pos=f"{x} {y} {h/2}",size=f'{ob["radius_m"]} {h/2}',rgba=".39 .48 .52 1")
    gx,gy=cfg["goal"]
    ET.SubElement(world,"geom",name="goal_area",type="cylinder",pos=f"{gx} {gy} .006",size=".20 .006",rgba=".22 .68 .48 1",contype="0",conaffinity="0")
    if name=="goal":
        # A 3-D flag is a destination annotation, not an object in experimental logs.
        ET.SubElement(world,"geom",name="flag_pole",type="cylinder",pos=f"{gx} {gy} .35",size=".014 .35",rgba=".18 .55 .40 1",contype="0",conaffinity="0")
        ET.SubElement(world,"geom",name="flag",type="box",pos=f"{gx+.16} {gy} .61",size=".16 .012 .085",rgba=".22 .68 .48 1",contype="0",conaffinity="0")
    if name=="budget":
        # The ribbon is the original assumed route, not a newly sampled trajectory.
        for j in range(12,51):
            p,q=rows[j],rows[j+1]
            endpoints=[float(p["x_exec_m"]),float(p["y_exec_m"]),.013,float(q["x_exec_m"]),float(q["y_exec_m"]),.013]
            color=".77 .40 .14 1" if j>=cc["frame"] else ".72 .59 .46 1"
            ET.SubElement(world,"geom",name=f"route_{j}",type="capsule",fromto=nums(endpoints),size=".024",rgba=color,contype="0",conaffinity="0")
    xml=ET.tostring(root,encoding="unicode")
    (out/(name+"_scene.xml")).write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    key=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,"home")
    assert key>=0
    mujoco.mj_resetDataKeyframe(model,data,key)
    pose=rows[cc["frame"]];yaw=float(pose["yaw_exec_rad"])
    data.qpos[:2]=[float(pose["x_exec_m"]),float(pose["y_exec_m"])]
    data.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    mujoco.mj_forward(model,data)
    feet=[j for j in range(model.ngeom) if model.geom_type[j]==mujoco.mjtGeom.mjGEOM_SPHERE and abs(float(model.geom_size[j,0])-.022)<1e-6]
    assert feet
    data.qpos[2]+=-min(float(data.geom_xpos[j,2]-model.geom_size[j,0]) for j in feet)+.001
    mujoco.mj_forward(model,data)
    cam=mujoco.MjvCamera();cam.lookat[:]=cc["lookat"]
    cam.distance=cc["distance"];cam.azimuth=cc["azimuth"];cam.elevation=cc["elevation"]
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    with mujoco.Renderer(model,height=cfg["height"],width=cfg["width"]) as renderer:
        renderer.update_scene(data,camera=cam,scene_option=opt)
        pixels=renderer.render().copy()
        c0,c1=renderer.scene.camera;position=(c0.pos+c1.pos)/2
        forward=np.array(c0.forward);up=np.array(c0.up);right=np.cross(forward,up)
        near=float(c0.frustum_near);top=float(c0.frustum_top);bottom=float(c0.frustum_bottom)
        def project(point):
            rel=np.asarray(point)-position;z=rel@forward
            return [float(.5+(rel@right)*near/z/((top-bottom)*cfg["width"]/cfg["height"])),
                    float(1-((rel@up)*near/z-bottom)/(top-bottom))]
        anchors={"robot":project([*data.qpos[:2],.30]),"goal":project([gx,gy,.04])}
    image=out/(name+".png");Image.fromarray(pixels).save(image)
    metadata={"name":name,"backend":"MuJoCo EGL","mujoco_version":mujoco.__version__,"physics_steps":0,
              "policy_executed":False,"model_sha256":sha(src),"config_sha256":sha(a.config),"poses_sha256":sha(a.poses),
              "image_sha256":sha(image),"shape":list(pixels.shape),"camera":cc,"source_frame":cc["frame"],
              "source_pose_is_assumed":True,"qpos":data.qpos.tolist(),"anchors_normalized":anchors,
              "provenance":cfg["provenance"],"goal_marker_is_annotation":name=="goal",
              "route_is_existing_assumed_data":name=="budget","render_script_sha256":sha(__file__)}
    (out/(name+"_metadata.json")).write_text(json.dumps(metadata,indent=2))
    print(json.dumps({"view":name,"success":True,"image_sha256":sha(image)}),flush=True)
assert sha(src)==cfg["model_sha256"]
