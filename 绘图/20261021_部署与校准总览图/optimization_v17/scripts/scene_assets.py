"""Architecture-led overview. Generic neural glyphs, unchanged illustrative inputs."""
from pathlib import Path
import argparse,json,hashlib,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Polygon,Ellipse,Arc,Rectangle
import matplotlib.patheffects as pe

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
L=read(ROOT/"config/layout.json");S=read(ROOT/"config/style.json");W,H=L["width_in"],L["height_in"]
INK=S["ink"];GRAY=S["muted"];BLUE=S["blue"];LIGHT=S["blue_light"];ORANGE=S["orange"];GREEN=S["green"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"svg.fonttype":"none","pdf.fonttype":42,"svg.hashsalt":"hdmcra-clarity-v7"})
def txt(ax,x,y,s,size=8,color=INK,ha="left",weight="normal",**kw):
    return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va="center",fontweight=weight,zorder=12,**kw)
def box(ax,x,y,w,h,fc,ec="#CAD5DC",radius=.012,lw=.8):
    q=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={radius}",fc=fc,ec=ec,lw=lw,zorder=0)
    ax.add_patch(q);return q
def arrow(ax,pts,color=INK,ls="-",lw=1.05):
    p=np.asarray(pts)
    if len(p)>2:ax.plot(p[:-1,0],p[:-1,1],color=color,lw=lw,ls=ls,zorder=8)
    ax.add_patch(FancyArrowPatch(p[-2],p[-1],arrowstyle="-|>",mutation_scale=8,lw=lw,color=color,linestyle=ls,zorder=8,shrinkA=0,shrinkB=0))
def clean(ax):ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
def save(fig,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf","svg"]:
        opts={"dpi":300} if ext=="png" else {"metadata":{"CreationDate":None,"ModDate":None}} if ext=="pdf" else {"metadata":{"Date":None}}
        fig.savefig(out.with_suffix("."+ext),facecolor="white",**opts)
def project(points,m):
    p=m["projection"];h,w=m["shape"][:2];r=np.asarray(points)-p["position"]
    f=np.asarray(p["forward"]);u=np.asarray(p["up"]);right=np.cross(f,u);z=r@f
    return np.c_[.5+(r@right)*p["near"]/z/((p["top"]-p["bottom"])*w/h),
                 ((r@u)*p["near"]/z-p["bottom"])/(p["top"]-p["bottom"])]
def base(ax,name):
    im=Image.open(ROOT/"assets/renders"/(name+".png")).convert("RGB");m=read(ROOT/"assets/renders"/(name+"_metadata.json"))
    assert sha(ROOT/"assets/renders"/(name+".png"))==m["image_sha256"]
    clean(ax);ax.imshow(im,extent=(0,1,0,1),aspect="auto",zorder=0);return im,m
def restore(ax,im):
    mask=np.asarray(Image.open(ROOT/"assets/renders/obstacle_mask.png"))
    ax.imshow(np.dstack([np.asarray(im),mask]),extent=(0,1,0,1),aspect="auto",zorder=5)
def route(ax,df,m,kind,col,ls,lw):
    uv=project(np.c_[df["x_"+kind+"_m"],df["y_"+kind+"_m"],np.full(len(df),.006)],m)
    ax.plot(*uv.T,color=col,ls=ls,lw=lw,zorder=3,path_effects=[pe.Stroke(linewidth=lw+.7,foreground="white"),pe.Normal()]);return uv
def cost_scene(ax):
    im,m=base(ax,"trajectory_scene")
    for j,(pid,col,ls) in enumerate(zip(["R1","R2","R3"],S["cost_colors"],S["cost_styles"])):
        df=pd.read_csv(ROOT/"data/cost_pairs"/("derived_"+pid+".csv"))
        uv=route(ax,df,m,"exec",col,ls,.9)
        # All line paths retain their original points; IDs only move labels.
        ax.annotate(pid,uv[65],xytext=(.91,.18+j*.29),fontsize=7,color=col,ha="center",va="center",
                    arrowprops={"arrowstyle":"-","color":col,"lw":.45},bbox={"fc":"white","ec":"none","pad":.6},zorder=9)
    restore(ax,im)
def safety_scene(ax):
    im,m=base(ax,"trajectory_scene");sc=read(ROOT/"data/reference/scenario.json")
    theta=np.linspace(0,2*np.pi,181)
    for ob in sc["obstacles"]:
        for extra,color,ls in [(sc["robot_radius_m"],GRAY,"--"),(sc["robot_radius_m"]+S["delta_m"],GREEN,":")]:
            rr=ob["radius_m"]+extra;x,y=ob["center"]
            uv=project(np.c_[x+rr*np.cos(theta),y+rr*np.sin(theta),np.full(len(theta),.006)],m)
            ax.plot(*uv.T,color=color,ls=ls,lw=.7,zorder=2)
    df=pd.read_csv(ROOT/"data/reference/derived_hd_mcra.csv")
    route(ax,df,m,"exec",BLUE,"-",1.2);route(ax,df,m,"nom",LIGHT,"--",.8);restore(ax,im)
    o=sc["obstacles"][0];r=o["radius_m"]+sc["robot_radius_m"]
    q=project([[o["center"][0],o["center"][1]+r,.006],[o["center"][0],o["center"][1]+r+S["delta_m"],.006]],m)
    ax.annotate("",q[1],q[0],arrowprops={"arrowstyle":"|-|","color":GREEN,"lw":.7},zorder=8)
    txt(ax,q[1,0]+.05,q[1,1]+.025,"δ",8,GREEN,bbox={"fc":"white","ec":"none","pad":.4})
    # Restore v5's same-time local view. All 75 path points remain unchanged.
    row=df.iloc[34]
    center=np.array([(row.x_nom_m+row.x_exec_m)/2,(row.y_nom_m+row.y_exec_m)/2])
    corners=project(np.c_[center+[[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]],np.full(4,.006)],m)
    ax.add_patch(Polygon(corners,fill=False,edgecolor=ORANGE,lw=.8,zorder=7))
    inset=ax.inset_axes([.04,.02,.30,.29])
    inset.set_facecolor("#F9FBFC")
    for kind,color,style in [("exec",BLUE,"-"),("nom",LIGHT,"--")]:
        inset.plot(df["x_"+kind+"_m"],df["y_"+kind+"_m"],color=color,ls=style,lw=1.1)
    inset.plot([row.x_nom_m,row.x_exec_m],[row.y_nom_m,row.y_exec_m],color=ORANGE,lw=1)
    inset.scatter([row.x_nom_m,row.x_exec_m],[row.y_nom_m,row.y_exec_m],s=9,c=[LIGHT,BLUE])
    inset.set_xlim(center[0]-.1,center[0]+.1)
    inset.set_ylim(center[1]-.1,center[1]+.1)
    inset.set_aspect("equal")
    inset.set_xticks([]);inset.set_yticks([])
    for spine in inset.spines.values():spine.set_edgecolor(ORANGE);spine.set_linewidth(.7)
    ax.plot([corners[0,0],.29],[corners[0,1],.34],color=ORANGE,lw=.7,zorder=7)
    txt(ax,.31,.17,"Paired\ndeviation",7,GRAY,bbox={"fc":"white","ec":"none","alpha":.93,"pad":.8})
    txt(ax,.87,.39,"Goal",7,INK,"center",bbox={"fc":"white","ec":"none","alpha":.93,"pad":.8})
    return {"source_frame":34,"time_s":float(row.t_s),"deviation_m":float(row.path_deviation_m),
            "window_world_m":[.20,.20],"equal_aspect":True,"deviation_is_not_delta":True}
def robot_scene(ax):
    im,m=base(ax,"interface_scene");crop=L["interface_crop"]
    # Coordinates are y-up; the crop is a view limit, not a distortion of scene geometry.
    ax.set_xlim(crop[0],crop[2]);ax.set_ylim(1-crop[3],1-crop[1])
    a=m["body_anchors"];p=np.asarray(a["origin"]);p[1]=1-p[1]
    for key,col in [("vx",BLUE),("vy",GREEN)]:
        q=np.asarray(a[key]);q[1]=1-q[1]
        ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=8,lw=1.3,color=col,
                    path_effects=[pe.Stroke(linewidth=2.5,foreground="white"),pe.Normal()],zorder=7))
    arc=np.asarray(a["yaw_arc"]);arc[:,1]=1-arc[:,1]
    ax.plot(*arc.T,color=ORANGE,lw=1.2,zorder=7,path_effects=[pe.Stroke(linewidth=2.3,foreground="white"),pe.Normal()])
    ax.add_patch(FancyArrowPatch(arc[-3],arc[-1],arrowstyle="-|>",mutation_scale=7,lw=1,color=ORANGE,zorder=7))
    txt(ax,.44,.61,"Velocity",7,BLUE,bbox={"fc":"white","ec":"none","pad":.6})
def lock(ax,x,y,color=GREEN):
    ax.add_patch(Arc((x,y+.008),.019,.030,theta1=0,theta2=180,color=color,lw=1.2,zorder=7))
    ax.add_patch(FancyBboxPatch((x-.012,y-.014),.024,.022,boxstyle="round,pad=0,rounding_size=.003",fc=color,ec=color,zorder=7))
    ax.plot([x,x],[y-.008,y+.001],color="white",lw=1,zorder=8)
def cost_bars(ax):
    rec=read(ROOT/"data/cost_pairs/summary.json")["records"];vals=[rec[p]["summary"]["relative_cost_overrun"]*100 for p in ["R1","R2","R3"]]
    ax.set(xlim=(0,22),ylim=(-.6,2.7));ax.spines[["top","left","right"]].set_visible(False);ax.spines["bottom"].set_color("#BFCBD3")
    ax.barh([2,1,0],vals,color=S["cost_colors"],height=.42)
    ax.set_yticks([2,1,0],labels=["R1","R2","R3"]);ax.tick_params(axis="y",length=0,pad=2,labelsize=7)
    ax.set_xticks([0,10,20]);ax.tick_params(axis="x",length=2,pad=1,labelsize=7,color=GRAY)
    ax.axvline(S["alpha"]*100,color=ORANGE,ls="--",lw=.9)
    for yy,v in zip([2,1,0],vals):ax.text(v+.5,yy,f"{v:.1f}",fontsize=7,color=INK,va="center")
    return vals
