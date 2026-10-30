import argparse
from pathlib import Path
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from plot_style import ROOT,plt,COLORS,LABELS,load_data,axis_style,export,method_handles,line_handles

def draw_trajectory(ax,data,details=True,publication=False):
    s,frames,events,summary=data
    delta=s["methods"]["hd_mcra"]["delta_m"] if publication else .12
    for o in s["obstacles"]:
        x,y=o["center"];r=o["radius_m"]
        ax.add_patch(Circle((x,y),r+s["robot_radius_m"]+delta,facecolor="none",
                           edgecolor="#a5adb4",ls=(0,(2,2)),lw=.65,zorder=1))
        ax.add_patch(Circle((x,y),r+s["robot_radius_m"],facecolor="#eef0f2",edgecolor="none",zorder=1))
        ax.add_patch(Circle((x,y),r,facecolor="#92999f",edgecolor="white",lw=.6,zorder=2))
        if details:ax.text(x,y,o["id"],ha="center",va="center",color="white",fontsize=7,zorder=3)
    for m,df in frames.items():
        ax.plot(df.x_nom_m,df.y_nom_m,color=COLORS[m],ls=(0,(3,2)),lw=.95,alpha=.55,zorder=4)
        ax.plot(df.x_exec_m,df.y_exec_m,color=COLORS[m],lw=1.6,zorder=5)
    start=s["initial_pose"][:2] if publication else (0,0)
    ax.plot(*start,"o",color="#263342",ms=4,zorder=7)
    ax.add_patch(Circle(s["goal_center"],s["goal_radius_m"],facecolor="#e7efdf",edgecolor="#768f50",lw=.6,zorder=1))
    ax.plot(*s["goal_center"],marker="*",color="#66883d",ms=8,zorder=7)
    if details:
        if publication:
            ax.annotate("Start",start,xytext=(0,11),textcoords="offset points",fontsize=7,ha="center")
            ax.annotate("Goal",s["goal_center"],xytext=(0,13),textcoords="offset points",fontsize=7,ha="center")
        else:
            ax.annotate("Start",(0,0),xytext=(-.05,.34),fontsize=7,ha="center")
            ax.annotate("Goal",s["goal_center"],xytext=(4.55,.40),fontsize=7,ha="center")
        for _,e in events.iterrows():
            ax.plot(e.x_m,e.y_m,"o",mfc="white",mec=COLORS[e.method],ms=4,mew=1,zorder=8)
            offset=(0,9) if e.method=="rcppo" else (0,-14)
            ax.annotate(e.event_id,(e.x_m,e.y_m),xytext=offset,textcoords="offset points",
                        color=COLORS[e.method],ha="center",fontsize=7,fontweight="bold",zorder=8)
    ax.set_aspect("equal")
    ax.set_xlim(-.4,5.0);ax.set_ylim(-1.5,1.8)
    ax.set_xticks([0,1,2,3,4,5]);ax.set_yticks([-1,0,1])
    if publication:
        xmin,xmax,ymin,ymax=s["map_bounds"]
        xs=[start[0],s["goal_center"][0]-s["goal_radius_m"],s["goal_center"][0]+s["goal_radius_m"]]
        ys=[start[1],s["goal_center"][1]-s["goal_radius_m"],s["goal_center"][1]+s["goal_radius_m"]]
        for o in s["obstacles"]:
            radius=o["radius_m"]+s["robot_radius_m"]+delta
            xs.extend([o["center"][0]-radius,o["center"][0]+radius])
            ys.extend([o["center"][1]-radius,o["center"][1]+radius])
        for df in frames.values():
            xs.extend([df.x_exec_m.min(),df.x_exec_m.max(),df.x_nom_m.min(),df.x_nom_m.max()])
            ys.extend([df.y_exec_m.min(),df.y_exec_m.max(),df.y_nom_m.min(),df.y_nom_m.max()])
        pad=.03*max(xmax-xmin,ymax-ymin)
        ax.set_xlim(min(xmin,min(xs)-pad),max(xmax,max(xs)+pad))
        ax.set_ylim(min(ymin,min(ys)-pad),max(ymax,max(ys)+pad))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6,steps=[1,2,5,10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,steps=[1,2,5,10]))
    ax.set_xlabel("x (m)");ax.set_ylabel("y (m)")
    axis_style(ax)

def draw_zoom(ax,data,publication=False):
    draw_trajectory(ax,data,details=False,publication=publication)
    s,frames,events,summary=data
    ax.set_xlim(1.45,2.30);ax.set_ylim(-.72,.52)
    ax.set_xticks([1.5,2.0]);ax.set_yticks([-.5,0,.5])
    if publication:
        a=events[events.event_id=="A"]
        if a.empty:
            ax.set_visible(False)
            return
        r=s["robot_radius_m"]
        ax.set_xlim(a.x_m.min()-2.5*r,a.x_m.max()+2.1*r)
        ax.set_ylim(a.y_m.min()-1.5*r,a.y_m.max()+3.5*r)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2,steps=[1,2,5,10],min_n_ticks=1))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3,steps=[1,2,5,10]))
    ax.set_xlabel("x (m)",fontsize=7);ax.set_ylabel("y (m)",fontsize=7)
    ax.tick_params(labelsize=6.5)
    for m,df in frames.items():
        matched=events[(events.method==m)&(events.event_id=="A")]
        if publication and matched.empty:continue
        row=matched.iloc[0]
        ax.add_patch(Circle((row.x_m,row.y_m),s["robot_radius_m"],fill=False,
                           edgecolor=COLORS[m],lw=.9,zorder=6))
    ax.set_title("At A: robot footprint",fontsize=7.5,pad=4)

def standalone(data,out,publication=False):
    if publication:
        fig=plt.figure(figsize=(3.9,2.9))
        ax=fig.add_axes([.14,.23,.82,.65]);draw_trajectory(ax,data,publication=True)
        ax.set_title("(a) Same-scene trajectories",loc="left",pad=8)
        fig.legend(handles=method_handles()+line_handles(),loc="lower center",
                   bbox_to_anchor=(.54,.01),ncol=3,frameon=False,fontsize=6.6)
        export(fig,out,publication=True,data_status=data[0]["data_status"])
        return
    fig=plt.figure(figsize=(3.9,3.3))
    ax=fig.add_axes([.14,.33,.82,.57]);draw_trajectory(ax,data)
    ax.set_title("(a) Same-scene trajectories",loc="left",pad=8)
    fig.legend(handles=method_handles(),loc="lower center",bbox_to_anchor=(.54,.15),ncol=2,frameon=False)
    fig.text(.5,.085,"Gray: obstacle; pale: footprint expansion; dotted: + delta",ha="center",fontsize=6.5)
    fig.text(.5,.035,"Solid: executed  |  dashed: nominal  |  Assumed data",ha="center",fontsize=7)
    export(fig,out)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path);p.add_argument("--output",type=Path,default=ROOT/"output/panels/trajectory_v1")
    a=p.parse_args();standalone(load_data(a.data_dir),a.output)

