import argparse
from pathlib import Path
from matplotlib.lines import Line2D
from plot_style import ROOT,plt,COLORS,load_data,export,method_handles,line_handles
from plot_trajectory import draw_trajectory,draw_zoom
from plot_metrics import draw_metric

def compose(data,out,publication=False):
    if publication:
        compose_publication(data,out)
        return
    fig=plt.figure(figsize=(184.2/25.4,100/25.4))
    ax=fig.add_axes([.065,.50,.40,.38]);draw_trajectory(ax,data)
    ax.set_title("(a) Same-scene trajectories",loc="left",pad=7)
    zoom=fig.add_axes([.085,.20,.125,.22]);draw_zoom(zoom,data)
    s,frames,events,summary=data
    fig.text(.24,.40,"Shared budget / interface",fontsize=7.3,fontweight="bold")
    fig.text(.24,.35,"Minimum clearance",fontsize=7,color="#5a626a")
    for m,x in [("hd_mcra",.24),("rcppo",.355)]:
        fig.text(x,.307,f"{summary[m]['min_clearance_m']:.3f} m",color=COLORS[m],fontsize=8.5)
    fig.text(.24,.26,"Arrival time",fontsize=7,color="#5a626a")
    for m,x in [("hd_mcra",.24),("rcppo",.355)]:
        fig.text(x,.217,f"{summary[m]['duration_s']:.1f} s",color=COLORS[m],fontsize=8.5)
    fig.text(.24,.16,r"$\alpha=0.18,\ \delta=0.12$ m",fontsize=7.5)
    for kind,y in zip(["clearance","cost","tracking"],[.71,.445,.18]):
        metric=fig.add_axes([.565,y,.405,.18])
        draw_metric(metric,data,kind)
        if kind!="tracking":metric.set_xlabel("");metric.tick_params(labelbottom=False)
    fig.legend(handles=method_handles()+[
        Line2D([],[],color="#656c73",ls=(0,(3,2)),lw=.9,label="Nominal"),
        Line2D([],[],color="#656c73",ls=(0,(5,2,1,2)),lw=.9,label="Calibrated nominal cost")],
        loc="upper center",bbox_to_anchor=(.52,.99),ncol=4,frameon=False,handlelength=2.1,columnspacing=1.4,fontsize=7.2)
    fig.text(.065,.070,"Gray: obstacle  |  pale: footprint expansion  |  dotted: + delta",fontsize=6.7,color="#555e66")
    fig.text(.065,.027,"ASSUMED DATA  •  A/B use each method's passage time  •  Illustrative cost units; neither trial exceeds budget",fontsize=6.5,color="#555e66")
    export(fig,out)

def compose_publication(data,out):
    fig=plt.figure(figsize=(184.2/25.4,90/25.4))
    ax=fig.add_axes([.065,.465,.40,.42]);draw_trajectory(ax,data,publication=True)
    ax.set_title("(a) Same-scene trajectories",loc="left",pad=7)
    zoom=fig.add_axes([.085,.115,.125,.25]);draw_zoom(zoom,data,publication=True)
    s,frames,events,summary=data
    fig.text(.24,.33,"Minimum clearance",fontsize=7,color="#5a626a")
    for m,x in [("hd_mcra",.24),("rcppo",.355)]:
        fig.text(x,.26,f"{summary[m]['min_clearance_m']:.3f} m",color=COLORS[m],fontsize=8.5)
    fig.text(.24,.185,"Arrival time",fontsize=7,color="#5a626a")
    for m,x in [("hd_mcra",.24),("rcppo",.355)]:
        fig.text(x,.115,f"{summary[m]['duration_s']:.1f} s",color=COLORS[m],fontsize=8.5)
    for kind,y in zip(["clearance","cost","tracking"],[.695,.405,.115]):
        metric=fig.add_axes([.565,y,.405,.20]);draw_metric(metric,data,kind,publication=True)
        if kind!="tracking":metric.set_xlabel("");metric.tick_params(labelbottom=False)
    fig.legend(handles=method_handles()+line_handles(calibrated=True),loc="upper center",
               bbox_to_anchor=(.52,.997),ncol=4,frameon=False,handlelength=2.1,
               columnspacing=1.4,fontsize=7.2)
    export(fig,out,publication=True,data_status=s["data_status"])

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path);p.add_argument("--output",type=Path,default=ROOT/"output/trajectory_sync_assumed_v1")
    a=p.parse_args();compose(load_data(a.data_dir),a.output)

