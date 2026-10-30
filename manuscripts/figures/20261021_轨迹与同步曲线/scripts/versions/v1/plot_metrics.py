import argparse
from pathlib import Path
import numpy as np
from plot_style import ROOT,plt,COLORS,LABELS,load_data,axis_style,export,method_handles

KINDS=["clearance","cost","tracking"]
def draw_metric(ax,data,kind,events_enabled=True):
    s,frames,events,summary=data
    for m,df in frames.items():
        c=COLORS[m]
        if kind=="clearance":
            ax.plot(df.t_s,df.clearance_exec_m,color=c)
        elif kind=="cost":
            ax.plot(df.t_s,df.J_nom,color=c,ls=(0,(2,2)),lw=.8,alpha=.55)
            ax.plot(df.t_s,df.J_exec,color=c)
            if m=="hd_mcra":
                ax.plot(df.t_s,df.J_charged,color=c,ls=(0,(5,2,1,2)),lw=.9,alpha=.85)
                ax.fill_between(df.t_s.to_numpy(),df.J_nom.to_numpy(),df.J_charged.to_numpy(),color=c,alpha=.06)
        else:
            ax.plot(df.t_s.iloc[:-1],df.tracking_error_mps.iloc[:-1],color=c,lw=1.15)
    if kind=="clearance":
        ax.axhline(0,color="#72777c",ls=(0,(3,2)),lw=.7)
        ax.set_ylim(-.04,1.45);ax.set_yticks([0,.5,1])
        ax.set_ylabel("Clearance (m)")
        ax.set_title("(b) Executed body clearance",loc="left",pad=5)
        if events_enabled:
            for _,e in events.iterrows():
                ax.axvline(e.t_s,color=COLORS[e.method],lw=.55,alpha=.4,ls=":")
                ax.text(e.t_s,1.27,e.event_id,ha="center",fontsize=6.8,color=COLORS[e.method])
    elif kind=="cost":
        ax.axhline(s["initial_budget"],color="#5e656b",ls=(0,(4,2)),lw=.8)
        ax.text(.3,86.6,"Common budget 85",fontsize=6.7,color="#4f565c")
        ax.set_ylim(0,98);ax.set_yticks([0,40,80])
        ax.set_ylabel("Cost (a.u.)")
        ax.set_title("(c) Actual cost and budget accounting",loc="left",pad=5)
        ax.text(14.8,75.5,r"$(1+\alpha)J_{\rm nom}$",color=COLORS["hd_mcra"],ha="right",fontsize=7)
    else:
        ax.set_ylim(0,.065);ax.set_yticks([0,.03,.06])
        ax.set_ylabel("Error (m/s)")
        ax.set_title("(d) Linear-velocity tracking error",loc="left",pad=5)
    if events_enabled and kind!="clearance":
        for _,e in events.iterrows():
            ax.axvline(e.t_s,color=COLORS[e.method],lw=.45,alpha=.22,ls=":")
    ax.set_xlim(0,15.3);ax.set_xticks([0,3,6,9,12,15])
    ax.set_xlabel("Elapsed time (s)")
    axis_style(ax)

def standalone(data,kind,out):
    fig=plt.figure(figsize=(4.5,2.4))
    ax=fig.add_axes([.16,.31,.80,.53])
    draw_metric(ax,data,kind)
    fig.legend(handles=method_handles(),loc="lower center",bbox_to_anchor=(.56,.07),ncol=2,frameon=False)
    fig.text(.5,.025,"Assumed data | "+("dashed: nominal; dash-dot: calibrated nominal" if kind=="cost" else "A/B: each method's own passage times"),ha="center",fontsize=6.6)
    export(fig,out)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path);p.add_argument("--output-dir",type=Path,default=ROOT/"output/panels")
    a=p.parse_args()
    for k in KINDS:standalone(load_data(a.data_dir),k,a.output_dir/(k+"_v1"))

