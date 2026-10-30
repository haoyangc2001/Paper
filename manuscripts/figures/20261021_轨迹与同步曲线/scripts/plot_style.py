from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
COLORS={"hd_mcra":"#0072B2","rcppo":"#D55E00"}
LABELS={"hd_mcra":"HD-MCRA","rcppo":"RC-PPO"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8.5,
 "axes.labelsize":8,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7,
 "axes.linewidth":.6,"lines.linewidth":1.5,"pdf.fonttype":42,"ps.fonttype":42,
 "savefig.dpi":300,"figure.facecolor":"white","axes.spines.top":True,"axes.spines.right":True})

def load_data(data_dir=None):
    d=Path(data_dir or ROOT/"data/versions/v2")
    s=json.loads((d/"scenario.json").read_text(encoding="utf-8"))
    frames={m:pd.read_csv(d/f"derived_{m}.csv") for m in COLORS}
    events=pd.read_csv(d/"events.csv")
    summary=json.loads((d/"summary.json").read_text(encoding="utf-8"))["methods"]
    return s,frames,events,summary

def axis_style(ax):
    ax.grid(True,color="#e6e8eb",lw=.4,zorder=0)
    ax.tick_params(length=2.5,width=.5,pad=2)
    for spine in ax.spines.values():spine.set_color("#6e747a")
    ax.set_axisbelow(True)

def method_handles():
    return [Line2D([],[],color=COLORS[m],lw=1.6,label=LABELS[m]) for m in COLORS]

def line_handles(calibrated=False):
    handles=[Line2D([],[],color="#656c73",ls=(0,(3,2)),lw=.9,label="Nominal")]
    if calibrated:
        handles.append(Line2D([],[],color="#656c73",ls=(0,(5,2,1,2)),lw=.9,
                              label="Calibrated nominal cost"))
    return handles

def export(fig,path,publication=False,data_status="assumed"):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix(".png"),dpi=300)
    metadata={"Title":"Assumed-data trajectory and synchronized diagnostics",
              "Author":"Illustrative case; no real experiments","CreationDate":None,"ModDate":None}
    if publication:
        metadata={"Title":"Trajectories and synchronized diagnostics",
                  "Subject":f"Data status: {data_status}; see accompanying caption and provenance.",
                  "CreationDate":None,"ModDate":None}
    fig.savefig(path.with_suffix(".pdf"),metadata=metadata)
    plt.close(fig)

