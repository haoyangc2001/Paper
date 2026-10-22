"""Input pictures rendered with the existing Go2 model on the authorized server.

These are static scene illustrations, not camera observations or policy rollouts.
The budget picture uses the existing assumed route; z_t denotes cost budget.
"""
from scene_assets import *

RENDERS = ROOT / "assets/input_renders"
COLORS = {"observation": BLUE, "goal": GREEN, "budget": ORANGE}


def scene(ax, name):
    path = RENDERS / (name + ".png")
    meta = read(RENDERS / (name + "_metadata.json"))
    assert sha(path) == meta["image_sha256"]
    assert meta["physics_steps"] == 0 and not meta["policy_executed"]
    clean(ax)
    ax.imshow(Image.open(path).convert("RGB"), extent=(0, 1, 0, 1), aspect="auto")
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=COLORS[name], lw=.7, zorder=5))
    if name == "budget":
        txt(ax, .15, .81, r"$z_t$", 10, ORANGE, "center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": .94})


def add_input_illustrations(fig, parent, text):
    """Three consistent scene views; short labels identify their input meanings."""
    width = .096
    height = width * W / H / 1.6
    for name, y, title in [("observation", .387, "Observation"),
                            ("goal", .267, "Goal"),
                            ("budget", .147, "Budget")]:
        axis = fig.add_axes([.024, y, width, height])
        scene(axis, name)
        text(parent, .072, y-.016, title, 7, COLORS[name], "center")


def export_input_components(out):
    for name in COLORS:
        fig = plt.figure(figsize=(1.8, 1.125))
        axis = fig.add_axes([0, 0, 1, 1])
        scene(axis, name)
        save(fig, out / "panels/inputs" / name)
        plt.close(fig)
