"""Two horizontal stages, parallel critics, and reusable multi-node network panels."""
from scene_assets import *
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle
from matplotlib.text import Text
from input_illustrations import add_input_illustrations, export_input_components
from command_illustrations import add_command_illustrations, export_command_components
from interaction_illustration import interaction_scene, interaction_aspect

LABELS = []
LINKS = []
NETWORKS = []


def label(ax, x, y, text, size=8, color=INK, ha="left", bold=False, **kwargs):
    artist = txt(ax, x, y, text, size, color, ha, "bold" if bold else "normal", **kwargs)
    LABELS.append(artist)
    return artist


def wire(ax, name, points, color=INK, training=False, lw=1.0):
    arrow(ax, points, color, "--" if training else "-", lw)
    LINKS.append({"id": name, "points": points, "kind": "training" if training else "execution", "arrowhead": True})


def training_branch(ax, name, points, color):
    """Both evaluations join one policy-update arrow without intermediate heads."""
    p = np.asarray(points)
    ax.plot(p[:, 0], p[:, 1], color=color, ls="--", lw=.75, zorder=8)
    LINKS.append({"id":name, "points":points, "kind":"training", "arrowhead":False})


def neural(ax, name, color, output_nodes=3):
    """Many circles and fine connections; layer widths are explicitly schematic."""
    clean(ax)
    sizes = [5, 7, 7, 5, output_nodes]
    xs = np.linspace(.07, .93, len(sizes))
    layers = [np.c_[np.full(n, x), np.linspace(.08, .92, n) if n > 1 else [.5]]
              for x, n in zip(xs, sizes)]
    edges = 0
    for left, right in zip(layers, layers[1:]):
        for p in left:
            for q in right:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, alpha=.24, lw=.34, zorder=1)
                edges += 1
    for layer in layers:
        ax.scatter(layer[:, 0], layer[:, 1], s=9.5, facecolor="white", edgecolor=color,
                   linewidth=.65, zorder=3)
    NETWORKS.append({"name": name, "nodes": sum(sizes), "edges": edges,
                     "layers_drawn": sizes, "schematic_not_actual_layer_widths": True})


def net_axes(fig, name, rect, color, outputs=3):
    ax = fig.add_axes(rect)
    neural(ax, name, color, outputs)
    return ax


def card(ax, rect, fill, edge="#B8C2CA", lw=.8):
    return box(ax, *rect, fill, edge, .012, lw)


def stage_heading(ax, y, number, title):
    card(ax, [.023, y-.020, .033, .038], INK, INK)
    label(ax, .0395, y, str(number), 10, "white", "center", True)
    label(ax, .068, y, title, 10, bold=True)


def physical_panel(fig, rect, function):
    axis = fig.add_axes(rect)
    function(axis)
    axis.patch.set_edgecolor("#CBD3D8")
    axis.patch.set_linewidth(.5)
    return axis


def rect_for_scene(x, y, width, robot=False):
    crop = L["interface_crop"]
    aspect = 1600/1050 if not robot else 1600*(crop[2]-crop[0])/(1050*(crop[3]-crop[1]))
    return [x, y, width, width*W/aspect/H]


def environment_scene(ax):
    interaction_scene(ax)


def label_geometry(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    values = []
    axes=list(fig.axes)
    for axis in list(axes):axes.extend(axis.child_axes)
    for item in [text for axis in axes for text in axis.texts if text.get_text()]:
        bounds = Text.get_window_extent(item, renderer).transformed(fig.transFigure.inverted())
        values.append({"text": item.get_text(), "font_pt": item.get_fontsize(),
                       "bbox_figure": list(bounds.extents)})
    return values


def draw_main(out, storyboard=False):
    """Two stacked calibration rows; parallel stacked critics left of the actor."""
    LABELS.clear(); LINKS.clear(); NETWORKS.clear()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); clean(ax)
    plt.rcParams["svg.hashsalt"] = "hdmcra-safety-inset-v17"
    stage_heading(ax, .970, 1, "Interface training & calibration")
    card(ax, [.015, .590, .970, .350], "#FBFCFD")
    label(ax, .145, .899, "Train & freeze", 8, BLUE, "center", True)
    label(ax, .376, .899, "Paired execution", 8, INK, "center", True)
    net_axes(fig, "stage1_velocity_tracker", [.065, .722, .158, .135], BLUE)
    lock(ax, .145, .680, BLUE)
    label(ax, .145, .628, "Velocity tracker", 7.4, BLUE, "center")
    wire(ax, "freeze_then_collect", [(.235, .778), (.274, .778)], BLUE)
    scenes = [(rect_for_scene(.284, .681, .190, True), robot_scene, "Robot scene"),
              (rect_for_scene(.535, .774, .145), cost_scene, "Cost routes"),
              (rect_for_scene(.535, .610, .145), safety_scene, "Paired paths")]
    for rect, function, title in scenes:
        if storyboard:
            card(ax, rect, "#EDF1F4")
            label(ax, rect[0]+rect[2]/2, rect[1]+rect[3]/2, title, 7, ha="center")
        else:
            physical_panel(fig, rect, function)
    label(ax, .379, .654, "Same commands", 7, ha="center")
    label(ax, .379, .622, "Nominal / executed", 7, GRAY, "center")
    # One collection branch, two independent calibration rows.
    ax.plot([.485, .510, .510], [.778, .778, .848], color=GRAY, lw=.8)
    ax.plot([.510, .510], [.778, .684], color=GRAY, lw=.8)
    wire(ax, "records_to_cost", [(.510, .848), (.530, .848)], GRAY, lw=.8)
    wire(ax, "records_to_safety", [(.510, .684), (.530, .684)], GRAY, lw=.8)
    label(ax, .705, .903, "Cost margin", 8, ORANGE, bold=True)
    label(ax, .705, .869, "Execution cost overrun", 7, GRAY)
    label(ax, .705, .824, r"$\alpha=Q(\varepsilon_c)$", 10, ORANGE)
    label(ax, .705, .739, "Safety margin", 8, GREEN, bold=True)
    label(ax, .705, .705, "Trajectory deviation", 7, GRAY)
    ax.plot([.705, .725], [.671, .671], color=LIGHT, ls="--", lw=1.0)
    label(ax, .732, .671, "Nominal", 7, GRAY)
    ax.plot([.835, .855], [.671, .671], color=BLUE, lw=1.0)
    label(ax, .862, .671, "Executed", 7, GRAY)
    label(ax, .705, .631, r"$\delta=Q(\varepsilon_d)$", 10, GREEN)

    stage_heading(ax, .548, 2, "Calibrated high-level learning & execution")
    card(ax, [.015, .075, .970, .440], "#F4F7F9")
    lower=read(ROOT/'config/execution_layout.json')
    actor=lower['actor'];tracker=lower['tracker'];chain_y=lower['chain_y']
    actor_x=lower['actor_center_x'];tracker_x=lower['tracker_center_x']
    card(ax, lower['navigation_card'], "white", "#8B9BA6", 1.0)
    label(ax, lower['navigation_heading_x'], .470, "High-level navigation", 8.5, ha="center", bold=True)
    label(ax, .247, .432, "Cost critic", 7.8, ORANGE, "center", True)
    label(ax, .247, .397, r"$\tilde{c}=(1+\alpha)c$", 10, ORANGE, "center")
    label(ax, .247, .272, "Reachability critic", 7.8, GREEN, "center", True)
    label(ax, .260, .234, r"$\mathcal{F}_\delta=\{x:d(x,\mathcal{F})\leq\delta\}$", 10, GREEN, "center")
    net_axes(fig, "cost_critic", [.177, .307, .140, .065], ORANGE, 1)
    net_axes(fig, "reachability_critic", [.177, .146, .140, .065], GREEN, 1)
    label(ax, actor_x, .417, "Actor", 8, BLUE, "center", True)
    label(ax, actor_x, .220, "Budget-conditioned", 7, BLUE, "center")
    net_axes(fig, "navigation_actor", actor, BLUE)
    # Parallel evaluation outputs join, without either critic feeding the other.
    merge=lower['critic_merge_x']
    training_branch(ax, "cost_to_joint_update", [(.317, .3395), (merge, .3395), (merge, chain_y)], ORANGE)
    training_branch(ax, "reach_to_joint_update", [(.317, .1785), (merge, .1785), (merge, chain_y)], GREEN)
    wire(ax, "joint_update_to_actor", [(merge, chain_y), (actor[0], chain_y)], GRAY, True, .8)
    # One module-level input; fan-out and feedback are omitted only visually.
    add_input_illustrations(fig, ax, label)
    wire(ax, "input_to_navigation", [(.125, .300), (.140, .300)], GRAY, lw=.8)
    label(ax, actor_x, .172, r"$z_{t+1}=z_t-\tilde{c}_t$", 10, ORANGE, "center")
    label(ax, tracker_x, .215, "Frozen tracker", 7.8, BLUE, "center", True)
    lock(ax, tracker_x, .160, BLUE)
    net_axes(fig, "stage2_velocity_tracker", tracker, BLUE)
    env_x=lower['environment_x'];env_width=lower['environment_width']
    interaction_layout=read(ROOT/'config/interaction_layout.json')
    label(ax, lower['environment_label_x'], .206, interaction_layout['environment_label'],
          interaction_layout['environment_label_font_pt'], ha="center", bold=interaction_layout['environment_label_bold'], linespacing=1.35)
    rect = [env_x,0,env_width,env_width*W/interaction_aspect()/H]
    rect[1]=chain_y-rect[3]/2
    if storyboard:
        card(ax, rect, "#EAF0F4")
        label(ax, env_x+env_width/2, chain_y, "Execution", 7, ha="center")
    else:
        physical_panel(fig, rect, environment_scene)
    velocity_link=[(actor[0]+actor[2],chain_y),(tracker[0],chain_y)]
    joint_link=[(tracker[0]+tracker[2],chain_y),(env_x,chain_y)]
    wire(ax, "actor_velocity_output", velocity_link, BLUE, lw=1.1)
    wire(ax, "tracker_joint_output", joint_link, INK, lw=1)
    commands=add_command_illustrations(fig, {'velocity':velocity_link,'joint':joint_link}, ax, label)
    geometry = label_geometry(fig)
    save(fig, out/("storyboard" if storyboard else "deployment_calibration_overview"))
    dump(out/"build_metadata.json", {
        "version":out.name, "width_in":W, "height_in":H, "minimum_base_font_pt":7,
        "layout":"two_stages_stacked_margin_images_right_text_stacked_critics_actor_right",
        "cross_stage_connections":0, "critics_parallel":True, "connections":LINKS,
        "lower_arrowheads":sum(v["arrowhead"] for v in LINKS if max(p[1] for p in v["points"]) < .55),
        "omitted_for_clarity":["observation_fanout", "state_feedback", "proprioceptive_feedback"],
        "input_illustrations":{"source":"server_mujoco_egl_static_go2_renders", "categories":["navigation_observation","goal_position","remaining_cost_budget"], "not_sensor_measurements":True,"not_battery_energy":True,"physics_steps":0,"policy_executed":False},
        "joint_update_label":False,
        "safety_inset":{"source_frame":34,"world_window_m":[.20,.20],"equal_aspect":True,"same_time_pair":True,"deviation_is_not_delta":True},
        "interaction_scene":{"source":"assets/interaction_renders_v2/interaction.png","source_frames":[0,36,74],"opacities":[0.45,0.70,1.0],"obstacles":3,"path_points":75,"one_robot_at_three_times":True,"physics_steps":0,"policy_executed":False,"data":"existing_assumed_R1","crop":interaction_layout['crop']},
        "execution_layout":{"actor":actor,"tracker":tracker,"environment":rect,"commands":commands,"chain_y":chain_y,"command_position":"centered_over_corresponding_arrow_with_0.015_gap","command_hierarchy_labels":True},
        "command_illustrations":{"source":"server_static_go2_home_pose","velocity":"body_translation_and_yaw","joint":"right_front_hip_and_knee_rotations","arrows":"projected_schematic_annotations","physics_steps":0,"policy_executed":False},
        "network_panels":NETWORKS, "labels":geometry, "alpha":S["alpha"], "delta_m":S["delta_m"],
        "margin_implementation":"design_only_not_yet_integrated_in_local_reference_code",
        "background":"unchanged_static_mujoco_egl", "data":"unchanged_assumed_pairs",
        "network_geometry":"generic_schematic_not_actual_layer_counts_or_widths"})
    plt.close(fig)


def components(out):
    export_input_components(out)
    export_command_components(out)
    for name, color, nout in [("cost_critic", ORANGE, 1), ("reachability_critic", GREEN, 1),
                               ("navigation_actor", BLUE, 3), ("velocity_tracker", BLUE, 3)]:
        fig = plt.figure(figsize=(1.25, .85))
        axis = fig.add_axes([.05, .05, .90, .90]); neural(axis, name, color, nout)
        save(fig, out/"panels/networks"/name); plt.close(fig)
    for name, function in [("cost_scene", cost_scene), ("safety_scene", safety_scene), ("robot_execution", environment_scene)]:
        crop = L["interface_crop"]
        ratio = 1600/1050 if name != "robot_execution" else interaction_aspect()
        fig = plt.figure(figsize=(2.5, 2.5/ratio)); axis=fig.add_axes([0,0,1,1]); function(axis)
        save(fig, out/"panels/scenes"/name); plt.close(fig)


def build(out, storyboard=False):
    out = Path(out)
    if out.exists():
        raise SystemExit("Choose a new output directory; previous versions are preserved.")
    out.mkdir(parents=True)
    draw_main(out, storyboard)
    if not storyboard:
        components(out)
    print("Built", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--storyboard", action="store_true")
    args = parser.parse_args()
    build(args.out, args.storyboard)
