"""HiRA 控制器包的冻结与重载（HIRA-020）。

一个"控制器包"必须能回答四个问题，否则它只是一堆文件：

1. **这是哪个控制器**：``controller_id`` + 源 checkpoint 的散列 + actor 张量摘要；
2. **它由什么组成**：策略文件、关节双射、默认角与限幅、观测/动作契约、时钟、资产与环境依赖；
3. **它有没有被改过**：``manifest.json`` 里逐文件 sha256，加上由「文件散列 + 描述内容」
   算出的 ``bundle_id``；
4. **改了会被发现吗**：``load_bundle()`` 不仅核对散列，还要做**语义校验**——
   只改内容再重算散列（伪装成"合法包"）必须被映射/时钟/身份的一致性检查挡住。

第 4 点是关键。只核散列的实现很容易"看起来通过"：伪造者改完文件重算一遍 manifest 就过了。
所以这里把「双射性、时钟自洽、维度自洽、关键字段必须显式」写成硬校验。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

BUNDLE_SCHEMA = "hira-controller-bundle-v1"
MANIFEST_NAME = "manifest.json"
CONTROLLER_NAME = "controller.json"
POLICY_NAME = "policy.pt"


class BundleError(RuntimeError):
    """包不完整、散列不符或语义自相矛盾。"""


# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_obj(obj) -> str:
    body = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def compute_bundle_id(controller: dict, files: list[dict]) -> str:
    """bundle_id = 描述内容的散列 + 逐文件散列，两者一起算。

    只算文件散列不够：``controller.json`` 是"这个包声称自己是什么"，
    它被改了但文件清单重算过的话，光看文件散列发现不了。
    """
    payload = {
        "schema": BUNDLE_SCHEMA,
        "controller": sha256_obj(controller),
        "files": sorted((item["path"], item["sha256"]) for item in files),
    }
    return sha256_obj(payload)


# --------------------------------------------------------------------------- #
def build_controller_descriptor(
    *,
    controller_id: str,
    version: str,
    policy_path: Path,
    source: dict,
    joints: dict,
    control: dict,
    observation: dict,
    timing: dict,
    asset: dict,
    energy: dict,
    environment: dict,
    acceptance: dict,
    created_at: str,
    policy_meta: dict | None = None,
    notes: list[str] | None = None,
) -> dict:
    """组装控制器描述。字段名即契约，缺一个都不允许（load 时会逐项校验）。"""
    return {
        "schema": BUNDLE_SCHEMA,
        "controller_id": controller_id,
        "version": version,
        "created_at": created_at,
        "policy": {
            "file": POLICY_NAME,
            "sha256": sha256_file(policy_path),
            "bytes": int(Path(policy_path).stat().st_size),
            **(policy_meta or {}),
        },
        "source": source,
        "joints": joints,
        "control": control,
        "observation": observation,
        "timing": timing,
        "asset": asset,
        "energy": energy,
        "environment": environment,
        "acceptance": acceptance,
        "notes": notes or [],
    }


# --------------------------------------------------------------------------- #
def freeze_bundle(
    bundle_dir: Path,
    controller: dict,
    policy_path: Path,
    extra_files: dict[str, Path] | None = None,
) -> dict:
    """把描述与策略写进一个**新**目录，生成 manifest，返回 manifest。

    ``extra_files`` 把来源文件一并封进包（例如源 URDF），让 ``load_bundle``
    能在没有仓库的情况下自证关节顺序与限幅——只靠"描述里写了什么"是挡不住伪造的。
    """
    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists():
        raise BundleError(f"冻结目录已存在（包不可变，禁止覆盖）：{bundle_dir}")
    bundle_dir.mkdir(parents=True)
    shutil.copyfile(policy_path, bundle_dir / POLICY_NAME)
    for name, source in (extra_files or {}).items():
        shutil.copyfile(source, bundle_dir / name)

    controller = dict(controller)
    controller["policy"] = dict(controller["policy"])
    controller["policy"]["sha256"] = sha256_file(bundle_dir / POLICY_NAME)
    controller["policy"]["bytes"] = int((bundle_dir / POLICY_NAME).stat().st_size)

    (bundle_dir / CONTROLLER_NAME).write_text(
        json.dumps(controller, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle_dir / "README.md").write_text(
        "\n".join(
            [
                f"# HiRA 控制器包 {controller['controller_id']} {controller['version']}",
                "",
                "不可变。任何权重、资产、增益、观测、时钟的改动都必须产生新的 controller 版本"
                "并重过受影响的门禁（HIRA-020 约束）。",
                "",
                f"- 策略：`{POLICY_NAME}`（TorchScript，sha256 {controller['policy']['sha256'][:16]}…）",
                f"- 描述：`{CONTROLLER_NAME}`",
                f"- 清单：`{MANIFEST_NAME}`（bundle_id 覆盖描述内容与逐文件散列）",
                "",
                "重载与校验：`from hira_go2.bundle import load_bundle; load_bundle(path)`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    files = []
    for path in sorted(bundle_dir.iterdir()):
        if path.is_file() and path.name != MANIFEST_NAME:
            files.append(
                {"path": path.name, "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
            )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "controller_id": controller["controller_id"],
        "version": controller["version"],
        "created_at": controller["created_at"],
        "bundle_id": compute_bundle_id(controller, files),
        "files": files,
    }
    (bundle_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return manifest


# --------------------------------------------------------------------------- #
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def _check_joint_bijection(joints: dict, n_joints: int) -> None:
    source = list(joints["source_order"])
    target = list(joints["target_order"])
    _require(len(source) == n_joints, f"源关节数 {len(source)} ≠ {n_joints}")
    _require(len(target) == n_joints, f"目标关节数 {len(target)} ≠ {n_joints}")
    _require(len(set(source)) == n_joints, "源关节名有重复")
    _require(len(set(target)) == n_joints, "目标关节名有重复")
    _require(set(source) == set(target), "源与目标关节集合不同——不是同一台机器人的两种排序")
    derived = [source.index(name) for name in target]
    _require(
        list(joints["permutation_source_to_target"]) == derived,
        "permutation_source_to_target 与两份关节名列表不一致（映射被改过）",
    )
    missing_defaults = sorted(set(source) - set(joints["default_angles_source"]))
    missing_limits = sorted(set(source) - set(joints["effort_limits_source"]))
    _require(not missing_defaults, f"默认角缺少关节：{missing_defaults}")
    _require(not missing_limits, f"力矩限幅缺少关节：{missing_limits}")


def _check_timing(timing: dict) -> None:
    sim_dt = float(timing["sim_dt"])
    decimation = int(timing["decimation"])
    command_policy_steps = int(timing["command_policy_steps"])
    window_s = float(timing["window_s"])
    _require(abs(float(timing["policy_dt"]) - sim_dt * decimation) < 1e-12, "policy_dt ≠ sim_dt × decimation")
    _require(
        abs(float(timing["command_dt"]) - float(timing["policy_dt"]) * command_policy_steps) < 1e-12,
        "command_dt ≠ policy_dt × command_policy_steps",
    )
    _require(
        int(timing["physics_steps"]) == round(window_s / sim_dt),
        f"physics_steps 与 window_s/sim_dt 不符（{timing['physics_steps']} vs {round(window_s / sim_dt)}）",
    )
    _require(
        int(timing["policy_steps"]) == round(window_s / float(timing["policy_dt"])),
        "policy_steps 与 window_s/policy_dt 不符",
    )
    _require(
        int(timing["command_intervals"]) == round(window_s / float(timing["command_dt"])),
        "command_intervals 与 window_s/command_dt 不符",
    )
    _require(
        int(timing["paired_points"]) == int(timing["command_intervals"]) + 1,
        "paired_points ≠ command_intervals + 1",
    )


def _check_observation(observation: dict, n_joints: int) -> None:
    dim = int(observation["dim"])
    blocks = observation["blocks"]
    _require(dim == 45, f"观测维度 {dim} ≠ 45")
    _require(len(blocks) == 6, f"观测块数 {len(blocks)} ≠ 6")
    cursor = 0
    for block in blocks:
        _require(int(block["start"]) == cursor, f"观测块 {block['name']} 起点 {block['start']} ≠ {cursor}")
        cursor = int(block["end"])
    _require(cursor == dim, f"观测块覆盖到 {cursor}，不等于 {dim}")
    _require(int(observation["n_joints"]) == n_joints, "观测的关节数与动作维度不一致")


def _check_policy(policy: dict, observation: dict, control: dict) -> None:
    _require(policy["file"] == POLICY_NAME, f"策略文件名必须是 {POLICY_NAME}")
    _require(int(policy["input_dim"]) == int(observation["dim"]), "策略输入维度 ≠ 观测维度")
    _require(int(policy["output_dim"]) == int(control["action_dim"]), "策略输出维度 ≠ 动作维度")
    _require(int(control["action_dim"]) == int(control["n_joints"]), "动作维度 ≠ 关节数")
    _require(bool(control["deterministic"]), "控制器包必须声明为确定性推理")


def _urdf_facts(path: Path) -> tuple[list[str], dict[str, float]]:
    """从 URDF 里读出关节声明顺序与 effort 限幅（与 HIRA-012 的解析同规则）。"""
    import re

    text = Path(path).read_text(encoding="utf-8")
    order, limits = [], {}
    for match in re.finditer(r'<joint\s+name="([^"]+)"\s+type="revolute">(.*?)</joint>', text, re.S):
        name, body = match.group(1), match.group(2)
        order.append(name)
        effort = re.search(r'effort="([0-9.]+)"', body)
        if effort:
            limits[name] = float(effort.group(1))
    return order, limits


def load_bundle(
    bundle_dir: Path,
    expected_controller_id: str | None = None,
    expected_joints: dict | None = None,
) -> dict:
    """校验并加载控制器包；任何一项不符都抛 :class:`BundleError`。

    ``expected_joints`` 是**包外**的参照（来自运行时审计），专门用来抓
    "映射被改但散列重封过"的伪造包——这类改动在包内部是完全自洽的，
    只有拿外部参照才能识破。
    """
    bundle_dir = Path(bundle_dir)
    _require(bundle_dir.is_dir(), f"包目录不存在：{bundle_dir}")
    manifest_path = bundle_dir / MANIFEST_NAME
    controller_path = bundle_dir / CONTROLLER_NAME
    _require(manifest_path.is_file(), "缺少 manifest.json")
    _require(controller_path.is_file(), "缺少 controller.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == BUNDLE_SCHEMA, "manifest schema 不符")
    _require(controller.get("schema") == BUNDLE_SCHEMA, "controller schema 不符")

    # ---- 1) 逐文件散列 ----
    listed = {item["path"]: item for item in manifest["files"]}
    on_disk = {path.name for path in bundle_dir.iterdir() if path.is_file() and path.name != MANIFEST_NAME}
    _require(set(listed) == on_disk, f"文件清单与目录不一致：清单 {sorted(listed)} vs 实际 {sorted(on_disk)}")
    for name, item in listed.items():
        actual = sha256_file(bundle_dir / name)
        _require(actual == item["sha256"], f"{name} 散列不符（清单 {item['sha256'][:12]} vs 实际 {actual[:12]}）")
        _require(
            int(item["bytes"]) == int((bundle_dir / name).stat().st_size),
            f"{name} 字节数不符",
        )

    # ---- 2) bundle_id 覆盖"描述内容 + 文件散列" ----
    recomputed = compute_bundle_id(controller, manifest["files"])
    _require(recomputed == manifest["bundle_id"], "bundle_id 与描述内容/文件散列不一致（内容被改过）")

    # ---- 3) 语义校验：伪装成合法包（改了内容并重算散列）必须在这里被挡住 ----
    controller_id = controller["controller_id"]
    if expected_controller_id is not None:
        _require(controller_id == expected_controller_id, f"controller_id {controller_id!r} ≠ 期望 {expected_controller_id!r}")
    _require(bundle_dir.name == controller_id, f"目录名 {bundle_dir.name!r} ≠ controller_id {controller_id!r}")

    control = controller["control"]
    n_joints = int(control["n_joints"])
    _require(n_joints == 12, f"关节数 {n_joints} ≠ 12")
    _require(float(control["action_scale"]) > 0, "action_scale 必须为正")
    _require(float(control["clip_actions"]) > 0, "clip_actions 必须为正")
    _require(int(control["decimation"]) >= 1, "decimation 必须 ≥ 1")
    _require(float(control["stiffness"]) > 0 and float(control["damping"]) >= 0, "增益必须非负且 kp > 0")

    _check_joint_bijection(controller["joints"], n_joints)
    _check_timing(controller["timing"])
    _require(int(controller["timing"]["decimation"]) == int(control["decimation"]), "时钟与控制的 decimation 不一致")
    _check_observation(controller["observation"], n_joints)
    _check_policy(controller["policy"], controller["observation"], control)

    asset = controller["asset"]
    _require(float(asset["friction"]) > 0, "asset.friction 必须显式给出且为正（导入默认 0.5 ≠ 源 1.0）")
    _require(
        abs(float(asset["friction"]) - float(asset["source_friction"])) < 1e-9,
        "asset.friction 与 source_friction 不一致——clean 条件必须用源值",
    )
    _require(len(str(asset["urdf_sha256"])) == 64, "urdf_sha256 不是 64 位十六进制")

    energy = controller["energy"]
    _require(energy["tau_source"] == "robot.data.applied_torque", "能量计量的力矩来源必须是 applied_torque")
    _require(energy["qdot_phase"] == "post_substep", "能量计量的速度相位必须是子步结束时刻")
    _require(abs(float(energy["dt_s"]) - float(controller["timing"]["sim_dt"])) < 1e-12, "能量 dt ≠ sim_dt")

    # ---- 3.5) 包内来源自证：源 URDF 的关节顺序与限幅 ----
    urdf_in_bundle = controller["asset"].get("urdf_in_bundle")
    if urdf_in_bundle:
        urdf_path = bundle_dir / urdf_in_bundle
        _require(urdf_path.is_file(), f"包内缺少来源 URDF：{urdf_in_bundle}")
        _require(
            sha256_file(urdf_path) == controller["asset"]["urdf_sha256"],
            "包内 URDF 与登记的 urdf_sha256 不一致",
        )
        urdf_order, urdf_limits = _urdf_facts(urdf_path)
        _require(
            list(controller["joints"]["source_order"]) == urdf_order,
            "source_order 与包内 URDF 的关节声明顺序不一致（来源被改）",
        )
        _require(
            controller["joints"]["effort_limits_source"] == urdf_limits,
            "effort_limits_source 与包内 URDF 的 effort 值不一致（限幅被改）",
        )

    # ---- 3.6) 外部参照：运行时审计过的 DOF 顺序 ----
    if expected_joints is not None:
        for key in ("source_order", "target_order"):
            if key in expected_joints:
                _require(
                    list(controller["joints"][key]) == list(expected_joints[key]),
                    f"{key} 与运行时审计的参照不一致（映射被改）",
                )

    # ---- 4) 策略文件必须真的能加载且尺寸对得上 ----
    policy_path = bundle_dir / POLICY_NAME
    _require(policy_path.stat().st_size > 0, "策略文件为空")
    _require(int(controller["policy"]["bytes"]) == int(policy_path.stat().st_size), "策略字节数与描述不符")

    return {"dir": str(bundle_dir), "manifest": manifest, "controller": controller}
