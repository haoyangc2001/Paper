"""Create a fresh r14 repair registration, preserving its frozen predecessor."""
import datetime
import hashlib
import json
from pathlib import Path

from camra_navigation.algorithm.validation import verify_training_bindings

NAV = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = NAV / "config/training_loop_prereg_r14.json"
    assert sha(previous) == Path(str(previous) + ".sha256").read_text().strip()
    p = json.loads(previous.read_bytes())
    p["version"] = "r14.1.0"
    p["status"] = "frozen_before_repaired_round_14"
    p["created_at"] = datetime.datetime.now().astimezone().isoformat()
    p["task"] = "NAV-064 r14原额度回修：obs-v5连续安全历史、T+1状态与真实critic边界、归一化min GAE、显式gamma_joint=1、启动身份核验。独立数值门禁通过后执行原六个600更新run；旧预登记和FAIL保留。"
    p["implementation_versions"] = {"observation": "obs-v5", "estimator": "rcppo-r14-v3"}
    p["authorization"]["scope"] = "用户同意当前阶段恢复安排，并明确允许当前tiger11会话直接tmux执行；原6run×600更新，首100更新计入600。新数值门禁替代相关系数硬门槛，开发关闭档及全部非时间资源边界保留。"
    p["hypothesis"] = "先验证实际边界与递推符合冻结目标，再检验原六个种子候选是否在development达到安全可行到达关闭档。负结果不自动归因于奖励尺度；corr只作分层诊断。"
    p["falsification"] = "独立数值门禁非PASS不得采样；训练或开发关闭档任一未通过则NAV-064=false，不进入NAV-065、不加r15。保留所有失败，不改变阈值。"
    p["method"]["baseline_comparison"] = "r13仅为多因素历史诊断，不能声称单变量因果对照。"
    p["contract_variant"]["comparability"] = "与r1-r13存在多项语义和环境差异，只作历史比较；旧checkpoint须用自己的观测与契约。"
    p["resource"]["max_rounds"] = 14
    p["resource"]["override_basis"] = "旧max_rounds=4字段为早期历史额度；用户逐轮授权至r14。此次仅使用尚未开始的r14六个run，不授权r15或额外性能smoke；墙钟上限已取消。"
    p["round_acceptance"]["energy_aggregation"] = "每种子六张图energy/budget中位数，再取三种子中位数；danger为每种子六图总步数后取三种子中位数。成功率为安全可行到达全分母。"
    p["round_acceptance"]["required_additional_audits"] = ["完整600次更新与18×227窗口", "无NaN/Inf与固定old_log_prob多epoch", "原始样本共同停止/预算/动作审计", "train/development数据路径、地图、散列及候选身份一致"]
    p["historical_references"] = {"gate_g3_declared_in_r14": p["binding"].pop("gate_g3")}
    p["historical_references"]["gate_g3_current_file_sha256"] = sha(NAV / "reports/gate_G3_alignment_v2.json")
    p["historical_references"]["note"] = "旧预登记的G3散列与当前报告不匹配；保留原声明及本次所见散列，不重算旧锁，不将它作为新版放行依据；当前有效放行仅用新数值门禁。"
    paths = {
        "navigation_contract": "config/navigation_contract_r14_repair_v2.json",
        "value_contract": "config/value_objective_contract_r14_repair_v1.json",
        "numerical_gate": "reports/gate_r14_repair_v1_attempt3.json",
        "h_semantics_test": "tests/test_h_semantics_v2.py",
        "split": "config/split_manifest_v1.json",
        "map_design": "config/map_design_v1.json",
        "geometry_contract": "config/geometry_contract_v1.json",
        "power_model": "../calibration/config/power_model_v1.json",
        "footprint": "../calibration/config/footprint_v1.json",
        "margins": "../calibration/config/margins_v1.json",
        "data_lock": "../calibration/config/data_lock_v1.json",
        "calibration_nominal_source": "../calibration/src/camra_calibration/nominal.py",
        "calibration_geometry_source": "../calibration/src/camra_calibration/geometry.py",
        "calibration_power_source": "../calibration/src/camra_calibration/power.py",
    }
    for name, rel in paths.items():
        p["binding"][name] = {"path": rel, "sha256": sha(NAV / rel)}
    p["amendments"].append({"version": p["version"], "date": p["created_at"],
                            "previous_prereg_sha256": sha(previous),
                            "trigger": "用户同意的2026-11-15恢复安排与同机授权，修复确定性观测/递推/边界/启动接线缺陷；原门禁相关性不足以判断正确性。",
                            "changes": p["task"], "unchanged": "6run×600更新、18×227窗口、T45/r_goal0.5、k抽样与开发网格、最后checkpoint、固定v2、train/dev、原关闭档阈值"})
    verify_training_bindings(p, NAV)
    target = NAV / "config/training_loop_prereg_r14_repair_v1.json"
    with target.open("x") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with Path(str(target) + ".sha256").open("x") as f:
        f.write(sha(target) + "\n")
    print(target, sha(target))


if __name__ == "__main__":
    main()
