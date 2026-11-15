"""NAV-064 r13 候选：预算内逐步成本的基础权重（base_weight）。

验收意图（先冻结判据，再写实现）：
  CB1  ``base_weight=0`` 时与旧公式逐位相同（关闭即回归保护）。
  CB2  ``base_weight>0`` 时：超支步仍用 ``weight``，未超支步用 ``base_weight``，
       终端项不变（终端无动作，不得被加成本）。
  CB3  不传 overspend_mask 时，全部步用 ``weight+base_weight``。
  CB4  非法输入必须显式报错（负权重、掩码长度不符）。
  CB5  默认构造的 ObjectiveConfig 该字段为 0.0，且 from_contracts 不改变它——
       保证「不打开就逐位等价」不是靠调用方自觉。
"""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_NAV_ROOT = os.path.join(_REPO, "CaMRA", "navigation")
sys.path.insert(0, os.path.join(_NAV_ROOT, "src"))
sys.path.insert(0, _NAV_ROOT)

import numpy as np  # noqa: E402

from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.algorithm import targets as T  # noqa: E402

VALUE_CONTRACT = os.path.join(_NAV_ROOT, "config", "value_objective_contract_v1.json")
RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: dict) -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("  OK   " if ok else "  FAIL ") + name, flush=True)
    if not ok:
        print("       " + json.dumps(detail, ensure_ascii=False, default=str)[:1200], flush=True)


def main() -> int:
    g_ext = np.array([0.10, 0.90, 0.40, 0.70, 1.00], dtype=np.float64)   # T=4, 末项为终端
    c = np.array([0.0021, 0.0024, 0.0019, 0.0026], dtype=np.float64)
    m = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)                 # 超支标志：2、4 步超支

    # ---- CB1 关闭即逐位相同 ------------------------------------------
    legacy = g_ext.copy()
    legacy[:-1] += 50.0 * c * m
    got = T.joint_training_sequence(g_ext, c, 50.0, overspend_mask=m, base_weight=0.0)
    check("CB1 base_weight=0 与旧公式逐位相同", np.array_equal(got, legacy),
          {"got": got.tolist(), "legacy": legacy.tolist()})

    # ---- CB2 未超支步用 base，超支步用 weight，终端不动 ----------------
    wb = 2.0
    got2 = T.joint_training_sequence(g_ext, c, 50.0, overspend_mask=m, base_weight=wb)
    want = g_ext.copy()
    want[:-1] += 50.0 * c * m + wb * c * (1.0 - m)
    check("CB2 超支/未超支分别用大权重与小权重", np.allclose(got2, want, rtol=0, atol=0),
          {"got": got2.tolist(), "want": want.tolist()})
    check("CB2 终端项未被加成本", got2[-1] == g_ext[-1], {"terminal": got2[-1], "expect": g_ext[-1]})
    check("CB2 未超支步的成本高于关闭时", got2[0] > legacy[0] and got2[2] > legacy[2],
          {"step1": [got2[0], legacy[0]], "step3": [got2[2], legacy[2]]})

    # ---- CB3 无掩码时全部步用 weight+base ----------------------------
    got3 = T.joint_training_sequence(g_ext, c, 50.0, overspend_mask=None, base_weight=wb)
    want3 = g_ext.copy()
    want3[:-1] += (50.0 + wb) * c
    check("CB3 无掩码时全部步用 weight+base", np.allclose(got3, want3, rtol=0, atol=0),
          {"got": got3.tolist(), "want": want3.tolist()})

    # ---- CB4 非法输入显式报错 ----------------------------------------
    bad = 0
    try:
        T.joint_training_sequence(g_ext, c, 50.0, overspend_mask=m, base_weight=-1.0)
    except ValueError:
        bad += 1
    try:
        T.joint_training_sequence(g_ext, c, -1.0, overspend_mask=m)
    except ValueError:
        bad += 1
    try:
        T.joint_training_sequence(g_ext, c, 50.0, overspend_mask=np.zeros(3), base_weight=1.0)
    except ValueError:
        bad += 1
    try:
        T.joint_training_sequence(g_ext, c[:-1], 50.0, overspend_mask=m)
    except ValueError:
        bad += 1
    check("CB4 负权重/长度不符均显式报错", bad == 4, {"errors_raised": bad})

    # ---- CB5 默认字段为 0，契约构造不改变它 --------------------------
    base = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
    check("CB5 from_contracts 的 base 权重默认为 0.0", float(base.joint_cost_base_weight) == 0.0,
          {"joint_cost_base_weight": base.joint_cost_base_weight})

    failed = [r["name"] for r in RESULTS if not r["ok"]]
    print("\n%d/%d PASS" % (len(RESULTS) - len(failed), len(RESULTS)))
    if failed:
        print("FAILED:", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
