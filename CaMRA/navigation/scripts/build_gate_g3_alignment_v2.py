"""Build reports/gate_G3_alignment_v2.json (NAV-063 G3 revalidation).

The preregistration is checked by hash before anything runs; the verdict is a
pure function of the test output, so a FAIL cannot be re-labelled by editing
prose.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

NAV = os.getcwd()
PREREG = os.path.join(NAV, "config", "gate_g3_alignment_v2_prereg.json")
PREREG_SHA = PREREG + ".sha256"
TEST = os.path.join(NAV, "tests", "test_rcppo_alignment.py")
OUT = os.path.join(NAV, "reports", "gate_G3_alignment_v2.json")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    if not os.path.exists(PREREG_SHA):
        print("prereg sidecar missing:", PREREG_SHA)
        return 2
    pinned = open(PREREG_SHA, encoding="utf-8").read().strip()
    actual = sha(PREREG)
    if pinned != actual:
        print(f"prereg hash mismatch: pinned {pinned[:16]} vs actual {actual[:16]}")
        return 2
    prereg = json.load(open(PREREG, encoding="utf-8"))

    tmp = tempfile.mkdtemp(prefix="g3a_")
    raw = os.path.join(tmp, "test.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(NAV, "src"), os.path.dirname(os.path.dirname(NAV)),
         os.path.join(os.path.dirname(os.path.dirname(NAV)), "CaMRA", "calibration", "src")])
    proc = subprocess.run([sys.executable, TEST, "--json", raw],
                          capture_output=True, text=True, env=env, cwd=NAV, timeout=1800)
    if not os.path.exists(raw):
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        return 4
    test = json.load(open(raw, encoding="utf-8"))
    by_name = {c["check"]: c for c in test["checks"]}

    rows = []
    for crit in prereg["criteria"]:
        got = by_name.get(crit["check"])
        rows.append({
            "criterion": crit["id"],
            "check": crit["check"],
            "requirement": crit["requirement"],
            "tolerance": crit["tolerance"],
            "status": "PASS" if got and got["ok"] else ("MISSING" if not got else "FAIL"),
            "detail": None if not got else got["detail"],
        })
    failed = [r for r in rows if r["status"] != "PASS"]
    alignment = json.load(open(os.path.join(NAV, "reports", "rcppo_alignment.json"),
                               encoding="utf-8"))
    gate = {
        "schema": "camra-gate-report-v1",
        "gate": "G3",
        "version": "alignment_v2",
        "task": "NAV-063",
        "prereg": os.path.relpath(PREREG, NAV),
        "prereg_sha256": pinned,
        "verdict": "PASS" if not failed else "FAIL",
        "n_criteria": len(rows),
        "n_failed": len(failed),
        "criteria": rows,
        "alignment_report": {
            "path": "reports/rcppo_alignment.json",
            "sha256": sha(os.path.join(NAV, "reports", "rcppo_alignment.json")),
            "verdict": alignment["verdict"],
            "deviations": alignment["deviations"],
        },
        "inputs_sha256": {
            p: sha(os.path.join(NAV, p))
            for p in ("tests/test_rcppo_alignment.py",
                      "src/camra_navigation/algorithm/targets.py",
                      "src/camra_navigation/algorithm/storage.py",
                      "src/camra_navigation/algorithm/ppo.py",
                      "config/value_objective_contract_v1.json")
        },
        "not_claimed": prereg["not_claimed"],
        "next_gate": "NAV-064 只能在本门禁 PASS 后开始学习；NAV-065 产出 gate_G4_alignment_v2",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(gate, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"gate": "G3_alignment_v2", "verdict": gate["verdict"],
                      "n_failed": gate["n_failed"],
                      "failed": [r["criterion"] for r in failed],
                      "written": OUT}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
