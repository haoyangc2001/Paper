#!/usr/bin/env bash
# NAV-064 round 13 evaluation: the 12 runs under the margin contract (T=45 s, k=1.15,
# E_ref recomputed at 45 s). Uses contract_margin_eval.py so the evaluation contract
# is byte-for-byte the one the prereg froze for training — the r12 lesson (eval that
# silently used a different budget) is not repeated.
set -u
NAV=/home/caohy/repositories/Paper/CaMRA/navigation
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
cd "$NAV" || exit 9
export PYTHONPATH="$NAV/src:$NAV/../calibration/src"

cat > /tmp/runs_r13.json <<'JSON'
{
 "nav064_r13_uncalibrated_s0": "training_loop_prereg_r13.json",
 "nav064_r13_uncalibrated_s1": "training_loop_prereg_r13.json",
 "nav064_r13_uncalibrated_s2": "training_loop_prereg_r13.json",
 "nav064_r13_full_s0": "training_loop_prereg_r13.json",
 "nav064_r13_full_s1": "training_loop_prereg_r13.json",
 "nav064_r13_full_s2": "training_loop_prereg_r13.json",
 "nav064_r13_uncalibrated_s0_b0": "training_loop_prereg_r13_b0.json",
 "nav064_r13_uncalibrated_s1_b0": "training_loop_prereg_r13_b0.json",
 "nav064_r13_uncalibrated_s2_b0": "training_loop_prereg_r13_b0.json",
 "nav064_r13_full_s0_b0": "training_loop_prereg_r13_b0.json",
 "nav064_r13_full_s1_b0": "training_loop_prereg_r13_b0.json",
 "nav064_r13_full_s2_b0": "training_loop_prereg_r13_b0.json"
}
JSON

LOG=/tmp/nav064_eval_r13.log
echo "eval r13 start $(date -Iseconds) host=$(hostname)" > "$LOG"
"$PY" scripts/contract_margin_eval.py --t-max 45 --k 1.15 \
  --runs /tmp/runs_r13.json --out reports/training_round_r13.json >> "$LOG" 2>&1
rc=$?
echo "EVAL_R13_RC=$rc end $(date -Iseconds)" >> "$LOG"
