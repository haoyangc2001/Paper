#!/bin/bash
# Evaluate NAV-064 round 12 on the development split only.
# Main arm (mask on): 6 runs. Control arm (mask off): 2 runs, evaluated separately.
set -u
NAV=/home/caohy/repositories/Paper/CaMRA/navigation
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
cd "$NAV" || exit 9
export PYTHONPATH="$NAV/src:$NAV/../calibration/src"

LOG=/tmp/nav064_eval_r12.log
echo "eval start $(date -Iseconds) host=$(hostname) cwd=$(pwd)" > "$LOG"

"$PY" scripts/eval_training_round.py --round 12 \
  --prereg training_loop_prereg_r12.json \
  --run-ids nav064_r12_uncalibrated_s0,nav064_r12_uncalibrated_s1,nav064_r12_uncalibrated_s2,nav064_r12_full_s0,nav064_r12_full_s1,nav064_r12_full_s2 \
  --out reports/training_round_r12.json >> "$LOG" 2>&1
rc=$?
echo "EVAL_MAIN_RC=$rc end $(date -Iseconds)" >> "$LOG"

"$PY" scripts/eval_training_round.py --round 12 \
  --prereg training_loop_prereg_r12_nomask.json \
  --run-ids nav064_r12_uncalibrated_s0_nomask,nav064_r12_full_s0_nomask \
  --out reports/training_round_r12_control.json >> "$LOG" 2>&1
rc2=$?
echo "EVAL_CONTROL_RC=$rc2 end $(date -Iseconds)" >> "$LOG"
