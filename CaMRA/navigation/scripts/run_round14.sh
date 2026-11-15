#!/usr/bin/env bash
# Original six r14 runs; no separate performance smoke or extra round.
set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo 'tmux is required'; exit 1; }
[[ "$(hostname -s)" == tiger11 ]] || { echo 'tiger11 is required'; exit 1; }
NAV=/home/caohy/repositories/Paper/CaMRA/navigation
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
TASK_ROOT=/pub/data/caohy/work/camra_navigation
ROUND_DIR="$TASK_ROOT/nav064_r14"
cd "$NAV"
export PYTHONPATH="$NAV/scripts:$NAV/src:$NAV/../calibration/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
"$PY" -c 'from train_formal import verify_prereg; verify_prereg("training_loop_prereg_r14_repair_v1.json")'
for variant in uncalibrated full; do
  for seed in 0 1 2; do
    [[ ! -e "$TASK_ROOT/nav064_r14_${variant}_s${seed}" ]] || { echo 'existing run, refusing duplicate'; exit 1; }
  done
done
mkdir "$ROUND_DIR"
date --iso-8601=seconds > "$ROUND_DIR/start.txt"
hostname -f >> "$ROUND_DIR/start.txt"
df -h "$TASK_ROOT" >> "$ROUND_DIR/start.txt"
pids=()
names=()
for variant in uncalibrated full; do
  for seed in 0 1 2; do
    rid="nav064_r14_${variant}_s${seed}"
    "$PY" -u scripts/train_formal.py --run-id "$rid" --variant "$variant" --seed "$seed" \
      --prereg-name training_loop_prereg_r14_repair_v1.json > "$ROUND_DIR/$rid.log" 2>&1 &
    pids+=("$!")
    names+=("$rid")
    printf '%s %s\n' "$!" "$rid" >> "$ROUND_DIR/pids.txt"
  done
done
fail=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  printf '%s %s\n' "$rc" "${names[$i]}" >> "$ROUND_DIR/exit_codes.txt"
  [[ "$rc" == 0 ]] || fail=$((fail+1))
done
date --iso-8601=seconds > "$ROUND_DIR/TRAINING_FINISHED"
[[ "$fail" == 0 ]] || { echo "training failed in $fail runs"; exit 1; }
"$PY" -u scripts/eval_r14_repair.py --out reports/training_round_r14_repair_v1.json \
  > "$ROUND_DIR/development_eval.log" 2>&1
date --iso-8601=seconds > "$ROUND_DIR/DEVELOPMENT_FINISHED"
