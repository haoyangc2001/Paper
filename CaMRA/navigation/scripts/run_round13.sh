#!/usr/bin/env bash
# NAV-064 round 13: contract-margin variant (T=45 s, k=1.15, E_ref recomputed at 45 s)
# with two arms that differ ONLY in joint_cost_base_weight (A=2.0, B=0.0).
#
# 12 runs = 2 arms x 2 methods x 3 seeds, spread over GPU 3-7 (three per card at most).
# Launched detached (setsid) so a monitoring SSH drop cannot kill training.
set -u
cd /home/caohy/repositories/Paper/CaMRA/navigation || exit 1
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
R=/pub/data/caohy/work/camra_navigation
DIR="$R/nav064_r13"
ABORT="$R/nav064_r13_attempts"

mkdir -p "$ABORT"
shopt -s nullglob
for d in "$R"/nav064_r13_*; do
  b=$(basename "$d")
  [ "$b" = "nav064_r13" ] && continue
  mv "$d" "$ABORT/${b}_$(date +%H%M%S)" 2>/dev/null || true
done

mkdir -p "$DIR"
START="$DIR/round13_start.txt"
{
  echo "round13 launch $(date --iso-8601=seconds) host=$(hostname) load=$(cat /proc/loadavg)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} > "$START"

pids=(); names=()
launch() {
  local run="$1" variant="$2" seed="$3" prereg="$4" gpu="$5"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_formal.py \
    --run-id "$run" --variant "$variant" --seed "$seed" \
    --prereg-name "$prereg" > "$DIR/$run.log" 2>&1 &
  pids+=("$!"); names+=("$run")
  echo "launched $run variant=$variant seed=$seed prereg=$prereg gpu=$gpu at $(date --iso-8601=seconds)" >> "$START"
}

# A 臂：预算内逐步成本 base_weight=2.0
launch nav064_r13_uncalibrated_s0        uncalibrated 0 training_loop_prereg_r13.json 3; sleep 12
launch nav064_r13_uncalibrated_s1        uncalibrated 1 training_loop_prereg_r13.json 4; sleep 12
launch nav064_r13_uncalibrated_s2        uncalibrated 2 training_loop_prereg_r13.json 5; sleep 12
launch nav064_r13_full_s0                full 0 training_loop_prereg_r13.json 6; sleep 12
launch nav064_r13_full_s1                full 1 training_loop_prereg_r13.json 7; sleep 12
launch nav064_r13_full_s2                full 2 training_loop_prereg_r13.json 3; sleep 12
# B 臂：同契约、base_weight=0.0（对照）
launch nav064_r13_uncalibrated_s0_b0     uncalibrated 0 training_loop_prereg_r13_b0.json 4; sleep 12
launch nav064_r13_uncalibrated_s1_b0     uncalibrated 1 training_loop_prereg_r13_b0.json 5; sleep 12
launch nav064_r13_uncalibrated_s2_b0     uncalibrated 2 training_loop_prereg_r13_b0.json 6; sleep 12
launch nav064_r13_full_s0_b0             full 0 training_loop_prereg_r13_b0.json 7; sleep 12
launch nav064_r13_full_s1_b0             full 1 training_loop_prereg_r13_b0.json 3; sleep 12
launch nav064_r13_full_s2_b0             full 2 training_loop_prereg_r13_b0.json 4

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    fail=$((fail + 1)); echo "wait_fail ${names[$i]}" >> "$START"
  fi
done

{ echo "end $(date --iso-8601=seconds) fail=$fail"; } >> "$START"
echo "fail=$fail" > "$DIR/exit_codes.txt"
for run in "$DIR"/nav064_r13_*.log; do
  base=$(basename "$run" .log)
  if grep -q "Traceback\|CUDA out of memory\|RuntimeError" "$run"; then
    echo "1 $base" >> "$DIR/exit_codes.txt"
  else
    echo "0 $base" >> "$DIR/exit_codes.txt"
  fi
done
touch "$DIR/ROUND13_DONE"
echo "ROUND13_DONE fail=$fail"
