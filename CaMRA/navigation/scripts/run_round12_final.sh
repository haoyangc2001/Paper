#!/usr/bin/env bash
# NAV-064 round 12, final launch: back on tiger11, all eight runs at once (the r11
# pattern that actually finished in 1.39 h).
#
# Why not tiger12: same isaaclab 0.47.2, but its GPU utilisation stayed at 0 % while a
# single pilot burned 97 % of one core for 8 min without logging a single update, and a
# 120-update speed probe (240 env calls) did not finish in 10 min. The identical smoke
# run on tiger11 logged 8 updates in 3.6 s. So tiger12's Isaac stack falls back to CPU
# here; it cannot be used, regardless of it being idle.
#
# tiger11 right now: load fell from 684 to ~220, GPUs 3-7 at 43-67 %, GPUs 0-2 still hold
# 17.7 GiB of someone else's memory with 0 % utilisation -> we stay off 0-2.
set -u
cd /home/caohy/repositories/Paper/CaMRA/navigation || exit 1
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
R=/pub/data/caohy/work/camra_navigation
DIR="$R/nav064_r12"
ABORT="$R/nav064_r12_aborted_rerun"
mkdir -p "$ABORT"

shopt -s nullglob
for d in "$R"/nav064_r12_full_s0 "$R"/nav064_r12_full_s0_nomask "$R"/nav064_r12_uncalibrated_s0 \
         "$R"/nav064_r12_uncalibrated_s0_nomask "$R"/nav064_r12_full_s1 "$R"/nav064_r12_full_s2 \
         "$R"/nav064_r12_uncalibrated_s1 "$R"/nav064_r12_uncalibrated_s2; do
  [ -d "$d" ] && mv "$d" "$ABORT/$(basename "$d")" 2>/dev/null || true
done
if [ -d "$DIR" ]; then mv "$DIR" "$ABORT/logs_$(date +%H%M)" 2>/dev/null || true; fi
mkdir -p "$DIR"

START="$DIR/round12_start.txt"
{
  echo "final launch $(date --iso-8601=seconds) host=$(hostname) load=$(cat /proc/loadavg)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} > "$START"

pids=()
names=()
launch() {
  local run="$1" variant="$2" seed="$3" prereg="$4" gpu="$5"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_formal.py \
    --run-id "$run" --variant "$variant" --seed "$seed" \
    --prereg-name "$prereg" > "$DIR/$run.log" 2>&1 &
  pids+=("$!")
  names+=("$run")
  echo "launched $run variant=$variant seed=$seed prereg=$prereg gpu=$gpu at $(date --iso-8601=seconds)" >> "$START"
}

# 8 runs over 5 GPUs (3-7). Two per card on 3/4/5, one on 6/7.
launch nav064_r12_full_s0                 full 0 training_loop_prereg_r12.json 3; sleep 15
launch nav064_r12_full_s0_nomask          full 0 training_loop_prereg_r12_nomask.json 4; sleep 15
launch nav064_r12_uncalibrated_s0         uncalibrated 0 training_loop_prereg_r12.json 5; sleep 15
launch nav064_r12_uncalibrated_s0_nomask  uncalibrated 0 training_loop_prereg_r12_nomask.json 6; sleep 15
launch nav064_r12_full_s1                 full 1 training_loop_prereg_r12.json 7; sleep 15
launch nav064_r12_full_s2                 full 2 training_loop_prereg_r12.json 3; sleep 15
launch nav064_r12_uncalibrated_s1         uncalibrated 1 training_loop_prereg_r12.json 4; sleep 15
launch nav064_r12_uncalibrated_s2         uncalibrated 2 training_loop_prereg_r12.json 5

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    fail=$((fail + 1)); echo "wait_fail ${names[$i]}" >> "$START"
  fi
done

{ echo "end $(date --iso-8601=seconds) fail=$fail"; } >> "$START"
echo "fail=$fail" > "$DIR/exit_codes.txt"
for run in "$DIR"/nav064_r12_*.log; do
  base=$(basename "$run" .log)
  if grep -q "Traceback\|CUDA out of memory\|RuntimeError" "$run"; then
    echo "1 $base" >> "$DIR/exit_codes.txt"
  else
    echo "0 $base" >> "$DIR/exit_codes.txt"
  fi
done
touch "$DIR/ROUND12_DONE"
echo "ROUND12_DONE fail=$fail"
