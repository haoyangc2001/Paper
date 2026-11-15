#!/usr/bin/env bash
# NAV-064 round 12: dead_step_mask ON (6 main runs) + mask-OFF control (2 runs).
#
# Two-phase launch because tiger11 is shared right now: GPU 0/1/2 only have ~6.9 GiB
# free (another user pinned 17.7 GiB on each), GPU 3-7 have ~20 GiB free but are at
# 95-100% utilisation by someone else. So we first probe the two tightest cards with
# one run each; only if both survive do we fan out the remaining six.
set -u
cd /home/caohy/repositories/Paper/CaMRA/navigation || exit 1
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
R=/pub/data/caohy/work/camra_navigation
DIR="$R/nav064_r12"
mkdir -p "$DIR"
START="$DIR/round12_start.txt"
{
  echo "start $(date --iso-8601=seconds) host=$(hostname) load=$(cat /proc/loadavg)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} > "$START"

pids=()
launch() {
  local run="$1" variant="$2" seed="$3" prereg="$4" gpu="$5"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_formal.py \
    --run-id "$run" --variant "$variant" --seed "$seed" \
    --prereg-name "$prereg" > "$DIR/$run.log" 2>&1 &
  pids+=("$!")
  echo "launched $run variant=$variant seed=$seed prereg=$prereg gpu=$gpu" >> "$START"
}

# ---- phase 1: probe the two tightest GPUs (one run each) ----
launch nav064_r12_full_s0        full 0 training_loop_prereg_r12.json        0
launch nav064_r12_full_s0_nomask full 0 training_loop_prereg_r12_nomask.json 1
sleep 150

alive=$(ps -o pid= -p "$(IFS=,; echo "${pids[*]}")" 2>/dev/null | wc -l)
echo "phase1 alive=$alive at $(date --iso-8601=seconds)" >> "$START"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$START" 2>/dev/null
if [ "$alive" -lt 2 ]; then
  echo "PHASE1_FAIL alive=$alive" >> "$START"
  for f in "$DIR"/nav064_r12_full_s0*.log; do
    echo "--- $f ---" >> "$START"
    tail -25 "$f" >> "$START"
  done
fi

# ---- phase 2: fan out the remaining six only if phase 1 is healthy ----
if [ "$alive" -ge 2 ]; then
  launch nav064_r12_uncalibrated_s0         uncalibrated 0 training_loop_prereg_r12.json 2
  launch nav064_r12_uncalibrated_s1         uncalibrated 1 training_loop_prereg_r12.json 3
  launch nav064_r12_uncalibrated_s2         uncalibrated 2 training_loop_prereg_r12.json 4
  launch nav064_r12_full_s1                 full 1 training_loop_prereg_r12.json 5
  launch nav064_r12_full_s2                 full 2 training_loop_prereg_r12.json 6
  launch nav064_r12_uncalibrated_s0_nomask  uncalibrated 0 training_loop_prereg_r12_nomask.json 7
fi

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=$((fail + 1)); fi
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
