#!/usr/bin/env bash
# NAV-064 round 12, attempt 5: tiger12, L40 cards only.
#
# Attempt 4 put the pilot on GPU 3 (RTX PRO 6000 Blackwell). It burned 6.5 min of a
# single core at 97% CPU with 0 MiB of GPU memory allocated and never logged an update:
# the Isaac 4.7.2 stack does not appear to drive that new architecture here, so it fell
# back to CPU. The identical smoke run on GPU 0 (L40) finished in seconds. So every run
# now goes on the four L40 cards (0,1,2,5); the two Blackwell cards are left alone.
set -u
cd /home/caohy/repositories/Paper/CaMRA/navigation || exit 1
PY=/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
R=/pub/data/caohy/work/camra_navigation
DIR="$R/nav064_r12"
ABORT="$R/nav064_r12_blackwell_abort"
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
  echo "attempt5 L40-only start $(date --iso-8601=seconds) host=$(hostname) load=$(cat /proc/loadavg)"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
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
count_updates() { grep -c '"kind": "update"' "$1" 2>/dev/null; }

# ---- pilot on L40 GPU 0 ----
launch nav064_r12_full_s0 full 0 training_loop_prereg_r12.json 0
pilot="$DIR/nav064_r12_full_s0.log"
ok=0
for i in $(seq 1 24); do          # 8 minutes max
  sleep 20
  n=$(count_updates "$pilot")
  echo "pilot poll $i updates=$n at $(date --iso-8601=seconds)" >> "$START"
  if [ "${n:-0}" -ge 1 ]; then ok=1; break; fi
  if ! ps -p "${pids[0]}" > /dev/null 2>&1; then
    echo "PILOT_DIED" >> "$START"; tail -30 "$pilot" >> "$START"; break
  fi
done

if [ "$ok" -ne 1 ]; then
  echo "PILOT_FAIL" >> "$START"
  pkill -f "train_formal.py --run-id nav064_r12_" || true
  touch "$DIR/ROUND12_DONE"
  echo "ROUND12_ABORT"
  exit 3
fi

echo "pilot healthy; fanning out 7 at $(date --iso-8601=seconds)" >> "$START"

# ---- 4 L40 cards, 2 runs each ----
launch nav064_r12_full_s1                full 1 training_loop_prereg_r12.json 0; sleep 20
launch nav064_r12_full_s2                full 2 training_loop_prereg_r12.json 1; sleep 20
launch nav064_r12_uncalibrated_s0        uncalibrated 0 training_loop_prereg_r12.json 1; sleep 20
launch nav064_r12_uncalibrated_s1        uncalibrated 1 training_loop_prereg_r12.json 2; sleep 20
launch nav064_r12_uncalibrated_s0_nomask uncalibrated 0 training_loop_prereg_r12_nomask.json 2; sleep 20
launch nav064_r12_uncalibrated_s2        uncalibrated 2 training_loop_prereg_r12.json 5; sleep 20
launch nav064_r12_full_s0_nomask         full 0 training_loop_prereg_r12_nomask.json 5

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
