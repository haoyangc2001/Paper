#!/usr/bin/env bash
# NAV-064 round 12, on tiger12.
#
# tiger11 became unusable mid-attempt: a colleague's 5-process GNN job (45-49 cores each)
# drove load from 110 to 684 while 3 GPUs were already pinned at 17.7 GiB by another user
# and 5 more ran at 95-99%. Our pilot never logged its first update in 5 minutes.
# tiger12 has the same repo (HEAD 694e900, shared home), a working env_isaaclab, six GPUs
# at 0% utilisation and load 26 -> verified with a 3-update smoke (prereg sha matches,
# both switches printed).
#
# fresh_init means no round-11 checkpoint is needed, so an independent work root is fine.
set -u
cd /home/caohy/repositories/Paper/CaMRA/navigation || exit 1
PY=/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
R=/pub/data/caohy/work/camra_navigation
DIR="$R/nav064_r12"
mkdir -p "$DIR"
START="$DIR/round12_start.txt"
{
  echo "tiger12 start $(date --iso-8601=seconds) host=$(hostname) load=$(cat /proc/loadavg)"
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

count_updates() { grep -c '"kind": "update"' "$1" 2>/dev/null; }

# ---- pilot ----
launch nav064_r12_full_s0 full 0 training_loop_prereg_r12.json 3
pilot="$DIR/nav064_r12_full_s0.log"
ok=0
for i in $(seq 1 30); do
  sleep 20
  n=$(count_updates "$pilot")
  echo "pilot poll $i updates=$n at $(date --iso-8601=seconds)" >> "$START"
  if [ "${n:-0}" -ge 1 ]; then ok=1; break; fi
  if ! ps -p "${pids[0]}" > /dev/null 2>&1; then
    echo "PILOT_DIED" >> "$START"
    tail -30 "$pilot" >> "$START"
    break
  fi
done

if [ "$ok" -ne 1 ]; then
  echo "PILOT_FAIL no update; aborting" >> "$START"
  pkill -f "train_formal.py --run-id nav064_r12_" || true
  touch "$DIR/ROUND12_DONE"
  echo "ROUND12_ABORT"
  exit 3
fi

echo "pilot healthy; fanning out 7 at $(date --iso-8601=seconds)" >> "$START"

# ---- fan out: 6 GPUs, 8 runs. GPU3 has 98 GiB free so it takes three. ----
launch nav064_r12_full_s0_nomask         full 0 training_loop_prereg_r12_nomask.json 3; sleep 20
launch nav064_r12_uncalibrated_s0        uncalibrated 0 training_loop_prereg_r12.json 3; sleep 20
launch nav064_r12_uncalibrated_s0_nomask uncalibrated 0 training_loop_prereg_r12_nomask.json 0; sleep 20
launch nav064_r12_full_s1                full 1 training_loop_prereg_r12.json 0; sleep 20
launch nav064_r12_full_s2                full 2 training_loop_prereg_r12.json 1; sleep 20
launch nav064_r12_uncalibrated_s1        uncalibrated 1 training_loop_prereg_r12.json 2; sleep 20
launch nav064_r12_uncalibrated_s2        uncalibrated 2 training_loop_prereg_r12.json 5

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    fail=$((fail + 1))
    echo "wait_fail ${names[$i]}" >> "$START"
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
