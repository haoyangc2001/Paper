#!/usr/bin/env bash
# NAV-064 round 12, attempt 3.
#
# What went wrong before:
#   attempt 1 (8 parallel): never logged a single update in 9 min -- eight Kit
#     instances raced to load the same multi-GB assets (all in folio_wait_bit_common,
#     rchar only ~0.4 MB/s). Not deadlocked, just page-cache thrash on a box at load 110.
#   attempt 2 (serial): the trainer creates its run dir at $R/$run_id, i.e. a SIBLING of
#     $R/nav064_r12, not a child. Moving only $R/nav064_r12 left the eight
#     nav064_r12_* run dirs in place, so the pilot died on "already exists and is
#     non-empty; retry_rule requires a new run id".
#
# This attempt parks every leftover nav064_r12_* run dir (and the old log dir) under
# nav064_r12_aborted/ -- nothing is deleted, the evidence stays -- then relaunches with
# the pre-registered run ids, one process at a time.
set -u
cd /home/caohy/repositories/Paper/CaMRA/navigation || exit 1
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
R=/pub/data/caohy/work/camra_navigation
ABORT="$R/nav064_r12_aborted"
mkdir -p "$ABORT"

shopt -s nullglob
for d in "$R"/nav064_r12_*; do
  b=$(basename "$d")
  case "$b" in
    nav064_r12_aborted|nav064_r12_aborted_*) continue ;;
  esac
  mv "$d" "$ABORT/$b" 2>/dev/null || true
done
if [ -d "$R/nav064_r12" ]; then
  mv "$R/nav064_r12" "$ABORT/logs_dir_$(date +%H%M)" 2>/dev/null || true
fi

DIR="$R/nav064_r12"
mkdir -p "$DIR"
START="$DIR/round12_start.txt"
{
  echo "attempt3 serial start $(date --iso-8601=seconds) host=$(hostname) load=$(cat /proc/loadavg)"
  echo "parked leftovers -> $ABORT"
  ls "$ABORT" 2>/dev/null
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

# ---- pilot on GPU 3 (20 GiB free; GPU 0-2 only have ~6.9 GiB) ----
launch nav064_r12_full_s0 full 0 training_loop_prereg_r12.json 3
pilot="$DIR/nav064_r12_full_s0.log"
ok=0
for i in $(seq 1 30); do
  sleep 30
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
  echo "PILOT_FAIL no update; aborting before fan-out" >> "$START"
  pkill -f "train_formal.py --run-id nav064_r12_" || true
  touch "$DIR/ROUND12_DONE"
  echo "ROUND12_ABORT"
  exit 3
fi

echo "pilot healthy; fanning out remaining 7 at $(date --iso-8601=seconds)" >> "$START"

# ---- fan out, 45 s apart: assets stay hot, no two processes race on cold IO ----
launch nav064_r12_full_s0_nomask         full 0 training_loop_prereg_r12_nomask.json 4; sleep 45
launch nav064_r12_uncalibrated_s0        uncalibrated 0 training_loop_prereg_r12.json 5; sleep 45
launch nav064_r12_uncalibrated_s0_nomask uncalibrated 0 training_loop_prereg_r12_nomask.json 6; sleep 45
launch nav064_r12_full_s1                full 1 training_loop_prereg_r12.json 7; sleep 45
launch nav064_r12_full_s2                full 2 training_loop_prereg_r12.json 0; sleep 45
launch nav064_r12_uncalibrated_s1        uncalibrated 1 training_loop_prereg_r12.json 1; sleep 45
launch nav064_r12_uncalibrated_s2        uncalibrated 2 training_loop_prereg_r12.json 2

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
