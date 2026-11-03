#!/bin/bash
# Development-set sweep over the checkpoints of one training run.
# In-repo replacement of /pub/data/caohy/scripts/t023_devsweep_r4.sh.
#
# Usage:
#   scripts/repro/devsweep.sh <RUN_DIR> [DEV_SEED] [EPISODES] [NUM_ENVS] [SUFFIX] [ITERS...]
#
# SUFFIX is appended to every output directory (the delivered run used "_r4" so
# that tuning-round-0 products in the same log root were never overwritten).
# EPISODES is per variant. The historical sweep asked for 30 per *scenario* while only
# variant 0 of multi-direction scenarios was executed, so the two are the same number
# for single-variant scenarios and stricter for lateral/combined now.
# Output: $CAMRA_LOG_ROOT/repro/devsweep/dev_it<iter><suffix>/
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" || exit 1

RUN_DIR="${1:?usage: devsweep.sh <RUN_DIR> [DEV_SEED] [EPISODES] [NUM_ENVS] [SUFFIX] [ITERS...]}"
DEV_SEED="${2:-100}"
EPS="${3:-30}"
ENVS="${4:-64}"
SUFFIX="${5:-}"
shift 5 2>/dev/null || shift $#
ITERS="${*:-500 1000 1500 2000 2500 2999}"

OUT_ROOT="$CAMRA_LOG_ROOT/repro/devsweep"
require_dir "$RUN_DIR" "run directory"
require_dir "$CAMRA_ASSET_ROOT/Go2" "Go2 asset root"
mkdir -p "$OUT_ROOT"

stamp_environment "$OUT_ROOT/devsweep${SUFFIX}.start.txt"
echo "[devsweep] run_dir=$RUN_DIR dev_seed=$DEV_SEED episodes=$EPS envs=$ENVS iters=$ITERS"

cd "$LOCOMOTION_ROOT" || die "cannot cd to $LOCOMOTION_ROOT"

missing=0
for IT in $ITERS; do
  CKPT="$RUN_DIR/model_${IT}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[devsweep] skip iteration $IT (no $CKPT)"
    missing=$((missing + 1))
    continue
  fi
  OUT="$OUT_ROOT/dev_it${IT}${SUFFIX}"
  if [ -s "$OUT/summary.json" ]; then
    echo "[devsweep] skip iteration $IT (already evaluated at $OUT)"
    continue
  fi
  echo "[devsweep] iteration $IT -> $OUT  $(date +%H:%M:%S)"
  "$PY" -u scripts/eval_locomotion.py \
    --headless \
    --task CaMRA-Go2-Velocity \
    --checkpoint "$CKPT" \
    --num_envs "$ENVS" \
    --scenarios all \
    --episodes_per_variant "$EPS" \
    --group nominal \
    --seed "$DEV_SEED" \
    --replay_snippets 0 \
    --out_dir "$OUT" \
    > "$OUT.log" 2> "$OUT.err.log"
  RC=$?
  echo "[devsweep] iteration $IT exit=$RC  $(date +%H:%M:%S)"
  if [ "$RC" -ne 0 ] || [ ! -s "$OUT/summary.json" ]; then
    # the Isaac entry point can exit 0 without artefacts, so check both
    echo "[devsweep] FAILED at iteration $IT (rc=$RC, summary=$( [ -s "$OUT/summary.json" ] && echo present || echo missing ))" >&2
    tail -20 "$OUT.err.log" >&2
    exit 3
  fi
done

if [ "$missing" -gt 0 ]; then
  echo "[devsweep] note: $missing requested iteration(s) had no checkpoint"
fi
echo "[devsweep] done $(date -Iseconds)"
