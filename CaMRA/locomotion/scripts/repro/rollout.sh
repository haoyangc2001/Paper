#!/bin/bash
# Closed-loop reload of an exported controller, three independent arms.
# In-repo replacement of /pub/data/caohy/scripts/t026_rollout.sh (TASK-034).
#
# Usage:
#   scripts/repro/rollout.sh <BUNDLE> <CHECKPOINT> [OUT_DIR] [NUM_ENVS] [SEED]
#
# The three arms run in separate processes: jit and onnx must not import the
# trainer (asserted inside the script by the no_trainer_dependency check), and the
# trainer arm is the numerical reference.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" || exit 1

BUNDLE="${1:?usage: rollout.sh <BUNDLE> <CHECKPOINT> [OUT_DIR] [NUM_ENVS] [SEED]}"
CKPT="${2:?checkpoint (trainer reference arm) required}"
OUT="${3:-$CAMRA_LOG_ROOT/repro/rollout}"
ENVS="${4:-32}"
SEED="${5:-100}"
LOG="$CAMRA_LOG_ROOT/repro/rollout"

require_dir "$BUNDLE" "exported candidate bundle"
require_file "$CKPT" "checkpoint"
require_dir "$CAMRA_ASSET_ROOT/Go2" "Go2 asset root"
mkdir -p "$OUT" "$LOG"

cd "$LOCOMOTION_ROOT" || die "cannot cd to $LOCOMOTION_ROOT"
stamp_environment "$LOG/rollout.start.txt"
echo "[rollout] bundle=$BUNDLE ckpt=$CKPT out=$OUT envs=$ENVS seed=$SEED"

for B in jit onnx trainer; do
  echo "[rollout] backend=$B start $(date +%H:%M:%S)"
  EXTRA=()
  [ "$B" = "trainer" ] && EXTRA=(--checkpoint "$CKPT")
  "$PY" -u scripts/rollout_exported_policy.py \
    --headless --task CaMRA-Go2-Velocity \
    --bundle "$BUNDLE" --backend "$B" "${EXTRA[@]}" \
    --num_envs "$ENVS" --seed "$SEED" --out_dir "$OUT" \
    > "$LOG/rollout_${B}.log" 2> "$LOG/rollout_${B}.err.log"
  echo "[rollout] backend=$B exit=$? $(date +%H:%M:%S)"
done

missing=0
for B in jit onnx trainer; do
  [ -s "$OUT/rollout_${B}.json" ] || { echo "[rollout] MISSING rollout_${B}.json" >&2; missing=1; }
done
if [ "$missing" -ne 0 ]; then
  echo "[rollout] FAILED: artefact check" >&2
  for B in jit onnx trainer; do echo "--- $B stderr"; tail -15 "$LOG/rollout_${B}.err.log"; done
  exit 2
fi

"$PY" -u scripts/aggregate_rollout_arms.py --out_dir "$OUT"
AGG_RC=$?
echo "[rollout] aggregate exit=$AGG_RC"
echo "[rollout] done $(date -Iseconds)"
exit "$AGG_RC"
