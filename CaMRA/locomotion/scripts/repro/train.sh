#!/bin/bash
# Training entry point (in-repo replacement of /pub/data/caohy/scripts/t023_train_r4.sh).
#
# Usage:
#   scripts/repro/train.sh [SEED] [ITERATIONS] [RUN_NAME] [--allow-existing]
#
# Defaults: seed 0, 3000 iterations, run name formal_s<seed>_<iters>it.
# Logs and the provenance stamp go to $CAMRA_LOG_ROOT/repro/train/.
# The rsl_rl run directory is created under $CAMRA_RSL_LOG_ROOT by train.py itself.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" || exit 1

ALLOW=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--allow-existing" ]; then ALLOW=1; else ARGS+=("$a"); fi
done

SEED="${ARGS[0]:-0}"
ITERS="${ARGS[1]:-3000}"
RUN_NAME="${ARGS[2]:-formal_s${SEED}_${ITERS}it}"
OUT="$CAMRA_LOG_ROOT/repro/train"

require_file "$LOCOMOTION_ROOT/scripts/rsl_rl/train.py" "training entry point"
require_dir "$CAMRA_ASSET_ROOT/Go2" "Go2 asset root"
require_new_dir "$OUT" "$ALLOW"

stamp_environment "$OUT/${RUN_NAME}.start.txt"
mkdir -p "$CAMRA_RUN_ROOT"
ln -sfn "$CAMRA_RSL_LOG_ROOT" "$CAMRA_RUN_ROOT/logs"

echo "[train] seed=$SEED iters=$ITERS run_name=$RUN_NAME"
echo "[train] log -> $OUT/${RUN_NAME}.log"

cd "$CAMRA_RUN_ROOT" || die "cannot cd to $CAMRA_RUN_ROOT"

"$PY" -u "$LOCOMOTION_ROOT/scripts/rsl_rl/train.py" \
  --task CaMRA-Go2-Velocity \
  --headless \
  --seed "$SEED" \
  --max_iterations "$ITERS" \
  --run_name "$RUN_NAME" \
  --device "$CAMRA_DEVICE" \
  > "$OUT/${RUN_NAME}.log" 2> "$OUT/${RUN_NAME}.err.log"
RC=$?

echo "TRAIN_EXIT=$RC" >> "$OUT/${RUN_NAME}.log"
echo "finished_at: $(date -Iseconds)" >> "$OUT/${RUN_NAME}.log"

if [ "$RC" -ne 0 ]; then
  echo "[train] FAILED rc=$RC; last stderr:" >&2
  tail -25 "$OUT/${RUN_NAME}.err.log" >&2
fi
exit "$RC"
