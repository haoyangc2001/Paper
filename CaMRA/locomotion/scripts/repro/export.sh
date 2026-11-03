#!/bin/bash
# Export entry point (in-repo replacement of /pub/data/caohy/scripts/t025_export.sh).
#
# Usage:
#   scripts/repro/export.sh <CHECKPOINT> [OUT_DIR] [TAG] [--allow-existing]
#
# OUT_DIR default: $CAMRA_BUNDLE_ROOT/candidates/<run-name>_<iteration>
# Success is decided by the exported artefacts, not by the process status: the
# Isaac entry point has been observed to exit 0 after a traceback.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" || exit 1

ALLOW=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--allow-existing" ]; then ALLOW=1; else ARGS+=("$a"); fi
done

CKPT="${ARGS[0]:?usage: export.sh <CHECKPOINT> [OUT_DIR] [TAG] [--allow-existing]}"
RUN_NAME=$(basename "$(dirname "$CKPT")")
ITER=$(basename "$CKPT" .pt)
OUT="${ARGS[1]:-$CAMRA_BUNDLE_ROOT/candidates/${RUN_NAME}_${ITER}}"
TAG="${ARGS[2]:-export_$(date +%Y%m%d)}"
LOG="$CAMRA_LOG_ROOT/repro/export"

require_file "$CKPT" "checkpoint"
require_dir "$CAMRA_ASSET_ROOT/Go2" "Go2 asset root"
mkdir -p "$LOG"
require_new_dir "$OUT" "$ALLOW"

cd "$LOCOMOTION_ROOT" || die "cannot cd to $LOCOMOTION_ROOT"
stamp_environment "$LOG/export.start.txt"
echo "[export] ckpt=$CKPT"
echo "[export] out=$OUT tag=$TAG"

"$PY" -u scripts/export_policy_candidate.py \
  --headless \
  --task CaMRA-Go2-Velocity \
  --checkpoint "$CKPT" \
  --out_dir "$OUT" \
  --num_envs "${EXPORT_ENVS:-64}" \
  --seed "${EXPORT_SEED:-100}" \
  --steps "${EXPORT_STEPS:-300}" \
  --device "$CAMRA_DEVICE" \
  --tag "$TAG" \
  > "$LOG/export_${ITER}.log" 2> "$LOG/export_${ITER}.err.log"
PY_RC=$?
echo "[export] python exit=$PY_RC"

missing=0
for f in exported/policy.pt exported/policy.onnx params/deploy.yaml \
         params/export_manifest.json samples/obs_samples.npz; do
  [ -s "$OUT/$f" ] || { echo "[export] MISSING $f" >&2; missing=1; }
done
if [ "$missing" -ne 0 ]; then
  echo "[export] FAILED: artefact check" >&2
  tail -25 "$LOG/export_${ITER}.err.log" >&2
  exit 2
fi

echo "[export] artefact check OK"
echo "[export] next: scripts/repro/rollout.sh $OUT $CKPT"
echo "[export] done $(date -Iseconds)"
