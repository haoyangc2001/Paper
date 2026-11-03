#!/bin/bash
# Final-test entry point (in-repo replacement of /pub/data/caohy/scripts/t024_finaltest.sh).
#
# Usage:
#   scripts/repro/finaltest.sh <CHECKPOINT> [GROUP] [SEEDS] [EPISODES_PER_VARIANT] [NUM_ENVS] [--allow-existing]
#
# GROUP: nominal | perturbed | both (default). SEEDS: space separated (default 200..209).
# EPISODES_PER_VARIANT defaults to 30; the confirmatory rerun uses 300 (supplement D-2).
# Note the unit: it is per *variant*, not per scenario. Until TASK-031 only variant 0 of a
# multi-direction scenario ran, so "30 per scenario" and "30 per variant" were the same
# number; they are not any more, and passing the old number here would silently give
# combined 30/4 = 7 per variant. Hence the explicit name.
#
# The final test is evaluated once per selected checkpoint; this script skips seeds
# that already have a summary.json so an interrupted run can be resumed, but it
# refuses to reuse a non-empty output root unless --allow-existing is given.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" || exit 1

ALLOW=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--allow-existing" ]; then ALLOW=1; else ARGS+=("$a"); fi
done

CKPT="${ARGS[0]:?usage: finaltest.sh <CHECKPOINT> [GROUP] [SEEDS] [EPISODES_PER_VARIANT] [NUM_ENVS] [--allow-existing]}"
GROUP="${ARGS[1]:-both}"
SEEDS="${ARGS[2]:-200 201 202 203 204 205 206 207 208 209}"
EPSV="${ARGS[3]:-30}"
ENVS="${ARGS[4]:-64}"

# accept "200-209" as well as an explicit list, so the documented command is typeable
if [[ "$SEEDS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
  SEEDS=$(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
fi

# FINALTEST_ROOT pins the output root exactly (the confirmatory rerun registers
# <CAMRA_LOG_ROOT>/TASK-036/<group>/s<seed>, which must not gain an extra level).
OUTROOT="${FINALTEST_ROOT:-$CAMRA_LOG_ROOT/repro/finaltest}"
require_file "$CKPT" "checkpoint"
require_dir "$CAMRA_ASSET_ROOT/Go2" "Go2 asset root"
require_new_dir "$OUTROOT" "$ALLOW"

case "$GROUP" in
  nominal) GROUP_LIST="nominal" ;;
  perturbed) GROUP_LIST="perturbed" ;;
  both) GROUP_LIST="nominal perturbed" ;;
  *) die "GROUP must be nominal, perturbed or both (got '$GROUP')" ;;
esac

stamp_environment "$OUTROOT/run.start.txt"
sha256sum "$CKPT" | tee "$OUTROOT/checkpoint.sha256"
echo "[finaltest] ckpt=$CKPT groups='$GROUP_LIST' seeds='$SEEDS' episodes_per_variant=$EPSV envs=$ENVS"

cd "$LOCOMOTION_ROOT" || die "cannot cd to $LOCOMOTION_ROOT"

FAILED=0
for G in $GROUP_LIST; do
  for S in $SEEDS; do
    OUT="$OUTROOT/$G/s$S"
    if [ -s "$OUT/summary.json" ]; then
      echo "[finaltest] skip $G seed $S (already done at $OUT)"
      continue
    fi
    echo "[finaltest] $G seed $S start $(date +%H:%M:%S)"
    "$PY" -u scripts/eval_locomotion.py \
      --headless \
      --task CaMRA-Go2-Velocity \
      --checkpoint "$CKPT" \
      --num_envs "$ENVS" \
      --scenarios all \
      --episodes_per_variant "$EPSV" \
      --group "$G" \
      --seed "$S" \
      --replay_snippets 0 \
      --out_dir "$OUT" \
      > "$OUTROOT/${G}_s${S}.log" 2> "$OUTROOT/${G}_s${S}.err.log"
    RC=$?
    # the Isaac entry point can exit 0 after a traceback: decide on the artefact
    if [ "$RC" -ne 0 ] || [ ! -s "$OUT/summary.json" ]; then
      echo "[finaltest] FAILED $G seed $S rc=$RC summary=$( [ -s "$OUT/summary.json" ] && echo present || echo missing )" >&2
      tail -20 "$OUTROOT/${G}_s${S}.err.log" >&2
      FAILED=$((FAILED + 1))
    else
      echo "[finaltest] $G seed $S ok $(date +%H:%M:%S)"
    fi
  done
done

echo "[finaltest] done $(date -Iseconds) failures=$FAILED"
[ "$FAILED" -ne 0 ] && exit 3
exit 0
