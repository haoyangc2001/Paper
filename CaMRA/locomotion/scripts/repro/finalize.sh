#!/bin/bash
# Dev sweep + checkpoint selection for one training run.
# In-repo replacement of /pub/data/caohy/scripts/t023_finalize_r4.sh.
#
# Usage:
#   scripts/repro/finalize.sh <RUN_DIR> [SUFFIX]
#
# Produces the sweep products and $CAMRA_LOG_ROOT/repro/devsweep/selection<suffix>.json
# The selection rule lives in scripts/repro/select_checkpoint.py and is printed in full.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" || exit 1

RUN_DIR="${1:?usage: finalize.sh <RUN_DIR> [SUFFIX]}"
SUFFIX="${2:-}"

DEV_SEED="${DEV_SEED:-100}"
EPS="${DEV_EPISODES:-30}"
ENVS="${DEV_ENVS:-64}"
ITERS="${DEV_ITERS:-500 1000 1500 2000 2500 2999}"

echo "[finalize] start $(date -Iseconds) run_dir=$RUN_DIR suffix='${SUFFIX}'"

bash "$LOCOMOTION_ROOT/scripts/repro/devsweep.sh" "$RUN_DIR" "$DEV_SEED" "$EPS" "$ENVS" "$SUFFIX" $ITERS
SWEEP_RC=$?
echo "SWEEP_EXIT=$SWEEP_RC"

"$PY" "$LOCOMOTION_ROOT/scripts/repro/select_checkpoint.py" \
  --dev-root "$CAMRA_LOG_ROOT/repro/devsweep" \
  --suffix "$SUFFIX"
SEL_RC=$?
echo "SELECT_EXIT=$SEL_RC"

echo "[finalize] done $(date -Iseconds)"
# propagate the first failure instead of returning a sum that could cancel out
[ "$SWEEP_RC" -ne 0 ] && exit "$SWEEP_RC"
exit "$SEL_RC"
