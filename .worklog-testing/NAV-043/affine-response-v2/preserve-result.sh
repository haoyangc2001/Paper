#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B .worklog-testing/NAV-043/affine-response-v2/preserve-result.py > .worklog-testing/NAV-043/affine-response-v2/preserve-result.log 2>&1
printf '0\n' > .worklog-testing/NAV-043/affine-response-v2/preserve-result.exit
