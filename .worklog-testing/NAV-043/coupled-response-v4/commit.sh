#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -u .worklog-testing/NAV-043/coupled-response-v4/commit-preparation.py
