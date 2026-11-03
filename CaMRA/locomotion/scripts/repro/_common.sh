#!/bin/bash
# Shared bootstrap for scripts/repro/*.sh (TASK-034).
#
# Replaces the copy-pasted preamble of the temporary scripts that used to live in
# /pub/data/caohy/scripts: they hard-coded the interpreter, the repository, the
# log root and the asset root, so nothing could run from another clone.
#
# Resolution: exported variable > config/repro.env > built-in default (delivery
# machine). Repo paths are derived from this file's location, so the scripts work
# from any working directory.
#
# This file is meant to be sourced:  source "$(dirname "$0")/_common.sh"

_repro_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# CaMRA/locomotion and the repository root, independent of $PWD and of symlinks
LOCOMOTION_ROOT=$(cd "$_repro_dir/../.." && pwd)
REPO_ROOT=$(cd "$LOCOMOTION_ROOT/../.." && pwd)
export LOCOMOTION_ROOT REPO_ROOT

_repro_cfg="$LOCOMOTION_ROOT/config/repro.env"
if [ ! -f "$_repro_cfg" ]; then
  echo "[repro] FATAL: missing $_repro_cfg" >&2
  echo "[repro] it ships with the repository; restore it from git or recreate it" >&2
  return 1 2>/dev/null || exit 1
fi

# only fill variables that are not already set, so exports keep winning
while IFS='=' read -r _k _v; do
  case "$_k" in ''|\#*) continue ;; esac
  if [ -z "${!_k:-}" ]; then
    export "$_k=$_v"
  fi
done < "$_repro_cfg"

# ---- Isaac Sim / simulator environment (same values as the historical runs) ----
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$CAMRA_XDG_DATA}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CAMRA_XDG_CACHE}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TMPDIR="${TMPDIR:-$CAMRA_TMPDIR}"
mkdir -p "$TMPDIR" || { echo "[repro] cannot create TMPDIR=$TMPDIR" >&2; return 1 2>/dev/null || exit 1; }

# GPU selection: respect an explicit GPU, otherwise default to device 0
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CAMRA_CUDA_VISIBLE_DEVICES:-0}"
fi

PY="${PY:-$CAMRA_PYTHON}"

# ---- helpers -------------------------------------------------------------------
die() {
  echo "[repro] FATAL: $*" >&2
  exit 2
}

require_file() {
  [ -s "$1" ] || die "$2 not found or empty: $1"
}

require_dir() {
  [ -d "$1" ] || die "$2 not found: $1"
}

# refuse to write into a directory that already holds results, unless allowed
require_new_dir() {
  local dir="$1" allow="${2:-0}"
  if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ] && [ "$allow" != "1" ]; then
    die "$dir already exists and is not empty; pass --allow-existing to append, or pick a new output root"
  fi
  mkdir -p "$dir"
}

# provenance block written next to every run; also proves which host ran what
stamp_environment() {
  local out="$1"
  {
    echo "=== $(basename "$0") ==="
    echo "started_at: $(date -Iseconds)"
    # the window this command actually runs in, not the session's active window
    echo "tmux: $(tmux display-message -p -t "${TMUX_PANE:-}" '#S:#W' 2>/dev/null || echo none)"
    echo "host: $(hostname -s) ($(hostname))"
    echo "user: $(id -un)"
    echo "pwd: $(pwd)"
    echo "repo_root: $REPO_ROOT"
    echo "locomotion_root: $LOCOMOTION_ROOT"
    echo "python: $PY ($("$PY" -V 2>&1))"
    echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty:"
    git -C "$REPO_ROOT" status --porcelain 2>/dev/null | sed 's/^/  /'
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "config:"
    for k in CAMRA_PYTHON CAMRA_LOG_ROOT CAMRA_ASSET_ROOT CAMRA_BUNDLE_ROOT CAMRA_DEVICE; do
      echo "  $k=${!k:-unset}"
    done
    echo "gpu: $(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null | head -1 || echo none)"
  } > "$out"
}
