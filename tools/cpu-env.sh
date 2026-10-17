#!/usr/bin/env bash
# Source before starting any Python, numerical command or local delegation.
set -Eeuo pipefail

aquacrop_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for aquacrop_setting in 'CUDA_VISIBLE_DEVICES=' 'NVIDIA_VISIBLE_DEVICES=none' \
    'JAX_PLATFORMS=cpu' 'JAX_PLATFORM_NAME=cpu' \
    'XLA_PYTHON_CLIENT_PREALLOCATE=false'; do
    aquacrop_name="${aquacrop_setting%%=*}"
    aquacrop_value="${aquacrop_setting#*=}"
    if [[ -v "$aquacrop_name" && "${!aquacrop_name}" != "$aquacrop_value" ]]; then
        printf 'CPU-only: configurazione incompatibile: %s (valore non modificato)\n' "$aquacrop_name" >&2
        exit 78
    fi
    export "$aquacrop_name=$aquacrop_value"
done

export AQUACROP_CPU_THREADS="${AQUACROP_CPU_THREADS:-1}"
if [[ ! "$AQUACROP_CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'AQUACROP_CPU_THREADS deve essere un intero positivo.' >&2
    exit 64
fi
export OMP_NUM_THREADS="$AQUACROP_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$AQUACROP_CPU_THREADS"
export MKL_NUM_THREADS="$AQUACROP_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$AQUACROP_CPU_THREADS"
export TF_NUM_INTRAOP_THREADS="$AQUACROP_CPU_THREADS"
export TF_NUM_INTEROP_THREADS=1
export JAX_ENABLE_X64=true
export PYTHONPATH="$aquacrop_root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${AQUACROP_PYTHON:-}" ]]; then
    if [[ -x "$aquacrop_root/.venv/bin/python" ]]; then
        export AQUACROP_PYTHON="$aquacrop_root/.venv/bin/python"
    else
        export AQUACROP_PYTHON="$(command -v python3)"
    fi
fi
