#!/usr/bin/env bash
set -euo pipefail

# Source: https://github.com/google-research/timesfm/tree/v2.0.2
# Reconstructs TimesFM 2.5 200M: one GPU, batch 1, context 512, horizon 512.
# Random weights preserve the forecast computation without a checkpoint download.
# Usage: run.sh [output_dir] [iterations=16] [warmup=10]
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=${1:-$(mktemp -d /tmp/timesfm-hlo.XXXXXX)}
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/timesfm-uv-cache}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$output_dir/dumps --xla_dump_hlo_pass_re=spmd-partitioner --xla_dump_hlo_module_re=jit__fun"

uv run --no-project --with 'jax[cuda13]==0.11.2' --with 'timesfm==2.0.2' \
  --with 'flax==0.12.10' --with einshape --with jaxtyping --with orbax-checkpoint \
  python3 "$script_dir/run.py" "$output_dir" "${2:-16}" "${3:-10}"
