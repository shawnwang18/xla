#!/bin/bash
set -euo pipefail
task_variant=${1:?variant required}
export PERF_VARIANT="$task_variant"
python3 /opt/campaign/check-imex.py
export PYTHONPATH="/opt/maxtext/src:/opt/campaign/build/jax:/opt/campaign/builds/$task_variant/jaxlibs/jaxlib:/opt/campaign/builds/$task_variant/jaxlibs/jax_cuda13_pjrt:/opt/campaign/builds/$task_variant/jaxlibs/jax_cuda13_plugin"
export JAX_COMPILATION_CACHE_DIR=/opt/results/jax-cache
export JAX_ENABLE_PGLE=true JAX_PGLE_PROFILING_RUNS=3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 CUDA_DEVICE_MAX_CONNECTIONS=16
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc NCCL_GRAPH_MIXING_SUPPORT=1
export XLA_PJRT_GPU_VALIDATE_TOPOLOGY=false
export NCCL_CFT_ENABLE=0
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/aarch64-linux-gnu
export NVTE_FUSED_ATTN=1 DECOUPLE_GCLOUD=TRUE
export XLA_PJRT_GPU_HOST_MEMORY_PREALLOCATE=false XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=180
export NVTE_MAXTEXT_DO_FSDP_AG_IN_BF16=1
export TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_MAX_LOG_LEVEL=10
export TF_CPP_VMODULE=compilation_stats=10,buffer_assignment=1,compile_module_to_llvm_ir=1
export XLA_FLAGS
XLA_FLAGS=$(python3 - <<'PY'
import json,os
data=json.load(open('/opt/campaign/workload.json'))
flags=data['xla_flags']+data['variant_flags'].get(os.environ['PERF_VARIANT'],[])
print(' '.join(flags))
PY
)
if [[ "$SLURM_PROCID" == 0 ]]; then
  XLA_FLAGS+=" --xla_dump_to=/opt/results/hlo --xla_dump_hlo_as_text=true --xla_dump_hlo_as_proto=true"
  python3 - <<'PY'
import json,os,pathlib
pathlib.Path('/opt/results/step.json').write_text(json.dumps({'step_id':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'nodes':os.environ['SLURM_JOB_NODELIST']})+'\n')
PY
fi
task_sampler=
if [[ "$SLURM_LOCALID" == 0 ]]; then
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits -lms 1000 > "/opt/results/gpu-memory-node-${SLURM_NODEID}.csv" 2> "/opt/results/gpu-memory-node-${SLURM_NODEID}.err" &
  task_sampler=$!
fi
stop_sampler() {
  if [[ -n "$task_sampler" ]]; then kill "$task_sampler" 2>/dev/null || true; wait "$task_sampler" 2>/dev/null || true; fi
}
trap stop_sampler EXIT
mapfile -t task_args < <(python3 - <<'PY'
import json
d=json.load(open('/opt/campaign/workload.json'))
for k,v in d['model_arguments'].items():print(f'{k}={v}')
PY
)
cd /opt/maxtext
numactl --cpunodebind=$((SLURM_LOCALID / 2)) --membind=$((SLURM_LOCALID / 2)) \
  python3 /opt/campaign/measure-training.py src/maxtext/configs/base.yml \
  run_name=ds3-ga2 base_output_directory=/opt/results/model-output \
  jax_cache_dir=/opt/results/jax-cache "${task_args[@]}"
