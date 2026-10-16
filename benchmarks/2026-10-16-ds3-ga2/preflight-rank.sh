#!/bin/bash
set -euo pipefail
export PYTHONPATH=/opt/maxtext/src:/opt/campaign/build/jax:/opt/campaign/builds/base/jaxlibs/jaxlib:/opt/campaign/builds/base/jaxlibs/jax_cuda13_pjrt:/opt/campaign/builds/base/jaxlibs/jax_cuda13_plugin
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_PGLE=true JAX_PGLE_PROFILING_RUNS=3
export XLA_PJRT_GPU_VALIDATE_TOPOLOGY=false
export NCCL_CFT_ENABLE=0
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/aarch64-linux-gnu
export CUDA_DEVICE_MAX_CONNECTIONS=16 NCCL_GRAPH_MIXING_SUPPORT=1
export DECOUPLE_GCLOUD=TRUE
export NCCL_DEBUG=INFO
export XLA_FLAGS
XLA_FLAGS=$(python3 -c "import json; print(' '.join(json.load(open('/opt/campaign/workload.json'))['xla_flags']))")
python3 /opt/campaign/check-imex.py
if [[ "$SLURM_PROCID" == 0 ]]; then
  python3 - <<'PY'
import hashlib,json,os,pathlib
root=pathlib.Path('/opt/campaign')
verified=[]
for variant in json.loads((root/'variants.json').read_text()):
    output=root/'builds'/variant['variant']/'jaxlibs'
    manifest=json.loads((output/'build-manifest.json').read_text())
    assert manifest['commit']==variant['commit']
    for name,digest in manifest['libraries'].items():
        path=output/name
        assert path.resolve().is_relative_to(output.resolve()), f'Non-isolated library: {path}'
        assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==digest, f'Changed library: {path}'
    verified.append(dict(variant=variant['variant'],commit=manifest['commit'],library_count=len(manifest['libraries'])))
(root/'preflight/libraries-verified.json').write_text(json.dumps(verified,indent=2)+'\n')
(root/'preflight/step.json').write_text(json.dumps({'step_id':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']})+'\n')
PY
fi
exec python3 /opt/campaign/preflight.py
