#!/bin/bash
set -euo pipefail
task_root=/home/scratch.shawnw_inf/workload-runs/ds3-ga2-pr-comparison-20261016
task_variant=${1:?variant required}
task_output="$task_root/builds/$task_variant"
mkdir -p "$task_output"
exec > >(tee -a "$task_output/build.log") 2>&1
trap 'rc=$?; python3 - "$task_output" "$rc" <<'"'"'PY'"'"'
import datetime,json,sys
from pathlib import Path
Path(sys.argv[1],"status.json").write_text(json.dumps({"state":"COMPLETED" if int(sys.argv[2])==0 else "FAILED","returncode":int(sys.argv[2]),"finished_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+"\n")
PY
' EXIT
test "$(uname -m)" = aarch64
test -x /usr/local/bin/build-jax.sh
export TMPDIR="$task_root/build/tmp" TEST_TMPDIR="$task_root/build/test_tmp"
export PIP_CACHE_DIR="$task_root/build/pip_cache" XDG_CACHE_HOME="$task_root/build/xdg_cache"
export PYTHONUSERBASE="$task_output/python-user"
mkdir -p "$TMPDIR" "$TEST_TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$PYTHONUSERBASE"
if [ ! -d "$task_root/build/jax" ]; then cp -a /opt/jax "$task_root/build/jax"; fi
cp /usr/local/bin/build-jax.sh "$task_root/build/build-jax.sh.snapshot"
python3 - "$task_root" "$task_variant" <<'PY'
import hashlib,json,os,pathlib,subprocess,sys
r=pathlib.Path(sys.argv[1]);v=sys.argv[2]
record=next(x for x in json.loads((r/'variants.json').read_text()) if x['variant']==v)
assert hashlib.file_digest((r/(v+'.tar')).open('rb'),'sha256').hexdigest()==record['archive_sha256']
revs={}
for name in ['jax','xla','maxtext','transformer-engine']:
 p=subprocess.run(['git','-C','/opt/'+name,'rev-parse','HEAD'],capture_output=True,text=True)
 revs[name]=p.stdout.strip() if p.returncode==0 else None
(r/'builds'/v/'container-sources.json').write_text(json.dumps(revs,indent=2)+'\n')
(r/'builds'/v/'step.json').write_text(json.dumps({'step_id':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':os.uname().nodename,'variant':v})+'\n')
print('SOURCE_VERIFIED',record['commit'],revs,flush=True)
PY
if [[ "${2:-}" != --finalize-only ]]; then
cp "$task_output/step.json" "$task_output/compilation-step.json"
# The image helper rewrites these after a build; restore its original build inputs.
cp /opt/jax/build/requirements.in "$task_root/build/jax/build/requirements.in"
cp /opt/jax/build/requirements_lock_3_12.txt "$task_root/build/jax/build/requirements_lock_3_12.txt"
/usr/local/bin/build-jax.sh \
  --src-path-jax "$task_root/build/jax" \
  --src-path-xla "$task_root/sources/$task_variant" \
  --build-path-jaxlib "$task_output/jaxlibs" \
  --bazel-cache http://bazel-tf-cache.nvidia.com:9091 --sm 10.0 \
  "--build-param=--bazel_startup_options=--output_user_root=$task_root/build/bazel" \
  --build-param=--bazel_startup_options=--batch \
  --build-param=--bazel_startup_options=--host_jvm_args=-Xmx16g \
  --build-param=--bazel_options=--remote_upload_local_results=false \
  --build-param=--bazel_options=--jobs=64 \
  --build-param=--bazel_options=--local_resources=memory=192000 --no-install
fi
python3 - "$task_root" "$task_variant" <<'PY'
import hashlib,json,os,pathlib,sys
r=pathlib.Path(sys.argv[1]);v=sys.argv[2];o=r/'builds'/v
record=next(x for x in json.loads((r/'variants.json').read_text()) if x['variant']==v)
libs={str(p.relative_to(o/'jaxlibs')):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (o/'jaxlibs').rglob('*.so')}
assert libs
record.update(libraries=libs,build_step=json.loads((o/'compilation-step.json').read_text())['step_id'],validation_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],image=json.loads((r/'image.json').read_text()))
(o/'jaxlibs/build-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
PY
python3 -m pip install --user --no-deps --no-build-isolation \
  -e "$task_output/jaxlibs/jaxlib" -e "$task_output/jaxlibs/jax_cuda13_pjrt" \
  -e "$task_output/jaxlibs/jax_cuda13_plugin"
export PYTHONPATH="$task_root/build/jax:$task_output/jaxlibs/jaxlib:$task_output/jaxlibs/jax_cuda13_pjrt:$task_output/jaxlibs/jax_cuda13_plugin"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 - "$task_variant" <<'PY'
import sys,json,pathlib
import jax,jaxlib,jax.numpy as jnp
import jax_plugins.xla_cuda13 as plugin
v=sys.argv[1]
assert '/builds/'+v+'/jaxlibs/' in jaxlib.__file__,jaxlib.__file__
assert '/builds/'+v+'/jaxlibs/' in plugin.__file__,plugin.__file__
options={'pr50278':{'xla_gpu_experimental_enable_early_buffer_release':True},'pr50281':{'xla_gpu_enable_heap_simulator_packing_search':True},'pr50286':{'xla_gpu_experimental_fusion_recomputation':2}}.get(v,{})
y=jax.jit(lambda x:x@x+1,compiler_options=options)(jnp.ones((32,32))).block_until_ready()
assert float(y[0,0])==33
print('SMOKE_PASS',v,jax.__version__,jaxlib.__version__,jax.devices(),flush=True)
PY
if [[ "$task_variant" == base ]]; then
  PYTHONPATH="$task_root/maxtext/src:$PYTHONPATH" python3 - <<'PY'
import maxtext.trainers.pre_train.train
print('MAXTEXT_IMPORT_PASS', flush=True)
PY
fi
touch "$task_output/BUILD_READY"
