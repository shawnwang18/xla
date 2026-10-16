"""Measure completed MaxText iterations; keep the same instrumentation for every tree."""
from pathlib import Path
import importlib
import hashlib
import json
import math
import os
import time
import traceback

from absl import app
import jax
import jaxlib
import jax_plugins.xla_cuda13 as plugin

ROOT = Path('/opt/campaign')
OUTPUT = Path('/opt/results')
VARIANT = os.environ['PERF_VARIANT']
RANK = int(os.environ['SLURM_PROCID'])
jax_patch = json.loads((ROOT / 'jax-pgle-profile-key.json').read_text())
for name, hashes in jax_patch['files'].items():
    assert hashlib.sha256((ROOT / 'build/jax' / name).read_bytes()).hexdigest() == hashes['patched_sha256'], name
manifest = json.loads((ROOT / 'builds' / VARIANT / 'jaxlibs/build-manifest.json').read_text())
expected = next(x for x in json.loads((ROOT / 'variants.json').read_text()) if x['variant'] == VARIANT)
assert manifest['commit'] == expected['commit']
assert f'/builds/{VARIANT}/jaxlibs/' in jaxlib.__file__, jaxlib.__file__
assert f'/builds/{VARIANT}/jaxlibs/' in plugin.__file__, plugin.__file__
print('PERF_PROVENANCE', json.dumps(dict(variant=VARIANT, rank=RANK, commit=manifest['commit'],
      tree=manifest['tree'], jax=jax.__version__, jaxlib=jaxlib.__version__,
      jax_path=jax.__file__, jaxlib_path=jaxlib.__file__, plugin_path=plugin.__file__), sort_keys=True), flush=True)

train = importlib.import_module('maxtext.trainers.pre_train.train')
original_iteration = train.training_loop_iteration
samples = []
loaded_libraries = []
run_config = {}


def measured_iteration(jax_device_state, python_vars, immutable_data):
    if not loaded_libraries:
        mapped = {line.split(maxsplit=5)[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                  if line.endswith(('/libjax_common.so', '/xla_cuda_plugin.so'))}
        assert len(mapped) == 2, mapped
        assert all(f'/builds/{VARIANT}/jaxlibs/' in path for path in mapped), mapped
        loaded_libraries.extend(sorted(mapped))
        print('PERF_LOADED_LIBRARIES', json.dumps(loaded_libraries), flush=True)
        config = immutable_data['config']
        names = ['model_name', 'num_decoder_layers', 'first_num_dense_layers', 'num_experts',
                 'gradient_accumulation_steps', 'per_device_batch_size', 'max_target_length',
                 'global_batch_size_to_load', 'global_batch_size_to_train_on',
                 'micro_batch_size_to_train_on', 'ici_fsdp_parallelism',
                 'ici_expert_parallelism', 'dcn_fsdp_parallelism']
        run_config.update({name: getattr(config, name, None) for name in names})
        assert run_config['num_decoder_layers'] == 17 and run_config['gradient_accumulation_steps'] == 2, run_config
        assert jax.process_count() == 32 and jax.device_count() == 32 and len(jax.local_devices()) == 1
        print('PERF_RESOLVED_CONFIG', json.dumps(run_config, sort_keys=True), flush=True)
    step = int(python_vars['step'])
    # Complete preceding work before starting this iteration's E2E timer.
    jax.block_until_ready(jax_device_state['state'])
    started = time.perf_counter_ns()
    metrics = original_iteration(jax_device_state, python_vars, immutable_data)
    # The stock iteration mutates this dict to hold the new state.
    jax.block_until_ready((jax_device_state['state'], metrics))
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    loss = float(metrics['scalar']['learning/loss'])
    if not math.isfinite(loss):
        raise RuntimeError(f'Nonfinite training loss at step {step}: {loss}')
    samples.append(dict(step=step, rank=RANK, e2e_ms=elapsed_ms, loss=loss))
    print('PERF_STEP', json.dumps(samples[-1], sort_keys=True), flush=True)
    return metrics


train.training_loop_iteration = measured_iteration


def main(argv):
    started = time.monotonic()
    state = 'FAILED'
    try:
        train.main(argv)
        state = 'COMPLETED'
    finally:
        devices = []
        try:
            for device in jax.local_devices():
                devices.append(dict(device=str(device), stats=device.memory_stats()))
        except Exception as exc:
            devices.append(dict(error=str(exc)))
        record = dict(state=state, variant=VARIANT, rank=RANK, samples=samples,
                      application_wall_seconds=time.monotonic()-started, devices=devices,
                      commit=manifest['commit'], tree=manifest['tree'], loaded_libraries=loaded_libraries,
                      jax_runtime_patch=jax_patch['files'],
                      resolved_config=run_config)
        (OUTPUT / f'rank-{RANK}.json').write_text(json.dumps(record, indent=2) + '\n')
        print('PERF_RESULT', json.dumps({k:v for k,v in record.items() if k != 'samples'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    try:
        app.run(main)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            raise
        # main() already persists failure evidence. Exit promptly so Slurm can
        # stop peer ranks instead of hanging in distributed shutdown at exit.
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
