"""Check the common runtime, device count and cross-node reduction before training."""
from pathlib import Path
import json
import importlib.metadata
import os
import platform
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src import compilation_cache, compiler, xla_bridge

ROOT = Path('/opt/campaign')
rank = int(os.environ['SLURM_PROCID'])
jax.distributed.initialize(local_device_ids=[int(os.environ['SLURM_LOCALID'])])
assert jax.process_count() == 32 and jax.device_count() == 32
assert len(jax.local_devices()) == 1
assert all(d.platform == 'gpu' for d in jax.local_devices())
assert platform.machine() == 'aarch64'
mesh = Mesh(np.array(jax.devices()), ('data',))
sharding = NamedSharding(mesh, P('data'))
x = jax.make_array_from_process_local_data(sharding, np.ones(1, dtype=np.float32), global_shape=(32,))
reduce_sum = jax.jit(lambda a: jnp.sum(a), out_shardings=NamedSharding(mesh, P()))
module = reduce_sum.lower(x).compiler_ir()
options = compiler.get_compile_options(num_replicas=1, num_partitions=32)
backend = xla_bridge.get_backend()
executable_key = compilation_cache.get_cache_key(module, np.array(jax.devices()), options, backend)
profile_key = compilation_cache.get_cache_key(
    module, np.array(jax.devices()), options, backend,
    profile_device_kinds_only=True,
)
reductions = [float(reduce_sum(x).block_until_ready()) for _ in range(6)]
y = reductions[-1]
assert reductions == [32.0] * 6
profiles = compiler._share_fdo_profiles.modules_profiles
assert profiles, 'Automatic PGLE did not exercise profile synchronization'
assert float(y) == 32
import maxtext.trainers.pre_train.train
record = dict(rank=rank, hostname=platform.node(), architecture=platform.machine(),
              devices=[str(d) for d in jax.local_devices()], process_count=jax.process_count(),
              device_count=jax.device_count(), reduction=float(y), jax_version=jax.__version__,
              jax_path=jax.__file__, reductions=reductions,
              executable_cache_key=executable_key, profile_sync_key=profile_key,
              shared_pgle_profiles={k:len(v) for k,v in profiles.items()})
if int(os.environ['SLURM_LOCALID']) == 0:
    record['gpu_inventory'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version,clocks.max.sm,clocks.sm,pstate,power.limit', '--format=csv,noheader'], text=True)
if rank == 0:
    record['python_version'] = platform.python_version()
    record['package_versions'] = sorted({(d.metadata.get('Name', 'unknown'), d.version) for d in importlib.metadata.distributions()})
(ROOT / 'preflight' / f'rank-{rank}.json').write_text(json.dumps(record, indent=2)+'\n')
print('PREFLIGHT_PASS', json.dumps(record, sort_keys=True), flush=True)
