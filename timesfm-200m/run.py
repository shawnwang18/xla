from pathlib import Path
import shutil
import statistics
import sys
import time

import jax
import jax.numpy as jnp
from timesfm.timesfm_2p5.timesfm_2p5_flax import TimesFM_2p5_200M_flax_module

assert len(jax.devices()) == 1, "Expose one GPU with CUDA_VISIBLE_DEVICES."
iterations, warmup = map(int, sys.argv[2:4])
assert iterations >= 2 and warmup >= 1, "Use >=2 iterations and >=1 warmup."
model = TimesFM_2p5_200M_flax_module()
model.compile(context=512, horizon=512, per_core_batch_size=1)
inputs = jnp.zeros((1, 1, 512), dtype=jnp.float32)
masks = jnp.zeros_like(inputs, dtype=jnp.bool_)
# Compile before warming up and timing synchronized calls.
jax.block_until_ready(model.compiled_decode(512, inputs, masks))
for _ in range(warmup):
    jax.block_until_ready(model.compiled_decode(512, inputs, masks))

durations_us = []
for _ in range(iterations):
    start = time.perf_counter_ns()
    jax.block_until_ready(model.compiled_decode(512, inputs, masks))
    durations_us.append((time.perf_counter_ns() - start) / 1000)

mean_us = statistics.mean(durations_us)
std_pct = 100 * statistics.stdev(durations_us) / mean_us
print(
    f"{mean_us:.1f} us +/- {std_pct:.2f}% "
    f"(synchronized wall time)"
)

output_dir = Path(sys.argv[1])
candidates = list((output_dir / "dumps").glob("*jit__fun*after_spmd_partitioner.txt"))
assert candidates, "No forecast HLO was dumped."
candidate = max(candidates, key=lambda path: path.stat().st_mtime_ns)
hlo = candidate.read_text()
assert 'custom_call_target="Sharding"' not in hlo
assert 'custom_call_target="CustomSPMDPartitioning"' not in hlo
destination = output_dir / "forecast_1gpu.after_spmd_partitioner.txt"
shutil.copyfile(candidate, destination)
print(f"Saved {destination}")
