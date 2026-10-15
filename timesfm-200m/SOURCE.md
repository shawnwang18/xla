# TimesFM 2.5 200M reproducer

Copied from https://gitlab-master.nvidia.com/xla/hlo-benchmark-collection/-/tree/dbb210fbfaad3bc794eb73448288f4899c598c16/inference/timesfm-200m

Source commit: `dbb210fbfaad3bc794eb73448288f4899c598c16`.

The four source files are unchanged. The HLO is the resolved Git LFS object,
verified against its pointer's size and SHA-256. `SHA256SUMS` covers those four
files. `SOURCE.md` and `SHA256SUMS` were added for provenance.

The original report describes about 20% shorter GPU time on B200 with
`--xla_gpu_enable_command_buffer=+WHILE --xla_gpu_command_buffer_unroll_loops`,
but approximately offsetting additional CPU time attributed to graph updates.
Whether those updates are unnecessary remains to be investigated.

The Python reproducer uses random weights, one GPU, batch 1, context 512, and
horizon 512. It compiles before warming up and measures synchronized wall time.
CPU/GPU time separation requires a profiler.

The provided run.sh uses uv and pins jax[cuda13]==0.11.2, timesfm==2.0.2,
and flax==0.12.10. These are the current source script's dependency pins;
the original measurement's exact software versions were not provided.

Example A/B comparison, running each command in a fresh process on the same B200:

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_FLAGS='--xla_gpu_enable_command_buffer=+WHILE --xla_gpu_command_buffer_unroll_loops=false' \
bash ./run.sh ./baseline 100 20

CUDA_VISIBLE_DEVICES=0 \
XLA_FLAGS='--xla_gpu_enable_command_buffer=+WHILE --xla_gpu_command_buffer_unroll_loops=true' \
bash ./run.sh ./unrolled 100 20
```

These example commands select 100 timed iterations and 20 warmups. The source
script defaults to 16 timed iterations and 10 warmups. This copy has not been
executed on a GPU as part of filing the issue.
