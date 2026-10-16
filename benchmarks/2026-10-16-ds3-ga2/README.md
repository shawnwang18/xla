# DS3 GA=2: independent PR performance comparison

Tracks https://github.com/shawnwang18/xla/issues/13.

Eight dlcluster GB200 nodes, four GPUs per node. This uses the bug's reduced
17-layer DeepSeek-V3 configuration (3 dense layers, 256 experts), GA=2,
`per_device_batch_size=6`, sequence length 4096, ICI FSDP=4 and EP=8.
It does not qualify the full 671B model. Exact model arguments and XLA flags
are in [workload.json](workload.json).

Each PR is applied independently to common baseline
`ab7771db404bbc19b519951e0099ee0939c28968`, which excludes all four PRs.
Source heads, comparison commits, tree hashes and archive hashes are recorded
in [variants.json](variants.json). For #50278, the new protobuf field is
renumbered from 556 to 558 to avoid a baseline field collision; the algorithm
is unchanged. The default container was resolved once and pinned by ARM64
digest in [image.json](image.json). The actual MaxText source is the frozen
TE-EP snapshot `67646f337b5781d6128a5584c9318262241fb807`, verified against
[maxtext-source-manifest.json](maxtext-source-manifest.json), mounted in place
of the image's MaxText. JAX source is `e0ec3646e180c64be20a63b1b2d50054f8c8995e`
and Transformer Engine is `33f3b3561943fcd9dd2f544f67dc2dfd62efdc9e`.

## Results

Positive reductions mean improvement; negative reductions mean regression.
Small latency differences comparable to repeat variation do not establish
a repeatable gain. Both trial means are included below.

| Variant   | E2E (ms) | Latency reduction (%) | Speedup (x) | Peak in-use (GiB/GPU) | Saved (GiB/GPU) | Memory reduction (%) | Sampled GPU peak (GiB/GPU) | Repeat means (ms)  |
| --------- | -------- | --------------------- | ----------- | --------------------- | --------------- | -------------------- | -------------------------- | ------------------ |
| Baseline  | 4920.768 | +0.000                | 1.0000      | 136.326               | +0.000          | +0.000               | 175.245                    | 4934.297, 4907.240 |
| PR #50182 | 4913.601 | +0.146                | 1.0015      | 136.326               | +0.000          | +0.000               | 175.245                    | 4930.879, 4896.323 |
| PR #50278 | 4913.370 | +0.150                | 1.0015      | 133.469               | +2.857          | +2.095               | 175.245                    | 4913.694, 4913.047 |
| PR #50281 | 4911.273 | +0.193                | 1.0019      | 136.326               | +0.000          | +0.000               | 175.245                    | 4912.269, 4910.278 |
| PR #50286 | 4907.416 | +0.271                | 1.0027      | 136.326               | +0.000          | +0.000               | 175.245                    | 4916.324, 4898.508 |

All 10 trials passed: two baseline trials and two independent trials per PR, with 320 completed rank records and successful Slurm steps `2568930.29`–`2568930.38`.

PR #50278 reduced the full-process JAX allocator peak by **2.857 GiB/GPU (2.095%)**, with the same peak in both repeats. The other PRs did not reduce that peak. Sampled physical GPU peak remained **175.245 GiB/GPU** for all variants under the common preallocation policy.

Observed latency reductions were **0.146%–0.271%**. The two baseline trial means differed by **0.550%**; with only two repeats, these results do not establish a repeatable latency improvement.


Latency reduction = `100*(base_ms-variant_ms)/base_ms`;
speedup = `base_ms/variant_ms`;
memory reduction = `100*(base_peak-variant_peak)/base_peak`.

## Measurement definitions

Trials run serially on the same nodes in forward/reverse variant order, with
fresh processes and per-trial compilation/PGLE caches. Each trial completes
65 steps. Primary latency is the mean over steps 40–60 of the maximum
synchronized end-to-end iteration time across all 32 ranks. It includes data
loading, training, and iteration bookkeeping. Device work is synchronized
at both timing boundaries. The comparison averages the two trial means.

Primary memory is the maximum `peak_bytes_in_use` across all 32 GPU allocators
and both trials. These are full-process lifetime peaks, including initialization
and compilation. Device-wide used memory is sampled once per second and is a
lower bound on its true peak. Peak allocator reservation and compiled total buffer
allocation are separate diagnostics. The compiled total sums all memory spaces,
including host offload allocations; it is not a device-memory peak. The common preallocation policy can keep
device-wide memory nearly constant despite differences in used allocations.

Application main wall time includes model setup and compilation but excludes
Python imports and container startup. Finite training losses, all-rank completion,
two compiler passes including PGLE, source/library hashes and exact successful
Slurm step accounting are required. Finite losses alone do not establish full
numerical equivalence or convergence.

## Additional diagnostics

| Variant | Peak pool (GiB/GPU) | Initial compiled total (GiB) | Post-PGLE compiled total (GiB) | Application main wall (s) |
| ------- | ------------------- | ---------------------------- | ------------------------------ | ------------------------- |
| base    | 149.828             | 136.310, 136.310             | 132.650, 132.940               | 433.4, 435.4              |
| pr50182 | 149.828             | 136.310, 136.310             | 132.940, 132.940               | 442.2, 440.9              |
| pr50278 | 149.828             | 133.450, 133.450             | 127.730, 129.430               | 445.7, 443.5              |
| pr50281 | 149.828             | 136.310, 136.310             | 130.430, 130.360               | 446.9, 434.0              |
| pr50286 | 149.828             | 136.310, 136.310             | 132.890, 132.890               | 442.4, 453.0              |

## Individual trials

| Trial | Variant | Slurm step | E2E step (ms) | Peak in-use (GiB/GPU) | Sampled GPU peak (GiB/GPU) |
| ----- | ------- | ---------- | ------------- | --------------------- | -------------------------- |
| 1     | base    | 2568930.29 | 4934.297      | 136.326               | 175.245                    |
| 2     | pr50182 | 2568930.30 | 4930.879      | 136.326               | 175.245                    |
| 3     | pr50278 | 2568930.31 | 4913.694      | 133.469               | 175.245                    |
| 4     | pr50281 | 2568930.32 | 4912.269      | 136.326               | 175.245                    |
| 5     | pr50286 | 2568930.33 | 4916.324      | 136.326               | 175.245                    |
| 6     | pr50286 | 2568930.34 | 4898.508      | 136.326               | 175.245                    |
| 7     | pr50281 | 2568930.35 | 4910.278      | 136.326               | 175.245                    |
| 8     | pr50278 | 2568930.36 | 4913.047      | 133.469               | 175.245                    |
| 9     | pr50182 | 2568930.37 | 4896.323      | 136.326               | 175.245                    |
| 10    | base    | 2568930.38 | 4907.240      | 136.326               | 175.245                    |

## Cluster configuration

This fixed node set reports 16 GPUs with 189471 MiB and 16 with 192527 MiB
of memory; all identify as GB200 with driver 580.159.04. JAX's strict topology
fingerprint check rejected the capacity difference. All trials therefore use
the same supported `XLA_PJRT_GPU_VALIDATE_TOPOLOGY=false` setting and require
a real 32-GPU collective preflight plus all-rank training completion.
See [hardware adjustment and inventory](hardware-adjustments.json). Containers
expose only the Slurm-assigned IMEX channel 926 through an explicit device bind;
each process verifies channel visibility and access before initializing JAX.
The image's CUDA compatibility library 620.65 returned an unsupported fabric
query error with host driver 580.159.04. Every trial instead loads the native
host driver through the recorded `LD_LIBRARY_PATH` and sets `NCCL_CFT_ENABLE=0`.
Preflight step 2568930.28 completed successfully on all 32 GPUs; NCCL recognized
the MNNVL domain and enabled NVLS. These common changes preserve the pinned
image and all five XLA binaries.
Distributed autotuning also encountered incompatible device fingerprints on
the mixed-capacity nodes. Every variant uses `--xla_gpu_shard_autotuning=false`
so each rank tunes locally; this changes compilation work and is included in
the shared benchmark configuration.
The initial 88% JAX pool reservation left insufficient memory for NCCL during
the first training step. All measured trials reserve 80%, leaving additional
space for communication buffers without changing the model or batch size.
JAX's PGLE synchronization key also included the capacity-dependent topology
fingerprint, leaving lower-capacity ranks waiting for a different key. A common
[Python-only JAX patch](jax-pgle-profile-key.patch) hashes device kinds for PGLE
profile sharing while retaining full topology fingerprints for executable cache
keys. Its source hashes are verified in every training process. The expanded
preflight requires one shared profile key, two distinct executable cache keys,
and six completed collective calls exercising automatic PGLE on all 32 ranks.

## Build validation

The initial incremental #50286 packages crashed during `CompileOptions`
construction, including with the optimization disabled. A fresh build directory
with remote compilation-cache reads disabled passed the GPU smoke check; no
optimization code changed. The measurements use only those validated packages.
See [clean build settings](clean-rebuild.json) and the
[clean build launcher](build-clean-pr50286.sh).

## Evidence and reproduction

- [Complete structured result](results.json), with per-step slowest-rank latency,
  loss samples and validation outcomes.
- [Raw measurements and provenance](raw-measurements.tar.gz), including all-rank
  JSON records, physical memory samples, exact launch commands, node inventory,
  package versions and frozen build-library checksums.
- [Scheduler accounting](accounting.txt) and [campaign status](campaign-status.json).
- [Independent raw-record aggregation audit](raw-data-audit.json).
- [Training instrumentation](measure-training.py), [rank launcher](run-rank.sh),
  [trial controller](run-campaign.py), [collector](collect.py), and independent
  PR patches beside this report.

The scripts record the original cluster paths for auditability. For a fresh run,
stage the baseline and apply each `comparison-pr*.patch` independently in an
isolated source tree (their recorded result tree hashes must match), use the
pinned container and frozen MaxText source revision from the issue, adapt storage
paths and obtain an equivalent allocation. Do not replay an old Slurm job ID.
Keep the same model, mesh, allocator, compiler defaults and profiling controls.
