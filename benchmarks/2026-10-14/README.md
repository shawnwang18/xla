# GB200/GB300 JAX benchmark results and reproduction environments — 2026-10-13–14

This records the full-size random-weight command-buffer comparison, three ten-workload compiler-flag suites, and four reduced multi-node Bug6811592 cases. Configurations, software provenance and frozen runners are archived for future reproduction. Failed and unstable measurements remain explicitly identified.

The command-buffer sweep improved decode throughput for Gemma 2 2B and Llama 3 8B; Qwen 3 8B was fastest with buffers OFF. The PR suites primarily showed allocator-memory savings, with mostly small latency changes and several qualified results.

## Reproduction files

[Environment index: one record per workload/campaign](environment-summary.md) · [Machine-readable environment records](environments.json) · [Archive checksums](SHA256SUMS)

| Download archive                                       | Environment records | Contents                                                       |
| ------------------------------------------------------ | ------------------- | -------------------------------------------------------------- |
| [command-buffers](command-buffers-reproduction.tar.gz) | 6                   | DeepSeek + five-model four-mode sweep, including OFF retry     |
| [pr50278](pr50278-reproduction.tar.gz)                 | 10                  | Early buffer release OFF/ON                                    |
| [pr50281](pr50281-reproduction.tar.gz)                 | 10                  | Heap packing search OFF/ON                                     |
| [pr50286](pr50286-reproduction.tar.gz)                 | 10                  | Fusion recomputation OFF/ON, including Qwen diagnostic repeats |
| [bug6811592](bug6811592-reproduction.tar.gz)           | 4                   | Reduced multi-node reproduction, OFF/ON                        |

Each archive contains a SHA-256 file manifest, configuration/environment records, measurement protocol, available runtime commands and hardware observations, runner/collector source, and build/library provenance. The snapshots are taken from the original runs, including final repaired variants. This publication did not rerun the benchmarks.

## Original command-buffer comparison

One node and four GB200 GPUs per model, BF16 weights/compute, random weights, skill-owned consistent-AOT runner. All image-supported command types except WHILE and CONDITIONAL were enabled in the three enabled modes. OFF disables command buffers. Modes ran sequentially on the same node for each model; models could run on separate nodes.

All 200 steps 101–300 were retained after 100 warmups for each phase. Each call was synchronized; a global step is the maximum of its four rank latencies. Both every rank and the global series must satisfy CV ≤10% and half-window mean drift ≤5%. No stalls were dropped and no faster window was selected.

### Prefill latency

| Model                  | Global batch | OFF (ms) | ALWAYS_UPDATE (ms) | SKIP_TEMP (ms) | SKIP_PROFILED (ms) |
| ---------------------- | ------------ | -------- | ------------------ | -------------- | ------------------ |
| DeepSeek V4 Flash 284B | 4            | 1110.972 | 1186.917           | 1147.946       | 1187.004           |
| Gemma 2 2B             | 256          | 13.900   | 15.734             | 16.431         | 17.076             |
| Llama 3 8B             | 256          | 16.386   | 18.207             | 19.529         | 20.012             |
| Qwen 3 8B              | 4            | 18.618   | 27.341             | 32.923         | 31.058             |

### Decode latency

| Model                  | Global batch | OFF (ms)           | ALWAYS_UPDATE (ms) | SKIP_TEMP (ms) | SKIP_PROFILED (ms) |
| ---------------------- | ------------ | ------------------ | ------------------ | -------------- | ------------------ |
| DeepSeek V4 Flash 284B | 4            | Unstable; excluded | 133.762            | 141.535        | 118.474            |
| Gemma 2 2B             | 256          | 11.514             | 10.948             | 11.089         | 11.322             |
| Llama 3 8B             | 256          | 12.351             | 12.388             | 11.493         | 11.877             |
| Qwen 3 8B              | 4            | 10.670             | 16.343             | 13.495         | 11.673             |

### Decode throughput

| Model                  | OFF (tokens/s) | ALWAYS_UPDATE (tokens/s) | SKIP_TEMP (tokens/s) | SKIP_PROFILED (tokens/s) |
| ---------------------- | -------------- | ------------------------ | -------------------- | ------------------------ |
| DeepSeek V4 Flash 284B | Excluded       | 29.904                   | 28.262               | 33.763                   |
| Gemma 2 2B             | 22233.331      | 23383.982                | 23086.504            | 22609.934                |
| Llama 3 8B             | 20726.505      | 20665.811                | 22273.831            | 21554.539                |
| Qwen 3 8B              | 374.889        | 244.749                  | 296.400              | 342.668                  |

- Gemma 2 2B: ALWAYS_UPDATE gives **5.2% higher decode throughput** than OFF.
- Llama 3 8B: SKIP_TEMP gives **7.5% higher decode throughput** than OFF.
- Qwen 3 8B: OFF is fastest for prefill and decode; enabled-mode decode latencies are 9.4–53.2% higher. Related investigation: [issue #6](https://github.com/shawnwang18/xla/issues/6).
- DeepSeek: SKIP_PROFILED gives **12.9% higher decode throughput than ALWAYS_UPDATE**. OFF decode remains excluded after its retry; no OFF-based decode speedup is valid.
- Gemma 2 27B (batch600) and Gemma 3 27B (batch480): OFF initialization ran out of GPU memory before timing; the remaining modes were not started. No performance numbers are available for those exact configurations.

DeepSeek OFF retry job2507484 used the same node as step2497742.6. Decode CV was65.638% and half-window drift19.454%; stalls at steps138/174/186/187 coincided with generation-2 Python GC events of588.817–647.491ms. This is correlation, not a controlled causal test. The original OFF window (CV59.628%, drift14.309%) and all retry samples remain archived. Other modes retain their original measurements.

OFF/ALWAYS_UPDATE used BFC; SKIP_TEMP/SKIP_PROFILED selected VMM, so the comparison includes allocator differences. True prompt lengths were4 tokens (DeepSeek/Gemma/Llama) or3 (Qwen), padded to4096 for DeepSeek or1024 for the other models. Maximum lengths were8192 and2048 respectively. These are single fixed-order sweeps, not randomized multi-run confidence estimates.

## Compiler PR comparisons: results per workload

Each PR compares one built binary with its feature OFF and ON. The switches are independent of the four-mode command-buffer sweep:

| PR    | Compiler switch                                  | OFF   | ON   | XLA commit                               |
| ----- | ------------------------------------------------ | ----- | ---- | ---------------------------------------- |
| 50278 | xla_gpu_experimental_enable_early_buffer_release | false | true | 4e5526cbc1d38484e39116573d5b3209d1a2e471 |
| 50281 | xla_gpu_enable_heap_simulator_packing_search     | false | true | f94df3539d9c655a86b5324264aaf01a0b5ce593 |
| 50286 | xla_gpu_experimental_fusion_recomputation        | 0     | 2    | 3c5d6db0a29728982cfc2aeda3f003c1fa128f61 |

Latency changes below are **100 × (ON/OFF −1)**: negative means lower latency. They are observed means, not blanket claims of statistical significance. Compare within each campaign; hardware, capacity adjustments and binaries differ across columns.

| Workload                   | PR50278 latency Δ | PR50281 latency Δ     | PR50286 latency Δ     |
| -------------------------- | ----------------- | --------------------- | --------------------- |
| Llama 3 8B FP8 training    | +0.86%            | -0.05%                | -0.22%                |
| Qwen 3 8B FP8 training     | +0.87%            | +1.30%                | +0.16%                |
| Mistral 7B FP8 training    | -0.06%            | -0.08%                | +0.35%                |
| Gemma 2 9B FP8 training    | +0.53%            | -0.05%                | -0.61%                |
| Gemma 3 4B FP8 training    | -0.83%            | -0.49%                | -1.32%                |
| Llama 3 8B BF16 inference  | -0.49%            | -1.04%                | -0.04%                |
| Qwen 3 8B BF16 inference   | +0.50%            | -6.56% (inconclusive) | +9.07% (inconclusive) |
| Gemma 2 2B BF16 inference  | +0.20%            | -0.22%                | -0.59%                |
| Gemma 2 27B BF16 inference | +0.21%            | -0.29% (conditional)  | -0.13%                |
| Gemma 3 27B BF16 inference | +0.16%            | +0.28%                | +0.03%                |

### Allocator peak memory

| Workload                   | PR50278 allocator Δ | PR50281 allocator Δ | PR50286 allocator Δ |
| -------------------------- | ------------------- | ------------------- | ------------------- |
| Llama 3 8B FP8 training    | +0.00%              | +0.00%              | -12.88%             |
| Qwen 3 8B FP8 training     | +0.00%              | -0.21%              | +0.00%              |
| Mistral 7B FP8 training    | -2.86%              | -0.43%              | -0.00%              |
| Gemma 2 9B FP8 training    | -1.44%              | -0.55%              | -8.14%              |
| Gemma 3 4B FP8 training    | +0.00%              | +0.00%              | -21.80%             |
| Llama 3 8B BF16 inference  | +0.00%              | +0.00%              | +0.00%              |
| Qwen 3 8B BF16 inference   | -0.36%              | -0.01%              | +0.00%              |
| Gemma 2 2B BF16 inference  | +0.00%              | +0.00%              | +0.00%              |
| Gemma 2 27B BF16 inference | +0.00%              | +0.00%              | +0.00%              |
| Gemma 3 27B BF16 inference | +0.00%              | +0.00%              | +0.00%              |

Allocator peaks cover the application including compilation, taking the maximum GPU per run and then averaging repeats. Sampled physical GPU peaks were essentially unchanged; reserved/preallocated pools can mask allocator-memory savings. Largest savings: PR50286 Llama3 training12.9%, Gemma2-9B training8.1%, Gemma3-4B training21.8%. Raw OFF/ON timings, memory values and repeat data are in [workload-results.csv](workload-results.csv) and the archives.

### Measurement and qualifications

- Training: FP8 catalog configurations, synthetic input, 101 total steps; synchronized iteration timing includes data fetch, forward/backward/optimizer, logging and synchronization, measured at steps40–99.
- Inference: BF16 random weights. Phase timings use100 warmups+200 samples. Request latency uses5 warmups+20 requests, with one active request, padded1024-token prefill/insert and64 sequential decode calls on the compiled batch shape. It excludes loading, tokenization, compilation and state initialization.
- Each PR planned a discarded OFF pilot followed by measured OFF/ON/ON/OFF. PR50286 Qwen inference includes four additional ON/OFF/OFF/ON trials: all four repeats per mode are included.
- Qwen inference repeat variation makes PR50281’s−6.56% and PR50286’s+9.07% means inconclusive. Phase stability does not establish request-timing stability. Other small changes can also be within repeat variation.
- PR50281 Gemma2-27B: one measured ON attempt failed with CUDA_ERROR_ILLEGAL_ADDRESS in prefill. The recheck passed, but the original failure remains unresolved. Its passing-run timing is conditional and does not qualify correctness.
- PR50286 Gemma2-9B: the original growing-pool OFF passed and ON failed with fragmentation OOM. Published paired results use supplemental preallocation for both modes. Gemma3-4B training and both27B inference cases also used preallocation.
- Later27B PR tests used smaller batches after baseline OOM: global32 for PR50278, global4 for PR50281, global64 for PR50286. These do not replace or invalidate the original batch600/480 four-mode failures.
- PR50281 Qwen training used the no_pipelining variant after the original OFF pilot timed out. PR50278 reduced some training batches; the per-workload environment index records the final settings.
- Inference checks cover finite outputs/format consistency and sampled logits, not checkpoint-backed accuracy. No accepted fusion recomputation rewrites were observed in PR50286 inference cases.

## Bug6811592 reduced multi-node reproduction

| Case   | Cluster   | OFF step (ms) | ON step (ms) | Latency Δ | OFF allocator (GiB) | ON allocator (GiB) |
| ------ | --------- | ------------- | ------------ | --------- | ------------------- | ------------------ |
| n4-ga2 | lyris     | 986.18        | 994.99       | +0.89%    | 75.811              | 71.586             |
| n4-ga3 | lyris     | 1462.06       | 1459.56      | -0.17%    | 77.792              | 76.482             |
| n8-ga2 | lyris     | 3095.84       | 3094.79      | -0.03%    | 138.823             | 138.823            |
| n8-ga3 | dlcluster | 4604.94       | 4608.03      | +0.07%    | 139.410             | 140.020            |

Four-node cases use16 GB300 GPUs, five layers,128 experts, FSDP4/EP4. Eight-node cases use32 GB300 GPUs,17 layers,256 experts, FSDP4/EP8. GA is gradient accumulation2 or3. OFF/ON/ON/OFF used the same reserved nodes and one PR50286 binary; PGLE was enabled. Timing is from unprofiled steps110–159 of165 total steps. These are reduced reproductions, not full284B inference. The dlcluster case required an assigned IMEX-channel mount; its exact setup and collective preflight records are archived.

## Software and hardware provenance

| Campaign                   | Runtime image                                                                                   | JAX / jaxlib                                                                               | MaxText revision                         |
| -------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------- |
| Command buffers            | dl/dgx/jax:maxtext-2026-10-12                                                                   | 0.12.0.dev20261012+290228f95c / 0.12.0.dev20261013                                         | b51e17d5dde9590909537673864cde99481f6f2e |
| PR50278                    | dl/dgx/jax@sha256:9c6acebc3c2af41015ea974148c055ba1517b5ce54592352eb797a92bfde091f              | 0.12.0.dev20261014 / 0.12.0.dev20261014                                                    | 391bf164b71cc0780c246bab899cd29effa06903 |
| PR50281                    | dl/dgx/jax:maxtext-2026-10-13                                                                   | 0.12.0.dev20261013+e0baa9fab1 / 0.12.0.dev20261013                                         | 391bf164b71cc0780c246bab899cd29effa06903 |
| PR50286 ten-workload suite | dl/dgx/jax:maxtext-2026-10-13                                                                   | 0.12.0.dev20261013+e0baa9fab1 / dev20261013 or dev20261014 by node; exact libraries hashed | 391bf164b71cc0780c246bab899cd29effa06903 |
| Bug6811592                 | abgoel/paxml-containers@sha256:8b47bdd072c8989ab91fc13c1beae73a633d80657e1ad7abc5d5fe26735c23bd | 0.12.0.dev20260929+e321d5cc54 / 0.12.0.dev20261013                                         | 67646f337b5781d6128a5584c9318262241fb807 |

All image names above use registry `gitlab-master.nvidia.com:5005`. Registry access is required. PR runtimes mount custom built JAX/XLA libraries over the base image: the base image alone is insufficient. Build manifests distinguish build-container provenance from runtime-container provenance. Dated tags without a captured digest are explicitly not immutable pins. The command-buffer image XLA revision is5cba10fbbb68e6d549d7670aa2d4d7b487494c86. PR50278 JAX source is8a09c826ff78fae72c55476c347ad924ffc32ded. Typeguard2.13.3 is the recorded runtime dependency override.

All ten-workload runs used one aarch64 node and four GPUs. PR50278 used GB200; PR50281 used GB300; PR50286 used a mixture recorded per workload. Node names, GPU identities/driver observations where captured, batch and sequence lengths, precision, mesh, allocator/preallocation, XLA/NCCL flags, model argv and source/library hashes are saved in the [per-workload environment index](environment-summary.md). The Qwen command-buffer run records GB200 driver595.84.01; consult each campaign’s own hardware records for its driver.

## Reproduce later

1. Download the desired archive and verify its SHA-256 against `SHA256SUMS`. Read its `README.txt` and `environments/<workload>.json`.
2. Restore the recorded container and, for a PR test, the matching custom-built JAX/XLA libraries. Check hashes and source revisions. Keep the recorded dependency overrides.
3. Stage the archived harness/configuration on shared storage. Adapt historical paths, Slurm IDs and node constraints to a new compatible allocation while preserving container mount targets and all model settings.
4. Run through Slurm on compute nodes. Access dlcluster nodes through the dlcluster head node. Keep compared modes sequential on the same GPU set; preserve allocator settings, mode order, warmups and measurement windows.
5. Run the archived collector, retain every required sample, and report failures/instability. Separate phase timings, request latency, application wall time, allocator peak and physical GPU memory.

These are environment/configuration and runner archives, not bootable image backups. Large binary wheels, CUDA libraries, full HLO dumps, Nsight captures and checkpoints are not included. Their available hashes and original locations are recorded. Historical commands require path/resource adaptation and were not newly validated as portable launchers. No credentials or authentication files are included.
