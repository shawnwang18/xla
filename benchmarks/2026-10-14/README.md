# GB200 JAX inference: command-buffer performance and reproduction environments

This report compares command buffers OFF, ALWAYS_UPDATE, SKIP_TEMP and SKIP_PROFILED for six full-size BF16 inference workloads on four GB200 GPUs. Gemma 2 2B and Llama 3 8B gained decode throughput; Qwen 3 8B was fastest with OFF. DeepSeek has stable enabled-mode results but no valid OFF decode baseline. Both 27B configurations failed before timing.

These are random-weight microbenchmarks. They do not establish checkpoint accuracy. The measurements were collected on 2026-10-13–14.

## Performance relative to OFF

Each table cell shows **absolute performance (speedup versus OFF)**. Speedup = **OFF mean latency / mode mean latency**; for decode this also equals mode throughput / OFF throughput. OFF is 1.000×, above 1× is faster, and below 1× is slower. Percentage change in performance is 100 × (speedup − 1), not percentage latency reduction. Ratios use unrounded measurements.

N/A means no stable OFF baseline exists for that phase. Failed or unstable timings are excluded from performance and speedup claims.

### Prefill latency and speedup

| Model                  | Global batch | OFF (ms; ×OFF)    | ALWAYS_UPDATE (ms; ×OFF) | SKIP_TEMP (ms; ×OFF) | SKIP_PROFILED (ms; ×OFF) |
| ---------------------- | ------------ | ----------------- | ------------------------ | -------------------- | ------------------------ |
| DeepSeek V4 Flash 284B | 4            | 1110.972 (1.000×) | 1186.917 (0.936×)        | 1147.946 (0.968×)    | 1187.004 (0.936×)        |
| Gemma 2 2B             | 256          | 13.900 (1.000×)   | 15.734 (0.883×)          | 16.431 (0.846×)      | 17.076 (0.814×)          |
| Llama 3 8B             | 256          | 16.386 (1.000×)   | 18.207 (0.900×)          | 19.529 (0.839×)      | 20.012 (0.819×)          |
| Qwen 3 8B              | 4            | 18.618 (1.000×)   | 27.341 (0.681×)          | 32.923 (0.566×)      | 31.058 (0.599×)          |

### Decode latency and speedup

| Model                  | Global batch | OFF (ms; ×OFF)     | ALWAYS_UPDATE (ms; ×OFF) | SKIP_TEMP (ms; ×OFF) | SKIP_PROFILED (ms; ×OFF) |
| ---------------------- | ------------ | ------------------ | ------------------------ | -------------------- | ------------------------ |
| DeepSeek V4 Flash 284B | 4            | Unstable; excluded | 133.762 (N/A)            | 141.535 (N/A)        | 118.474 (N/A)            |
| Gemma 2 2B             | 256          | 11.514 (1.000×)    | 10.948 (1.052×)          | 11.089 (1.038×)      | 11.322 (1.017×)          |
| Llama 3 8B             | 256          | 12.351 (1.000×)    | 12.388 (0.997×)          | 11.493 (1.075×)      | 11.877 (1.040×)          |
| Qwen 3 8B              | 4            | 10.670 (1.000×)    | 16.343 (0.653×)          | 13.495 (0.791×)      | 11.673 (0.914×)          |

### Decode throughput and speedup

| Model                  | Global batch | OFF (tokens/s; ×OFF) | ALWAYS_UPDATE (tokens/s; ×OFF) | SKIP_TEMP (tokens/s; ×OFF) | SKIP_PROFILED (tokens/s; ×OFF) |
| ---------------------- | ------------ | -------------------- | ------------------------------ | -------------------------- | ------------------------------ |
| DeepSeek V4 Flash 284B | 4            | Unstable; excluded   | 29.904 (N/A)                   | 28.262 (N/A)               | 33.763 (N/A)                   |
| Gemma 2 2B             | 256          | 22233.331 (1.000×)   | 23383.982 (1.052×)             | 23086.504 (1.038×)         | 22609.934 (1.017×)             |
| Llama 3 8B             | 256          | 20726.505 (1.000×)   | 20665.811 (0.997×)             | 22273.831 (1.075×)         | 21554.539 (1.040×)             |
| Qwen 3 8B              | 4            | 374.889 (1.000×)     | 244.749 (0.653×)               | 296.400 (0.791×)           | 342.668 (0.914×)               |

- **Gemma 2 2B:** ALWAYS_UPDATE is 1.052× OFF for decode, a 5.2% throughput gain.
- **Llama 3 8B:** SKIP_TEMP is 1.075× OFF for decode, a 7.5% throughput gain.
- **Qwen 3 8B:** enabled-mode decode throughput is 0.653× / 0.791× / 0.914× OFF, respectively: −34.7% / −20.9% / −8.6%. Investigation: [issue #6](https://github.com/shawnwang18/xla/issues/6).
- **DeepSeek V4 Flash 284B:** no OFF-based decode speedup is valid. Among enabled modes, SKIP_PROFILED has 12.9% higher decode throughput than ALWAYS_UPDATE.
- **Prefill:** OFF is fastest for each of the four models with measurements.

Exact accepted values and OFF-relative ratios are also saved in [command-buffer-results.csv](command-buffer-results.csv). The CSV leaves rejected measurements and unavailable ratios empty; the reproduction archive preserves their raw evidence.

## Measurement protocol

Each model used one aarch64 node and four GB200 GPUs. Modes ran sequentially on the same GPU set in the order OFF → ALWAYS_UPDATE → SKIP_TEMP → SKIP_PROFILED. Independent models could run on separate nodes. DeepSeek's later OFF retry used its original node.

Every phase used 100 warmup steps and **all 200 measured steps 101–300**. Each call was synchronized; a global step is the maximum latency across the four ranks. Reported latency is the mean of these 200 global steps. Every rank and the global series must pass CV ≤10% and half-window mean drift ≤5%. No stalls were dropped or faster windows selected. Stability is checked separately for prefill and decode.

Decode throughput is global batch × 1000 / mean decode latency in milliseconds. These are single fixed-order sweeps, not randomized repeated trials with confidence intervals. Compare modes within a workload; batches and node topology differ between workloads.

## Failed configurations and DeepSeek OFF retry

Gemma 2 27B (global batch 600, step 2521171.0) and Gemma 3 27B (global batch 480, step 2521172.0) ran out of GPU memory during OFF state initialization, before any timing. Their supervisors returned 143 after termination; the remaining modes did not start. No performance or speedup is reported for those configurations.

DeepSeek step 2497742.6 completed, and OFF retry job 2507484 also completed on gb-nvl-053-compute09. The original OFF decode window failed stability (CV 59.628%, half-window drift 14.309%); the retry also failed (CV 65.638%, drift 19.454%). Retry stalls at steps 138, 174, 186 and 187 coincided with generation-2 Python GC events lasting 588.817–647.491 ms. This is correlation, not a controlled causal test. Both complete windows and GC observations remain archived. The tables use stable retry OFF prefill and the original enabled-mode measurements; OFF decode is excluded.

## Reproduction environments

[Per-workload environment index](environment-summary.md) · [All six environment records](environments.json) · [Reproduction archive](command-buffers-reproduction.tar.gz) · [SHA-256 checksums](SHA256SUMS)

The archive preserves the six configurations, original and retry samples, available runtime commands, hardware observations, source revisions, frozen runners and collectors, scheduler/status records and GC evidence. The individual JSON records save exact model settings, flags, mode-specific provenance and evidence paths.

| Workload / environment record                                                    | Node (4 × GB200)     | Global batch | Allocation/job or step   | Outcome                            |
| -------------------------------------------------------------------------------- | -------------------- | ------------ | ------------------------ | ---------------------------------- |
| [DeepSeek V4 Flash 284B](environments/command-buffers/deepseek4-flash-284b.json) | gb-nvl-053-compute09 | 4            | 2497742.6; retry 2507484 | Completed; OFF decode unstable     |
| [Gemma 2 2B](environments/command-buffers/gemma2-2b.json)                        | gb-nvl-053-compute09 | 256          | 2507927                  | Stable, all four modes             |
| [Llama 3 8B](environments/command-buffers/llama3-8b.json)                        | gb200-nvl4-ts2-79    | 256          | 2521076                  | Stable, all four modes             |
| [Qwen 3 8B](environments/command-buffers/qwen3-8b.json)                          | gb200-nvl4-ts2-88    | 4            | 2521170                  | Stable, all four modes             |
| [Gemma 2 27B](environments/command-buffers/gemma2-27b.json)                      | gb200-nvl4-ts2-82    | 600          | 2521171                  | OFF initialization OOM; no timings |
| [Gemma 3 27B](environments/command-buffers/gemma3-27b.json)                      | gb200-nvl4-ts2-81    | 480          | 2521172                  | OFF initialization OOM; no timings |

All models use BF16 weights/compute, random weights and the skill-owned consistent-AOT runner (`inference_aot_mode=consistent`), with typeguard 2.13.3. No MaxText or JAX source edits were applied. True prompt lengths are 4 tokens for DeepSeek/Gemma/Llama and 3 for Qwen; padded prefill lengths are 4096 for DeepSeek and 1024 for the other models. Maximum sequence lengths are 8192 and 2048, respectively. Exact meshes and model arguments are retained in each environment record.

OFF disables command buffers. Enabled modes include every image-supported command type except WHILE and CONDITIONAL:

```text
FUSION,CUBLAS,CUDNN,COLLECTIVES,CUSTOM_CALL,CUBLASLT,DYNAMIC_SLICE_FUSION,CONVOLUTION,HOST_EXECUTE,COLLECTIVES_KERNEL
```

Command-buffer scheduling is LHS and command-buffer loop unrolling is false. OFF/ALWAYS_UPDATE use BFC; SKIP_TEMP/SKIP_PROFILED select VMM, so the measured comparison includes allocator differences. Preserve the exact archived flags when reproducing.

| Software component          | Recorded value                                                |
| --------------------------- | ------------------------------------------------------------- |
| Container                   | `gitlab-master.nvidia.com:5005/dl/dgx/jax:maxtext-2026-10-12` |
| XLA revision                | `5cba10fbbb68e6d549d7670aa2d4d7b487494c86`                    |
| MaxText revision            | `b51e17d5dde9590909537673864cde99481f6f2e`                    |
| JAX revision                | `290228f95c2435d1cc110469e785b83e9859760a`                    |
| JAX / jaxlib versions       | `0.12.0.dev20261012+290228f95c` / `0.12.0.dev20261013`        |
| Runtime dependency override | `typeguard==2.13.3`                                           |

The dated image tag has no captured immutable digest. Registry access is required; verify saved source revisions and available library hashes before accepting a reproduction. GPU identities, driver and topology observations are saved per mode where captured. The Qwen node records driver 595.84.01 and NV6 links between every GPU pair; consult the other workloads' own hardware records.

## Reproduce later

1. Download the command-buffer reproduction archive and check its SHA-256 against `SHA256SUMS`; verify its internal `MANIFEST.json` entries.
2. Select `environments/<workload>.json` in the archive. Restore its container, dependency override, model size, precision, batch, sequence lengths and mesh.
3. Stage the archived runner/configuration on shared storage. Adapt historical paths, Slurm IDs and node constraints to compatible resources while preserving the recorded mount targets and flags. Use the saved consistent-AOT runner.
4. Access dlcluster compute nodes through the dlcluster head node and launch through Slurm. Run the four modes sequentially on the same GPU set, preserving allocator choices, mode order, warmups and the full measurement window.
5. Use the archived `deepseek/collect_stable.py` or `suite/collect_stable.py` to evaluate every required sample. For DeepSeek, `deepseek/off-retry-01/comparison-updated/stable-comparison.json` identifies the retry-OFF/original-enabled combination reported here. Report failures and instability explicitly.

The archive saves configurations, measurements and runner code. Container images, large binary wheels, CUDA libraries, full HLO dumps, Nsight captures and checkpoints are not bundled; available hashes and original paths identify those dependencies. Historical launch commands require resource/path adaptation and have not been newly validated as portable launchers. This report update did not rerun benchmarks.
