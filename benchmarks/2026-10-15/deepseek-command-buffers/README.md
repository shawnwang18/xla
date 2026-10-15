# DeepSeek V4 Flash 284B command-buffer rerun, 2026-10-15 UTC

A stable OFF decode baseline was obtained: **95.902 ms/step, 41.709 tokens/s**. SKIP_PROFILED is **0.962× OFF (−3.79% throughput)**; SKIP_TEMP is **0.801× OFF (−19.92%)**. ALWAYS_UPDATE decode is excluded because its complete fixed window failed drift stability. All four modes completed successfully; execution success and measurement stability are separate checks.

See `deepseek-comparison.md`, `deepseek-results.csv`, and `stable-comparison.json`. `command-buffer-results.csv` combines this new DeepSeek sweep with the unchanged other-model rows from artifact snapshot `9efe517b25ee943b404b9273942800134a6a046c`. Rejected absolute performance and ratios are blank in the CSVs; raw rejected measurements remain in `stable-comparison.json` and the evidence archive.

## Protocol

One aarch64 GB200 NVL72 node, four GPUs, global batch 4, full 284,333,639,255-parameter BF16 model, random weights. OFF → ALWAYS_UPDATE → SKIP_TEMP → SKIP_PROFILED ran sequentially in step `2550798.5`, on the same GPUs and original `maxtext-2026-10-12` image. Source revisions, package versions, model configuration, command-buffer flags and allocator choices match the original issue #7 experiment. `environment.json` records these identities. The node changed from `gb-nvl-053-compute09` to `gb-nvl-081-compute06`; both recorded driver 580.159.04. No framework source edits were applied.

Each phase performs a full Python cyclic collection before timing, then disables automatic cyclic GC for the entire warmup plus measurement phase and restores its prior policy afterward. Reference counting remains active. All four ranks and all four modes used this policy; before/after counters confirm no cyclic collections during any phase. Cross-rank barriers are outside timing. Prefill uses 100 warmups then all 200 steps 101–300. Decode uses 1,000 warmups then all 200 steps 1001–1200. The warmup counts were fixed before submission. No samples were removed or faster subwindows chosen. Each call blocks until ready; each global step uses the maximum latency across four ranks. Both every rank and the global series must have CV ≤10% and half-window mean drift ≤5%. Full raw warmup and measurement samples are retained.

The earlier GC-controlled attempt (`2550798.3`) used 100 warmups for both phases. OFF completed, but decode drift was 5.922%, so its baseline was rejected and enabled controls did not start. Its quality-gate assertion caused Slurm to terminate the step (`CANCELLED`, `0:15`, supervisor 143); this was not a user cancellation. The subsequent longer-warmup sweep completed with scheduler `0:0` and supervisor 0. ALWAYS_UPDATE's new decode drift is 10.676% globally, with every rank also failing; its decode speedup is unavailable.

These fresh results are not divided by the older enabled-mode timings: node, GC policy and decoded context positions changed. The original October 13–14 results and both rejected original OFF runs remain in the pinned original artifact snapshot. The new result is a single fixed-order sweep, without randomized repetitions or confidence intervals. It does not establish checkpoint accuracy or attribute cross-run differences solely to GC or warmup.

## Evidence and reproduction

`deepseek-rerun-evidence.tar.gz` contains both new attempts' complete per-rank samples, rank reports, GC events/counters, provenance, startup hardware records, scheduler/supervisor results, the final sweep's GPU clock telemetry, frozen run-local scripts, and the shared consistent-AOT runner. The prior issue body is preserved as historical context. Full process logs remain in the local/cluster run directories; the archive omits environment dumps, credentials, container images, binary wheels and checkpoints. `MANIFEST.json` gives file hashes inside the archive; `SHA256SUMS` covers all published files.

After extracting, validate collected results with:

```bash
python3 long-warmup/summarize_results.py long-warmup/result
```

The collector intentionally exits 1 because ALWAYS_UPDATE decode is unstable. `gc-quality-checks.json` must show valid GC controls for all four modes, and the other seven mode/phase rows must be stable. This is not a failed model execution.

For a new cluster run, use the archived `environment.sh`, `supervise.py`, `mode_sweep.sh`, and runner as the exact configuration source, adapting historical Slurm IDs, claims/locks and filesystem paths to a compatible idle allocation. Route all dlcluster node access through its head node. Keep the original image/revisions, full model, batch, mesh, allocator differences, GC policy, phase-specific warmup counts, 200 measured steps and stability limits. Restore the archived runner mounts plus `typeguard==2.13.3`. Historical launchers are not turnkey portable scripts; container registry access and the dependency wheel are required. The archived source image has no captured immutable digest.
