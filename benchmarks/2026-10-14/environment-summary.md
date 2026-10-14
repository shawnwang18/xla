# Recorded environments by workload

These are the final recorded configurations, including capacity/allocator adjustments. Historical failed variants remain in the archives. The JSON files retain exact per-trial environment variables, XLA flags, runtime argv where captured, mesh settings, library versions and evidence paths.

## command-buffers

| Workload / complete record                                                     | GPU       | Node                 | Global batch | Outcome                        |
| ------------------------------------------------------------------------------ | --------- | -------------------- | ------------ | ------------------------------ |
| [gemma2-2b](environments/command-buffers/gemma2-2b.json)                       | 4 × GB200 | gb-nvl-053-compute09 | 256          | COMPLETED                      |
| [llama3-8b](environments/command-buffers/llama3-8b.json)                       | 4 × GB200 | gb200-nvl4-ts2-79    | 256          | COMPLETED                      |
| [qwen3-8b](environments/command-buffers/qwen3-8b.json)                         | 4 × GB200 | gb200-nvl4-ts2-88    | 4            | COMPLETED                      |
| [gemma2-27b](environments/command-buffers/gemma2-27b.json)                     | 4 × GB200 | gb200-nvl4-ts2-82    | 600          | FAILED                         |
| [gemma3-27b](environments/command-buffers/gemma3-27b.json)                     | 4 × GB200 | gb200-nvl4-ts2-81    | 480          | FAILED                         |
| [deepseek4-flash-284b](environments/command-buffers/deepseek4-flash-284b.json) | 4 × GB200 | gb-nvl-053-compute09 | 4            | Completed; OFF decode unstable |

## pr50278

All entries use one aarch64 node with four GPUs. Batch below is per device. The exact mesh keys are in each JSON; -1 denotes automatic resolution.

| Workload / complete record                                                               | GPU   | Node               | Batch / device | Max length | Recorded mesh                             | Trials |
| ---------------------------------------------------------------------------------------- | ----- | ------------------ | -------------- | ---------- | ----------------------------------------- | ------ |
| [llama3-8b-fp8](environments/pr50278/llama3-8b-fp8.json)                                 | GB200 | gb200-nvl4-ts2-97  | 2              | 8192       | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [qwen3-8b-fp8](environments/pr50278/qwen3-8b-fp8.json)                                   | GB200 | gb200-nvl4-ts2-90  | 6              | 4096       | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [mistral-7b-fp8](environments/pr50278/mistral-7b-fp8.json)                               | GB200 | gb200-nvl4-ts2-80  | 12.0           | 2048       | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [gemma2-9b-fp8](environments/pr50278/gemma2-9b-fp8.json)                                 | GB200 | gb200-nvl4-ts2-77  | 2              | 4096       | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [gemma3-4b-fp8](environments/pr50278/gemma3-4b-fp8.json)                                 | GB200 | gb200-nvl4-ts2-74  | 4              | 4096       | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [infer-llama3-8b-bf16-1n4g](environments/pr50278/infer-llama3-8b-bf16-1n4g.json)         | GB200 | gb200-nvl4-ts2-73  | 64             | 2048       | dcn[1,1,1,1,1] ici[1,1,1,1,1,1,1,-1,1,1]  | 5      |
| [infer-qwen3-8b-bf16-1n4g](environments/pr50278/infer-qwen3-8b-bf16-1n4g.json)           | GB200 | gb200-nvl4-ts2-72  | 1              | 2048       | dcn[1,1,1,1,1] ici[1,1,1,1,4,1,1,1,1,1]   | 5      |
| [infer-gemma2-2b-bf16-1n4g](environments/pr50278/infer-gemma2-2b-bf16-1n4g.json)         | GB200 | gb200-nvl4-ts2-100 | 64             | 2048       | dcn[1,1,1,1,1] ici[1,1,1,1,1,1,1,-1,1,1]  | 5      |
| [ci-infer-gemma2-27b-bf16-1n4g](environments/pr50278/ci-infer-gemma2-27b-bf16-1n4g.json) | GB200 | gb200-nvl4-ts2-86  | 8              | 2048       | dcn[1,1,1,1,1] ici[1,4,1,1,1,1,1,1,1,1]   | 5      |
| [ci-infer-gemma3-27b-bf16-1n4g](environments/pr50278/ci-infer-gemma3-27b-bf16-1n4g.json) | GB200 | gb200-nvl4-ts2-73  | 8              | 2048       | dcn[1,1,1,1,1] ici[1,4,1,1,1,1,1,1,1,1]   | 5      |

## pr50281

All entries use one aarch64 node with four GPUs. Batch below is per device. The exact mesh keys are in each JSON; -1 denotes automatic resolution.

| Workload / complete record                                                               | GPU   | Node                    | Batch / device                               | Max length              | Recorded mesh                             | Trials |
| ---------------------------------------------------------------------------------------- | ----- | ----------------------- | -------------------------------------------- | ----------------------- | ----------------------------------------- | ------ |
| [llama3-8b-fp8](environments/pr50281/llama3-8b-fp8.json)                                 | GB300 | gb300-nvl-022-compute01 | 2                                            | 8192                    | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [qwen3-8b-fp8](environments/pr50281/qwen3-8b-fp8.json)                                   | GB300 | gb300-nvl-022-compute04 | 6                                            | 4096                    | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [mistral-7b-fp8](environments/pr50281/mistral-7b-fp8.json)                               | GB300 | gb300-nvl-022-compute07 | inherited model default; see archived config | inherited model default | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [gemma2-9b-fp8](environments/pr50281/gemma2-9b-fp8.json)                                 | GB300 | gb300-nvl-022-compute08 | 6                                            | 4096                    | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [gemma3-4b-fp8](environments/pr50281/gemma3-4b-fp8.json)                                 | GB300 | gb300-nvl-022-compute09 | 8                                            | 4096                    | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [infer-llama3-8b-bf16-1n4g](environments/pr50281/infer-llama3-8b-bf16-1n4g.json)         | GB300 | gb300-nvl-022-compute10 | 64                                           | 2048                    | dcn[1,1,1,1,1] ici[1,1,1,1,1,1,1,-1,1,1]  | 5      |
| [infer-qwen3-8b-bf16-1n4g](environments/pr50281/infer-qwen3-8b-bf16-1n4g.json)           | GB300 | gb300-nvl-022-compute11 | 1                                            | 2048                    | dcn[1,1,1,1,1] ici[1,1,1,1,4,1,1,1,1,1]   | 5      |
| [infer-gemma2-2b-bf16-1n4g](environments/pr50281/infer-gemma2-2b-bf16-1n4g.json)         | GB300 | gb300-nvl-022-compute14 | 64                                           | 2048                    | dcn[1,1,1,1,1] ici[1,1,1,1,1,1,1,-1,1,1]  | 5      |
| [ci-infer-gemma2-27b-bf16-1n4g](environments/pr50281/ci-infer-gemma2-27b-bf16-1n4g.json) | GB300 | gb300-nvl-022-compute10 | 1                                            | 2048                    | dcn[1,1,1,1,1] ici[1,4,1,1,1,1,1,1,1,1]   | 5      |
| [ci-infer-gemma3-27b-bf16-1n4g](environments/pr50281/ci-infer-gemma3-27b-bf16-1n4g.json) | GB300 | gb300-nvl-022-compute11 | 1                                            | 2048                    | dcn[1,1,1,1,1] ici[1,4,1,1,1,1,1,1,1,1]   | 5      |

## pr50286

All entries use one aarch64 node with four GPUs. Batch below is per device. The exact mesh keys are in each JSON; -1 denotes automatic resolution.

| Workload / complete record                                                               | GPU   | Node                    | Batch / device                               | Max length              | Recorded mesh                             | Trials |
| ---------------------------------------------------------------------------------------- | ----- | ----------------------- | -------------------------------------------- | ----------------------- | ----------------------------------------- | ------ |
| [llama3-8b-fp8](environments/pr50286/llama3-8b-fp8.json)                                 | GB300 | gb300-nvl-012-compute15 | 2                                            | 8192                    | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [qwen3-8b-fp8](environments/pr50286/qwen3-8b-fp8.json)                                   | GB300 | gb300-nvl-012-compute15 | 6                                            | 4096                    | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [mistral-7b-fp8](environments/pr50286/mistral-7b-fp8.json)                               | GB200 | gb200-nvl4-ts2-79       | inherited model default; see archived config | inherited model default | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [gemma2-9b-fp8](environments/pr50286/gemma2-9b-fp8.json)                                 | GB200 | gb200-nvl4-ts2-82       | 6                                            | 4096                    | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [gemma3-4b-fp8](environments/pr50286/gemma3-4b-fp8.json)                                 | GB200 | gb200-nvl4-ts2-81       | 8                                            | 4096                    | dcn[1,-1,1,1,1] ici[1,-1,1,1,1,1,1,1,1,1] | 5      |
| [infer-llama3-8b-bf16-1n4g](environments/pr50286/infer-llama3-8b-bf16-1n4g.json)         | GB200 | gb200-nvl4-ts2-82       | 64                                           | 2048                    | dcn[1,1,1,1,1] ici[1,1,1,1,1,1,1,-1,1,1]  | 5      |
| [infer-qwen3-8b-bf16-1n4g](environments/pr50286/infer-qwen3-8b-bf16-1n4g.json)           | GB200 | gb200-nvl4-ts2-79       | 1                                            | 2048                    | dcn[1,1,1,1,1] ici[1,1,1,1,4,1,1,1,1,1]   | 9      |
| [infer-gemma2-2b-bf16-1n4g](environments/pr50286/infer-gemma2-2b-bf16-1n4g.json)         | GB200 | gb200-nvl4-ts2-82       | 64                                           | 2048                    | dcn[1,1,1,1,1] ici[1,1,1,1,1,1,1,-1,1,1]  | 5      |
| [ci-infer-gemma2-27b-bf16-1n4g](environments/pr50286/ci-infer-gemma2-27b-bf16-1n4g.json) | GB300 | gb300-nvl-012-compute15 | 16                                           | 2048                    | dcn[1,1,1,1,1] ici[1,4,1,1,1,1,1,1,1,1]   | 5      |
| [ci-infer-gemma3-27b-bf16-1n4g](environments/pr50286/ci-infer-gemma3-27b-bf16-1n4g.json) | GB300 | gb300-nvl-012-compute15 | 16                                           | 2048                    | dcn[1,1,1,1,1] ici[1,4,1,1,1,1,1,1,1,1]   | 5      |

## bug6811592

| Case / complete record                        | Cluster   | Nodes | GA | Layers | Experts | FSDP | EP |
| --------------------------------------------- | --------- | ----- | -- | ------ | ------- | ---- | -- |
| [n4-ga2](environments/bug6811592/n4-ga2.json) | lyris     | 4     | 2  | 5      | 128     | 4    | 4  |
| [n4-ga3](environments/bug6811592/n4-ga3.json) | lyris     | 4     | 3  | 5      | 128     | 4    | 4  |
| [n8-ga2](environments/bug6811592/n8-ga2.json) | lyris     | 8     | 2  | 17     | 256     | 4    | 8  |
| [n8-ga3](environments/bug6811592/n8-ga3.json) | dlcluster | 8     | 3  | 17     | 256     | 4    | 8  |
