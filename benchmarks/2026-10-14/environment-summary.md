# Command-buffer reproduction environments

One aarch64 node and four GB200 GPUs per workload. The JSON records preserve exact settings, mode-specific flags, provenance and evidence paths.

| Workload / environment record                                                    | Node (4 × GB200)     | Global batch | Allocation/job or step   | Outcome                            |
| -------------------------------------------------------------------------------- | -------------------- | ------------ | ------------------------ | ---------------------------------- |
| [DeepSeek V4 Flash 284B](environments/command-buffers/deepseek4-flash-284b.json) | gb-nvl-053-compute09 | 4            | 2497742.6; retry 2507484 | Completed; OFF decode unstable     |
| [Gemma 2 2B](environments/command-buffers/gemma2-2b.json)                        | gb-nvl-053-compute09 | 256          | 2507927                  | Stable, all four modes             |
| [Llama 3 8B](environments/command-buffers/llama3-8b.json)                        | gb200-nvl4-ts2-79    | 256          | 2521076                  | Stable, all four modes             |
| [Qwen 3 8B](environments/command-buffers/qwen3-8b.json)                          | gb200-nvl4-ts2-88    | 4            | 2521170                  | Stable, all four modes             |
| [Gemma 2 27B](environments/command-buffers/gemma2-27b.json)                      | gb200-nvl4-ts2-82    | 600          | 2521171                  | OFF initialization OOM; no timings |
| [Gemma 3 27B](environments/command-buffers/gemma3-27b.json)                      | gb200-nvl4-ts2-81    | 480          | 2521172                  | OFF initialization OOM; no timings |

[Performance report](README.md) · [All records](environments.json) · [Reproduction archive](command-buffers-reproduction.tar.gz)
