/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_THROUGHPUT_DATA_H_
#define XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_THROUGHPUT_DATA_H_

namespace xla {
namespace gpu {

// Elementwise op costs measured as issue throughput: FMA-equivalent flops per
// element with every SM of the device saturated by independent work. Obtained
// with xla/service/gpu/model:hlo_op_profiler_run --mode=throughput, which runs
// a device-filling kernel instead of the single-thread dependency chain of the
// default --mode=latency (which measures latency). When HloOpProfiles selects
// this table, a device listed here replaces its entries from
// hlo_op_profiles_data.h as a whole; devices missing here keep them.
//
// The `clock_cycles` field name is kept for compatibility with the loader; the
// values are flops per element in the unit GpuHloCostAnalysis reports.
inline constexpr char kDeviceHloOpProfilesThroughput[] = R"pb(
  entries {
    key: "sm_120"  # NVIDIA RTX PRO 6000 Blackwell, 188 SMs, 2430 MHz, CUDA 13.5
    value {
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S8 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 84
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 81
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S16 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 86
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 78
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 84
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 57
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S64 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 320
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 245
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S64 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U8 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 69
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 55
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U16 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 70
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 59
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 27
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 44
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U64 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 301
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 215
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U64 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 47
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 94
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F16 }
        }
        clock_cycles: 90
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 53
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 73
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 48
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F16 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 100
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F16 }
        }
        clock_cycles: 82
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 45
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 211
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 270
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 45
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 108
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F32 }
        }
        clock_cycles: 56
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 21
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 50
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 60
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 74
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 41
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 106
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F32 }
        }
        clock_cycles: 110
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 50
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 246
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 265
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 1942
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 2753
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F64 }
        }
        clock_cycles: 6866
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 2794
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 3207
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 308
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 5576
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 3388
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F64 }
        }
        clock_cycles: 301
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F64 }
        }
        clock_cycles: 130
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 1198
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 2868
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 1909
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F64 }
        }
        clock_cycles: 4528
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 2151
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 157
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 6958
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 2020
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 159
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 14352
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 164
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 180
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 129
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 330
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 351
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 603
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C64 }
        }
        clock_cycles: 650
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 698
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 131
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 980
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C64 }
        }
        clock_cycles: 524
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 522
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 221
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 19
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 1281
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 11146
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 12430
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 11749
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 14487
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 19057
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C128 }
        }
        clock_cycles: 28278
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 20760
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 11152
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 31954
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C128 }
        }
        clock_cycles: 18045
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 17843
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 315
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 13720
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 956
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 39145
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 320
      }
    }
  }
  entries {
    key: "sm_107"  # NVIDIA R100, 204 SMs, 2424 MHz, CUDA 13.5
    value {
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S8 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 112
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 91
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 113
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 130
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 118
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 86
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S64 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 323
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 312
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U8 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 85
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 105
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 75
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 94
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 35
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 57
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U64 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 267
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 277
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 67
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 150
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F16 }
        }
        clock_cycles: 134
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 24
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 96
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 12
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 120
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 63
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F16 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F16 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 23
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 145
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 23
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F16 }
        }
        clock_cycles: 147
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 64
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 264
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 14
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 335
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 63
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 202
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F32 }
        }
        clock_cycles: 85
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 29
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 78
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 94
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 108
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 56
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 196
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F32 }
        }
        clock_cycles: 197
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 79
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 334
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 322
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 186
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 233
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F64 }
        }
        clock_cycles: 468
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 200
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 308
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 126
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 614
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 310
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F64 }
        }
        clock_cycles: 19
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F64 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 113
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 245
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 196
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F64 }
        }
        clock_cycles: 385
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 188
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 667
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 237
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 1192
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 411
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 678
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 424
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 380
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 669
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C64 }
        }
        clock_cycles: 624
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 660
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 351
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 944
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C64 }
        }
        clock_cycles: 488
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 486
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 1842
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 214
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 21
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 1589
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 1327
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 1711
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1458
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 1985
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 2570
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C128 }
        }
        clock_cycles: 3363
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 2588
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 1285
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 3467
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C128 }
        }
        clock_cycles: 2295
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 2337
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 19
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 1632
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 57
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 4458
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 18
      }
    }
  }
  entries {
    key: "sm_103"  # NVIDIA B300A, 72 SMs, 2085 MHz, CUDA 13.5
    value {
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S8 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 148
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 162
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S16 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 131
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 134
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 98
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 60
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S64 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 323
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 265
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U8 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 77
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 95
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U16 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 70
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 85
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 28
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 44
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U64 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 284
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 240
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 55
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 121
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F16 }
        }
        clock_cycles: 133
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 22
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 84
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 22
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 109
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 49
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F16 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 21
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 118
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 21
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F16 }
        }
        clock_cycles: 100
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 56
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 249
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 13
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 337
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 51
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 105
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F32 }
        }
        clock_cycles: 70
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 23
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 63
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 78
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 88
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 45
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 100
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F32 }
        }
        clock_cycles: 92
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 63
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 335
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 308
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 1901
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 2824
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F64 }
        }
        clock_cycles: 6820
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 2758
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 3180
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 308
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 5415
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 3323
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F64 }
        }
        clock_cycles: 295
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F64 }
        }
        clock_cycles: 129
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 1219
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 2827
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 1481
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F64 }
        }
        clock_cycles: 4486
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 2130
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 153
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 6887
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 2002
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 154
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 14240
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 153
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 279
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 319
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 351
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 385
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 712
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C64 }
        }
        clock_cycles: 583
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 671
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 288
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 929
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C64 }
        }
        clock_cycles: 471
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 480
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 186
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 1321
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 11082
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 12891
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 11753
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 14271
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 19524
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C128 }
        }
        clock_cycles: 28248
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 21145
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 11082
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 31663
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C128 }
        }
        clock_cycles: 17431
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 17485
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 310
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 13894
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 932
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 311
      }
    }
  }
  entries {
    key: "sm_100"  # NVIDIA B100, 148 SMs, 1965 MHz, CUDA 13.5
    value {
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S8 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 148
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 173
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S16 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 131
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 133
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 102
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 60
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S64 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 328
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 264
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U8 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 78
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 96
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U8 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U16 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 73
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 88
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 33
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 50
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U64 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 244
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 228
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 58
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 133
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F16 }
        }
        clock_cycles: 150
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 25
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 89
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 23
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 106
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 51
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F16 }
        }
        clock_cycles: 20
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 22
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 123
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 21
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F16 }
        }
        clock_cycles: 108
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 54
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 290
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 14
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 361
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 59
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 98
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F32 }
        }
        clock_cycles: 77
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 26
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 72
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 85
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 93
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 54
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 90
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F32 }
        }
        clock_cycles: 93
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 73
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 325
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 325
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 159
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 172
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F64 }
        }
        clock_cycles: 328
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 124
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 233
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 85
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 530
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 218
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F64 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F64 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 102
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 169
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 141
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F64 }
        }
        clock_cycles: 290
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 155
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 628
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 172
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 951
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 282
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 289
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 371
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 354
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 667
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C64 }
        }
        clock_cycles: 612
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 696
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 272
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 990
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C64 }
        }
        clock_cycles: 499
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 508
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 194
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 19
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 1500
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 1079
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 1389
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1223
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 1627
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 2129
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C128 }
        }
        clock_cycles: 2778
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 2226
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 1097
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 3090
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C128 }
        }
        clock_cycles: 1816
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 1896
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 1110
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 33
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 10
      }
    }
  }
  entries {
    key: "sm_90"  # NVIDIA H100 PCIe, 114 SMs, 1755 MHz, CUDA 13.5
    value {
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S8 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 107
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 102
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S16 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 107
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 99
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 91
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 68
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S64 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 322
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 255
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U8 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 68
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 63
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U16 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 68
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 66
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 26
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 42
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U64 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 240
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 218
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 45
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 84
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F16 }
        }
        clock_cycles: 97
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 55
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 81
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 45
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 79
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F16 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 40
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 201
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 254
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 40
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 78
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F32 }
        }
        clock_cycles: 55
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 20
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 54
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 66
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 78
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 39
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 77
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 16
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F32 }
        }
        clock_cycles: 89
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 51
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 257
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 255
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 137
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 165
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F64 }
        }
        clock_cycles: 312
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 104
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 236
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 89
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 567
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 242
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F64 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F64 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 85
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 159
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 104
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F64 }
        }
        clock_cycles: 281
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 158
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 577
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 162
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 935
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 307
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 335
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 343
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 380
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 700
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C64 }
        }
        clock_cycles: 587
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 695
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 301
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 963
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C64 }
        }
        clock_cycles: 424
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 417
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 1897
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 199
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 18
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 1301
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 893
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 1191
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1063
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 1327
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1845
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C128 }
        }
        clock_cycles: 2189
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 1709
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 871
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 2740
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C128 }
        }
        clock_cycles: 1432
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 1433
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 1134
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 26
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 3641
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 9
      }
    }
  }
  entries {
    key: "sm_89"  # NVIDIA L40S, 142 SMs, 2520 MHz, CUDA 13.5
    value {
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S8 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 56
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 48
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 56
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 46
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 46
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 32
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S64 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 162
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 121
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U8 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 34
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 30
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 35
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 30
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 22
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U64 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 142
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 106
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 22
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 39
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F16 }
        }
        clock_cycles: 42
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 25
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 35
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 22
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 42
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F16 }
        }
        clock_cycles: 50
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 22
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 104
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 120
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 22
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 36
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F32 }
        }
        clock_cycles: 27
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 27
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 30
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 33
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 24
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 37
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F32 }
        }
        clock_cycles: 43
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 27
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 115
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 122
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 949
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 1406
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F64 }
        }
        clock_cycles: 3393
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 1357
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1582
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 154
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 2665
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 1653
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F64 }
        }
        clock_cycles: 147
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F64 }
        }
        clock_cycles: 64
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 606
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 1413
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 736
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F64 }
        }
        clock_cycles: 2233
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 1059
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 3431
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 996
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 7086
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 99
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 92
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 134
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 163
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 395
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C64 }
        }
        clock_cycles: 270
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 325
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 87
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 454
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C64 }
        }
        clock_cycles: 201
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 198
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 116
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 5642
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 6689
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 5897
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 7250
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 9574
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C128 }
        }
        clock_cycles: 14465
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 10476
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 5401
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 16042
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C128 }
        }
        clock_cycles: 8739
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 8751
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 158
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 7059
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 474
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 158
      }
    }
  }
  entries {
    key: "sm_86"  # NVIDIA A2, 10 SMs, 1770 MHz, CUDA 13.5
    value {
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S8 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 55
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 48
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 55
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 46
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 44
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 30
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S64 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 154
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 109
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U8 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 34
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 30
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 34
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 30
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 20
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U64 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 140
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 4
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 102
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 23
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 33
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F16 }
        }
        clock_cycles: 41
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 24
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 32
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 21
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 33
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F16 }
        }
        clock_cycles: 34
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 21
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 93
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 110
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 20
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 68
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F32 }
        }
        clock_cycles: 25
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 23
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 29
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 30
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 67
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F32 }
        }
        clock_cycles: 73
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 23
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 108
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 105
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 944
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 1398
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F64 }
        }
        clock_cycles: 3378
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 1365
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1575
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 153
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 2614
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 1646
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F64 }
        }
        clock_cycles: 146
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F64 }
        }
        clock_cycles: 64
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 603
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 1405
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 733
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F64 }
        }
        clock_cycles: 2221
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 1055
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 3413
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 991
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 7054
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 136
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 153
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 168
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 146
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 258
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C64 }
        }
        clock_cycles: 233
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 272
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 137
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C64 }
        }
        clock_cycles: 192
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 193
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 78
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 5426
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 6435
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 5748
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 6977
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 9113
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C128 }
        }
        clock_cycles: 13897
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 10049
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 5427
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 15410
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C128 }
        }
        clock_cycles: 8398
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 8396
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 152
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 6788
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 456
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 152
      }
    }
  }
  entries {
    key: "sm_80"  # NVIDIA A100-SXM4-80GB, 108 SMs, 1410 MHz, CUDA 13.5
    value {
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S8 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 55
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 70
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 55
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 80
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S32 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 62
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 45
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: S64 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 197
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 164
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: S64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U8 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 34
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 36
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U8 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 40
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 42
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U32 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 19
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 25
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U32 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "popcnt"
          shape { element_type: U64 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 150
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 135
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: U64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 29
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 54
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F16 }
        }
        clock_cycles: 74
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 39
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 61
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 30
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F16 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F16 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 53
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F16 }
        }
        clock_cycles: 57
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 26
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 166
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 8
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 219
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 1
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 23
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 55
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F32 }
        }
        clock_cycles: 44
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 36
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 48
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 56
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 25
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F32 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F32 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 58
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F32 }
        }
        clock_cycles: 56
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 40
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 205
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 237
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 97
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 124
      }
      entries {
        instruction {
          opcode: "erf"
          shape { element_type: F64 }
        }
        clock_cycles: 212
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 96
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 157
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 114
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 331
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 146
      }
      entries {
        instruction {
          opcode: "round-nearest-afz"
          shape { element_type: F64 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "round-nearest-even"
          shape { element_type: F64 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 59
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 124
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: F64 }
        }
        clock_cycles: 180
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 90
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 365
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 101
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 690
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 213
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 328
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 440
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 314
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 472
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C64 }
        }
        clock_cycles: 479
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 496
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 199
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 777
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C64 }
        }
        clock_cycles: 379
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 378
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 160
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 17
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 643
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 818
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 709
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 873
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1703
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: C128 }
        }
        clock_cycles: 1804
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 1313
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 608
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 1940
      }
      entries {
        instruction {
          opcode: "tan"
          shape { element_type: C128 }
        }
        clock_cycles: 1025
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 1063
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 847
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 30
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 10
      }
    }
  }
)pb";

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_THROUGHPUT_DATA_H_
