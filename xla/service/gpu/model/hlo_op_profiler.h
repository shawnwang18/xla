/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILER_H_
#define XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILER_H_

#include <cstdint>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class HloOpProfiler {
 public:
  class KernelTracer {
   public:
    virtual ~KernelTracer() = default;
    virtual uint64_t getMedianKernelTimeNs() && = 0;
    virtual uint64_t getSumKernelTimeNs() && = 0;
  };

  // What a measurement reports in the `clock_cycles` field of the profile.
  enum class Mode {
    // Latency of one thread's dependency chain: the measured kernel processes
    // a single element, and the result is clock cycles per op (counted as 2
    // per cycle to match the FMA convention). This is how
    // hlo_op_profiles_data.h was generated.
    kLatency,
    // Issue throughput: the measured kernel processes enough independent
    // elements to saturate every SM, and the result is the number of
    // FMA-equivalent flops the whole device could have issued while it
    // processed one element of the op. This is the unit
    // GpuPerformanceModelBase::ComputeTime divides by.
    kThroughput,
  };

  HloOpProfiler(HloRunnerInterface* absl_nonnull runner,
                const stream_executor::DeviceDescription* absl_nonnull
                    device_description,
                Mode mode = Mode::kLatency);

  Mode mode() const { return mode_; }

  static std::unique_ptr<KernelTracer> GetKernelTracer();

  static absl::Span<const HloOpcode> AllSupportedOps();

  static const absl::flat_hash_set<HloOpcode>& Unsupported();

  static const absl::flat_hash_set<HloOpcode>& TooFastToMeasure();

  static absl::Span<const PrimitiveType> AllSupportedDtypes();

  // Measures the cost of `op` on `data_type` in the unit selected by the
  // profiler's Mode (see above) and stores it in the `clock_cycles` field of
  // the profile.
  absl::StatusOr<HloInstructionProfile> MeasureClockCyclesPerOp(
      HloOpcode op, PrimitiveType data_type);

 private:
  static std::unique_ptr<HloModule> MakeModuleForMeasurements(
      HloOpcode op, PrimitiveType data_type, int chain_length,
      int64_t num_elements);

  absl::StatusOr<absl::Duration> MeasureOpChainDuration(HloOpcode op,
                                                        PrimitiveType data_type,
                                                        int chain_length);

  // Duration of a copy kernel (chain length 0) over NumElements(data_type)
  // elements of `data_type`, measured once per type. Throughput mode only.
  absl::StatusOr<absl::Duration> CopyDuration(PrimitiveType data_type);

  // Number of independent elements the measured kernel processes: one in
  // latency mode, a fixed byte budget's worth of `data_type` in throughput
  // mode.
  int64_t NumElements(PrimitiveType data_type) const;

  HloRunnerInterface& runner_;
  const se::DeviceDescription& dev_info_;
  const Mode mode_;
  // Kernel runtime below which a measurement is dominated by fixed costs: twice
  // an F32 copy kernel. Throughput mode raises it to twice the copy kernel of
  // the measured data type when that is slower (see CopyDuration).
  absl::Duration min_duration_;
  absl::flat_hash_map<PrimitiveType, absl::Duration> copy_durations_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILER_H_
