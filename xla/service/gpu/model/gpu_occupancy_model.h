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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_OCCUPANCY_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_OCCUPANCY_MODEL_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/time/time.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Occupancy model for the GPU cost model.
//
// The default cost model judges how busy a kernel keeps the GPU from the
// number of blocks it launches: min(num_blocks, core_count) SMs run at
// min(threads_per_block, fpus_per_core) lanes, any launch with enough blocks
// is credited the full HBM bandwidth, and compute and memory accesses always
// overlap by a fixed factor. None of these depend on how many warps are
// resident on an SM, which is what actually sets the issue rate, the memory
// requests in flight, and the ability to hide latency.
//
// This file estimates that missing quantity. From the launch dimensions and a
// few hints about the kernel it derives how many blocks fit on one SM, splits
// the launch into waves of blocks that run concurrently, and reports, per
// wave, how many SMs are busy and how many warps the busiest SM and the whole
// chip hold. GpuPerformanceModelBase turns these into time with an issue rate
// that ramps with resident warps, a Little's law bandwidth cap (bytes in flight
// divided by latency), and an overlap factor that grows with resident warps.

// Constants of the occupancy model. The defaults are starting values for an
// H100 SXM and are meant to be measured per GPU generation.
struct OccupancyModelParams {
  // Round trip of one memory request to HBM.
  absl::Duration dram_latency = absl::Nanoseconds(600);
  // Round trip of one memory request that hits in L2.
  absl::Duration l2_latency = absl::Nanoseconds(250);
  // Fraction of the datasheet HBM bandwidth that a well-written kernel
  // reaches.
  double achievable_bandwidth_fraction = 0.9;
  // Resident warps per SM at which the SM issues at its full rate: four
  // schedulers that each need about four warps to cover the dependent-issue
  // latency of an FMA.
  int64_t warps_for_full_issue_rate = 16;
  // Resident warps per SM at which compute and memory accesses overlap as well
  // as GpuPerformanceModelBase::kMemoryComputeParallelism assumes.
  int64_t warps_for_full_overlap = 16;
  // Bounds on the number of memory requests one warp keeps in flight.
  int64_t min_requests_in_flight_per_warp = 1;
  int64_t max_requests_in_flight_per_warp = 8;
  // Registers per thread assumed when a kernel's register usage is unknown.
  int64_t default_registers_per_thread = 32;
  // Resident blocks per SM assumed when the device does not report a limit.
  int64_t default_max_blocks_per_sm = 32;
};

// Returns the occupancy model parameters selected by `debug_options`, or
// nullopt if the occupancy model is disabled. Passes that want the occupancy
// model pass the result to the performance model explicitly; the models
// default to nullopt, so the flag reaches only passes that opt in.
std::optional<OccupancyModelParams> OccupancyModelParamsFromDebugOptions(
    const DebugOptions& debug_options);

// Facts about a kernel that the occupancy model needs beyond its launch
// dimensions. Values left at zero are unknown and replaced by defaults.
struct KernelOccupancyHints {
  int64_t registers_per_thread = 0;
  int64_t shared_memory_per_block_bytes = 0;
  // Elements of the largest array each thread touches. Together with
  // `num_inputs` it bounds the number of memory requests a warp keeps in
  // flight and the number of dependent round trips a block has to wait for.
  int64_t elements_per_thread = 1;
  // Distinct operands the kernel reads.
  int64_t num_inputs = 1;
  // Size in bytes of one element of the kernel's memory traffic.
  int64_t bytes_per_element = 4;
};

// Resident state of the GPU during one wave: a set of blocks that run at the
// same time.
struct OccupancyWave {
  int64_t num_blocks = 0;
  // SMs that hold at least one block of this wave.
  int64_t active_sms = 0;
  // Blocks and warps on the SM that received the most blocks. The wave ends
  // when this SM finishes.
  int64_t blocks_on_busiest_sm = 0;
  int64_t warps_on_busiest_sm = 0;
  // Warps resident on the whole chip during this wave.
  int64_t total_warps = 0;
  // Share of the kernel's blocks, and therefore of its work and bytes, that
  // run in this wave.
  double fraction_of_kernel = 0.0;
};

struct OccupancyEstimate {
  int64_t num_blocks = 0;
  int64_t threads_per_block = 0;
  int64_t warps_per_block = 0;
  // Blocks that fit on one SM: the tightest of the block, thread, register and
  // shared memory limits.
  int64_t blocks_per_sm = 0;
  // Blocks the whole chip holds at once.
  int64_t slots = 0;
  int64_t num_waves = 0;
  // All full waves look alike, so they are described once with a count.
  int64_t num_full_waves = 0;
  OccupancyWave full_wave;
  // The trailing wave when `num_blocks` is not a multiple of `slots`.
  std::optional<OccupancyWave> partial_wave;
  // Memory requests each warp keeps in flight, and the size of each.
  int64_t requests_in_flight_per_warp = 1;
  int64_t bytes_per_request = 128;
  // Round trips a block has to wait for one after another. A wave cannot be
  // shorter than this many memory latencies.
  int64_t dependent_round_trips_per_block = 1;

  // Calls `fn(wave, count)` for the full wave (if any) and the partial wave
  // (if any).
  template <typename Fn>
  void ForEachWave(Fn&& fn) const {
    if (num_full_waves > 0) {
      fn(full_wave, num_full_waves);
    }
    if (partial_wave.has_value()) {
      fn(*partial_wave, int64_t{1});
    }
  }

  std::string ToString() const;
};

// Estimates how `launch_dimensions` occupy `device_info`.
OccupancyEstimate EstimateOccupancy(const se::DeviceDescription& device_info,
                                    const LaunchDimensions& launch_dimensions,
                                    const KernelOccupancyHints& hints,
                                    const OccupancyModelParams& params);

// Fraction of its peak issue rate that an SM reaches with `warps_per_sm`
// resident warps: min(1, warps_per_sm / warps_for_full_issue_rate).
double IssueRateFraction(int64_t warps_per_sm,
                         const OccupancyModelParams& params);

// Fraction of the full compute-memory overlap that an SM reaches with
// `warps_per_sm` resident warps: min(1, warps_per_sm / warps_for_full_overlap).
double OverlapFraction(int64_t warps_per_sm,
                       const OccupancyModelParams& params);

// Bandwidth in bytes per second that `total_warps` warps sustain by Little's
// law when each keeps `requests_in_flight_per_warp` requests of
// `bytes_per_request` bytes outstanding over `latency`.
double LittlesLawBandwidth(int64_t total_warps,
                           int64_t requests_in_flight_per_warp,
                           int64_t bytes_per_request, absl::Duration latency);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_GPU_OCCUPANCY_MODEL_H_
