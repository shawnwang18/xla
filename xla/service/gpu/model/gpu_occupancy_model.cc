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

#include "xla/service/gpu/model/gpu_occupancy_model.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

std::optional<OccupancyModelParams> OccupancyModelParamsFromDebugOptions(
    const DebugOptions& debug_options) {
  if (!debug_options.xla_gpu_experimental_occupancy_cost_model()) {
    return std::nullopt;
  }
  return OccupancyModelParams{};
}

namespace {

// Number of blocks that fit within `limit` when each uses `per_block`, or
// nullopt when the device does not report the limit. A block that exceeds the
// limit on its own still has to run somewhere, so the result is at least one.
std::optional<int64_t> BlocksWithinLimit(int64_t limit, int64_t per_block) {
  if (limit <= 0 || per_block <= 0) {
    return std::nullopt;
  }
  return std::max<int64_t>(1, limit / per_block);
}

OccupancyWave MakeWave(int64_t num_blocks, int64_t total_blocks,
                       int64_t sm_count, int64_t warps_per_block) {
  OccupancyWave wave;
  wave.num_blocks = num_blocks;
  wave.active_sms = std::min(num_blocks, sm_count);
  wave.blocks_on_busiest_sm = CeilOfRatio(num_blocks, wave.active_sms);
  wave.warps_on_busiest_sm = wave.blocks_on_busiest_sm * warps_per_block;
  wave.total_warps = num_blocks * warps_per_block;
  wave.fraction_of_kernel =
      static_cast<double>(num_blocks) / static_cast<double>(total_blocks);
  return wave;
}

std::string WaveToString(const OccupancyWave& wave) {
  return absl::StrFormat(
      "{blocks: %d, active_sms: %d, blocks_on_busiest_sm: %d, "
      "warps_on_busiest_sm: %d, total_warps: %d, fraction: %.4f}",
      wave.num_blocks, wave.active_sms, wave.blocks_on_busiest_sm,
      wave.warps_on_busiest_sm, wave.total_warps, wave.fraction_of_kernel);
}

}  // namespace

OccupancyEstimate EstimateOccupancy(const se::DeviceDescription& device_info,
                                    const LaunchDimensions& launch_dimensions,
                                    const KernelOccupancyHints& hints,
                                    const OccupancyModelParams& params) {
  OccupancyEstimate estimate;
  estimate.num_blocks = std::max<int64_t>(
      1, static_cast<int64_t>(launch_dimensions.num_blocks()));
  estimate.threads_per_block = std::max<int64_t>(
      1, static_cast<int64_t>(launch_dimensions.num_threads_per_block()));
  const int64_t warp_size =
      device_info.threads_per_warp() > 0 ? device_info.threads_per_warp() : 32;
  estimate.warps_per_block = CeilOfRatio(estimate.threads_per_block, warp_size);

  // Blocks per SM: the tightest of the four limits. Limits the device does not
  // report are skipped.
  int64_t blocks_per_sm = device_info.max_blocks_per_multiprocessor() > 0
                              ? device_info.max_blocks_per_multiprocessor()
                              : params.default_max_blocks_per_sm;
  if (std::optional<int64_t> by_threads = BlocksWithinLimit(
          device_info.threads_per_core_limit(), estimate.threads_per_block)) {
    blocks_per_sm = std::min(blocks_per_sm, *by_threads);
  }
  const int64_t registers_per_thread =
      hints.registers_per_thread > 0 ? hints.registers_per_thread
                                     : params.default_registers_per_thread;
  if (std::optional<int64_t> by_registers = BlocksWithinLimit(
          device_info.registers_per_core_limit(),
          registers_per_thread * estimate.threads_per_block)) {
    blocks_per_sm = std::min(blocks_per_sm, *by_registers);
  }
  if (hints.shared_memory_per_block_bytes > 0) {
    if (std::optional<int64_t> by_shared_memory =
            BlocksWithinLimit(device_info.shared_memory_per_core(),
                              hints.shared_memory_per_block_bytes)) {
      blocks_per_sm = std::min(blocks_per_sm, *by_shared_memory);
    }
  }
  estimate.blocks_per_sm = std::max<int64_t>(1, blocks_per_sm);

  // Waves. Every full wave looks the same, so it is described once.
  const int64_t sm_count = std::max<int64_t>(1, device_info.core_count());
  estimate.slots = sm_count * estimate.blocks_per_sm;
  estimate.num_full_waves = estimate.num_blocks / estimate.slots;
  const int64_t remainder = estimate.num_blocks % estimate.slots;
  estimate.num_waves = estimate.num_full_waves + (remainder > 0 ? 1 : 0);
  if (estimate.num_full_waves > 0) {
    estimate.full_wave = MakeWave(estimate.slots, estimate.num_blocks, sm_count,
                                  estimate.warps_per_block);
  }
  if (remainder > 0) {
    estimate.partial_wave = MakeWave(remainder, estimate.num_blocks, sm_count,
                                     estimate.warps_per_block);
  }

  // Memory-level parallelism. A thread that touches `elements_per_thread`
  // elements of each of `num_inputs` operands has that many loads to issue;
  // simple code keeps a bounded number of them in flight and waits for the
  // rest in dependent round trips.
  const int64_t loads_per_thread =
      std::max<int64_t>(1, hints.elements_per_thread) *
      std::max<int64_t>(1, hints.num_inputs);
  const int64_t min_requests =
      std::max<int64_t>(1, params.min_requests_in_flight_per_warp);
  const int64_t max_requests =
      std::max(min_requests, params.max_requests_in_flight_per_warp);
  estimate.requests_in_flight_per_warp =
      std::clamp(loads_per_thread, min_requests, max_requests);
  estimate.bytes_per_request =
      warp_size * std::max<int64_t>(1, hints.bytes_per_element);
  estimate.dependent_round_trips_per_block =
      CeilOfRatio(loads_per_thread, estimate.requests_in_flight_per_warp);
  return estimate;
}

double IssueRateFraction(int64_t warps_per_sm,
                         const OccupancyModelParams& params) {
  if (params.warps_for_full_issue_rate <= 0) {
    return 1.0;
  }
  return std::min(1.0,
                  static_cast<double>(std::max<int64_t>(1, warps_per_sm)) /
                      static_cast<double>(params.warps_for_full_issue_rate));
}

double OverlapFraction(int64_t warps_per_sm,
                       const OccupancyModelParams& params) {
  if (params.warps_for_full_overlap <= 0) {
    return 1.0;
  }
  return std::min(1.0, static_cast<double>(std::max<int64_t>(1, warps_per_sm)) /
                           static_cast<double>(params.warps_for_full_overlap));
}

double LittlesLawBandwidth(int64_t total_warps,
                           int64_t requests_in_flight_per_warp,
                           int64_t bytes_per_request, absl::Duration latency) {
  const double seconds = absl::ToDoubleSeconds(latency);
  if (seconds <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(total_warps) *
         static_cast<double>(requests_in_flight_per_warp) *
         static_cast<double>(bytes_per_request) / seconds;
}

std::string OccupancyEstimate::ToString() const {
  std::string result = absl::StrFormat(
      "OccupancyEstimate{blocks: %d, threads_per_block: %d, warps_per_block: "
      "%d, blocks_per_sm: %d, slots: %d, waves: %d, "
      "requests_in_flight_per_warp: "
      "%d, bytes_per_request: %d, dependent_round_trips_per_block: %d",
      num_blocks, threads_per_block, warps_per_block, blocks_per_sm, slots,
      num_waves, requests_in_flight_per_warp, bytes_per_request,
      dependent_round_trips_per_block);
  if (num_full_waves > 0) {
    absl::StrAppendFormat(&result, ", full_wave x%d: %s", num_full_waves,
                          WaveToString(full_wave));
  }
  if (partial_wave.has_value()) {
    absl::StrAppendFormat(&result, ", partial_wave: %s",
                          WaveToString(*partial_wave));
  }
  result += "}";
  return result;
}

}  // namespace xla::gpu
