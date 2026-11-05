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

#include <gtest/gtest.h>

#include <cstdint>

#include "absl/time/time.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

// H100 SXM: 132 SMs, 2048 threads, 65536 registers and 228 KB of shared memory
// per SM. The test device does not report a block limit, so the default of 32
// applies.
class GpuOccupancyModelTest : public ::testing::Test {
 protected:
  se::DeviceDescription device_info_{TestGpuDeviceInfo::H100SXMDeviceInfo()};
  OccupancyModelParams params_;

  OccupancyEstimate Estimate(int64_t num_blocks, int64_t threads_per_block,
                             KernelOccupancyHints hints = {}) {
    return EstimateOccupancy(device_info_,
                             LaunchDimensions(num_blocks, threads_per_block),
                             hints, params_);
  }
};

TEST_F(GpuOccupancyModelTest, BlocksPerSmIsTheTightestLimit) {
  // 128 threads at 32 registers: threads and registers both allow 16.
  EXPECT_EQ(Estimate(1, 128).blocks_per_sm, 16);
  // 1024 threads at 64 registers use the whole register file.
  EXPECT_EQ(Estimate(1, 1024, {.registers_per_thread = 64}).blocks_per_sm, 1);
  // 1024 threads at the default 32 registers: threads allow 2.
  EXPECT_EQ(Estimate(1, 1024).blocks_per_sm, 2);
  // 256 threads with 48 KB of shared memory: 228 / 48 rounds down to 4.
  EXPECT_EQ(Estimate(1, 256, {.shared_memory_per_block_bytes = 48 * 1024})
                .blocks_per_sm,
            4);
  // One-warp blocks: the block limit binds before threads or registers.
  EXPECT_EQ(Estimate(1, 32).blocks_per_sm, 32);
}

TEST_F(GpuOccupancyModelTest, SplitsTheLaunchIntoWaves) {
  OccupancyEstimate estimate =
      Estimate(1100, 256, {.shared_memory_per_block_bytes = 48 * 1024});
  EXPECT_EQ(estimate.warps_per_block, 8);
  EXPECT_EQ(estimate.blocks_per_sm, 4);
  EXPECT_EQ(estimate.slots, 528);
  EXPECT_EQ(estimate.num_waves, 3);
  EXPECT_EQ(estimate.num_full_waves, 2);

  EXPECT_EQ(estimate.full_wave.num_blocks, 528);
  EXPECT_EQ(estimate.full_wave.active_sms, 132);
  EXPECT_EQ(estimate.full_wave.blocks_on_busiest_sm, 4);
  EXPECT_EQ(estimate.full_wave.warps_on_busiest_sm, 32);
  EXPECT_EQ(estimate.full_wave.total_warps, 4224);
  EXPECT_DOUBLE_EQ(estimate.full_wave.fraction_of_kernel, 528.0 / 1100.0);

  ASSERT_TRUE(estimate.partial_wave.has_value());
  EXPECT_EQ(estimate.partial_wave->num_blocks, 44);
  EXPECT_EQ(estimate.partial_wave->active_sms, 44);
  EXPECT_EQ(estimate.partial_wave->blocks_on_busiest_sm, 1);
  EXPECT_EQ(estimate.partial_wave->warps_on_busiest_sm, 8);
  EXPECT_EQ(estimate.partial_wave->total_warps, 352);
  EXPECT_DOUBLE_EQ(estimate.partial_wave->fraction_of_kernel, 44.0 / 1100.0);
}

TEST_F(GpuOccupancyModelTest, BusiestSmTakesTheRemainder) {
  // 1088 blocks over 132 SMs: 8.24 per SM on average, so 32 SMs get 9.
  OccupancyEstimate estimate = Estimate(1088, 128);
  EXPECT_EQ(estimate.num_waves, 1);
  EXPECT_EQ(estimate.num_full_waves, 0);
  ASSERT_TRUE(estimate.partial_wave.has_value());
  EXPECT_EQ(estimate.partial_wave->active_sms, 132);
  EXPECT_EQ(estimate.partial_wave->blocks_on_busiest_sm, 9);
  EXPECT_EQ(estimate.partial_wave->warps_on_busiest_sm, 36);
}

TEST_F(GpuOccupancyModelTest, ExactMultipleOfSlotsHasNoPartialWave) {
  OccupancyEstimate estimate =
      Estimate(132, 1024, {.registers_per_thread = 64});
  EXPECT_EQ(estimate.slots, 132);
  EXPECT_EQ(estimate.num_waves, 1);
  EXPECT_EQ(estimate.num_full_waves, 1);
  EXPECT_FALSE(estimate.partial_wave.has_value());
  EXPECT_EQ(estimate.full_wave.warps_on_busiest_sm, 32);
  EXPECT_EQ(estimate.full_wave.total_warps, 4224);
}

TEST_F(GpuOccupancyModelTest, RequestsInFlightAndRoundTrips) {
  // Four elements per thread from two inputs: eight loads, all in flight.
  OccupancyEstimate estimate =
      Estimate(1, 128, {.elements_per_thread = 4, .num_inputs = 2});
  EXPECT_EQ(estimate.requests_in_flight_per_warp, 8);
  EXPECT_EQ(estimate.bytes_per_request, 128);
  EXPECT_EQ(estimate.dependent_round_trips_per_block, 1);

  // Sixty-four elements per thread: eight in flight, eight round trips.
  estimate = Estimate(1, 128, {.elements_per_thread = 64});
  EXPECT_EQ(estimate.requests_in_flight_per_warp, 8);
  EXPECT_EQ(estimate.dependent_round_trips_per_block, 8);

  // Two-byte elements shrink the request.
  estimate = Estimate(1, 128, {.bytes_per_element = 2});
  EXPECT_EQ(estimate.bytes_per_request, 64);
}

TEST_F(GpuOccupancyModelTest, IssueRateRampsToOneAtSixteenWarps) {
  EXPECT_DOUBLE_EQ(IssueRateFraction(1, params_), 1.0 / 16.0);
  EXPECT_DOUBLE_EQ(IssueRateFraction(8, params_), 0.5);
  EXPECT_DOUBLE_EQ(IssueRateFraction(16, params_), 1.0);
  EXPECT_DOUBLE_EQ(IssueRateFraction(64, params_), 1.0);
  EXPECT_DOUBLE_EQ(OverlapFraction(4, params_), 0.25);
  EXPECT_DOUBLE_EQ(OverlapFraction(32, params_), 1.0);
}

TEST_F(GpuOccupancyModelTest, LittlesLawBandwidth) {
  // 53 warps, four 128-byte requests each, 600 ns round trip: 45.2 GB/s.
  EXPECT_NEAR(LittlesLawBandwidth(53, 4, 128, absl::Nanoseconds(600)) / 1e9,
              45.2, 0.1);
  // 1,696 warps: 1,447 GB/s.
  EXPECT_NEAR(LittlesLawBandwidth(1696, 4, 128, absl::Nanoseconds(600)) / 1e9,
              1447.3, 0.5);
}

TEST(OccupancyModelParamsFromDebugOptionsTest, FollowsTheFlag) {
  DebugOptions debug_options;
  EXPECT_FALSE(OccupancyModelParamsFromDebugOptions(debug_options).has_value());
  debug_options.set_xla_gpu_experimental_occupancy_cost_model(true);
  EXPECT_TRUE(OccupancyModelParamsFromDebugOptions(debug_options).has_value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
