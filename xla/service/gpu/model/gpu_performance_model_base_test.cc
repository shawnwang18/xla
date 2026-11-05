/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_performance_model_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_occupancy_model.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::MLIRContext;

class GpuPerformanceModelBaseTest : public HloHardwareIndependentTestBase {
 public:
  GpuHloCostAnalysis::Options options_;
  // The reference times in the test cases below are measured
  // on A6000 by profiling the execution of the HLOs.
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  std::unique_ptr<GpuHloCostAnalysis> analysis_;

  GpuPerformanceModelBaseTest() {
    options_.count_multiple_input_accesses = true;
    analysis_ = std::make_unique<GpuHloCostAnalysis>(options_, device_info_);
  }
};

TEST_F(GpuPerformanceModelBaseTest, SharedOperandBytesAccessed_InPlaceDUS) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = f32[8,16] parameter(0)
  param_1 = f32[4,4] parameter(1)
  c_0 = s32[] constant(0)
  log = f32[4,4] log(param_1)
  ROOT dynamic-update-slice = f32[8,16] dynamic-update-slice(param_0, log, c_0, c_0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(analysis_.get()));

  auto dus_consumer = computation->root_instruction();
  auto log_producer = dus_consumer->mutable_operand(1);

  auto get_shared_operand_bytes_accessed = [&](const HloInstruction* operand) {
    return GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
        analysis_.get(), log_producer, dus_consumer, operand);
  };

  EXPECT_EQ(get_shared_operand_bytes_accessed(dus_consumer->operand(0)), 0);
  EXPECT_EQ(get_shared_operand_bytes_accessed(log_producer->operand(0)), 64);
}

TEST_F(GpuPerformanceModelBaseTest, SharedOperandBytesAccessed_DUS) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = f32[8,16] parameter(0)
  param_1 = f32[4,4] parameter(1)
  c_0 = s32[] constant(0)
  log = f32[8,16] log(param_0)
  ROOT dynamic-update-slice = f32[8,16] dynamic-update-slice(log, param_1, c_0, c_0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(analysis_.get()));

  auto dus_consumer = computation->root_instruction();
  auto log_producer = dus_consumer->mutable_operand(0);

  auto get_shared_operand_bytes_accessed = [&](const HloInstruction* operand) {
    return GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
        analysis_.get(), log_producer, dus_consumer, operand);
  };

  EXPECT_EQ(get_shared_operand_bytes_accessed(dus_consumer->operand(1)), 64);
  EXPECT_EQ(get_shared_operand_bytes_accessed(log_producer->operand(0)), 448);
}

// This test documents current behaviour. See comments below how the correct
// result should look like.
TEST_F(GpuPerformanceModelBaseTest,
       ReduceBroadcastedDim_IncorrectBytesAccessed) {
  absl::string_view hlo_string = R"(
HloModule m

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

f1 {
  p0 = f32[128] parameter(0)
  c0 = f32[] constant(0)
  broadcast = f32[128,256] broadcast(p0), dimensions={0}
  ROOT reduce = f32[128] reduce(broadcast, c0), dimensions={1}, to_apply=add
}

ENTRY entry_computation {
  param_0 = f32[128] parameter(0)
  ROOT fusion = f32[128] fusion(param_0), kind=kLoop, calls=f1
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(analysis_.get()));

  auto root = computation->root_instruction();

  // Cost Model estimates that input element we be re-read in reduce. Each
  // element of reduce output needs only one input element. Bytes accessed
  // should be 4*128=512.
  EXPECT_EQ(GpuPerformanceModelBase::GetOperandBytesAccessed(
                analysis_.get(), root, root->operand(0)),
            /*4*128*256=*/131072);
}

TEST_F(GpuPerformanceModelBaseTest,
       GetOperandBytesAccessedReturnsZeroForUnusedOperand) {
  absl::string_view hlo_string = R"(
HloModule m

f1 {
  p0 = f32[128] parameter(0)
  ROOT reduce = f32[128] exponential(p0)
}

ENTRY entry_computation {
  p0 = f32[128] parameter(0)
  p1 = f32[128] parameter(1)
  fusion = f32[128] fusion(p0), kind=kLoop, calls=f1
  ROOT add = f32[128] add(p1, fusion)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(analysis_.get()));

  HloInstruction* root = computation->root_instruction();
  HloInstruction* p1 = root->mutable_operand(0);
  HloInstruction* fusion = root->mutable_operand(1);

  EXPECT_EQ(GpuPerformanceModelBase::GetOperandBytesAccessed(analysis_.get(),
                                                             fusion, p1),
            0);
}

// This test documents current behaviour. See comments below how the correct
// result should look like.
TEST_F(GpuPerformanceModelBaseTest, ElementwiseBitcast_IncorrectBytesAccessed) {
  absl::string_view hlo_string = R"(
HloModule m

f1 {
  p0 = f32[128] parameter(0)
  bitcast.1 = f32[8,16] bitcast(p0)
  log = f32[128] log(p0)
  bitcast.2 = f32[8,16] bitcast(log)
  ROOT add = f32[8,16] add(bitcast.1, bitcast.2)
}

ENTRY entry_computation {
  param_0 = f32[128] parameter(0)
  ROOT fusion = f32[8,16] fusion(param_0), kind=kLoop, calls=f1
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(analysis_.get()));

  auto root = computation->root_instruction();

  // Bitcast breaks the chain of elementwise utilization even if the bitcast
  // doesn't change physical layout. Each element of `param_0` should be read
  // only once, but Cost Model estimates that it will be accessed twice. Bytes
  // accessed should be 4*128=512.
  EXPECT_EQ(GpuPerformanceModelBase::GetOperandBytesAccessed(
                analysis_.get(), root, root->operand(0)),
            /*2*4*128=*/1024);
}

TEST_F(GpuPerformanceModelBaseTest, EstimateFusionLaunchDimensions_LoopFusion) {
  absl::string_view hlo_string = R"(
HloModule m

f1 {
  p0 = f32[8,16,128] parameter(0)
  log = f32[8,16,128] log(p0)
  ROOT add = f32[8,16,128] add(p0, log)
}

ENTRY entry_computation {
  param_0 = f32[8,16,128] parameter(0)
  ROOT fusion = f32[8,16,128] fusion(param_0), kind=kLoop, calls=f1
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  auto fusion_analysis = HloFusionAnalysis::Create(
      *module->entry_computation()->root_instruction(), device_info_);
  auto launch_dimensions =
      GpuPerformanceModelBase::EstimateFusionLaunchDimensions(fusion_analysis);

  EXPECT_EQ(launch_dimensions.num_blocks(), 128);
  EXPECT_EQ(launch_dimensions.num_threads_per_block(), 128);
}

TEST_F(GpuPerformanceModelBaseTest,
       EstimateFusionLaunchDimensions_TritonSoftMaxFusion) {
  absl::string_view hlo_string = R"(
max {
  p1 = f32[] parameter(1)
  p0 = f32[] parameter(0)
  ROOT m = f32[] maximum(p0, p1)
}

triton_softmax_computation {
  p0 = f32[16,970] parameter(0)
  constant = f32[] constant(-inf)
  reduce = f32[16] reduce(p0, constant), dimensions={1}, to_apply=max
  broadcast = f32[16,970] broadcast(reduce), dimensions={0}
  ROOT subtract = f32[16,970] subtract(p0, broadcast)
}

ENTRY e {
  p0 = f32[16,970]{1,0} parameter(0)
  ROOT r = f32[16,970]{1,0} fusion(p0), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={"fusion_backend_config": {kind: "__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1","970"]}],"num_warps":"2"}}}
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  auto fusion_analysis = HloFusionAnalysis::Create(
      *module->entry_computation()->root_instruction(), device_info_);
  auto launch_dimensions =
      GpuPerformanceModelBase::EstimateFusionLaunchDimensions(fusion_analysis);

  EXPECT_EQ(launch_dimensions.num_blocks(), 16);
  EXPECT_EQ(launch_dimensions.num_threads_per_block(), 64);
}

TEST_F(GpuPerformanceModelBaseTest,
       EstimateFusionLaunchDimensions_CudnnFusion) {
  absl::string_view hlo_string = R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,256] parameter(1)
  ROOT r = f32[32,256] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,256] parameter(1)
  ROOT _ = f32[32,256] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  auto fusion_analysis = HloFusionAnalysis::Create(
      *module->entry_computation()->root_instruction(), device_info_);
  auto launch_dimensions =
      GpuPerformanceModelBase::EstimateFusionLaunchDimensions(fusion_analysis);

  // CuNnnFusion doesn't implement KernelLaunchInsterface, so
  // EstimateFusionLaunchDimensions returns a default estimate.
  EXPECT_EQ(launch_dimensions.num_blocks(), 64);
  EXPECT_EQ(launch_dimensions.num_threads_per_block(), 128);
}

TEST_F(GpuPerformanceModelBaseTest,
       CalculateEffectiveFlopsPerNsForFullOccupancyH100) {
  se::DeviceDescription h100_device_info =
      TestGpuDeviceInfo::H100SXMDeviceInfo();
  int64_t flops_per_ns = GpuPerformanceModelBase::CalculateEffectiveFlopsPerNs(
      h100_device_info, /*num_blocks=*/h100_device_info.core_count(),
      /*num_threads_per_block=*/h100_device_info.fpus_per_core());
  // H100 has a peak of 66.9 TFLOPS/s for TF32.
  EXPECT_GT(flops_per_ns, 66000);
  EXPECT_LT(flops_per_ns, 68000);
}

TEST_F(GpuPerformanceModelBaseTest, CalculatePeakBF16OpsPerNsH100) {
  se::DeviceDescription h100_device_info =
      TestGpuDeviceInfo::H100SXMDeviceInfo();
  int64_t flops_per_ns = GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(
      h100_device_info, xla::PrimitiveType::BF16);
  // H100 has a peak of 989.4 TFLOPS/s for BF16.
  EXPECT_GT(flops_per_ns, 988000);
  EXPECT_LT(flops_per_ns, 991000);
}

TEST_F(GpuPerformanceModelBaseTest, CalculatePeakF64OpsPerNsH100) {
  se::DeviceDescription h100_device_info =
      TestGpuDeviceInfo::H100SXMDeviceInfo();
  int64_t flops_per_ns = GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(
      h100_device_info, xla::PrimitiveType::F64);
  // H100 has a peak of 66.8 TFLOPS/s for FP64.
  EXPECT_GT(flops_per_ns, 66000);
  EXPECT_LT(flops_per_ns, 68000);
}

// MI300X MFMA peaks below are from the CDNA 3 white paper.
TEST_F(GpuPerformanceModelBaseTest, CalculatePeakBF16OpsPerNsMI300X) {
  se::DeviceDescription mi300_device_info =
      TestGpuDeviceInfo::AMDMI300DeviceInfo();
  int64_t flops_per_ns = GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(
      mi300_device_info, xla::PrimitiveType::BF16);
  // MI300X has a peak of 1307.4 TFLOPS/s for BF16.
  EXPECT_GT(flops_per_ns, 1300000);
  EXPECT_LT(flops_per_ns, 1315000);
}

TEST_F(GpuPerformanceModelBaseTest, CalculatePeakFP8OpsPerNsMI300X) {
  se::DeviceDescription mi300_device_info =
      TestGpuDeviceInfo::AMDMI300DeviceInfo();
  int64_t flops_per_ns = GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(
      mi300_device_info, xla::PrimitiveType::F8E4M3);
  // MI300X has a peak of 2614.9 TFLOPS/s for FP8.
  EXPECT_GT(flops_per_ns, 2600000);
  EXPECT_LT(flops_per_ns, 2620000);
}

// Guards against matrix_unit_description silently going unset, in which case
// CalculatePeakMatrixOpsPerNs degrades to the vector FP32 peak instead of
// failing, and the ratio below collapses to 1.
TEST_F(GpuPerformanceModelBaseTest, MFMAPathExceedsScalarFallbackMI300X) {
  se::DeviceDescription mi300_device_info =
      TestGpuDeviceInfo::AMDMI300DeviceInfo();
  int64_t mfma_bf16 = GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(
      mi300_device_info, xla::PrimitiveType::BF16);
  int64_t scalar_fallback =
      GpuPerformanceModelBase::CalculateEffectiveFlopsPerNs(
          mi300_device_info, /*num_blocks=*/mi300_device_info.core_count(),
          /*num_threads_per_block=*/mi300_device_info.fpus_per_core());
  // 1307.4 TF against 163.4 TF, so 5x leaves ample margin.
  EXPECT_GT(mfma_bf16, 5 * scalar_fallback);
}

TEST_F(GpuPerformanceModelBaseTest, RecordEstimatedRunTimeWithName) {
  EstimateRunTimeData data = {/*flops=*/100,
                              /*bytes_read=*/200,
                              /*bytes_written=*/300,
                              /*read_time=*/absl::Microseconds(10),
                              /*write_time=*/absl::Microseconds(5),
                              /*compute_time=*/absl::Microseconds(50),
                              /*exec_time=*/absl::Microseconds(60)};

  ReificationCost cost =
      GpuPerformanceModelBase::MakeReificationCostFromRuntime(
          data, device_info_, "test-model");

  EXPECT_EQ(cost.name(), "test-model");
  EXPECT_DOUBLE_EQ(cost.compute_time_us(), 50.0);
  EXPECT_DOUBLE_EQ(cost.memory_access_time_us(), 15.0);
  EXPECT_DOUBLE_EQ(cost.exec_time_us(), 60.0);
  EXPECT_DOUBLE_EQ(cost.end_to_end_cycles(),
                   absl::ToDoubleNanoseconds(absl::Microseconds(60)) *
                       device_info_.clock_rate_ghz());
}

TEST_F(GpuPerformanceModelBaseTest, RecordEstimatedRunTimeWithoutName) {
  ReificationCost cost =
      GpuPerformanceModelBase::MakeReificationCostFromRuntime(
          EstimateRunTimeData{}, device_info_);

  EXPECT_TRUE(cost.name().empty());
}

TEST_F(GpuPerformanceModelBaseTest,
       CacheContainsConsumersReturnsFalseWhenAllConsumersInvalidated) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = f32[8,16] parameter(0)
  log = f32[8,16] log(param_0)
  exp = f32[8,16] exponential(log)
  neg = f32[8,16] negate(log)
  ROOT root = tuple(exp, neg)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto* producer = module->entry_computation()->GetInstructionWithName("log");
  auto* consumer_1 = module->entry_computation()->GetInstructionWithName("exp");
  auto* consumer_2 = module->entry_computation()->GetInstructionWithName("neg");

  GpuPerformanceModelCache cache;
  EXPECT_FALSE(cache.ContainsConsumers(*producer));

  cache.Set(*producer, *consumer_1, absl::Microseconds(10));
  cache.Set(*producer, *consumer_2, absl::Microseconds(20));
  EXPECT_TRUE(cache.ContainsConsumers(*producer));
  EXPECT_EQ(cache.GetAllConsumers(*producer).size(), 2);

  // Invalidate first consumer: producer still has consumer_2.
  cache.Invalidate(*consumer_1);
  EXPECT_TRUE(cache.ContainsConsumers(*producer));
  EXPECT_EQ(cache.GetAllConsumers(*producer).size(), 1);

  // Invalidate second (and final) consumer: producer now has no consumers in
  // cache.
  cache.Invalidate(*consumer_2);
  EXPECT_FALSE(cache.ContainsConsumers(*producer));
  EXPECT_TRUE(cache.GetAllConsumers(*producer).empty());

  // Re-populating and invalidating producer itself also resets
  // ContainsConsumers.
  cache.Set(*producer, *consumer_1, absl::Microseconds(10));
  EXPECT_TRUE(cache.ContainsConsumers(*producer));
  cache.Invalidate(*producer);
  EXPECT_FALSE(cache.ContainsConsumers(*producer));
}

// The occupancy model on an H100 SXM, with the kernel used throughout the
// model's design notes: 16 MiB of f32 read from one input, four elements per
// thread, so each warp keeps four 128-byte requests in flight.
class OccupancyRunTimeEstimatorTest : public ::testing::Test {
 protected:
  static constexpr int64_t kBytes = int64_t{16} << 20;

  se::DeviceDescription device_info_{TestGpuDeviceInfo::H100SXMDeviceInfo()};
  OccupancyModelParams params_;

  OccupancyRunTimeEstimator MakeEstimator(int64_t num_blocks,
                                          int64_t threads_per_block,
                                          KernelOccupancyHints hints = {
                                              .elements_per_thread = 4}) {
    return OccupancyRunTimeEstimator(
        device_info_,
        EstimateOccupancy(device_info_,
                          LaunchDimensions(num_blocks, threads_per_block),
                          hints, params_),
        params_);
  }

  double ReadMicros(int64_t num_blocks, int64_t threads_per_block,
                    KernelOccupancyHints hints = {.elements_per_thread = 4}) {
    OccupancyRunTimeEstimator estimator =
        MakeEstimator(num_blocks, threads_per_block, hints);
    estimator.AddRead(kBytes, kBytes, /*hbm_bandwidth_utilization_rate=*/1.0);
    return absl::ToDoubleMicroseconds(estimator.read_time());
  }
};

TEST_F(OccupancyRunTimeEstimatorTest, NarrowBlocksGetLittleBandwidth) {
  // The default model credits any launch of 53 or more blocks the full HBM
  // bandwidth, whatever the block size.
  EXPECT_NEAR(absl::ToDoubleMicroseconds(
                  GpuPerformanceModelBase::ReadTimeWithDRAMHeuristic(
                      device_info_, /*num_blocks=*/53, kBytes, kBytes, F32,
                      /*hbm_bandwidth_utilization_rate=*/1.0)),
              5.0, 0.1);

  // 53 one-warp blocks keep 27 KB in flight: about 45 GB/s.
  EXPECT_NEAR(ReadMicros(53, 32), 371, 4);
  // 53 blocks of 32 warps keep 868 KB in flight: about 1,450 GB/s.
  EXPECT_NEAR(ReadMicros(53, 1024), 11.6, 0.2);
  // 32 warps on a single SM: about 27 GB/s.
  EXPECT_NEAR(ReadMicros(1, 1024), 614, 6);
}

TEST_F(OccupancyRunTimeEstimatorTest, TrailingWaveCostsNearlyAFullWave) {
  // 1024 threads at 64 registers use the whole register file: one block per
  // SM, so 132 blocks fill the chip exactly and saturate 90% of peak.
  KernelOccupancyHints hints{.registers_per_thread = 64,
                             .elements_per_thread = 4};
  EXPECT_NEAR(ReadMicros(132, 1024, hints), 5.6, 0.1);
  // The 133rd block runs alone in a second wave at one SM's bandwidth.
  EXPECT_NEAR(ReadMicros(133, 1024, hints), 10.1, 0.2);
}

TEST_F(OccupancyRunTimeEstimatorTest, SmallBlocksPackWithoutACliff) {
  // Sixteen 128-thread blocks fit per SM, so 133 blocks are still one wave.
  double t_132 = ReadMicros(132, 128);
  double t_133 = ReadMicros(133, 128);
  EXPECT_NEAR(t_132, 37.2, 0.5);
  EXPECT_NEAR(t_133, t_132, 0.5);
}

TEST_F(OccupancyRunTimeEstimatorTest, IssueRateFollowsResidentWarps) {
  constexpr int64_t kFlopsPerBlock = 41984;
  // 2112 blocks of 128 threads: 16 per SM, 64 warps, full issue rate.
  OccupancyRunTimeEstimator full = MakeEstimator(2112, 128);
  full.AddCompute(2112 * kFlopsPerBlock);
  EXPECT_NEAR(absl::ToDoubleMicroseconds(full.compute_time()), 1.33, 0.01);

  // 132 one-warp blocks: one warp per SM, a sixteenth of the issue rate, so
  // the same time for a sixteenth of the work.
  OccupancyRunTimeEstimator narrow = MakeEstimator(132, 32);
  narrow.AddCompute(132 * kFlopsPerBlock);
  EXPECT_NEAR(absl::ToDoubleMicroseconds(narrow.compute_time()), 1.33, 0.01);

  // The default model credits a 32-thread block a quarter of the SM.
  EXPECT_NEAR(absl::ToDoubleMicroseconds(GpuPerformanceModelBase::ComputeTime(
                  device_info_, 132 * kFlopsPerBlock, /*num_blocks=*/132,
                  /*num_threads_per_block=*/32)),
              0.33, 0.01);
}

TEST_F(OccupancyRunTimeEstimatorTest, OverlapFollowsResidentWarps) {
  const absl::Duration ten_us = absl::Microseconds(10);
  double one_warp = GpuPerformanceModelBase::kMemoryComputeParallelism *
                    OverlapFraction(1, params_);
  EXPECT_NEAR(absl::ToDoubleMicroseconds(
                  GpuPerformanceModelBase::CombineComputeAndMemoryAccessTime(
                      ten_us, ten_us, one_warp)),
              19.4, 0.1);
  double many_warps = GpuPerformanceModelBase::kMemoryComputeParallelism *
                      OverlapFraction(32, params_);
  EXPECT_NEAR(absl::ToDoubleMicroseconds(
                  GpuPerformanceModelBase::CombineComputeAndMemoryAccessTime(
                      ten_us, ten_us, many_warps)),
              10.5, 0.1);
}

TEST_F(OccupancyRunTimeEstimatorTest, WaveTakesAtLeastItsDependentRoundTrips) {
  // 64 loads per thread with 8 in flight: 8 round trips of 600 ns, however
  // few bytes they move.
  OccupancyRunTimeEstimator estimator =
      MakeEstimator(1, 128, {.elements_per_thread = 64});
  estimator.AddRead(1024, 1024, /*hbm_bandwidth_utilization_rate=*/1.0);
  EXPECT_LT(absl::ToDoubleMicroseconds(estimator.read_time()), 1.0);
  EXPECT_NEAR(absl::ToDoubleMicroseconds(estimator.exec_time()), 4.8, 0.01);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
