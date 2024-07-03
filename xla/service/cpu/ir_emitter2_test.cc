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

#include "xla/service/cpu/ir_emitter2.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/IR/LLVMContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/target_machine_features_fake.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_ordering.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

using IrEmitter2Test = HloTestBase;

TEST_F(IrEmitter2Test, BuildKernelPrototype) {
  auto hlo = std::make_unique<HloModule>("test", HloModuleConfig());

  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  auto shape = ShapeUtil::MakeShape(PrimitiveType::F32, {4, 2});

  std::vector<IrEmitter2::KernelParameter> arguments = {{shape}};
  std::vector<IrEmitter2::KernelParameter> results = {{shape}};

  IrEmitter2 ir_emitter(*hlo, module.get(), /*nested_ir_emitter=*/nullptr);
  IrEmitter2::KernelPrototype prototype =
      ir_emitter.EmitKernelPrototype("test", arguments, results);

  ASSERT_TRUE(*RunFileCheck(llvm_ir::DumpToString(module.get()), R"(
    CHECK: define ptr @test(ptr %0) #0 {

    CHECK-NEXT: getelementptr inbounds %SE_HOST_KernelCallFrame, {{.*}} i32 0
    CHECK:      getelementptr inbounds %SE_HOST_KernelThreadDim, {{.*}} i32 0
    CHECK:      getelementptr inbounds %SE_HOST_KernelThreadDim, {{.*}} i32 1
    CHECK:      getelementptr inbounds %SE_HOST_KernelThreadDim, {{.*}} i32 2
    CHECK:      load i64
    CHECK:      load i64
    CHECK:      load i64

    CHECK-NEXT: getelementptr inbounds %SE_HOST_KernelCallFrame, {{.*}} i32 1
    CHECK:      getelementptr inbounds %SE_HOST_KernelThread, {{.*}} i32 0
    CHECK:      getelementptr inbounds %SE_HOST_KernelThread, {{.*}} i32 1
    CHECK:      getelementptr inbounds %SE_HOST_KernelThread, {{.*}} i32 2
    CHECK:      load i64
    CHECK:      load i64
    CHECK:      load i64

    CHECK-NEXT: getelementptr inbounds %SE_HOST_KernelCallFrame, {{.*}} i32 3
    CHECK:      load ptr
    CHECK:      getelementptr %SE_HOST_KernelArg, {{.*}} i32 0, i32 0
    CHECK:      load ptr, {{.*}} !align !0

    CHECK-NEXT: getelementptr inbounds %SE_HOST_KernelCallFrame, {{.*}} i32 3
    CHECK:      load ptr
    CHECK:      getelementptr %SE_HOST_KernelArg, {{.*}} i32 1, i32 0
    CHECK:      load ptr, {{.*}} !align !0

    CHECK:   ret ptr null
    CHECK: }

    CHECK: !0 = !{i64 16}
  )"));
}

TEST_F(IrEmitter2Test, EmitElementalKernel) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      ROOT convert = s32[2,2] convert(p0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  HloInstruction* convert = FindInstruction(hlo.get(), "convert");
  ASSERT_NE(convert, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          hlo.get(), std::make_unique<DependencyHloOrdering>(hlo.get()),
          backend().compiler()->BufferSizeBytesFunction(),
          [](LogicalBuffer::Color) { return /*alignment=*/1; }));

  TargetMachineFeaturesWithFakeAlignmentLogic target_machine(
      [](int64_t size) { return 1; });

  IrEmitter nested_ir_emitter(nullptr, *hlo, *buffer_assignment, module.get(),
                              {}, {}, {}, &target_machine, false);

  IrEmitter2 ir_emitter(*hlo, module.get(), &nested_ir_emitter);
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2::KernelInfo kernel,
                          ir_emitter.EmitElementalHostKernel(convert));

  ASSERT_TRUE(*RunFileCheck(llvm_ir::DumpToString(module.get()), R"(
    CHECK: define ptr @convert(ptr %0) #0 {
    CHECK:   fptosi float {{.*}} to i32
    CHECK: }
  )"));
}

TEST_F(IrEmitter2Test, EmitParallelKernel) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[1,2,1,16384,256] parameter(0)
      ROOT convert = s32[1,2,1,16384,256] convert(p0),
        backend_config={"outer_dimension_partitions":["1","2","1","4"]}
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  HloInstruction* convert = FindInstruction(hlo.get(), "convert");
  ASSERT_NE(convert, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          hlo.get(), std::make_unique<DependencyHloOrdering>(hlo.get()),
          backend().compiler()->BufferSizeBytesFunction(),
          [](LogicalBuffer::Color) { return /*alignment=*/1; }));

  TargetMachineFeaturesWithFakeAlignmentLogic target_machine(
      [](int64_t size) { return 1; });

  IrEmitter nested_ir_emitter(nullptr, *hlo, *buffer_assignment, module.get(),
                              {}, {}, {}, &target_machine, false);

  IrEmitter2 ir_emitter(*hlo, module.get(), &nested_ir_emitter);
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2::KernelInfo kernel,
                          ir_emitter.EmitElementalHostKernel(convert));

  ASSERT_TRUE(*RunFileCheck(llvm_ir::DumpToString(module.get()), R"(
    CHECK: @convert_parallel_bounds = private constant [8 x [4 x [2 x i64]]]

    CHECK: define ptr @convert(ptr %0) #0 {
    CHECK:   %lo_dim_0_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 0, i32 0
    CHECK:   %up_dim_0_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 0, i32 1
    CHECK:   %lo_dim_1_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 1, i32 0
    CHECK:   %up_dim_1_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 1, i32 1
    CHECK:   %lo_dim_2_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 2, i32 0
    CHECK:   %up_dim_2_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 2, i32 1
    CHECK:   %lo_dim_3_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 3, i32 0
    CHECK:   %up_dim_3_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 3, i32 1
    CHECK:   fptosi float {{.*}} to i32
    CHECK: }
  )"));
}

}  // namespace
}  // namespace xla::cpu
