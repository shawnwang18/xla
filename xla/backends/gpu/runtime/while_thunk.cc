/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/while_thunk.h"

#include <cstdint>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/host_memory_pool.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

#ifndef VDBG
#define VDBG() VLOG(0) << "while_thunk.cc:" << __LINE__ << " " << __func__
#endif

namespace xla {
namespace gpu {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

struct RunningLoop {
  const HloInstruction* loop_instr;
  int64_t counter;
};

static std::list<RunningLoop>& RunningLoops() {
  // TODO(b/343294327): Do not rely on thread-local storage.
  static thread_local std::list<RunningLoop> loops;
  VDBG();
  return loops;
}

bool WhileThunk::RunningWhileThunkLoop() { VDBG(); return RunningLoops().size() > 0; }

absl::StatusOr<int64_t> WhileThunk::CurrentLoopIteration(int64_t depth) {
  if (depth >= RunningLoops().size()) {
    VDBG();
    return absl::InvalidArgumentError(absl::StrFormat(
        "Loop depth %d is greater than the number of tracked loops %d", depth,
        RunningLoops().size()));
  }

  auto loop = RunningLoops().begin();
  VDBG();
  std::advance(loop, depth);
  VDBG();
  return loop->counter;
}

absl::StatusOr<int64_t> WhileThunk::CurrentLoopIteration(
    const HloInstruction* while_instr) {
  for (const auto& loop : RunningLoops()) {
    VDBG();
    if (loop.loop_instr == while_instr) {
      VDBG();
      return loop.counter;
    }
    VDBG();
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Loop %s is not currently running", while_instr->name()));
}

WhileThunk::WhileThunk(
    ThunkInfo thunk_info, const HloInstruction* loop,
    const BufferAllocation::Slice& condition_result_buffer_index,
    std::unique_ptr<SequentialThunk> condition_thunk_sequence,
    std::unique_ptr<SequentialThunk> body_thunk_sequence,
    std::optional<int64_t> trip_count)
    : Thunk(Kind::kWhile, thunk_info),
      loop_(loop),
      condition_result_buffer_index_(condition_result_buffer_index),
      condition_thunk_sequence_(std::move(condition_thunk_sequence)),
      body_thunk_sequence_(std::move(body_thunk_sequence)),
      trip_count_(trip_count) { VDBG(); }

absl::Status WhileThunk::Prepare(const PrepareParams& params,
                                 ResourceRequestsInterface& resource_requests) {
  TF_RETURN_IF_ERROR(
      condition_thunk_sequence_->Prepare(params, resource_requests));
  VDBG();
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Prepare(params, resource_requests));
  VDBG();
  return absl::OkStatus();
}

absl::Status WhileThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(condition_thunk_sequence_->Initialize(params));
  VDBG();
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Initialize(params));
  VDBG();

  absl::MutexLock lock(mutex_);
  VDBG();
  if (!host_memory_pools_.contains(params.executor)) {
    VDBG();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HostMemoryPool> pool,
        HostMemoryPool::Create(params.executor, PrimitiveType::PRED));
    VDBG();
    host_memory_pools_[params.executor] = std::move(pool);
    VDBG();
  }
  return absl::OkStatus();
}

absl::Status WhileThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;
  VDBG();

  RunningLoop& loop = RunningLoops().emplace_front();
  VDBG();
  loop.loop_instr = loop_;
  VDBG();
  int64_t& iter = loop.counter;
  VDBG();
  absl::Cleanup cleanup = [&] { RunningLoops().pop_front(); };
  VDBG();

  int device_ordinal = stream.parent()->device_ordinal();
  VDBG();
  if (trip_count_.has_value()) {
    VDBG();
    VLOG(2) << "[" << device_ordinal << "] Executing WhileThunk for "
            << *trip_count_ << " iterations";
    VDBG();
    for (iter = 0; iter < trip_count_; ++iter) {
      VDBG();
      VLOG(3) << "[" << device_ordinal << "] Executing iteration # " << iter
              << " (Device: " << stream.parent()->device_ordinal() << ")";
      VDBG();
      TF_RETURN_IF_ERROR(body_thunk_sequence_->ExecuteOnStream(params));
      VDBG();
    }
    VDBG();
    return absl::OkStatus();
  }

  HostMemoryPool* pool;
  VDBG();
  {
    absl::MutexLock lock(mutex_);
    VDBG();
    pool = host_memory_pools_.at(stream.parent()).get();
    VDBG();
  }
  VDBG();
  TF_ASSIGN_OR_RETURN(HostMemoryPool::Handle handle, pool->Acquire());
  VDBG();
  bool* condition_result = handle.get<bool>();
  VDBG();
  se::DeviceMemoryBase condition_result_data =
      params.buffer_allocations->GetDeviceAddress(
          condition_result_buffer_index_);
  VDBG();

  while (true) {
    VDBG();
    TraceMe trace(
        [&] { return TraceMeEncode("While", {{"iteration:", iter}}); });
    VDBG();
    VLOG(3) << "[" << device_ordinal
            << "] Executing WhileThunk condition computation; iter=" << iter;
    VDBG();
    TF_RETURN_IF_ERROR(condition_thunk_sequence_->ExecuteOnStream(params));
    VDBG();

    // Copy the result of condition computation and break the loop if 'false'.
    TF_RETURN_IF_ERROR(
        stream.Memcpy(condition_result, condition_result_data, sizeof(bool)));
    VDBG();

    if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
      VDBG();
      return absl::InternalError(absl::StrFormat(
          "Failed to complete all kernels launched on stream %p: %s", &stream,
          blocked.message()));
    }
    VDBG();

    VLOG(3) << "[" << device_ordinal
            << "] condition_result = " << *condition_result;
    VDBG();
    if (!*condition_result) {
      VDBG();
      VLOG(3) << "[" << device_ordinal
              << "] Break WhileThunk loop; iter=" << iter;
      VDBG();
      break;
    }
    VDBG();

    VLOG(3) << "[" << device_ordinal
            << "] Executing WhileThunk body computation; iter=" << iter
            << " (Device: " << stream.parent()->device_ordinal() << ")";
    VDBG();
    TF_RETURN_IF_ERROR(body_thunk_sequence_->ExecuteOnStream(params));
    VDBG();
    ++iter;
    VDBG();
  }
  return absl::OkStatus();
}

void WhileThunk::ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
  VDBG();
  condition_thunk_sequence_->ForAllThunks(fn);
  VDBG();
  body_thunk_sequence_->ForAllThunks(fn);
  VDBG();
}

void WhileThunk::ForAllThunksMutable(absl::FunctionRef<void(Thunk*)> fn) {
  fn(this);
  VDBG();
  condition_thunk_sequence_->ForAllThunksMutable(fn);
  VDBG();
  body_thunk_sequence_->ForAllThunksMutable(fn);
  VDBG();
}

std::string WhileThunk::ToString(int indent) const {
  std::string indent_str(indent * 2, ' ');
  VDBG();
  std::string result;
  VDBG();
  absl::StrAppend(&result, indent_str, "\ncondition:\n");
  VDBG();
  absl::StrAppend(&result, condition_thunk_sequence_->ToString(indent + 1));
  VDBG();
  absl::StrAppend(&result, indent_str, "body:\n");
  VDBG();
  absl::StrAppend(&result, body_thunk_sequence_->ToString(indent + 1));
  VDBG();
  return result;
}

absl::StatusOr<ThunkProto> WhileThunk::ToProto() const {
  ThunkProto proto;
  VDBG();
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  VDBG();

  auto* while_proto = proto.mutable_while_thunk();
  VDBG();
  TF_ASSIGN_OR_RETURN(*while_proto->mutable_condition_result_buffer_index(),
                      condition_result_buffer_index_.ToProto());
  VDBG();

  if (condition_thunk_sequence_) {
    VDBG();
    TF_ASSIGN_OR_RETURN(ThunkProto thunk_proto,
                        condition_thunk_sequence_->ToProto());
    VDBG();
    *while_proto->mutable_condition_thunk_sequence() =
        thunk_proto.sequential_thunk();
    VDBG();
  }
  VDBG();

  if (body_thunk_sequence_) {
    VDBG();
    TF_ASSIGN_OR_RETURN(ThunkProto thunk_proto,
                        body_thunk_sequence_->ToProto());
    VDBG();
    *while_proto->mutable_body_thunk_sequence() =
        thunk_proto.sequential_thunk();
    VDBG();
  }
  VDBG();

  if (trip_count_.has_value()) {
    VDBG();
    while_proto->set_trip_count(*trip_count_);
    VDBG();
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<WhileThunk>> WhileThunk::FromProto(
    ThunkInfo thunk_info, const WhileThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const Deserializer& deserializer) {
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice condition_result_buffer_index,
      BufferAllocation::Slice::FromProto(
          thunk_proto.condition_result_buffer_index(), buffer_allocations));
  VDBG();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<SequentialThunk> condition_thunk_sequence,
      SequentialThunk::FromProto(
          thunk_info, thunk_proto.condition_thunk_sequence(), deserializer));
  VDBG();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<SequentialThunk> body_thunk_sequence,
      SequentialThunk::FromProto(thunk_info, thunk_proto.body_thunk_sequence(),
                                 deserializer));
  VDBG();
  std::optional<int64_t> trip_count;
  VDBG();
  if (thunk_proto.has_trip_count()) {
    VDBG();
    trip_count = thunk_proto.trip_count();
    VDBG();
  }
  return std::make_unique<WhileThunk>(
      std::move(thunk_info), /*loop=*/nullptr, condition_result_buffer_index,
      std::move(condition_thunk_sequence), std::move(body_thunk_sequence),
      trip_count);
}

}  // namespace gpu
}  // namespace xla
