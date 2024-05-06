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

#ifndef XLA_SERVICE_GPU_RUNTIME_FUSED_MHA_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_FUSED_MHA_THUNK_H_

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_fused_mha_runner.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class FusedMHABackwardThunk : public Thunk {
 public:
  // Constructs a thunk for launching a DNN FMHA backward.
  FusedMHABackwardThunk(ThunkInfo thunk_info, GpufMHABackwardConfig config,
                        BufferAllocation::Slice bmm1_grad_gemm1_rhs_slice,
                        BufferAllocation::Slice bmm1_grad_gemm2_rhs_slice,
                        BufferAllocation::Slice bmm2_grad_gemm1_lhs_slice,
                        BufferAllocation::Slice bmm2_grad_gemm2_rhs_slice,
                        BufferAllocation::Slice d_output_slice,
                        BufferAllocation::Slice scratch_slice,
                        BufferAllocation::Slice d_bmm1_lhs_slice,
                        BufferAllocation::Slice d_bmm1_rhs_slice,
                        BufferAllocation::Slice d_bmm2_rhs_slice,
                        BufferAllocation::Slice d_s_slice,
                        BufferAllocation::Slice mask_slice,
                        BufferAllocation::Slice d_bias_slice,
                        BufferAllocation::Slice fwd_output_slice,
                        BufferAllocation::Slice bias_slice,
                        BufferAllocation::Slice seqlen_q_slice,
                        BufferAllocation::Slice seqlen_k_slice);

  FusedMHABackwardThunk(const FusedMHABackwardThunk&) = delete;
  FusedMHABackwardThunk& operator=(const FusedMHABackwardThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  BufferAllocation::Slice bmm1_grad_gemm1_rhs_buffer_;
  BufferAllocation::Slice bmm1_grad_gemm2_rhs_buffer_;
  BufferAllocation::Slice bmm2_grad_gemm1_lhs_buffer_;
  BufferAllocation::Slice bmm2_grad_gemm2_rhs_buffer_;
  BufferAllocation::Slice d_output_buffer_;
  BufferAllocation::Slice scratch_buffer_;
  BufferAllocation::Slice d_bmm1_lhs_buffer_;
  BufferAllocation::Slice d_bmm1_rhs_buffer_;
  BufferAllocation::Slice d_bmm2_rhs_buffer_;
  BufferAllocation::Slice d_s_buffer_;
  BufferAllocation::Slice d_bias_buffer_;
  BufferAllocation::Slice fwd_output_buffer_;
  BufferAllocation::Slice bias_buffer_;
  BufferAllocation::Slice seqlen_q_buffer_;
  BufferAllocation::Slice seqlen_k_buffer_;

  FusedMultiHeadedAttentionBackwardRunner& GetOrCreateRunner(
      const stream_executor::Stream* stream);

  // FusedMHA backward config
  const GpufMHABackwardConfig config_;
  absl::Mutex mu_;
  absl::flat_hash_map<const stream_executor::Stream*,
                      std::unique_ptr<FusedMultiHeadedAttentionBackwardRunner>>
      runner_cache_ ABSL_GUARDED_BY(mu_);
};
}  // namespace gpu
}  // namespace xla
#endif  // XLA_SERVICE_GPU_RUNTIME_FUSED_MHA_THUNK_H_
