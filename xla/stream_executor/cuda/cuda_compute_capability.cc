/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_compute_capability.h"

#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"

namespace stream_executor {

absl::StatusOr<CudaComputeCapability> CudaComputeCapability::FromString(
    absl::string_view cuda_arch_name) {
  std::vector<absl::string_view> split = absl::StrSplit(cuda_arch_name, '.');
  if (split.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid CUDA architecture name: ", cuda_arch_name));
  }

  FeatureExtension feature_extension = FeatureExtension::kNone;
  if (!split[1].empty() && (split[1].back() == 'a' || split[1].back() == 'A')) {
    feature_extension = FeatureExtension::kAcceleratedFeatures;
    split[1].remove_suffix(1);
  }

  if (!split[1].empty() && (split[1].back() == 'f' || split[1].back() == 'F')) {
    feature_extension = FeatureExtension::kFamilyCompatibleFeatures;
    split[1].remove_suffix(1);
  }

  int major, minor;
  if (!absl::SimpleAtoi(split[0], &major) ||
      !absl::SimpleAtoi(split[1], &minor)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid CUDA architecture name: ", cuda_arch_name));
  }
  return CudaComputeCapability{major, minor, feature_extension};
}

absl::StatusOr<CudaComputeCapability>
CudaComputeCapability::FromPtxAsTargetName(absl::string_view cuda_arch_name) {
  const auto invalid_arg = [name = cuda_arch_name](absl::string_view phase) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid CUDA architecture name: %s [%s]", name, phase));
  };
  if (!absl::ConsumePrefix(&cuda_arch_name, "sm_")) {
    return invalid_arg("prefix");
  }
  FeatureExtension feature_extension = FeatureExtension::kNone;
  if (absl::ConsumeSuffix(&cuda_arch_name, "a")) {
    feature_extension = FeatureExtension::kAcceleratedFeatures;
  } else if (absl::ConsumeSuffix(&cuda_arch_name, "f")) {
    feature_extension = FeatureExtension::kFamilyCompatibleFeatures;
  }
  int minor;
  constexpr auto minor_size = 1;
  if (cuda_arch_name.size() < minor_size) {
    return invalid_arg("minor");
  }
  if (auto minor_str =
          cuda_arch_name.substr(cuda_arch_name.size() - minor_size, minor_size);
      !absl::SimpleAtoi(minor_str, &minor)) {
    return invalid_arg(absl::StrCat("minor=", minor_str));
  }
  cuda_arch_name.remove_suffix(minor_size);
  int major;
  if (!absl::SimpleAtoi(cuda_arch_name, &major) || major < 0) {
    return invalid_arg(absl::StrCat("major=", cuda_arch_name));
  }
  return CudaComputeCapability{major, minor, feature_extension};
}

static std::string FeatureExtensionToString(
    CudaComputeCapability::FeatureExtension feature_extension) {
  switch (feature_extension) {
    case CudaComputeCapability::FeatureExtension::kNone:
      return "";
    case CudaComputeCapability::FeatureExtension::kAcceleratedFeatures:
      return "a";
    case CudaComputeCapability::FeatureExtension::kFamilyCompatibleFeatures:
      return "f";
  }
}

std::string CudaComputeCapability::ToString() const {
  return absl::StrCat(major, ".", minor,
                      FeatureExtensionToString(feature_extension));
}

std::string CudaComputeCapability::GetPtxAsTargetName(
    CompileMode compile_mode) const {
  absl::string_view prefix = [&]() {
    switch (compile_mode) {
      case CompileMode::kPtx:
        return "compute";
      case CompileMode::kLto:
        return "lto";
      case CompileMode::kSass:
        return "sm";
    }
  }();
  return absl::StrFormat("%s_%d%d%s", prefix, major, minor,
                         FeatureExtensionToString(feature_extension));
}

std::string CudaComputeCapability::GetHighestKnownCompatiblePtxAsTargetName()
    const {
  auto gpu_compute_capability = *this;
  gpu_compute_capability.feature_extension =
      CudaComputeCapability::FeatureExtension::kNone;
  // If the current compute capability isn't known, fallback to the
  // most recent version before it.
  constexpr stream_executor::CudaComputeCapability kSupportedVersions[] = {
      {12, 1}, {12, 0}, {11, 0}, {10, 3}, {10, 0}, {9, 0}, {8, 9}, {8, 7},
      {8, 6},  {8, 0},  {7, 5},  {7, 2},  {7, 0},  {6, 2}, {6, 1}, {6, 0},
      {5, 3},  {5, 2},  {5, 0},  {3, 7},  {3, 5},  {3, 2}, {3, 0}};
  // Initialize to the least supported version, which acts as a safe fallback
  auto target_compute_capability =
      kSupportedVersions[std::size(kSupportedVersions) - 1];

  for (const auto& v : kSupportedVersions) {
    if (gpu_compute_capability.SupportsAllFeaturesOf(v)) {
      // Found the most advanced supported capability
      target_compute_capability = v;
      break;
    }
  }

  if (target_compute_capability.major == gpu_compute_capability.major &&
      target_compute_capability.minor == gpu_compute_capability.minor) {
    // If we support the requested compute capability, then we can also enable
    // the requested feature extension.
    target_compute_capability.feature_extension = feature_extension;
  } else if (target_compute_capability.major >=
                 CudaComputeCapabilities::kBlackwell &&
             target_compute_capability.major <= kSupportedVersions[0].major &&
             target_compute_capability.major == gpu_compute_capability.major &&
             target_compute_capability.minor <= gpu_compute_capability.minor) {
    // If we don't support the requested compute capability, but an
    // earlier one with the same major version, then we can enable
    // the forward compatible feature extension - if the particular
    // major version supports the forward compatible feature
    // extension.
    target_compute_capability.feature_extension =
        CudaComputeCapability::FeatureExtension::kFamilyCompatibleFeatures;
  }

  // If the current CC isn't supported by LLVM and it is newer then
  // the max supported LLVM version, do not warn about it. The end
  // user can't do anything about this. E.g., PTX compiled for SM75 will
  // run on SM80 too.
  if (target_compute_capability != *this &&
      target_compute_capability.major != kSupportedVersions[0].major &&
      target_compute_capability.minor != kSupportedVersions[0].minor) {
    LOG(WARNING)
        << "Unknown compute capability " << ToString()
        << ". Defaulting to telling LLVM that we're compiling for "
        << target_compute_capability.GetPtxAsTargetName(
               stream_executor::CudaComputeCapability::CompileMode::kSass);
  }
  return target_compute_capability.GetPtxAsTargetName(
      stream_executor::CudaComputeCapability::CompileMode::kSass);
}

absl::StatusOr<CudaComputeCapability> CudaComputeCapability::FromProto(
    const CudaComputeCapabilityProto& proto) {
  CudaComputeCapability cc;
  cc.major = proto.major();
  cc.minor = proto.minor();
  switch (proto.feature_extension()) {
    case CudaComputeCapabilityProto::UNSPECIFIED:
      // For backward compatibility we assume sm_90a and sm_100a for Hopper and
      // Blackwell generation GPUs.
      if (cc.major == 9 || cc.major == 10) {
        cc.feature_extension = FeatureExtension::kAcceleratedFeatures;
      } else {
        cc.feature_extension = FeatureExtension::kNone;
      }
      break;
    case CudaComputeCapabilityProto::NONE:
      cc.feature_extension = FeatureExtension::kNone;
      break;
    case CudaComputeCapabilityProto::ACCELERATED_FEATURES:
      cc.feature_extension = FeatureExtension::kAcceleratedFeatures;
      break;
    case CudaComputeCapabilityProto::FAMILY_COMPATIBLE_FEATURES:
      cc.feature_extension = FeatureExtension::kFamilyCompatibleFeatures;
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid feature extension: ", proto.feature_extension()));
  }
  return cc;
}

CudaComputeCapabilityProto CudaComputeCapability::ToProto() const {
  CudaComputeCapabilityProto proto;
  proto.set_major(major);
  proto.set_minor(minor);

  switch (feature_extension) {
    case FeatureExtension::kNone:
      proto.set_feature_extension(CudaComputeCapabilityProto::NONE);
      break;
    case FeatureExtension::kAcceleratedFeatures:
      proto.set_feature_extension(
          CudaComputeCapabilityProto::ACCELERATED_FEATURES);
      break;
    case FeatureExtension::kFamilyCompatibleFeatures:
      proto.set_feature_extension(
          CudaComputeCapabilityProto::FAMILY_COMPATIBLE_FEATURES);
      break;
  }
  return proto;
}

}  // namespace stream_executor
