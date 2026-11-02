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

#include "xla/service/gpu/model/hlo_op_profiles.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "tsl/platform/protobuf.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles_data.h"
#include "xla/service/gpu/model/hlo_op_profiles_throughput_data.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

/*static*/ const HloOpProfiles& HloOpProfiles::Singleton() {
  static const auto* hlo_op_profiles = [] {
    // The singleton is shared by every compilation in the process, so the
    // table is selected once from the process-wide flags (XLA_FLAGS).
    const bool use_throughput_profiles =
        GetDebugOptionsFromFlags()
            .xla_gpu_experimental_throughput_op_profiles();
    return HloOpProfiles::Load(
               kDeviceHloOpProfiles,
               /*default_profile_name=*/"sm_86",
               use_throughput_profiles
                   ? absl::string_view(kDeviceHloOpProfilesThroughput)
                   : absl::string_view())
        .release();
  }();
  return *hlo_op_profiles;
}

/*static*/ std::string HloOpProfiles::GetProfileName(
    const se::DeviceDescription& device_info) {
  if (auto* ptr =
          device_info.gpu_compute_capability().cuda_compute_capability()) {
    return absl::StrCat("sm_", ptr->major, ptr->minor);
  }
  if (auto* ptr =
          device_info.gpu_compute_capability().rocm_compute_capability()) {
    return ptr->gfx_version();
  }
  return "<unknown>";
}

/*static*/ std::string HloOpProfiles::GetDeviceSpecificProfileName(
    const se::DeviceDescription& device_info) {
  std::string profile_name = GetProfileName(device_info);
  if (profile_name == "sm_100") {
    CHECK(device_info.name() != se::DeviceDescription::kUndefinedString)
        << "Device name must be set for sm_100 device-specific profiles.";
    std::vector<std::string> full_name =
        absl::StrSplit(device_info.name(), ' ');
    return absl::StrCat(profile_name, "_", full_name.back());
  }
  return profile_name;
}

namespace {

HloOpProfiles::ProfilesNestedMap ParseProfiles(
    absl::string_view profiles_text_proto) {
  HloOpProfiles::ProfilesNestedMap profiles_map;
  DeviceHloInstructionProfiles all_device_profiles;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(profiles_text_proto,
                                                   &all_device_profiles));
  for (const auto& device_profile : all_device_profiles.entries()) {
    for (const auto& entry : device_profile.second.entries()) {
      auto op_code = StringToHloOpcode(entry.instruction().opcode()).value();
      auto element_type = entry.instruction().shape().element_type();
      profiles_map[device_profile.first][std::make_pair(
          op_code, element_type)] = entry.clock_cycles();
    }
  }
  return profiles_map;
}

}  // namespace

/*static*/ std::unique_ptr<HloOpProfiles> HloOpProfiles::Load(
    absl::string_view profiles_text_proto,
    absl::string_view default_profile_name) {
  return Load(profiles_text_proto, default_profile_name,
              /*overrides_text_proto=*/"");
}

/*static*/ std::unique_ptr<HloOpProfiles> HloOpProfiles::Load(
    absl::string_view profiles_text_proto,
    absl::string_view default_profile_name,
    absl::string_view overrides_text_proto) {
  ProfilesNestedMap profiles_map = ParseProfiles(profiles_text_proto);
  std::vector<std::string> overridden_devices;
  for (auto& [device, profile] : ParseProfiles(overrides_text_proto)) {
    overridden_devices.push_back(device);
    profiles_map[device] = std::move(profile);
  }
  if (!overridden_devices.empty()) {
    std::sort(overridden_devices.begin(), overridden_devices.end());
    LOG(INFO) << "Using override op profiles for devices: "
              << absl::StrJoin(overridden_devices, ", ")
              << ". All other devices keep their base profiles.";
  }
  return absl::WrapUnique(
      new HloOpProfiles(std::move(profiles_map), default_profile_name));
}

const HloOpProfiles::HloOpProfile& HloOpProfiles::GetProfile(
    const se::DeviceDescription& device_info) const {
  auto it = profiles_.find(GetProfileName(device_info));
  if (it != profiles_.end()) {
    return it->second;
  }
  return default_profile_;
}

}  // namespace gpu
}  // namespace xla
