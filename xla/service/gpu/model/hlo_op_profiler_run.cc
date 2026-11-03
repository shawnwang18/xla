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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiler.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/service/pjrt_gpu_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {
namespace {

constexpr absl::string_view kUsage = R"(
This tool measures the cost of elementwise HLO operations on GPU.

--mode=latency (default) measures the latency of a dependent chain on a single
element in clock cycles, the unit of hlo_op_profiles_data.h.
--mode=throughput measures the issue cost on a device-filling kernel in
FMA-equivalent flops per element, the unit GpuPerformanceModelBase::ComputeTime
divides by.
--opcodes and --data_types restrict the measurement to the listed names. A name
the tool cannot measure is an error rather than silently skipped.
)";

absl::StatusOr<HloOpProfiler::Mode> ParseMode(absl::string_view mode) {
  if (mode == "latency") {
    return HloOpProfiler::Mode::kLatency;
  }
  if (mode == "throughput") {
    return HloOpProfiler::Mode::kThroughput;
  }
  return InvalidArgument("Unknown --mode '%s'; expected latency or throughput",
                         mode);
}

// Splits a comma-separated flag value into trimmed, lower-cased tokens.
std::vector<std::string> SplitFlagList(absl::string_view value) {
  std::vector<std::string> tokens;
  for (absl::string_view token :
       absl::StrSplit(value, ',', absl::SkipWhitespace())) {
    tokens.push_back(absl::AsciiStrToLower(absl::StripAsciiWhitespace(token)));
  }
  return tokens;
}

// Parses --opcodes. A name that is not an HLO opcode, or that the profiler does
// not measure, is fatal: skipping it silently would write an incomplete or
// empty profile and exit successfully.
absl::flat_hash_set<HloOpcode> ParseOpcodesFilter(absl::string_view value) {
  absl::flat_hash_set<HloOpcode> opcodes;
  for (const std::string& name : SplitFlagList(value)) {
    absl::StatusOr<HloOpcode> op = StringToHloOpcode(name);
    if (!op.ok()) {
      LOG(QFATAL) << "--opcodes: '" << name << "' is not an HLO opcode";
    }
    if (!absl::c_linear_search(HloOpProfiler::AllSupportedOps(), *op) ||
        HloOpProfiler::Unsupported().contains(*op) ||
        HloOpProfiler::TooFastToMeasure().contains(*op)) {
      LOG(QFATAL) << "--opcodes: " << name
                  << " is not measured by this tool (not elementwise, "
                     "unsupported, or too fast to measure)";
    }
    opcodes.insert(*op);
  }
  return opcodes;
}

// Parses --data_types; names are case-insensitive. Same policy as above.
absl::flat_hash_set<PrimitiveType> ParseDataTypesFilter(
    absl::string_view value) {
  absl::flat_hash_set<PrimitiveType> data_types;
  for (const std::string& name : SplitFlagList(value)) {
    absl::StatusOr<PrimitiveType> type =
        primitive_util::StringToPrimitiveType(name);
    if (!type.ok() ||
        !absl::c_linear_search(HloOpProfiler::AllSupportedDtypes(), *type)) {
      LOG(QFATAL) << "--data_types: '" << name
                  << "' is not a type this tool measures; supported: "
                  << absl::StrJoin(
                         HloOpProfiler::AllSupportedDtypes(), ",",
                         [](std::string* out, PrimitiveType type) {
                           absl::StrAppend(
                               out, primitive_util::LowercasePrimitiveTypeName(
                                        type));
                         });
    }
    data_types.insert(*type);
  }
  return data_types;
}

void WriteOutput(const DeviceHloInstructionProfiles& literal,
                 absl::string_view name) {
  std::string file_name;
  std::string output_directory;
  if (absl::StrContains(name, '/')) {
    // A path was given: write exactly there.
    file_name = std::string(name);
  } else if (tsl::io::GetTestUndeclaredOutputsDir(&output_directory)) {
    file_name = tsl::io::JoinPath(
        output_directory,
        absl::StrFormat("profiles-%d-%s", tsl::Env::Default()->NowMicros(),
                        name));
    absl::StrAppend(&file_name, ".textproto");
  } else {
    file_name = tsl::io::GetTempFilename(absl::StrCat(name, ".textproto"));
  }
  VLOG(0) << "Writing output to " << file_name;
  CHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(), file_name,
                                  tsl::LegacyUnredactedDebugString(literal)));
}

std::pair<std::unique_ptr<HloRunnerInterface>,
          stream_executor::DeviceDescription>
MakeRunnerAndGetDeviceDescription() {
  GpuAllocatorConfig gpu_config;
  gpu_config.kind = GpuAllocatorConfig::Kind::kDefault;
  gpu_config.preallocate = false;
  gpu_config.collective_memory_size = 0;
  GpuClientOptions options;
  options.allocator_config = std::move(gpu_config);

  absl::StatusOr<std::unique_ptr<PjRtClient>> client =
      GetXlaPjrtGpuClient(options);
  CHECK_OK(client);
  GpuTargetConfig gpu_target_config = GetGpuTargetConfig(client->get());
  return {std::make_unique<HloRunner>(*std::move(client)),
          gpu_target_config.device_description};
}

int RunProfiler(int argc, char** argv) {
  std::string output_file;
  std::string opcodes_filter;
  std::string data_types_filter;
  std::string mode_name = "latency";
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("mode", &mode_name,
                "What to measure: 'latency' (default) reports clock cycles "
                "of a dependent op chain on one element, as in "
                "hlo_op_profiles_data.h; 'throughput' reports FMA-equivalent "
                "flops per element on a device-filling kernel, the unit "
                "GpuPerformanceModelBase::ComputeTime divides by."),
      tsl::Flag("output_file", &output_file,
                "Where to write the measured profiles (text proto). A value "
                "containing '/' is used as the file path; a bare name is "
                "placed under TEST_UNDECLARED_OUTPUTS_DIR or a temp "
                "directory as profiles-<time>-<name>.textproto."),
      tsl::Flag("opcodes", &opcodes_filter,
                "Comma-separated HLO opcode names to measure, e.g. "
                "exponential,divide. Default: all supported opcodes. A name "
                "the tool does not measure is an error."),
      tsl::Flag("data_types", &data_types_filter,
                "Comma-separated primitive type names to measure, "
                "case-insensitive, e.g. f32,f16. Default: all supported "
                "types. A name the tool does not measure is an error."),
  };
  // Allow setting flags as command line argument (in addition to XLA_FLAGS
  // environment variable).
  AppendDebugOptionsFlags(&flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsage.data(), &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << "Error parsing flags";
  }

  absl::StatusOr<HloOpProfiler::Mode> mode = ParseMode(mode_name);
  if (!mode.ok()) {
    LOG(QFATAL) << mode.status();
  }
  const absl::flat_hash_set<HloOpcode> opcodes =
      ParseOpcodesFilter(opcodes_filter);
  const absl::flat_hash_set<PrimitiveType> data_types =
      ParseDataTypesFilter(data_types_filter);

  auto [runner, dev_info] = MakeRunnerAndGetDeviceDescription();
  HloOpProfiler profiler(runner.get(), &dev_info, *mode);
  VLOG(0) << dev_info.name() << " @ " << dev_info.clock_rate_ghz() << " GHz, "
          << dev_info.core_count() << " cores x " << dev_info.fpus_per_core()
          << " FP32 lanes; measuring " << mode_name;

  HloInstructionProfileList instr_profiles;

  for (const PrimitiveType data_type : HloOpProfiler::AllSupportedDtypes()) {
    if (!data_types.empty() && !data_types.contains(data_type)) {
      continue;
    }
    for (const HloOpcode op : HloOpProfiler::AllSupportedOps()) {
      if (HloOpProfiler::TooFastToMeasure().count(op) ||
          HloOpProfiler::Unsupported().count(op)) {
        continue;
      }
      if (!opcodes.empty() && !opcodes.contains(op)) {
        continue;
      }
      auto result = profiler.MeasureClockCyclesPerOp(op, data_type);
      if (result.ok()) {
        instr_profiles.add_entries()->Swap(&*result);
      } else {
        LOG(ERROR) << result.status();
      }
    }
  }

  VLOG(1) << "\n" << instr_profiles.DebugString();

  DeviceHloInstructionProfiles device_profiles;
  device_profiles.mutable_entries()->insert(
      {HloOpProfiles::GetProfileName(dev_info), instr_profiles});
  if (!output_file.empty()) {
    WriteOutput(device_profiles, output_file);
  }

  return 0;
}

}  // namespace
}  // namespace gpu
}  // namespace xla

int main(int argc, char** argv) { return xla::gpu::RunProfiler(argc, argv); }
