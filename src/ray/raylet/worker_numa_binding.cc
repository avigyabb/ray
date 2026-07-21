// Copyright 2026 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/raylet/worker_numa_binding.h"

#ifdef __linux__
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "ray/common/scheduling/scheduling_ids.h"

namespace ray {
namespace raylet {

namespace {

/// Parse a Linux-style cpu list ("0-15,32-47") into cpu ids. Returns
/// std::nullopt on any parse error.
std::optional<std::vector<int>> ParseCpuList(const std::string &cpu_list) {
  std::vector<int> cpus;
  for (const absl::string_view part : absl::StrSplit(cpu_list, ',')) {
    if (part.empty()) {
      return std::nullopt;
    }
    const size_t dash = part.find('-');
    if (dash == absl::string_view::npos) {
      int cpu = 0;
      if (!absl::SimpleAtoi(part, &cpu) || cpu < 0) {
        return std::nullopt;
      }
      cpus.push_back(cpu);
    } else {
      int lo = 0;
      int hi = 0;
      if (!absl::SimpleAtoi(part.substr(0, dash), &lo) ||
          !absl::SimpleAtoi(part.substr(dash + 1), &hi) || lo < 0 || hi < lo) {
        return std::nullopt;
      }
      for (int cpu = lo; cpu <= hi; cpu++) {
        cpus.push_back(cpu);
      }
    }
  }
  return cpus;
}

}  // namespace

std::vector<GpuNumaInfo> ParseGpuNumaTopology(const std::string &spec) {
  std::vector<GpuNumaInfo> topology;
  if (spec.empty()) {
    return topology;
  }
  for (const absl::string_view entry : absl::StrSplit(spec, ';')) {
    const size_t colon = entry.find(':');
    if (colon == absl::string_view::npos) {
      return {};
    }
    int64_t numa_node = 0;
    if (!absl::SimpleAtoi(entry.substr(0, colon), &numa_node) || numa_node < 0) {
      return {};
    }
    auto cpus = ParseCpuList(std::string(entry.substr(colon + 1)));
    if (!cpus.has_value() || cpus->empty()) {
      return {};
    }
    topology.push_back(GpuNumaInfo{numa_node, std::move(*cpus)});
  }
  return topology;
}

std::optional<NumaBindSpec> ComputeNumaBindSpec(
    const std::vector<GpuNumaInfo> &gpu_numa_topology,
    const TaskResourceInstances &allocated_instances) {
  if (gpu_numa_topology.empty()) {
    return std::nullopt;
  }
  const auto gpu_id = scheduling::ResourceID::GPU();
  if (!allocated_instances.Has(gpu_id)) {
    return std::nullopt;
  }
  // Any GPU instance with a positive allocation (whole or fractional) is used
  // by this worker.
  const std::vector<FixedPoint> &gpu_allocations = allocated_instances.Get(gpu_id);
  std::optional<int64_t> numa_node;
  std::set<int> cpus;
  bool any_gpu_used = false;
  for (size_t i = 0; i < gpu_allocations.size(); i++) {
    if (gpu_allocations[i] <= 0) {
      continue;
    }
    any_gpu_used = true;
    if (i >= gpu_numa_topology.size()) {
      // Allocation references a GPU the topology doesn't cover; fail open.
      return std::nullopt;
    }
    const GpuNumaInfo &info = gpu_numa_topology[i];
    if (numa_node.has_value() && *numa_node != info.numa_node) {
      // GPUs span NUMA nodes; binding to one node would be wrong. Fail open.
      return std::nullopt;
    }
    numa_node = info.numa_node;
    cpus.insert(info.cpus.begin(), info.cpus.end());
  }
  if (!any_gpu_used || cpus.empty()) {
    return std::nullopt;
  }
  return NumaBindSpec{std::vector<int>(cpus.begin(), cpus.end()), *numa_node};
}

#ifdef __linux__

// From <numaif.h> (which ships with libnuma headers, not glibc); defined here
// so no libnuma dependency is needed for the raw set_mempolicy syscall.
#ifndef MPOL_PREFERRED
#define MPOL_PREFERRED 1
#endif

void ApplyNumaBindingInChild(const NumaBindSpec &spec) {
  // Runs in the forked child before exec: raw syscalls only, no allocation, no
  // logging, and errors are ignored (binding is best effort by design).
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  bool any_cpu = false;
  for (const int cpu : spec.cpus) {
    if (cpu >= 0 && cpu < CPU_SETSIZE) {
      CPU_SET(cpu, &cpuset);
      any_cpu = true;
    }
  }
  if (any_cpu) {
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
  }
  // MPOL_PREFERRED (not MPOL_BIND) so allocation degrades to remote nodes
  // instead of failing/OOM when the preferred node is full. Both the CPU
  // affinity and the memory policy are preserved across execve.
  if (spec.numa_node >= 0 &&
      spec.numa_node < static_cast<int64_t>(sizeof(unsigned long) * 8)) {
    unsigned long nodemask = 1UL << spec.numa_node;
    syscall(SYS_set_mempolicy, MPOL_PREFERRED, &nodemask, sizeof(nodemask) * 8);
  }
}

#else

void ApplyNumaBindingInChild(const NumaBindSpec &spec) {}

#endif

}  // namespace raylet
}  // namespace ray
