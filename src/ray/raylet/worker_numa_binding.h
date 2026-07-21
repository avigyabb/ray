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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ray/common/scheduling/cluster_resource_data.h"

namespace ray {
namespace raylet {

/// NUMA locality of one GPU instance, parsed from the --gpu_numa_topology flag.
struct GpuNumaInfo {
  /// NUMA node the GPU is attached to.
  int64_t numa_node;
  /// Logical CPUs local to that NUMA node.
  std::vector<int> cpus;
};

/// Parse the --gpu_numa_topology raylet flag into per-GPU NUMA info.
///
/// The spec contains one ';'-separated entry per GPU instance index, each of
/// the form "<numa_node>:<cpulist>" where <cpulist> is a Linux-style cpu list
/// (e.g. "0-15,32-47"). Example for a 4-GPU dual-socket host:
///   "0:0-15,32-47;0:0-15,32-47;1:16-31,48-63;1:16-31,48-63"
///
/// Returns an empty vector if the spec is empty or malformed (fail open).
std::vector<GpuNumaInfo> ParseGpuNumaTopology(const std::string &spec);

/// The CPU/memory binding to apply to a worker process at spawn.
struct NumaBindSpec {
  /// Logical CPUs to pin the process to (sched_setaffinity).
  std::vector<int> cpus;
  /// NUMA node to prefer for memory allocation (set_mempolicy MPOL_PREFERRED).
  int64_t numa_node;
};

/// Compute the NUMA binding for a worker given the node's GPU topology and the
/// worker's allocated resource instances.
///
/// Returns a bind spec iff the allocation uses at least one GPU instance, every
/// used GPU instance index is covered by the topology, all used GPUs share a
/// single NUMA node, and that node's cpu list is non-empty. Otherwise returns
/// nullopt (fail open: the worker is spawned unbound, exactly as today).
std::optional<NumaBindSpec> ComputeNumaBindSpec(
    const std::vector<GpuNumaInfo> &gpu_numa_topology,
    const TaskResourceInstances &allocated_instances);

/// Pin the calling process to spec.cpus and prefer memory allocation from
/// spec.numa_node. Both settings survive execve, so a worker forked+exec'd
/// after this call faults all of its memory (interpreter, imports, user code)
/// onto the GPU-local NUMA node via first-touch.
///
/// Intended to be called in the forked child before exec. Only raw syscalls,
/// no allocation and no logging (async-signal-safe); errors are ignored (best
/// effort). No-op on non-Linux platforms.
void ApplyNumaBindingInChild(const NumaBindSpec &spec);

}  // namespace raylet
}  // namespace ray
