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

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace ray {
namespace raylet {

namespace {

/// 4-GPU dual-socket host: GPUs 0,1 on node 0 (cpus 0-3), GPUs 2,3 on node 1
/// (cpus 4-7).
constexpr char kDualSocketSpec[] = "0:0-3;0:0-3;1:4-7;1:4-7";

TaskResourceInstances MakeGpuAllocation(std::vector<FixedPoint> gpu_instances) {
  TaskResourceInstances allocation;
  allocation.Set(scheduling::ResourceID::GPU(), std::move(gpu_instances));
  return allocation;
}

}  // namespace

TEST(ParseGpuNumaTopologyTest, ParsesDualSocketSpec) {
  auto topology = ParseGpuNumaTopology(kDualSocketSpec);
  ASSERT_EQ(topology.size(), 4u);
  EXPECT_EQ(topology[0].numa_node, 0);
  EXPECT_EQ(topology[0].cpus, std::vector<int>({0, 1, 2, 3}));
  EXPECT_EQ(topology[3].numa_node, 1);
  EXPECT_EQ(topology[3].cpus, std::vector<int>({4, 5, 6, 7}));
}

TEST(ParseGpuNumaTopologyTest, ParsesSingleCpusAndRanges) {
  auto topology = ParseGpuNumaTopology("0:0,2,4-5");
  ASSERT_EQ(topology.size(), 1u);
  EXPECT_EQ(topology[0].cpus, std::vector<int>({0, 2, 4, 5}));
}

TEST(ParseGpuNumaTopologyTest, EmptyAndMalformedSpecsFailOpen) {
  EXPECT_TRUE(ParseGpuNumaTopology("").empty());
  EXPECT_TRUE(ParseGpuNumaTopology("garbage").empty());
  EXPECT_TRUE(ParseGpuNumaTopology("0:").empty());
  EXPECT_TRUE(ParseGpuNumaTopology(":0-3").empty());
  EXPECT_TRUE(ParseGpuNumaTopology("0:3-1").empty());      // inverted range
  EXPECT_TRUE(ParseGpuNumaTopology("-1:0-3").empty());     // negative node
  EXPECT_TRUE(ParseGpuNumaTopology("0:0-3;bad").empty());  // one bad entry
}

TEST(ComputeNumaBindSpecTest, SingleNodeGpusBind) {
  auto topology = ParseGpuNumaTopology(kDualSocketSpec);
  // GPUs 2 and 3, both on node 1.
  auto allocation =
      MakeGpuAllocation({FixedPoint(0), FixedPoint(0), FixedPoint(1), FixedPoint(1)});
  auto spec = ComputeNumaBindSpec(topology, allocation);
  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->numa_node, 1);
  EXPECT_EQ(spec->cpus, std::vector<int>({4, 5, 6, 7}));
}

TEST(ComputeNumaBindSpecTest, FractionalGpuBinds) {
  auto topology = ParseGpuNumaTopology(kDualSocketSpec);
  // Half of GPU 0 still identifies GPU 0 as the worker's device.
  auto allocation =
      MakeGpuAllocation({FixedPoint(0.5), FixedPoint(0), FixedPoint(0), FixedPoint(0)});
  auto spec = ComputeNumaBindSpec(topology, allocation);
  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->numa_node, 0);
  EXPECT_EQ(spec->cpus, std::vector<int>({0, 1, 2, 3}));
}

TEST(ComputeNumaBindSpecTest, CrossNodeGpusFailOpen) {
  auto topology = ParseGpuNumaTopology(kDualSocketSpec);
  // GPUs 1 (node 0) and 2 (node 1) span sockets: no binding.
  auto allocation =
      MakeGpuAllocation({FixedPoint(0), FixedPoint(1), FixedPoint(1), FixedPoint(0)});
  EXPECT_FALSE(ComputeNumaBindSpec(topology, allocation).has_value());
}

TEST(ComputeNumaBindSpecTest, NoGpuAllocationFailsOpen) {
  auto topology = ParseGpuNumaTopology(kDualSocketSpec);
  TaskResourceInstances cpu_only;
  cpu_only.Set(scheduling::ResourceID::CPU(), {FixedPoint(4)});
  EXPECT_FALSE(ComputeNumaBindSpec(topology, cpu_only).has_value());
  auto zero_gpu =
      MakeGpuAllocation({FixedPoint(0), FixedPoint(0), FixedPoint(0), FixedPoint(0)});
  EXPECT_FALSE(ComputeNumaBindSpec(topology, zero_gpu).has_value());
}

TEST(ComputeNumaBindSpecTest, AllocationBeyondTopologyFailsOpen) {
  // Topology only covers 1 GPU but the allocation uses instance 1.
  auto topology = ParseGpuNumaTopology("0:0-3");
  auto allocation = MakeGpuAllocation({FixedPoint(0), FixedPoint(1)});
  EXPECT_FALSE(ComputeNumaBindSpec(topology, allocation).has_value());
}

TEST(ComputeNumaBindSpecTest, EmptyTopologyFailsOpen) {
  auto allocation = MakeGpuAllocation({FixedPoint(1)});
  EXPECT_FALSE(ComputeNumaBindSpec({}, allocation).has_value());
}

TEST(ApplyNumaBindingInChildTest, DoesNotCrash) {
  // On Linux this actually pins the test process; on other platforms it is a
  // no-op. Either way it must not crash. Use the full CPU range of node 0 so
  // the test does not depend on host topology.
  NumaBindSpec spec;
  for (int cpu = 0; cpu < 4; cpu++) {
    spec.cpus.push_back(cpu);
  }
  spec.numa_node = 0;
  ApplyNumaBindingInChild(spec);
}

}  // namespace raylet
}  // namespace ray
