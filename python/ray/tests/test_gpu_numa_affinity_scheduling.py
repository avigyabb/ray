"""Integration tests for NUMA-aware GPU scheduling.

These use a fake multi-GPU local cluster with an injected per-GPU NUMA topology
(the reserved ``_ray_gpu_numa_nodes`` node label), so they exercise the full
path -- numa_affinity option -> task/lease spec -> local allocator
(NodeResourceInstanceSet::TryAllocate) -> Component 0 topology delivery --
without requiring real GPUs or a multi-socket host.
"""
import sys

import pytest

import ray
import ray.exceptions


@pytest.fixture
def numa_gpu_cluster(shutdown_only):
    # GPUs 0,1 on NUMA node 0; GPUs 2,3 on NUMA node 1.
    ray.init(
        num_cpus=8,
        num_gpus=4,
        labels={"_ray_gpu_numa_nodes": "0,0,1,1"},
        include_dashboard=False,
    )
    yield


def test_strict_pends_when_not_colocatable(numa_gpu_cluster):
    # 3 GPUs exist in aggregate (a plain actor gets them)...
    @ray.remote(num_gpus=3)
    class ThreePlain:
        def ping(self):
            return "ok"

    p = ThreePlain.remote()
    assert ray.get(p.ping.remote(), timeout=60) == "ok"
    ray.kill(p)

    # ...but no single NUMA node has 3 GPUs, so a strict 3-GPU actor pends.
    @ray.remote(num_gpus=3, numa_affinity="strict")
    class ThreeStrict:
        def ping(self):
            return "ok"

    b = ThreeStrict.remote()
    with pytest.raises(ray.exceptions.GetTimeoutError):
        ray.get(b.ping.remote(), timeout=8)
    ray.kill(b)


def test_strict_colocates_two_gpus(numa_gpu_cluster):
    @ray.remote(num_gpus=2, numa_affinity="strict")
    class TwoStrict:
        def gpu_ids(self):
            return sorted(int(x) for x in ray.get_gpu_ids())

    a = TwoStrict.remote()
    ids = set(ray.get(a.gpu_ids.remote(), timeout=60))
    # Both GPUs must share a NUMA node.
    assert ids in ({0, 1}, {2, 3}), ids


def test_soft_falls_back_cross_node(numa_gpu_cluster):
    # Occupy GPU 1 (node 0) and GPU 2 (node 1) so no single node has 2 free.
    @ray.remote(num_gpus=1)
    class Holder:
        def gpu(self):
            return int(ray.get_gpu_ids()[0])

    holders = [Holder.remote() for _ in range(2)]
    ray.get([h.gpu.remote() for h in holders], timeout=60)

    # Soft 2-GPU cannot co-locate now; it must still schedule (cross-node).
    @ray.remote(num_gpus=2, numa_affinity="soft")
    class TwoSoft:
        def ping(self):
            return "ok"

    s = TwoSoft.remote()
    assert ray.get(s.ping.remote(), timeout=60) == "ok"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
