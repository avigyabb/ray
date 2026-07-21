"""Integration tests for NUMA-aware worker binding at spawn.

These inject a fake 4-GPU dual-socket topology into the raylet command line by
monkeypatching the node-startup detection hook (which runs in the driver
process during ``ray.init``), then assert on raylet logs that the spawn-time
binding path fired for exactly the right workers. This exercises the full
chain -- ``--gpu_numa_topology`` flag -> raylet parse -> lease allocation ->
``ComputeNumaBindSpec`` -> worker spawn -- without GPUs or a NUMA host. The
``sched_setaffinity``/``set_mempolicy`` syscalls themselves are Linux-only and
no-op elsewhere.
"""
import glob
import os
import sys
import time

import pytest

import ray
import ray._private.gpu_numa_topology as gnt

# GPUs 0,1 on NUMA node 0 (cpus 0-3); GPUs 2,3 on node 1 (cpus 4-7).
FAKE_SPEC = "0:0-3;0:0-3;1:4-7;1:4-7"


def _raylet_log_text():
    session = ray._private.worker._global_node.get_session_dir_path()
    text = ""
    for path in glob.glob(os.path.join(session, "logs", "raylet.*")):
        try:
            with open(path, errors="replace") as f:
                text += f.read()
        except OSError:
            pass
    return text


def _bind_log_lines():
    return [
        line
        for line in _raylet_log_text().splitlines()
        if "NUMA-binding worker at spawn" in line
    ]


@pytest.fixture
def numa_fake_gpu_cluster(shutdown_only, monkeypatch):
    # Freshly spawn workers (skip both prestart mechanisms) so the
    # spawn-binding path is exercised deterministically instead of popping an
    # idle prestarted worker (which is never bound; documented limitation).
    monkeypatch.setenv("RAY_enable_worker_prestart", "0")
    monkeypatch.setenv("RAY_prestart_worker_first_driver", "0")
    monkeypatch.setattr(
        gnt, "get_gpu_numa_topology_spec_if_enabled", lambda num_gpus: FAKE_SPEC
    )
    ray.init(num_cpus=8, num_gpus=4, include_dashboard=False)
    yield


def test_colocated_gpu_actor_is_bound_at_spawn(numa_fake_gpu_cluster):
    @ray.remote(num_gpus=2)
    class TwoGpuActor:
        def gpu_ids(self):
            return sorted(int(x) for x in ray.get_gpu_ids())

    a = TwoGpuActor.remote()
    # First-fit allocates GPUs 0,1 -- both on NUMA node 0.
    assert ray.get(a.gpu_ids.remote(), timeout=60) == [0, 1]

    time.sleep(1)
    assert "GPU NUMA topology configured for 4 GPUs" in _raylet_log_text()
    lines = _bind_log_lines()
    assert len(lines) == 1, lines
    assert "numa_node=0" in lines[0] and "num_cpus=4" in lines[0], lines[0]


def test_cross_socket_actor_and_tasks_are_not_bound(numa_fake_gpu_cluster):
    @ray.remote(num_gpus=3)
    class ThreeGpuActor:  # GPUs 0,1,2 span both nodes: fail open, no binding.
        def ping(self):
            return "ok"

    @ray.remote(num_gpus=1)
    def gpu_task():  # Non-actor GPU work is never bound at spawn.
        return "ok"

    b = ThreeGpuActor.remote()
    assert ray.get(b.ping.remote(), timeout=60) == "ok"
    assert ray.get(gpu_task.remote(), timeout=60) == "ok"

    time.sleep(1)
    assert _bind_log_lines() == []


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
