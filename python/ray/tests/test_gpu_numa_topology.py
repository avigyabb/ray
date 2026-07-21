"""Unit tests for GPU NUMA topology detection (ray._private.gpu_numa_topology).

Pure-logic tests: NVML is mocked, so these run on any platform without GPUs.
"""

import sys
from unittest import mock

import pytest

from ray._private import gpu_numa_topology as gnt


# ---------------------------------------------------------------------------
# cpu_affinity_words_to_cpu_set
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "words,expected",
    [
        ([0b0], set()),
        ([0b1], {0}),
        ([0b1011], {0, 1, 3}),
        # Second word covers CPUs 64..127.
        ([0, 0b101], {64, 66}),
        ([0xFFFFFFFFFFFFFFFF], set(range(64))),
    ],
)
def test_cpu_affinity_words_to_cpu_set(words, expected):
    assert gnt.cpu_affinity_words_to_cpu_set(words) == expected


# ---------------------------------------------------------------------------
# cpu_set_to_cpulist
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cpus,expected",
    [
        (set(), ""),
        ({0}, "0"),
        ({0, 1, 2, 3}, "0-3"),
        ({0, 2, 4}, "0,2,4"),
        ({0, 1, 2, 8, 10, 11}, "0-2,8,10-11"),
        ({16, 17, 18, 19, 48, 49, 50, 51}, "16-19,48-51"),
    ],
)
def test_cpu_set_to_cpulist(cpus, expected):
    assert gnt.cpu_set_to_cpulist(cpus) == expected


# ---------------------------------------------------------------------------
# build_gpu_numa_topology_spec (mocked NVML)
# ---------------------------------------------------------------------------
def _fake_pynvml(numa_nodes, cpu_masks, device_count=None):
    """Fake pynvml where GPU i has numa_nodes[i] and affinity words cpu_masks[i]."""
    fake = mock.MagicMock()
    fake.nvmlInit.return_value = None
    fake.nvmlDeviceGetCount.return_value = (
        len(numa_nodes) if device_count is None else device_count
    )
    fake.nvmlDeviceGetHandleByIndex.side_effect = lambda i: i
    fake.nvmlDeviceGetNumaNodeId.side_effect = lambda i: numa_nodes[i]
    fake.nvmlDeviceGetCpuAffinity.side_effect = lambda i, size: cpu_masks[i]
    return fake


def _with_fake_pynvml(fake):
    return mock.patch.dict("sys.modules", {"ray._private.thirdparty.pynvml": fake})


def test_build_spec_dual_socket():
    # GPUs 0,1 on node 0 (cpus 0-3); GPUs 2,3 on node 1 (cpus 4-7).
    fake = _fake_pynvml(
        numa_nodes=[0, 0, 1, 1],
        cpu_masks=[[0b1111], [0b1111], [0b11110000], [0b11110000]],
    )
    with _with_fake_pynvml(fake):
        assert gnt.build_gpu_numa_topology_spec(4) == "0:0-3;0:0-3;1:4-7;1:4-7"


def test_build_spec_subset_of_devices():
    # Node configured with num_gpus=1 out of 2 physical GPUs.
    fake = _fake_pynvml(numa_nodes=[1, 0], cpu_masks=[[0b1100], [0b0011]])
    with _with_fake_pynvml(fake):
        assert gnt.build_gpu_numa_topology_spec(1) == "1:2-3"


def test_build_spec_unknown_numa_node_fails_open():
    fake = _fake_pynvml(numa_nodes=[0, -1], cpu_masks=[[0b1111], [0b1111]])
    with _with_fake_pynvml(fake):
        assert gnt.build_gpu_numa_topology_spec(2) is None


def test_build_spec_empty_cpu_mask_fails_open():
    fake = _fake_pynvml(numa_nodes=[0], cpu_masks=[[0]])
    with _with_fake_pynvml(fake):
        assert gnt.build_gpu_numa_topology_spec(1) is None


def test_build_spec_fewer_devices_than_gpus_fails_open():
    fake = _fake_pynvml(numa_nodes=[0], cpu_masks=[[0b1]], device_count=1)
    with _with_fake_pynvml(fake):
        assert gnt.build_gpu_numa_topology_spec(2) is None


def test_build_spec_nvml_init_failure_fails_open():
    fake = mock.MagicMock()
    fake.nvmlInit.side_effect = RuntimeError("no driver")
    with _with_fake_pynvml(fake):
        assert gnt.build_gpu_numa_topology_spec(2) is None


@pytest.mark.parametrize("num_gpus", [0, -1, None, "x"])
def test_build_spec_no_gpus(num_gpus):
    assert gnt.build_gpu_numa_topology_spec(num_gpus) is None


# ---------------------------------------------------------------------------
# opt-in gate
# ---------------------------------------------------------------------------
def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(gnt.RAY_GPU_NUMA_AFFINITY_ENV_VAR, raising=False)
    assert gnt.gpu_numa_affinity_enabled() is False
    # The startup entry point returns None without touching NVML.
    assert gnt.get_gpu_numa_topology_spec_if_enabled(4) is None


def test_enabled_entry_point(monkeypatch):
    monkeypatch.setenv(gnt.RAY_GPU_NUMA_AFFINITY_ENV_VAR, "1")
    fake = _fake_pynvml(numa_nodes=[0], cpu_masks=[[0b11]])
    with _with_fake_pynvml(fake):
        assert gnt.get_gpu_numa_topology_spec_if_enabled(1) == "0:0-1"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
