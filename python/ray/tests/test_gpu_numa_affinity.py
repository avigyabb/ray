"""Unit tests for GPU NUMA CPU-affinity binding (ray._private.numa_affinity).

These are pure-logic tests: no GPU and no Linux host required. The OS affinity
syscalls and NVML are mocked, so the tests exercise the topology math and the
apply/reset/guard behavior on any platform.
"""

import sys
from unittest import mock

import pytest

from ray._private import numa_affinity as na


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
    assert na.cpu_affinity_words_to_cpu_set(words) == expected


# ---------------------------------------------------------------------------
# compute_gpu_local_cpu_set
# ---------------------------------------------------------------------------
# A synthetic dual-socket topology: GPUs 0,1 on socket 0 (CPUs 0-3),
# GPUs 2,3 on socket 1 (CPUs 4-7).
TOPO = {
    "0": {0, 1, 2, 3},
    "1": {0, 1, 2, 3},
    "2": {4, 5, 6, 7},
    "3": {4, 5, 6, 7},
}


def test_compute_single_socket_gpus():
    # Two co-located GPUs -> that socket's cores.
    assert na.compute_gpu_local_cpu_set(["0", "1"], TOPO) == {0, 1, 2, 3}


def test_compute_cross_socket_gpus_unions():
    # GPUs on different sockets -> union of both (soft-fallback case).
    assert na.compute_gpu_local_cpu_set(["1", "2"], TOPO) == {0, 1, 2, 3, 4, 5, 6, 7}


def test_compute_empty_gpu_ids_returns_none():
    assert na.compute_gpu_local_cpu_set([], TOPO) is None


def test_compute_unknown_gpu_returns_none():
    # If any assigned GPU's topology is unknown, don't guess.
    assert na.compute_gpu_local_cpu_set(["0", "99"], TOPO) is None


def test_compute_intersects_allowed_cpus_never_widens():
    # Process is already restricted to CPUs {2,3}; result must stay within it.
    assert na.compute_gpu_local_cpu_set(["0"], TOPO, allowed_cpus={2, 3, 100}) == {2, 3}


def test_compute_disjoint_allowed_returns_none():
    # GPU-local cores are entirely outside the allowed cpuset -> leave alone.
    assert na.compute_gpu_local_cpu_set(["0"], TOPO, allowed_cpus={64, 65}) is None


def test_compute_accepts_non_string_gpu_ids():
    assert na.compute_gpu_local_cpu_set([0, 1], TOPO) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# set_gpu_numa_cpu_affinity / reset_cpu_affinity  (mock the syscalls)
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_sched():
    """Mock os.sched_get/setaffinity (absent on macOS -> create=True) and force
    binding support on regardless of host platform."""
    state = {"affinity": set(range(8))}

    def getaffinity(_pid):
        return set(state["affinity"])

    def setaffinity(_pid, cpus):
        state["affinity"] = set(cpus)

    with mock.patch.object(
        na, "numa_affinity_binding_supported", return_value=True
    ), mock.patch.object(
        na.os, "sched_getaffinity", create=True, side_effect=getaffinity
    ), mock.patch.object(
        na.os, "sched_setaffinity", create=True, side_effect=setaffinity
    ) as set_mock:
        yield state, set_mock


def test_set_affinity_pins_to_socket(fake_sched):
    state, set_mock = fake_sched
    original = na.set_gpu_numa_cpu_affinity(["0", "1"], gpu_to_cpus=TOPO)
    assert original == set(range(8))  # returns the original for restoration
    assert state["affinity"] == {0, 1, 2, 3}  # pinned to socket 0
    set_mock.assert_called_once_with(0, {0, 1, 2, 3})


def test_set_affinity_noop_when_already_matching(fake_sched):
    state, set_mock = fake_sched
    state["affinity"] = {0, 1, 2, 3}
    # Target equals current affinity -> no syscall, returns None.
    assert na.set_gpu_numa_cpu_affinity(["0"], gpu_to_cpus=TOPO) is None
    set_mock.assert_not_called()


def test_set_affinity_noop_when_unknown_topology(fake_sched):
    _state, set_mock = fake_sched
    assert na.set_gpu_numa_cpu_affinity(["99"], gpu_to_cpus=TOPO) is None
    set_mock.assert_not_called()


def test_set_affinity_noop_when_no_gpus(fake_sched):
    _state, set_mock = fake_sched
    assert na.set_gpu_numa_cpu_affinity([], gpu_to_cpus=TOPO) is None
    set_mock.assert_not_called()


def test_reset_restores_affinity(fake_sched):
    state, _set_mock = fake_sched
    original = na.set_gpu_numa_cpu_affinity(["2"], gpu_to_cpus=TOPO)
    assert state["affinity"] == {4, 5, 6, 7}
    na.reset_cpu_affinity(original)
    assert state["affinity"] == {0, 1, 2, 3, 4, 5, 6, 7}


def test_reset_none_is_noop(fake_sched):
    state, set_mock = fake_sched
    na.reset_cpu_affinity(None)
    set_mock.assert_not_called()
    assert state["affinity"] == set(range(8))


def test_set_affinity_noop_when_unsupported():
    with mock.patch.object(na, "numa_affinity_binding_supported", return_value=False):
        assert na.set_gpu_numa_cpu_affinity(["0"], gpu_to_cpus=TOPO) is None


# ---------------------------------------------------------------------------
# enable flag
# ---------------------------------------------------------------------------
def test_enabled_flag_default_off():
    with mock.patch.dict("os.environ", {}, clear=False):
        na.os.environ.pop(na.RAY_GPU_NUMA_AFFINITY_ENV_VAR, None)
        assert na.gpu_numa_affinity_enabled() is False


def test_enabled_flag_on():
    with mock.patch.dict(
        "os.environ", {na.RAY_GPU_NUMA_AFFINITY_ENV_VAR: "1"}, clear=False
    ):
        assert na.gpu_numa_affinity_enabled() is True


# ---------------------------------------------------------------------------
# maybe_bind_worker_to_gpu_numa (orchestrator used by the execution path)
# ---------------------------------------------------------------------------
def test_maybe_bind_disabled_is_noop():
    with mock.patch.object(na, "gpu_numa_affinity_enabled", return_value=False):
        with mock.patch.object(na, "set_gpu_numa_cpu_affinity") as set_mock:
            assert na.maybe_bind_worker_to_gpu_numa() is None
            set_mock.assert_not_called()


def test_maybe_bind_no_gpus_is_noop():
    with mock.patch.object(
        na, "gpu_numa_affinity_enabled", return_value=True
    ), mock.patch.object(
        na, "numa_affinity_binding_supported", return_value=True
    ), mock.patch.object(
        na, "get_assigned_gpu_ids", return_value=[]
    ), mock.patch.object(
        na, "set_gpu_numa_cpu_affinity"
    ) as set_mock:
        assert na.maybe_bind_worker_to_gpu_numa() is None
        set_mock.assert_not_called()


def test_maybe_bind_calls_through_when_enabled():
    with mock.patch.object(
        na, "gpu_numa_affinity_enabled", return_value=True
    ), mock.patch.object(
        na, "numa_affinity_binding_supported", return_value=True
    ), mock.patch.object(
        na, "get_assigned_gpu_ids", return_value=["0", "1"]
    ), mock.patch.object(
        na, "set_gpu_numa_cpu_affinity", return_value={0, 1, 2, 3}
    ) as set_mock:
        assert na.maybe_bind_worker_to_gpu_numa() == {0, 1, 2, 3}
        set_mock.assert_called_once_with(["0", "1"])


# ---------------------------------------------------------------------------
# Component 0 topology detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cpu_list,expected",
    [
        ("0-3", {0, 1, 2, 3}),
        ("0,2,4", {0, 2, 4}),
        ("0-1,4-5", {0, 1, 4, 5}),
        ("", set()),
        ("  0-2 \n", {0, 1, 2}),
    ],
)
def test_parse_cpu_list(cpu_list, expected):
    assert na.parse_cpu_list(cpu_list) == expected


def test_get_cpu_to_numa_node_reads_sysfs(tmp_path):
    (tmp_path / "node0").mkdir()
    (tmp_path / "node1").mkdir()
    (tmp_path / "node0" / "cpulist").write_text("0-1\n")
    (tmp_path / "node1" / "cpulist").write_text("2-3\n")
    # Non-node dirs are ignored.
    (tmp_path / "has_cpu").mkdir()
    mapping = na.get_cpu_to_numa_node(sysfs_root=str(tmp_path))
    assert mapping == {0: 0, 1: 0, 2: 1, 3: 1}


def test_get_cpu_to_numa_node_missing_root_is_empty():
    assert na.get_cpu_to_numa_node(sysfs_root="/nonexistent/path/xyz") == {}


def _install_fake_pynvml(node_ids):
    """Return a fake pynvml module mapping device index -> NUMA node id."""
    fake = mock.MagicMock()
    fake.nvmlInit.return_value = None
    fake.nvmlDeviceGetCount.return_value = len(node_ids)
    fake.nvmlDeviceGetHandleByIndex.side_effect = lambda i: i
    fake.nvmlDeviceGetNumaNodeId.side_effect = lambda i: node_ids[i]
    return fake


def test_get_gpu_numa_nodes_via_nvml():
    fake = _install_fake_pynvml([0, 0, 1, 1])
    with mock.patch.dict("sys.modules", {"ray._private.thirdparty.pynvml": fake}):
        assert na.get_gpu_numa_nodes_via_nvml() == {"0": 0, "1": 0, "2": 1, "3": 1}


def test_get_gpu_numa_nodes_omits_unknown_negative():
    fake = _install_fake_pynvml([0, -1, 1])
    with mock.patch.dict("sys.modules", {"ray._private.thirdparty.pynvml": fake}):
        # GPU 1 has an unknown (-1) node and is omitted.
        assert na.get_gpu_numa_nodes_via_nvml() == {"0": 0, "2": 1}


# ---------------------------------------------------------------------------
# numa_affinity remote-option validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, "soft", "strict"])
def test_numa_affinity_option_accepts_valid(value):
    from ray._common.ray_option_utils import _common_options

    opt = _common_options["numa_affinity"]
    assert opt.validate("numa_affinity", value) is None


@pytest.mark.parametrize("value", ["STRICT", "hard", "1", ""])
def test_numa_affinity_option_rejects_invalid(value):
    from ray._common.ray_option_utils import _common_options

    opt = _common_options["numa_affinity"]
    with pytest.raises(ValueError):
        opt.validate("numa_affinity", value)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
