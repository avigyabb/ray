"""Detection of per-GPU NUMA topology for NUMA-aware worker binding.

At node startup, when ``RAY_EXPERIMENTAL_GPU_NUMA_AFFINITY=1``, Ray detects
which NUMA node each local NVIDIA GPU is attached to (and that node's CPUs) and
passes the result to the raylet via the ``--gpu_numa_topology`` flag. The
raylet then pins workers started for GPU actor leases to the CPUs / memory of
their GPUs' NUMA node at process spawn (see
src/ray/raylet/worker_numa_binding.h).

The spec format is one ';'-separated entry per GPU index, each
``"<numa_node>:<cpulist>"`` with a Linux-style cpu list, e.g. for a 4-GPU
dual-socket host: ``"0:0-15,32-47;0:0-15,32-47;1:16-31,48-63;1:16-31,48-63"``.

Everything here fails open: any detection problem yields ``None`` and the
raylet behaves exactly as today (no binding).

Only standard-library imports at module level so this is importable in
isolation; NVML is imported lazily.
"""

import logging
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Cluster-wide opt-in for NUMA binding of GPU actor workers.
RAY_GPU_NUMA_AFFINITY_ENV_VAR = "RAY_EXPERIMENTAL_GPU_NUMA_AFFINITY"


def gpu_numa_affinity_enabled() -> bool:
    """Whether GPU NUMA worker binding is opted in via env var."""
    from ray._private.ray_constants import env_bool

    return env_bool(RAY_GPU_NUMA_AFFINITY_ENV_VAR, False)


def cpu_affinity_words_to_cpu_set(words: Iterable[int]) -> Set[int]:
    """Decode an NVML CPU-affinity bitmask into a set of CPU indices.

    NVML returns affinity as an array of 64-bit words; bit ``b`` of word ``w``
    set means logical CPU ``w * 64 + b`` is local to the GPU.
    """
    cpus: Set[int] = set()
    for word_idx, word in enumerate(words):
        word = int(word)
        bit = 0
        while word:
            if word & 1:
                cpus.add(word_idx * 64 + bit)
            word >>= 1
            bit += 1
    return cpus


def cpu_set_to_cpulist(cpus: Set[int]) -> str:
    """Format a CPU set as a Linux-style cpu list, e.g. {0,1,2,8} -> "0-2,8"."""
    if not cpus:
        return ""
    sorted_cpus = sorted(cpus)
    ranges: List[Tuple[int, int]] = []
    start = prev = sorted_cpus[0]
    for cpu in sorted_cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append((start, prev))
        start = prev = cpu
    ranges.append((start, prev))
    return ",".join(f"{lo}-{hi}" if lo != hi else f"{lo}" for lo, hi in ranges)


def _detect_gpu_numa_topology(
    num_gpus: int,
) -> Optional[Dict[int, Tuple[int, Set[int]]]]:
    """Query NVML for each GPU's NUMA node and local CPU set.

    Returns {gpu_index: (numa_node, cpu_set)} covering all of 0..num_gpus-1, or
    ``None`` if NVML is unavailable or the topology of any GPU is unknown.
    """
    try:
        # import_module resolves through sys.modules, which keeps this
        # mockable in tests even after the real module was imported elsewhere.
        import importlib

        pynvml = importlib.import_module("ray._private.thirdparty.pynvml")
    except Exception:  # noqa: BLE001 - fail open without NVML.
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001
        return None
    try:
        import os

        device_count = pynvml.nvmlDeviceGetCount()
        if device_count < num_gpus:
            return None
        # NVML expects the CPU set size in units of 64-bit words.
        num_cpus = os.cpu_count() or 64
        cpu_set_size = max(1, (num_cpus + 63) // 64)
        result: Dict[int, Tuple[int, Set[int]]] = {}
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            numa_node = int(pynvml.nvmlDeviceGetNumaNodeId(handle))
            if numa_node < 0:
                return None  # Unknown node for this GPU: fail open.
            cpus = cpu_affinity_words_to_cpu_set(
                pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)
            )
            if not cpus:
                return None
            result[i] = (numa_node, cpus)
        return result
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def build_gpu_numa_topology_spec(num_gpus: int) -> Optional[str]:
    """Build the --gpu_numa_topology raylet flag value for this node.

    Returns the spec string, or ``None`` when there are no GPUs, detection is
    unavailable, or any GPU's topology is unknown (fail open).
    """
    try:
        num_gpus = int(num_gpus)
    except (TypeError, ValueError):
        return None
    if num_gpus <= 0:
        return None
    topology = _detect_gpu_numa_topology(num_gpus)
    if topology is None:
        return None
    entries = []
    for i in range(num_gpus):
        numa_node, cpus = topology[i]
        cpulist = cpu_set_to_cpulist(cpus)
        if not cpulist:
            return None
        entries.append(f"{numa_node}:{cpulist}")
    return ";".join(entries)


def get_gpu_numa_topology_spec_if_enabled(num_gpus: int) -> Optional[str]:
    """Entry point for node startup: the spec, or None unless opted in."""
    if not gpu_numa_affinity_enabled():
        return None
    try:
        spec = build_gpu_numa_topology_spec(num_gpus)
    except Exception:  # noqa: BLE001
        logger.debug("GPU NUMA topology detection failed", exc_info=True)
        return None
    if spec:
        logger.info("Detected GPU NUMA topology: %s", spec)
    else:
        logger.debug(
            "GPU NUMA affinity enabled but topology unavailable; "
            "workers will not be NUMA-bound on this node."
        )
    return spec
