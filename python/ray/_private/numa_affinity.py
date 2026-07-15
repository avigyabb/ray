"""NUMA / CPU-affinity binding for GPU workers.

When a worker is assigned GPUs, the cores physically local to those GPUs live on
one CPU socket / NUMA node. Pinning the worker's CPU affinity to those cores
avoids cross-socket host<->device traffic. This module contains the (Linux-only)
mechanism for computing and applying that affinity.

The heavy lifting is split into small pure functions so the logic can be unit
tested without a GPU or a Linux host:

  * ``compute_gpu_local_cpu_set`` -- pure: (assigned GPUs, GPU->CPU topology,
    current affinity) -> the CPU set to pin to (or ``None`` to leave alone).
  * ``get_gpu_cpu_affinity_via_nvml`` -- queries NVML for the GPU->CPU topology.
  * ``set_gpu_numa_cpu_affinity`` / ``reset_cpu_affinity`` -- apply / restore.

Only ``os.sched_setaffinity`` is used (CPU affinity). NUMA *memory* binding
(libnuma / numactl) is intentionally out of scope for v1.

Top-level imports are restricted to the standard library so this module can be
imported and tested in isolation; ray-specific imports are done lazily inside
the functions that need them.
"""

import logging
import os
import sys
from typing import Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# Cluster-wide opt-in for CPU-affinity binding of GPU workers.
RAY_GPU_NUMA_AFFINITY_ENV_VAR = "RAY_EXPERIMENTAL_GPU_NUMA_AFFINITY"

# A GPU->local-CPU topology map: global GPU index (as a string, matching the
# accelerator-id representation Ray uses) -> set of local logical CPU indices.
GpuToCpuMap = Dict[str, Set[int]]


def gpu_numa_affinity_enabled() -> bool:
    """Whether GPU NUMA CPU-affinity binding is opted in via env var."""
    from ray._private.ray_constants import env_bool

    return env_bool(RAY_GPU_NUMA_AFFINITY_ENV_VAR, False)


def numa_affinity_binding_supported() -> bool:
    """CPU-affinity binding requires Linux' ``os.sched_*affinity`` APIs."""
    return (
        sys.platform.startswith("linux")
        and hasattr(os, "sched_setaffinity")
        and hasattr(os, "sched_getaffinity")
    )


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


def compute_gpu_local_cpu_set(
    gpu_ids: Iterable[str],
    gpu_to_cpus: GpuToCpuMap,
    allowed_cpus: Optional[Set[int]] = None,
) -> Optional[Set[int]]:
    """Compute the CPU set to pin a worker to, given its GPUs and the topology.

    Args:
        gpu_ids: GPUs assigned to this worker (physical/global indices).
        gpu_to_cpus: Map of GPU index -> set of local CPU indices.
        allowed_cpus: The process' currently-allowed CPU set. If provided, the
            result is intersected with it so we never *widen* affinity beyond
            an existing restriction (e.g. a cgroup/cpuset pin).

    Returns:
        The set of CPUs to bind to, or ``None`` if affinity should be left
        untouched -- when there are no GPUs, the topology of any assigned GPU is
        unknown, or the local CPUs fall entirely outside ``allowed_cpus``.
    """
    ids = [str(g) for g in gpu_ids]
    if not ids:
        return None

    # Union the local CPUs of every assigned GPU. With NUMA-aware (strict)
    # scheduling all GPUs share a socket, so this is a single node's cores. With
    # a soft fallback the GPUs may span sockets; binding to the union is the
    # best we can do and still beats no binding.
    local_cpus: Set[int] = set()
    for gpu_id in ids:
        cpus = gpu_to_cpus.get(gpu_id)
        if not cpus:
            # Topology for at least one assigned GPU is unknown -- don't guess.
            return None
        local_cpus |= set(cpus)

    if not local_cpus:
        return None

    if allowed_cpus is not None:
        local_cpus &= set(allowed_cpus)
        if not local_cpus:
            # GPU-local CPUs are entirely outside the process' cpuset; leaving
            # affinity untouched is safer than pinning to nothing.
            return None

    return local_cpus


def get_gpu_cpu_affinity_via_nvml(num_cpus: Optional[int] = None) -> GpuToCpuMap:
    """Query NVML for each GPU's local CPU set. Returns ``{}`` on any failure."""
    result: GpuToCpuMap = {}
    try:
        import ray._private.thirdparty.pynvml as pynvml
    except Exception:  # noqa: BLE001 - fail open if NVML is unavailable.
        return result

    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001
        return result

    try:
        if num_cpus is None:
            num_cpus = os.cpu_count() or 64
        # NVML expects the CPU set size in units of 64-bit words.
        cpu_set_size = max(1, (num_cpus + 63) // 64)
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            words = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)
            result[str(i)] = cpu_affinity_words_to_cpu_set(words)
    except Exception:  # noqa: BLE001
        result = {}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return result


def set_gpu_numa_cpu_affinity(
    gpu_ids: Iterable[str],
    gpu_to_cpus: Optional[GpuToCpuMap] = None,
) -> Optional[Set[int]]:
    """Pin the current process to the CPUs local to its assigned GPUs.

    Args:
        gpu_ids: GPUs assigned to this worker.
        gpu_to_cpus: Optional precomputed topology map; queried from NVML if not
            provided.

    Returns:
        The process' *original* affinity set if it was changed (so a caller can
        restore it after a task), or ``None`` if affinity was left untouched.
    """
    if not numa_affinity_binding_supported():
        return None

    if gpu_to_cpus is None:
        gpu_to_cpus = get_gpu_cpu_affinity_via_nvml()

    original = set(os.sched_getaffinity(0))
    target = compute_gpu_local_cpu_set(gpu_ids, gpu_to_cpus, allowed_cpus=original)
    if target is None or target == original:
        return None

    try:
        os.sched_setaffinity(0, target)
    except OSError:
        logger.debug("Failed to set GPU NUMA CPU affinity", exc_info=True)
        return None

    logger.debug(
        "Pinned worker to GPU-local CPUs %s for GPUs %s",
        sorted(target),
        list(gpu_ids),
    )
    return original


def reset_cpu_affinity(original: Optional[Set[int]]) -> None:
    """Restore a CPU affinity set previously returned by ``set_...`` (no-op if
    ``None``)."""
    if not original or not numa_affinity_binding_supported():
        return
    try:
        os.sched_setaffinity(0, original)
    except OSError:
        logger.debug("Failed to reset CPU affinity", exc_info=True)


def get_assigned_gpu_ids() -> List[str]:
    """Return the GPU ids assigned to the current worker (``[]`` if none).

    Note: these ids are in Ray's accelerator-id space, which for a worker that
    Ray started with no pre-existing ``CUDA_VISIBLE_DEVICES`` matches the
    physical/NVML device index. If the environment pre-set
    ``CUDA_VISIBLE_DEVICES``, the caller-provided topology map must be in the
    same (relative) index space.
    """
    import ray

    try:
        accel = ray.get_runtime_context().get_accelerator_ids()
    except Exception:  # noqa: BLE001
        return []
    return [str(g) for g in accel.get("GPU", [])]


def maybe_bind_worker_to_gpu_numa() -> Optional[Set[int]]:
    """Bind the worker to its GPUs' local CPUs, if the feature is enabled.

    Safe to call unconditionally from the task-execution path: it no-ops unless
    the feature is opted in, the platform supports it, and the worker actually
    has GPUs with known topology.

    Returns the process' original affinity set (so a normal task can restore it
    afterwards) or ``None`` if affinity was left untouched.
    """
    if not gpu_numa_affinity_enabled() or not numa_affinity_binding_supported():
        return None
    gpu_ids = get_assigned_gpu_ids()
    if not gpu_ids:
        return None
    return set_gpu_numa_cpu_affinity(gpu_ids)
