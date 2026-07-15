# Proposal: NUMA-Aware GPU Scheduling & CPU Affinity for Ray Actors and Tasks

Status: Draft
Author: Avigya Basnet
Scope: Ray Core (raylet scheduler + worker startup)

---

## 1. Motivation

On multi-socket GPU hosts (e.g. dual-socket DGX-style nodes), each GPU is
physically attached to the PCIe/NUMA domain of one CPU socket. When a worker
runs on cores in socket A but uses GPUs attached to socket B, every
host↔device transfer and pinned-memory allocation crosses the inter-socket
link (UPI/xGMI). This adds latency, wastes cross-socket bandwidth, and hurts
data-loading and H2D-heavy workloads.

Ray today is **NUMA-blind** in two ways:

1. **Selection** — when an actor/task requests GPUs, the raylet picks GPU slots
   greedily in index order with no topology awareness
   (`NodeResourceInstanceSet::TryAllocate`,
   `src/ray/common/scheduling/resource_instance_set.cc:295`). A 2-GPU actor can
   get GPUs 0 and 4 that sit on different sockets.
2. **Binding** — the worker process is never pinned to any CPU/NUMA domain
   (`Process::Spawnvpe`, `src/ray/util/process.cc`, has no `sched_setaffinity`
   call). The kernel scheduler is free to run it anywhere.

This proposal makes Ray **prefer or require** GPUs that share a NUMA node when
scheduling an actor/task, and **pin the worker's CPU affinity** to the cores
local to those GPUs.

### Prior art

An existing branch
([tohtana/ray-core-numa-affinity-opt-in](https://github.com/ray-project/ray/compare/master...tohtana:ray:tohtana/ray-core-numa-affinity-opt-in-squashed))
implements the **binding** half in pure Python: it uses
`pynvml.nvmlDeviceGetCpuAffinity()` + `os.sched_setaffinity()` in
`_raylet.pyx`, gated by `RAY_EXPERIMENTAL_NVIDIA_GPU_NUMA_AFFINITY`. It only
binds **non-actor tasks** (it deliberately skips actors) and does **not** touch
GPU selection. This proposal generalizes that work: it adds the actor path and
adds NUMA-aware selection with a soft/strict scheduling flag.

## 2. Goals / Non-goals

**Goals**

- NUMA-aware GPU **selection**: prefer (or require) a GPU set co-located on one
  NUMA node, and prefer a socket that also satisfies the actor's CPU demand.
- **CPU-affinity binding**: pin the worker to the cores local to its GPUs, for
  **both actors and non-actor tasks**.
- A per-workload **flag with two modes**:
  - **soft** — best-effort; fall back to today's first-fit if no affine set is
    available *now*. Never blocks.
  - **strict** — only schedule when a NUMA-affine set is available; otherwise
    the actor stays **pending** until it can be satisfied (blocks).
- Backward compatible: default off; unchanged behavior when disabled or on
  unsupported platforms.

**Non-goals (v1)**

- NUMA **memory** binding (`libnuma`/`numactl` membind). CPU affinity only; the
  kernel will still favor local allocation for a CPU-pinned process. Memory
  binding is a documented follow-up.
- Cross-node topology / network locality (that's `scheduling_strategy`'s job).
- Non-NVIDIA topology-aware *selection* in v1 (AMD binding is feasible via
  `pyamdsmi` and is listed as a follow-up; the framework is vendor-neutral).

## 3. Current code path (grounding)

Selection → allocation → env, end to end:

| Step | Location |
| --- | --- |
| GPU slot picking (first-fit, index order) | `resource_instance_set.cc:295` `NodeResourceInstanceSet::TryAllocate` |
| Multi-resource orchestration (GPU + CPU + wildcard) | `resource_instance_set.cc:140-293` |
| Allocation bound to worker | `local_lease_manager.cc:989-1052` (`SetAllocatedInstances` / `SetLifetimeAllocatedInstances`) |
| Allocation serialized into lease reply | `local_lease_manager.cc:1028-1046` (`resource_mapping` / instance `index`) |
| Flows to worker | `normal_task_submitter.cc:530` → `task_receiver.cc:182-193` → `core_worker.cc:2933` (`resource_ids_`) |
| Exposed to Python | `_raylet.pyx:4243` `resource_ids()` → `runtime_context.get_accelerator_ids()` |
| `CUDA_VISIBLE_DEVICES` set (tasks) | `_raylet.pyx:2351-2356` → `utils.set_visible_accelerator_ids()` |
| Worker fork/exec (no affinity today) | `worker_pool.cc:394-459` (env) → `process.cc:257` (`fork`) |
| GPU→NUMA topology source (available, unused) | `pynvml.nvmlDeviceGetNumaNodeId` / `nvmlDeviceGetCpuAffinity`; `pyamdsmi.smi_get_device_topo_numa_affinity` |

Two useful facts: the GPU→NUMA mapping already ships in vendored `pynvml`, and
the per-worker **CPU** instance indices are already tracked raylet-side
(`local_lease_manager.cc:1096-1153`). `set_omp_num_threads_if_unset`
(`utils.py:235`) is a direct precedent for "read assigned instance indices →
configure the process."

## 4. Design overview

Two loosely-coupled components:

- **Component A — NUMA-aware selection (C++, raylet).** Make `TryAllocate`
  topology-aware and honor a per-request soft/strict mode.
- **Component B — CPU-affinity binding (Python, worker).** After the worker
  knows its GPUs, pin `sched_setaffinity` to the local cores, for actors and
  tasks.

They are independently useful and independently landable:

- A alone → GPUs are co-located, kernel still free to place the process.
- B alone → matches the prior-art branch (bind to whatever GPUs you got).
- A + B → the full win: co-located GPUs **and** a pinned worker.

Both are fed by a new **node topology map** (Component 0).

### Component 0 — Node topology map

The raylet needs, per node: `gpu_instance_index → numa_node` and
`cpu_instance_index → numa_node`. This does not exist today.

- **Where computed:** at node startup on the Python side, where accelerators
  are already detected (`_private/accelerators/nvidia_gpu.py`). Add a helper
  that builds:
  - GPU→node via `pynvml.nvmlDeviceGetNumaNodeId` (fallback:
    `nvmlDeviceGetCpuAffinity` → intersect with node cpulists).
  - CPU→node via `/sys/devices/system/node/node*/cpulist` (or `libnuma`
    `numa_node_of_cpu`, but we avoid the dep — sysfs parsing is enough).
- **How delivered to the raylet:** pass the serialized map through node
  startup config (same channel that already carries accelerator/resource info
  to the raylet), stored alongside `NodeResourceInstanceSet`.
- **Degradation:** if topology is unavailable (no NUMA, container without
  sysfs, NVML error), the map is empty → selection silently behaves like today
  and binding is skipped. Fail-open.

### Component A — NUMA-aware selection

**Per-request mode plumbing.** Add a `numa_affinity` field to the resource
request / task spec, carrying `NONE | SOFT | STRICT`. It is set from the remote
option (§6), travels with the lease request, and reaches `TryAllocate`.

**Whole-GPU picking.** Today's whole-unit loop
(`resource_instance_set.cc:325-337`) takes GPUs first-fit. Replace with
node-grouped selection:

1. Bucket available GPU instance indices by NUMA node.
2. Prefer a **single node** whose available GPU count ≥ demand. Among
   candidates, prefer the node that also has ≥ the request's CPU demand
   available (this realizes "pick a socket that satisfies the GPU *and* CPU
   requirement"). Tie-break by most-free to reduce fragmentation.
3. If no single node fits, prefer the **minimum number of nodes**.

**Joint CPU co-allocation.** For the socket-satisfies-CPU goal, the
orchestration layer (`TryAllocate(ResourceSet)`, `resource_instance_set.cc:140`)
chooses a target node `N` first, then constrains **both** the GPU sub-allocation
and the CPU sub-allocation to instances on `N`. This keeps GPU and CPU picks
consistent so Component B can bind to a single clean core set.
*Phasing:* v1 may ship GPU-only co-location (bind to the union of the GPUs'
node cpulists); the joint CPU-pinned-to-same-node refinement is a fast follow.

**Soft vs strict** — the entire difference is the fallback branch:

```text
alloc = numa_aware_try_allocate(demand, mode)
if alloc is None:
    if mode == STRICT:
        return None          # lease not granted → actor stays pending ("blocks")
    else:  # SOFT or NONE
        alloc = first_fit_try_allocate(demand)   # today's behavior
return alloc
```

**Strict "blocking" reuses existing semantics.** A `nullopt` from `TryAllocate`
means the lease isn't granted; the actor sits in the existing pending/infeasible
queue and is retried as resources free up — identical to an actor requesting
more GPUs than are currently free. **No new blocking machinery.** Open question:
whether to add an optional scheduling timeout (§9); v1 recommends indefinite
pending for consistency with Ray today.

### Component B — CPU-affinity binding

Mirror the prior-art branch, generalized to actors, in `_private/utils.py` +
call sites in `_raylet.pyx`.

- `set_gpu_numa_cpu_affinity_if_enabled()`:
  1. Read assigned GPU ids (`runtime_context.get_accelerator_ids()`), which
     already flow from `resource_ids_`.
  2. Guard: enabled? Linux? whole (non-fractional) GPUs? at least one GPU?
  3. For each GPU get its local CPU set (`nvmlDeviceGetCpuAffinity`, or node
     cpulist from Component 0). Intersect the union with the process's current
     `os.sched_getaffinity(0)` so we never *widen* affinity.
  4. If the resulting set is non-empty, `os.sched_setaffinity(0, cpu_set)`.
- `reset_gpu_numa_cpu_affinity(original)`: restore for the task path.

**Actor vs task application:**

- **Actor:** bind **once** at actor-creation execution and **leave it** for the
  actor's lifetime (allocation is lifetime-scoped;
  `_raylet.pyx` actor-creation branch). Do **not** reset per task.
- **Task:** bind at task start, **reset after** (like prior art), because slots
  differ per task (`task_receiver.cc:180-181`).

**Soft-mode caveat:** after a soft fallback the GPUs may span nodes. Binding
degrades gracefully — bind to the union of their cpulists, or skip if the union
is empty. Log at debug so users can see when affinity wasn't applied.

## 5. Interaction between A and B

In **strict** mode, selection guarantees a single-node GPU set, so binding is
exact. In **soft** mode, binding does the best it can with whatever was
allocated. B is safe to ship without A (matches prior art); A is safe without B
(kernel placement unchanged). This is why they're separable PRs.

## 6. User-facing API

Two controls:

1. **Cluster enable (binding), env var** — cheap global on-switch, consistent
   with prior art:
   `RAY_EXPERIMENTAL_GPU_NUMA_AFFINITY=1`.
2. **Per-workload selection mode, remote option** — strict/soft is inherently
   per-actor:

   ```python
   @ray.remote(num_gpus=2, num_cpus=8, numa_affinity="strict")
   class Trainer: ...

   # or per-instance
   Trainer.options(numa_affinity="soft").remote()
   ```

   Accepts `None` (default), `"soft"`, `"strict"`. Plumbed into the task spec /
   resource request → raylet `TryAllocate` mode.

*Alternative (simpler v1):* a single global env
`RAY_GPU_NUMA_AFFINITY=off|soft|strict` for both selection and binding, deferring
per-actor granularity. Recommend the per-option API since strict's whole value
is per-actor blocking, but the global env is an acceptable first cut if the
task-spec plumbing is deferred.

## 7. Edge cases

- **Non-Linux / no NUMA / single socket:** no-op (fail-open).
- **Fractional GPUs:** skip binding; selection uses best-fit as today.
- **`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`:** CPU binding is independent
  and still applies as long as assigned ids are known.
- **Placement groups:** GPU resources are already sliced per-bundle
  (`GPU_<pg_id>` wildcards, `resource_instance_set.cc:152-181`); node-grouping
  operates within the bundle's instances. Verify STRICT_PACK interplay.
- **CPU-only actors:** out of scope (no GPU → nothing to co-locate).
- **`OMP_NUM_THREADS`:** already derived from assigned CPUs
  (`utils.py:235`); ensure our narrower affinity stays consistent with the
  thread count it advertises.
- **AMD:** binding via `pyamdsmi`; topology via
  `smi_get_device_topo_numa_affinity`. Framework stays vendor-neutral;
  NVIDIA-first.

## 8. Testing

- **Unit (C++):** extend `resource_instance_set_test.cc` with a synthetic
  topology map: assert soft falls back, strict returns `nullopt`, joint CPU+GPU
  node selection, tie-breaking, fragmentation.
- **Unit (Python):** mirror the prior-art `test_nvidia_gpu_numa_affinity.py`;
  add actor-lifetime binding (bind once, persists across actor tasks), soft
  degradation, guard conditions. Mock `pynvml` / `sched_*`.
- **Integration:** on a real multi-socket GPU node, assert
  `os.sched_getaffinity(0)` ⊆ the GPUs' local cpulist; strict actor stays
  pending when no affine set is free, then schedules once one frees.
- **Regression:** feature-off path byte-for-byte unchanged.

## 9. Open questions

1. **Strict timeout** — indefinite pending (recommended, consistent) vs an
   optional scheduling deadline (new plumbing, clearer failure for users)?
2. **Global env vs per-actor option** for the selection mode — how much
   task-spec plumbing to take on in v1.
3. **Fragmentation policy** — should soft mode prefer the *least-loaded* node or
   the *first* fitting node? Affects packing under mixed workloads.
4. **Topology delivery** — reuse the existing accelerator-info startup channel,
   or a dedicated field on the node resource message?

## 10. Rollout plan

- **PR 1 — Component 0 + B (binding), actors + tasks.** Env-gated, no scheduler
  change. Lands the prior-art win plus the actor path. Lowest risk.
- **PR 2 — Component A soft mode.** NUMA-grouped GPU selection with first-fit
  fallback; global env or `numa_affinity="soft"`. Behavior only *improves*
  locality, never blocks.
- **PR 3 — Component A strict mode + per-actor option.** Task-spec plumbing for
  `numa_affinity`, strict `nullopt` → pending, joint CPU+GPU node selection.
- **Follow-ups:** NUMA memory binding (`libnuma`), AMD selection, optional
  strict timeout.

---

### Contribution-policy checklist (per `AGENTS.md`)

- [ ] Confirm no open PR/issue already covers NUMA affinity beyond the
      referenced draft branch (`gh pr list --search "numa"`).
- [ ] Human-authored, reviewed line-by-line; tests run locally and reported.
- [ ] PR description states why it's not a duplicate, tests run, and that AI
      assistance was used.
- [ ] Commit with `-s` (DCO); run `pre-commit`.
