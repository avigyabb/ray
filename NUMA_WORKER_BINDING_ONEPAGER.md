# One-Pager: NUMA-Correct Worker Startup for GPU Actors

**Scope:** Deliver GPU assignment at worker *spawn* instead of first-task execution, then pin the
worker's CPUs and memory policy to the GPUs' NUMA node before it allocates anything.
**Explicit non-goals:** no changes to actor scheduling or GPU *selection* (first-fit stays), no
strict/soft scheduling modes, no cross-node awareness.

## Problem

For a GPU actor, the raylet picks the GPU instances **before** the worker process exists
(`local_lease_manager.cc:362` allocates → `:387` stores on the lease → `:393` `PopWorker` →
`worker_pool.cc:1597` `StartWorkerProcess`). But the worker only *learns* its GPUs when the
actor-creation task runs in Python (`_raylet.pyx` → `set_visible_accelerator_ids()`). By then the
interpreter has booted and imported heavy libraries (torch alone is hundreds of MB), and Linux
first-touch has placed all of that memory on whatever NUMA node the kernel happened to run the
process on — possibly not the node the GPUs are attached to. Result: cross-socket traffic for the
actor's lifetime. Runtime `setenv` at task time is also the thread-safety hazard behind
[#63138](https://github.com/ray-project/ray/issues/63138).

## Prior art

- **[#63327](https://github.com/ray-project/ray/pull/63327)** (WIP, **reopened 2026-07-18 — active**)
  — moves accelerator env vars from task execution to worker startup for #63138: threads
  `allocated_instances_` through `PopWorkerRequest`/`Worker`, injects `CUDA_VISIBLE_DEVICES`/
  `OMP_NUM_THREADS` via the runtime env agent, deletes the task-time setenv path, and adds
  `RESOURCE_MISMATCH`/`ALLOCATED_INSTANCES_MISMATCH` reuse guards. This is effectively our PR 1;
  we coordinate with and build on it rather than duplicate it. No human review yet (last push
  2026-07-19); two unresolved findings from the automated review bot look real and need
  addressing: prestart-reuse over-strictness (`node_manager.cc`; mitigate by scoping strict
  matching to accelerator workers only) and an agent env-isolation bug (`pop(env_var)` inherits
  the raylet's value instead of clearing it). Verify: whether workers with an empty runtime env
  bypass the agent (if so, PR 2 also needs the raylet spawn-env channel).
- **[tohtana's branch](https://github.com/ray-project/ray/compare/master...tohtana:ray:tohtana/ray-core-numa-affinity-opt-in-squashed)**
  — `pynvml.nvmlDeviceGetCpuAffinity` + `os.sched_setaffinity` at *task* time, tasks only. We reuse
  its bind-set computation but apply it at *process birth* and add memory policy.

## Design — three small changes

**1. Raylet: pass the allocation at spawn (plumbing).** Thread the lease's already-computed GPU
instance ids from `PopWorker` into `StartWorkerProcess` → the env assembly at
`worker_pool.cc:394-459`. Set the accelerator env var (`CUDA_VISIBLE_DEVICES`, via the existing
per-accelerator env-var names) plus a marker (`RAY_ACCELERATOR_IDS_SET_AT_SPAWN=1`) in the child's
environment. Applies only to lease-driven fresh starts; prestarted/idle-reused workers keep today's
behavior (see §Reuse).

**2. Raylet: bind the child pre-exec (the NUMA fix).** In `Process::Spawnvpe`'s child branch
(`process.cc:257-262`, next to the existing `add_to_cgroup` hook), when the lease's GPUs all map to
one NUMA node: `sched_setaffinity(0, <GPU-local cpuset>)` and `set_mempolicy(MPOL_PREFERRED,
<node>)` (raw syscall — no libnuma dependency). Both survive `execve`, so **every page the worker
ever faults — interpreter, imports, model — is node-local from byte zero**. `MPOL_PREFERRED` (not
`BIND`) so a full node degrades to remote allocation instead of OOM. The GPU→NUMA/cpuset table is
detected once at node start by the Python bootstrap (pynvml `nvmlDeviceGetNumaNodeId` /
sysfs cpulists — code exists from the prior branch) and handed to the raylet via startup config;
empty table ⇒ everything below is a no-op.

**3. Python: skip task-time setenv when the marker is present.** In the `_raylet.pyx` execution
callback, if `RAY_ACCELERATOR_IDS_SET_AT_SPAWN` is set, skip `set_visible_accelerator_ids()` (fixes
the #63138 hazard for this path). Keep tohtana-style task-time bind/reset as the **fallback** for
reused workers — affinity still helps future allocations even though startup memory is already
placed.

## Reuse & correctness

Actor workers die with their actor, so bind-once-at-spawn is safe for the whole actor lifetime. The
risk #63327 hit: a worker spawned with `CUDA_VISIBLE_DEVICES=X` must never serve a lease with GPUs
Y. Mitigation: a spawn-bound worker is tagged with its allocated instances and is only handed to the
lease that started it (dedicated-worker path); if the pop races to a different lease, mark unfit
(`WorkerUnfitForLeaseReason`) and start a fresh worker. GPUs spanning NUMA nodes, non-Linux, no
topology, fractional-GPU multi-tenant workers ⇒ skip binding, log at debug (fail-open, zero behavior
change when disabled).

## Rollout

Opt-in: `RAY_EXPERIMENTAL_GPU_NUMA_AFFINITY=1` (binding), `..._MEMBIND=0/1` (memory policy,
default on when binding is on).
**PR 1:** #63327 (active upstream) — coordinate: help resolve its open bot findings (scope strict
matching to accelerator workers; fix agent env isolation) rather than open a competing PR.
**PR 2:** topology table + pre-exec `sched_setaffinity`/`set_mempolicy`, consuming
`PopWorkerRequest.allocated_instances_` from #63327; cover the empty-runtime-env path if the
agent is bypassed.
**PR 3:** task-time fallback binding for reused workers (port of tohtana, actors included).

**Tests:** C++ unit (env assembly, child-bind args, fitness guard); Python integration with fake
GPUs + injected topology (env var present at actor process birth; `sched_getaffinity` ⊆ expected
cpuset via `/proc/self/status`); real multi-socket validation (`g4dn.metal`) measuring
`numastat -p` locality before/after.
