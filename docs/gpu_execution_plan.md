# GPU Execution Plan: SA One-Thread-Per-Sample

This document captures the execution plan for implementing CUDA acceleration
for simulated annealing (`sa_backend="gpu_sa"`), where each CUDA thread runs
one sample/read.

## Objectives

- Implement a GPU path for SA with one CUDA thread per sample.
- Keep Python API stable via `sa_backend`.
- Maintain CPU defaults and CPU-only build compatibility.
- Enable incremental correctness-first delivery before optimization.

## Constraints (Phase 1)

- Initial GPU support targets:
  - `randomize_order=False` (sequential variable order)
  - `proposal_acceptance_criteria="Metropolis"`
- `interrupt_function` not supported for GPU in Phase 1.
- Deterministic per-sample seeding within GPU backend.

---

## Milestones

### 1. Build + Feature Flag Wiring

1. Add optional CUDA build path behind env flag:
   - `DWAVE_SA_ENABLE_CUDA=1`
2. Detect `nvcc` from `CUDA_HOME`/`CUDA_PATH` or `PATH`.
3. Compile `dwave/samplers/sa/src/gpu_sa.cu` to object and link into
   `dwave.samplers.sa.simulated_annealing`.
4. Define `DWAVE_SA_WITH_CUDA` when linked.

Status: **Started**

---

### 2. Host-Side Data Flattening

1. In `gpu_sa.cpp`, convert graph inputs to flat CSR-like arrays:
   - `neighbor_offsets[num_vars+1]`
   - `neighbor_indices[nnz]`
   - `neighbor_couplings[nnz]`
2. Validate coupler shapes and indices.
3. Switch CUDA call interface to POD pointers/sizes.

Status: **Started**

---

### 3. CUDA Kernel (One Thread = One Sample)

1. Launch kernel over `num_samples`.
2. Each thread:
   - Uses per-thread RNG state.
   - Computes initial `delta_energy` for all variables.
   - Runs betas/sweeps/variables with Metropolis updates.
   - Applies neighbor `delta_energy` updates.
   - Computes final sample energy.
3. Copy final states + energies back to host.

Status: **Started**

---

### 4. Runtime Status Codes and Python Mapping

Use explicit status codes to avoid C++ exception boundary issues:

- `-2`: unsupported interrupt callback in GPU backend
- `-3`: built without CUDA support
- `-4`: unsupported options in GPU Phase 1
- `-6`: CUDA runtime failure (alloc/copy/kernel)

Map these in Cython to clear Python exceptions.

Status: **Started**

---

### 5. Correctness Validation

1. Add GPU-specific tests (with CUDA availability checks).
2. Validate:
   - output shapes/types
   - deterministic behavior with fixed seeds (within backend)
   - energy consistency against host recomputation
3. Compare statistical parity to CPU backend on small problems.

Status: Pending

---

### 6. Performance Benchmarking

1. Reuse/add benchmark scripts to compare:
   - `cpu_sa`, `fast_cpu_sa`, `gpu_sa`
2. Sweep `num_reads` and report:
   - end-to-end wall time
   - transfer overhead vs kernel time (if instrumented)

Status: Pending

---

### 7. Phase 2 Optimizations

1. Reduce per-thread memory pressure from `delta_energy`.
2. Improve occupancy and memory coalescing.
3. Consider block-level cooperative variants if needed.
4. Add support for:
   - `randomize_order=True`
   - `proposal_acceptance_criteria="Gibbs"`

Status: Pending

---

## Implementation Notes (Current Branch)

- `gpu_sa.cpp` now prepares CSR-style adjacency on host and calls a POD-based
  CUDA entrypoint when `DWAVE_SA_WITH_CUDA` is defined.
- `gpu_sa.cu` now contains a first-pass kernel implementation with one thread
  per sample.
- `setup.py` now has optional CUDA object compilation/linking path controlled
  by `DWAVE_SA_ENABLE_CUDA`.
- `simulated_annealing.pyx` now maps additional GPU status codes to Python
  exceptions.

## Usage (When CUDA Toolchain Is Present)

```bash
export DWAVE_SA_ENABLE_CUDA=1
export CUDA_HOME=/usr/local/cuda
python setup.py build_ext --inplace
```

Then:

```python
from dwave.samplers import SimulatedAnnealingSampler
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, sa_backend="gpu_sa")
```
