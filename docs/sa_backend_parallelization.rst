.. _sa_backend_parallelization:

=========================================
SA Backend Parallelization (Low-Level)
=========================================

Scope
=====

This document captures the implemented backend split and parallelization
plumbing for simulated annealing under ``dwave/samplers/sa/``.

Implemented Backends
====================

The SA stack now supports three backend options exposed through Python:

* ``cpu_sa``: baseline CPU backend (default).
* ``fast_cpu_sa``: parallel CPU backend.
* ``gpu_sa``: GPU backend entrypoint wired through C++/Cython (CUDA kernel
  file present but currently a placeholder in this branch).

Public API Changes
==================

``SimulatedAnnealingSampler.sample(...)`` now accepts:

* ``sa_backend: str = "cpu_sa"``

Accepted values:

* ``"cpu_sa"``
* ``"fast_cpu_sa"``
* ``"gpu_sa"``

Validation is enforced in ``dwave/samplers/sa/sampler.py``.

Sampler parameter metadata now includes:

* ``self.parameters["sa_backend"] = []``

Cython Bridge and Dispatch
==========================

File: ``dwave/samplers/sa/simulated_annealing.pyx``

Changes:

* Linked additional C++ sources:

  * ``src/cpu_sa.cpp``
  * ``src/fast_cpu_sa.cpp``
  * ``src/gpu_sa.cpp``

* Added C extern declarations for:

  * ``cpu_general_simulated_annealing(...)``
  * ``fast_cpu_general_simulated_annealing(...)``
  * ``gpu_general_simulated_annealing(...)``

* Added ``sa_backend`` function argument to Python-facing
  ``simulated_annealing(...)`` wrapper.
* Dispatch logic selects backend at runtime based on ``sa_backend``.

Backend status/error codes returned by C++ and mapped in Cython:

* ``-1``: unknown backend (raises ``ValueError``).
* ``-2``: ``interrupt_function`` unsupported for ``fast_cpu_sa``
  (raises ``ValueError``).
* ``-3``: ``gpu_sa`` requested without CUDA-enabled build
  (raises ``RuntimeError``).

C++ Backend Structure
=====================

Baseline CPU backend
--------------------

Files:

* ``dwave/samplers/sa/src/cpu_sa.h``
* ``dwave/samplers/sa/src/cpu_sa.cpp``

Change:

* Exported entrypoint renamed to
  ``cpu_general_simulated_annealing(...)`` (from
  ``general_simulated_annealing(...)``) to avoid symbol collisions when
  multiple SA backends are linked together.

Fast CPU backend
----------------

Files:

* ``dwave/samplers/sa/src/fast_cpu_sa.h``
* ``dwave/samplers/sa/src/fast_cpu_sa.cpp``

Implementation details:

* Backend entrypoint: ``fast_cpu_general_simulated_annealing(...)``.
* Parallelization target implemented: parallel execution over
  ``num_samples`` (independent reads).
* Uses OpenMP parallel-for only when both conditions are true:

  * OpenMP is available at compile time (``_OPENMP``), and
  * ``num_samples > 10000`` at runtime.

* For ``num_samples <= 10000``, uses ``std::thread`` + atomic work queue.
* Falls back to ``std::thread`` + atomic work queue when OpenMP is unavailable.
* Each sample uses thread-local RNG state seeded deterministically from
  ``seed`` and sample index.
* ``interrupt_function`` is currently unsupported in this backend (returns
  status ``-2``).

GPU backend plumbing
--------------------

Files:

* ``dwave/samplers/sa/src/gpu_sa.h``
* ``dwave/samplers/sa/src/gpu_sa.cpp``
* ``dwave/samplers/sa/src/gpu_sa.cu``

Implementation details:

* Backend entrypoint: ``gpu_general_simulated_annealing(...)``.
* ``gpu_sa.cpp`` dispatches to ``gpu_general_simulated_annealing_cuda(...)``
  only when compiled with ``DWAVE_SA_WITH_CUDA``.
* Without CUDA-enabled compile flags, backend returns status ``-3`` and Python
  raises ``RuntimeError``.
* ``gpu_sa.cu`` is present as a kernel-integration placeholder and currently
  throws if reached.

Build-System Changes
====================

File: ``setup.py``

Changes:

* Fixed extension link-arg assignment bug:

  * ``ext.extra_link_args`` is now set correctly (instead of accidentally
    writing into ``extra_compile_args`` twice).

* Added SA-extension OpenMP flags:

  * MSVC: ``/openmp``
  * Unix (non-macOS): ``-fopenmp`` for compile and link

Current CUDA status:

* No CUDA compilation path is yet enabled in ``setup.py`` for ``.cu`` files.
  GPU API plumbing exists; CUDA kernel build integration remains future work.

Parallelization Validation Notes
================================

The implementation aligns with the validated parallelization priorities:

1. ``num_samples`` parallelization: implemented in ``fast_cpu_sa``.
2. Initial per-variable ``delta_energy`` calculation: valid optimization
   target, not implemented in this patch.
3. Neighbor ``delta_energy`` updates after accepted spin flips: correctness is
   order-sensitive and not directly parallelized in this patch.

Tests Added/Updated
===================

Files:

* ``tests/test_simulated_annealing.py``
* ``tests/test_simulated_annealing_sampler.py``

Coverage added:

* ``fast_cpu_sa`` backend execution path.
* invalid backend value handling.
* ``gpu_sa`` failure path on non-CUDA builds.
* sampler-level backend argument acceptance/validation.

Verification performed:

* ``python -m pytest tests/test_simulated_annealing.py tests/test_simulated_annealing_sampler.py -q``
* Result: all tests passed in the targeted SA suites after rebuild.
