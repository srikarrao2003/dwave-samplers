// Copyright 2026
//
// Licensed under the Apache License, Version 2.0

#include <stdexcept>

#include "gpu_sa.h"

// TODO: replace with CUDA kernels that parallelize per-read spin updates.
int gpu_general_simulated_annealing_cuda(
    std::int8_t *states,
    double *energies,
    const int num_samples,
    const std::vector<double> h,
    const std::vector<int> coupler_starts,
    const std::vector<int> coupler_ends,
    const std::vector<double> coupler_values,
    const int sweeps_per_beta,
    const std::vector<double> beta_schedule,
    const uint64_t seed,
    const VariableOrder varorder,
    const Proposal proposal_acceptance_criteria,
    callback interrupt_callback,
    void * const interrupt_function
) {
    (void)states;
    (void)energies;
    (void)num_samples;
    (void)h;
    (void)coupler_starts;
    (void)coupler_ends;
    (void)coupler_values;
    (void)sweeps_per_beta;
    (void)beta_schedule;
    (void)seed;
    (void)varorder;
    (void)proposal_acceptance_criteria;
    (void)interrupt_callback;
    (void)interrupt_function;
    throw std::runtime_error("gpu_sa.cu placeholder reached without kernel implementation");
}
