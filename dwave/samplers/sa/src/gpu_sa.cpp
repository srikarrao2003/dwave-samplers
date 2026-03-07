// Copyright 2026
//
// Licensed under the Apache License, Version 2.0

#include "gpu_sa.h"

#ifdef DWAVE_SA_WITH_CUDA
extern int gpu_general_simulated_annealing_cuda(
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
);
#endif

int gpu_general_simulated_annealing(
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
#ifdef DWAVE_SA_WITH_CUDA
    return gpu_general_simulated_annealing_cuda(
        states, energies, num_samples, h, coupler_starts, coupler_ends, coupler_values,
        sweeps_per_beta, beta_schedule, seed, varorder, proposal_acceptance_criteria,
        interrupt_callback, interrupt_function);
#else
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
    return -3;
#endif
}
