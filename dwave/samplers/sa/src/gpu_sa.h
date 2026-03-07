// Copyright 2026
//
// Licensed under the Apache License, Version 2.0

#ifndef _gpu_sa_h
#define _gpu_sa_h

#include <cstdint>
#include <vector>

#include "cpu_sa.h"

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
);

#endif
