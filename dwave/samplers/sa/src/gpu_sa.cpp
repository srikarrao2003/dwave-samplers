// Copyright 2026
//
// Licensed under the Apache License, Version 2.0

#include "gpu_sa.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

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
    if (!((coupler_starts.size() == coupler_ends.size()) &&
          (coupler_starts.size() == coupler_values.size()))) {
        throw std::runtime_error("coupler vectors have mismatched lengths");
    }

    if (interrupt_function != nullptr) {
        // GPU backend currently does not support host callback interrupts.
        return -2;
    }

    const int num_vars = static_cast<int>(h.size());
    const int num_couplers = static_cast<int>(coupler_starts.size());
    const int num_betas = static_cast<int>(beta_schedule.size());

    std::vector<int> degrees(num_vars, 0);
    for (int c = 0; c < num_couplers; ++c) {
        const int u = coupler_starts[c];
        const int v = coupler_ends[c];
        if ((u < 0) || (v < 0) || (u >= num_vars) || (v >= num_vars)) {
            throw std::runtime_error("coupler indexes contain an invalid variable");
        }
        degrees[u]++;
        degrees[v]++;
    }

    std::vector<int> neighbor_offsets(num_vars + 1, 0);
    for (int v = 0; v < num_vars; ++v) {
        neighbor_offsets[v + 1] = neighbor_offsets[v] + degrees[v];
    }

    std::vector<int> fill = neighbor_offsets;
    std::vector<int> neighbor_indices(neighbor_offsets[num_vars], 0);
    std::vector<double> neighbor_couplings(neighbor_offsets[num_vars], 0.0);

    for (int c = 0; c < num_couplers; ++c) {
        const int u = coupler_starts[c];
        const int v = coupler_ends[c];
        const double w = coupler_values[c];

        const int idx_u = fill[u]++;
        neighbor_indices[idx_u] = v;
        neighbor_couplings[idx_u] = w;

        const int idx_v = fill[v]++;
        neighbor_indices[idx_v] = u;
        neighbor_couplings[idx_v] = w;
    }

#ifdef DWAVE_SA_WITH_CUDA
    return gpu_general_simulated_annealing_cuda(
        states,
        energies,
        num_samples,
        h.data(),
        num_vars,
        coupler_starts.data(),
        coupler_ends.data(),
        coupler_values.data(),
        num_couplers,
        neighbor_offsets.data(),
        neighbor_indices.data(),
        neighbor_couplings.data(),
        sweeps_per_beta,
        beta_schedule.data(),
        num_betas,
        seed,
        varorder,
        proposal_acceptance_criteria);
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
    (void)num_vars;
    (void)num_couplers;
    (void)num_betas;
    (void)degrees;
    (void)neighbor_offsets;
    (void)fill;
    (void)neighbor_indices;
    (void)neighbor_couplings;
    return -3;
#endif
}
