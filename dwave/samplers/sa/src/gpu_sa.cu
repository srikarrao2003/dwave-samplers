// Copyright 2026
//
// Licensed under the Apache License, Version 2.0

#include <cstdint>
#include <cmath>

#include <cuda_runtime.h>

#include "gpu_sa.h"

#define GPU_RANDMAX ((uint64_t)-1LL)

namespace {

inline int check_cuda(cudaError_t err) {
    if (err == cudaSuccess) return 0;
    return -1000 - static_cast<int>(err);
}

__device__ inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ inline uint64_t next_u64(uint64_t& s0, uint64_t& s1) {
    uint64_t x = s0;
    const uint64_t y = s1;
    s0 = y;
    x ^= x << 23;
    s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s1 + y;
}

__global__ void gpu_sa_kernel(
    std::int8_t* states,
    double* energies,
    double* delta_energy,
    const int num_samples,
    const double* h,
    const int num_vars,
    const int* coupler_starts,
    const int* coupler_ends,
    const double* coupler_values,
    const int num_couplers,
    const int* neighbor_offsets,
    const int* neighbor_indices,
    const double* neighbor_couplings,
    const int sweeps_per_beta,
    const double* beta_schedule,
    const int num_betas,
    const uint64_t seed
) {
    for (int sample = blockDim.x * blockIdx.x + threadIdx.x;
         sample < num_samples;
         sample += blockDim.x * gridDim.x) {

        std::int8_t* state = states + static_cast<size_t>(sample) * num_vars;
        double* delta = delta_energy + static_cast<size_t>(sample) * num_vars;

        uint64_t s0 = splitmix64(seed ^ static_cast<uint64_t>(sample));
        uint64_t s1 = splitmix64(seed + static_cast<uint64_t>(sample) + 1ULL);
        if ((s0 | s1) == 0ULL) s1 = 1ULL;

        // Initialize per-variable delta energies.
        for (int var = 0; var < num_vars; ++var) {
            double local = h[var];
            for (int n = neighbor_offsets[var]; n < neighbor_offsets[var + 1]; ++n) {
                local += state[neighbor_indices[n]] * neighbor_couplings[n];
            }
            delta[var] = -2.0 * state[var] * local;
        }

        for (int beta_idx = 0; beta_idx < num_betas; ++beta_idx) {
            const double beta = beta_schedule[beta_idx];
            const double threshold = 44.36142 / beta;

            for (int sweep = 0; sweep < sweeps_per_beta; ++sweep) {
                for (int var = 0; var < num_vars; ++var) {
                    if (delta[var] >= threshold) continue;

                    bool flip = false;
                    if (delta[var] <= 0.0) {
                        flip = true;
                    } else {
                        const uint64_t r = next_u64(s0, s1);
                        if (exp(-delta[var] * beta) * (double)GPU_RANDMAX > (double)r) {
                            flip = true;
                        }
                    }

                    if (flip) {
                        const std::int8_t multiplier = 4 * state[var];
                        for (int n = neighbor_offsets[var]; n < neighbor_offsets[var + 1]; ++n) {
                            const int neighbor = neighbor_indices[n];
                            delta[neighbor] += multiplier * neighbor_couplings[n] * state[neighbor];
                        }
                        state[var] *= -1;
                        delta[var] *= -1.0;
                    }
                }
            }
        }

        // Final energy.
        double energy = 0.0;
        for (int var = 0; var < num_vars; ++var) {
            energy += state[var] * h[var];
        }
        for (int c = 0; c < num_couplers; ++c) {
            energy += state[coupler_starts[c]] * coupler_values[c] * state[coupler_ends[c]];
        }
        energies[sample] = energy;
    }
}

}  // namespace

int gpu_general_simulated_annealing_cuda(
    std::int8_t *states,
    double *energies,
    const int num_samples,
    const double *h,
    const int num_vars,
    const int *coupler_starts,
    const int *coupler_ends,
    const double *coupler_values,
    const int num_couplers,
    const int *neighbor_offsets,
    const int *neighbor_indices,
    const double *neighbor_couplings,
    const int sweeps_per_beta,
    const double *beta_schedule,
    const int num_betas,
    const uint64_t seed,
    const VariableOrder varorder,
    const Proposal proposal_acceptance_criteria
) {
    if (varorder != Sequential || proposal_acceptance_criteria != Metropolis) {
        // First CUDA milestone only supports sequential Metropolis.
        return -4;
    }

    if (num_samples <= 0 || num_vars < 0) return 0;

    int device_count = 0;
    int rc = check_cuda(cudaGetDeviceCount(&device_count));
    if (rc) return rc;
    if (device_count < 1) return -5;

    std::int8_t* d_states = nullptr;
    double* d_energies = nullptr;
    double* d_h = nullptr;
    int* d_coupler_starts = nullptr;
    int* d_coupler_ends = nullptr;
    double* d_coupler_values = nullptr;
    int* d_neighbor_offsets = nullptr;
    int* d_neighbor_indices = nullptr;
    double* d_neighbor_couplings = nullptr;
    double* d_beta_schedule = nullptr;
    double* d_delta_energy = nullptr;

    const size_t states_size = static_cast<size_t>(num_samples) * num_vars * sizeof(std::int8_t);
    const size_t energies_size = static_cast<size_t>(num_samples) * sizeof(double);
    const size_t h_size = static_cast<size_t>(num_vars) * sizeof(double);
    const size_t coupler_i_size = static_cast<size_t>(num_couplers) * sizeof(int);
    const size_t coupler_w_size = static_cast<size_t>(num_couplers) * sizeof(double);
    const size_t offsets_size = static_cast<size_t>(num_vars + 1) * sizeof(int);
    const int nnz = neighbor_offsets[num_vars];
    const size_t nnz_i_size = static_cast<size_t>(nnz) * sizeof(int);
    const size_t nnz_w_size = static_cast<size_t>(nnz) * sizeof(double);
    const size_t beta_size = static_cast<size_t>(num_betas) * sizeof(double);
    const size_t delta_size = static_cast<size_t>(num_samples) * num_vars * sizeof(double);
    const int threads = 128;
    const int blocks = (num_samples + threads - 1) / threads;

    rc = 0;
    rc = check_cuda(cudaMalloc(&d_states, states_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_energies, energies_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_h, h_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_coupler_starts, coupler_i_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_coupler_ends, coupler_i_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_coupler_values, coupler_w_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_neighbor_offsets, offsets_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_neighbor_indices, nnz_i_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_neighbor_couplings, nnz_w_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_beta_schedule, beta_size)); if (rc) goto cleanup;
    rc = check_cuda(cudaMalloc(&d_delta_energy, delta_size)); if (rc) goto cleanup;

    rc = check_cuda(cudaMemcpy(d_states, states, states_size, cudaMemcpyHostToDevice)); if (rc) goto cleanup;
    rc = check_cuda(cudaMemcpy(d_h, h, h_size, cudaMemcpyHostToDevice)); if (rc) goto cleanup;
    rc = check_cuda(cudaMemcpy(d_coupler_starts, coupler_starts, coupler_i_size, cudaMemcpyHostToDevice)); if (rc) goto cleanup;
    rc = check_cuda(cudaMemcpy(d_coupler_ends, coupler_ends, coupler_i_size, cudaMemcpyHostToDevice)); if (rc) goto cleanup;
    rc = check_cuda(cudaMemcpy(d_coupler_values, coupler_values, coupler_w_size, cudaMemcpyHostToDevice)); if (rc) goto cleanup;
    rc = check_cuda(cudaMemcpy(d_neighbor_offsets, neighbor_offsets, offsets_size, cudaMemcpyHostToDevice)); if (rc) goto cleanup;
    rc = check_cuda(cudaMemcpy(d_neighbor_indices, neighbor_indices, nnz_i_size, cudaMemcpyHostToDevice)); if (rc) goto cleanup;
    rc = check_cuda(cudaMemcpy(d_neighbor_couplings, neighbor_couplings, nnz_w_size, cudaMemcpyHostToDevice)); if (rc) goto cleanup;
    rc = check_cuda(cudaMemcpy(d_beta_schedule, beta_schedule, beta_size, cudaMemcpyHostToDevice)); if (rc) goto cleanup;

    gpu_sa_kernel<<<blocks, threads>>>(
        d_states,
        d_energies,
        d_delta_energy,
        num_samples,
        d_h,
        num_vars,
        d_coupler_starts,
        d_coupler_ends,
        d_coupler_values,
        num_couplers,
        d_neighbor_offsets,
        d_neighbor_indices,
        d_neighbor_couplings,
        sweeps_per_beta,
        d_beta_schedule,
        num_betas,
        seed
    );
    rc = check_cuda(cudaGetLastError()); if (rc) goto cleanup;
    rc = check_cuda(cudaDeviceSynchronize()); if (rc) goto cleanup;

    rc = check_cuda(cudaMemcpy(states, d_states, states_size, cudaMemcpyDeviceToHost)); if (rc) goto cleanup;
    rc = check_cuda(cudaMemcpy(energies, d_energies, energies_size, cudaMemcpyDeviceToHost)); if (rc) goto cleanup;

cleanup:
    cudaFree(d_states);
    cudaFree(d_energies);
    cudaFree(d_h);
    cudaFree(d_coupler_starts);
    cudaFree(d_coupler_ends);
    cudaFree(d_coupler_values);
    cudaFree(d_neighbor_offsets);
    cudaFree(d_neighbor_indices);
    cudaFree(d_neighbor_couplings);
    cudaFree(d_beta_schedule);
    cudaFree(d_delta_energy);

    if (rc) return rc;
    return num_samples;
}
