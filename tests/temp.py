import numpy as np
from dwave.samplers.sa.simulated_annealing import simulated_annealing

num_samples = 4
h = [0.0, -1.0]
coupler_starts = [0]
coupler_ends = [1]
coupler_weights = [-1.0]
beta_schedule = np.linspace(0.1, 2.0, 10)
states = (2*np.random.randint(2, size=(num_samples, 2)).astype(np.int8)-1)

samples, energies = simulated_annealing(
  num_samples, h, coupler_starts, coupler_ends, coupler_weights,
  1, beta_schedule, 123, states, sa_backend="gpu_sa"
)
print(samples.shape, energies.shape)
