import numpy as np
import scipy as s

"""good matching parameters, d = 1, epsilon = 0.1, q0 = 2.0, p0 = 1.0, final_time = 10.0"""
epsilon = 0.1
d = 1
d1 = 0.1
q0 = 2.0
p0 = 1.0
N = 100000 # number of samples
dt = 0.01
final_time = 5.0
n_steps = int(round(final_time / dt))

# matrix for initial gaussian wavepacket
D1 = np.diag([d1])
D2 = s.linalg.block_diag(D1, np.reciprocal(D1))

# how many H_alpha(z) we want
basis_size = 5

number_of_trials = 20
target_time = 5.0
