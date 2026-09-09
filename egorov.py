import numpy as np
from vel_verlet import flow_vel_verlet
from config import *

z0 = np.array([q10,q20,p10,p20]) # gaussian wavepacket centre
covariance = (epsilon / 2) * np.linalg.inv(D2) # Covariance corresponding exactly to the Gaussian Wigner density
scaled_covariance = epsilon * np.linalg.inv(D2) # scaled for W^1/2

def MC_sample(sample_size,rng):
    samples = rng.multivariate_normal(mean=z0, cov=covariance, size=sample_size) # returns matrix of shape (N,4) since the mean is (4,) which is 1dim array with 4 entries
    q1_samples = samples[:,0]
    q2_samples = samples[:,1]
    p1_samples = samples[:,2]
    p2_samples = samples[:,3]

    return q1_samples, q2_samples, p1_samples, p2_samples

def scaled_MC_sample(sample_size,rng):
    samples = rng.multivariate_normal(mean=z0, cov=scaled_covariance, size=sample_size) # returns matrix of shape (N,4) since the mean is (4,) which is 1dim array with 4 entries
    q1_samples = samples[:,0]
    q2_samples = samples[:,1]
    p1_samples = samples[:,2]
    p2_samples = samples[:,3]

    return q1_samples, q2_samples, p1_samples, p2_samples

# computes observables without computing expectation value
def compute_observable(observable, q1, q2, p1, p2):
    observable_values = observable(q1, q2, p1, p2) # composes points with observable
    return observable_values
