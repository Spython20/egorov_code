import numpy as np
from vel_verlet import flow_vel_verlet
from config import *

rng = np.random.default_rng(seed=42) # MC sampling seed
z0 = np.array([q0, p0]) # z0 = (q0, p0)
covariance = (epsilon / 2.0) * np.linalg.inv(D2) # Covariance corresponding exactly to the Gaussian Wigner density

def MC_sample(sample_size,rng):
    samples = rng.multivariate_normal(
        mean=z0,
        cov=covariance,
        size=sample_size,
    )
    return samples[:, 0], samples[:, 1]

def exp_flow(observable, q0, p0, dt, n_steps):
    times = []
    expectation_values = []
    observable_values_stored = []
    q_stored = []
    p_stored = []
    for t, q_t, p_t in flow_vel_verlet(q0=q0, p0=p0, dt=dt, n_steps=n_steps): # loops for each time step
        observable_values = observable(q_t, p_t) # composes points with observable
        times.append(t)
        expectation_values.append(np.mean(observable_values)) # Monte Carlo approximation of the phase-space integral, which is just a mean since we have even-weight quadrature

        observable_values_stored.append(observable_values.copy())
        q_stored.append(q_t.copy())
        p_stored.append(p_t.copy())

        print("computed " + str(observable.__name__) + " at time " + str(t))    
    
    return (np.asarray(times),np.asarray(expectation_values),np.asarray(observable_values_stored), np.asarray(q_stored), np.asarray(p_stored))
