import numpy as np
from config import *

def grad_V(q):
    q1 = q[0, :]
    q2 = q[1, :]

    return np.transpose(np.column_stack((q1 + 2 * lamb * q1 * q2, q2 + lamb * (q1**2 - q2**2))))

# takes in array of positions and momentums of each sampled point
def flow_vel_verlet(q0, p0, dt, n_steps):
    q = np.asarray(q0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()

    # Phi^0, initial state
    yield 0.0, q.copy(), p.copy()

    # velocity verlet algorithm
    for n in range(n_steps):
        p_half = p - 0.5 * dt * grad_V(q)
        q = q + dt * p_half
        p = p_half - 0.5 * dt * grad_V(q)
        yield (n + 1) * dt, q.copy(), p.copy()
