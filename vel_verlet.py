import numpy as np

# V(q) = 1 - cos(q)
def grad_V(q):
    return np.sin(q)

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
