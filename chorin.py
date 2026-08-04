import numpy as np
from scipy.special import comb, factorial
from config import *

# phase space centred z
def z_origin_squared(q, p):
    return d1 * (q - q0) ** 2 + ((p - p0) ** 2) / d1

# laguerre polynomial coefficients
def laguerre_coefficient(j,k):
    return((-1)**j * comb(k,k-j)) / factorial(j)

def H(k,q,p):
    value = 0
    i = k
    while i >= 0:
        value = value + (-1.0) ** np.abs(k) * laguerre_coefficient(i, k) * ((1 / epsilon) * (z_origin_squared(q,p)))**i
        print("computed H coefficient " + str(i))
        i = i - 1    
    return value

def chorin_step_one(observable_flow_composed, q_samples, p_samples):
    c = []
    for i in range(0, basis_size):
        c.append((1 / N) * np.sum(observable_flow_composed * H(i, q_samples, p_samples),axis = 1))
        print("computed chorin coefficient " + str(i))

    return c

def chorin_step_two(observable_flow_composed, c_alpha, q_samples, p_samples):
    c_alpha = np.asarray(c_alpha)
    correction = np.zeros_like(observable_flow_composed, dtype=float) # empty correction array

    # sum over cn * Hn
    for i in range(1, c_alpha.shape[0]):
        correction += c_alpha[i][:, None] * H(i, q_samples, p_samples)[None, :]
        print("computed corrected coefficient " + str(i))

    corrected_values = observable_flow_composed - correction

    # final summation and mean computation
    return np.mean(corrected_values, axis=1)
