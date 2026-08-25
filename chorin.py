import numpy as np
from scipy.special import comb, factorial
from config import *

# phase space centred z
def z_1(q1, p1):
    return d1 * (q1 - q10) ** 2 + ((p1 - p10) ** 2) / d1

def z_2(q2, p2):
    return d2 * (q2 - q20) ** 2 + ((p2 - p20) ** 2) / d2

# laguerre polynomial coefficients
def laguerre_coefficient(j,k):
    return((-1)**j * comb(k,k-j)) / factorial(j)

def laguerre(k, x):
    value = 0
    i = k
    while i >= 0:
        value += laguerre_coefficient(i, k) * x**i
        #print("computed H coefficient " + str(i))
        i -= 1
    return value

def H(k, q, p):
    k1, k2 = k
    return (-1.0) ** (k1 + k2) * laguerre(k1, z_1(q[:, 0], p[:, 0]) / epsilon) * laguerre(k2, z_2(q[:, 1], p[:, 1]) / epsilon)

def chorin_step_one(observable_flow_composed, q_samples, p_samples):
    c = []
    for k in basis_indices:
        c.append((1 / N) * np.sum(observable_flow_composed * H(k, q_samples, p_samples),axis = 1))
        #print("computed chorin coefficient " + str(i))

    return c

def chorin_step_two(observable_flow_composed, c_alpha, q_samples, p_samples):
    c_alpha = np.asarray(c_alpha)
    correction = np.zeros_like(observable_flow_composed, dtype=float) # empty correction array

    # sum over cn * Hn
    for i in range(1, c_alpha.shape[0]):
        k = basis_indices[i]
        correction += c_alpha[i][:, None] * H(k, q_samples, p_samples)[None, :]
        #print("computed corrected coefficient " + str(i))

    corrected_values = observable_flow_composed - correction

    # final summation and mean computation
    return np.mean(corrected_values, axis=1)
