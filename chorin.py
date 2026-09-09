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
        i -= 1
    return value

def H(k, q1,q2,p1, p2):
    k1, k2 = k
    return (-1.0) ** (k1 + k2) * laguerre(k1, z_1(q1, p1) / (2*epsilon)) * laguerre(k2, z_2(q2, p2) / (2*epsilon))

# scaling factor for observable, see Latex. needed because we are sampling from rho now, so need to rescale observable so that our Monte Carlo expectation integral is approx correctly
def scaling_factor(q1, q2, p1, p2):
    return 4 * np.exp(-(z_1(q1, p1) + z_2(q2, p2)) / (2 * epsilon))

def chorin_step_one(observable_flow_composed, q1_samples,q2_samples, p1_samples,p2_samples):
    c = []
    scaled_observable = observable_flow_composed * scaling_factor(q1_samples, q2_samples, p1_samples, p2_samples)
    for k in basis_indices:
        c.append((1 / N) * np.sum(scaled_observable * H(k, q1_samples,q2_samples, p1_samples,p2_samples),axis = 1))

    return c

# MAKE SURE TO USE NEW SAMPLE
def chorin_step_two(observable_flow_composed, c_alpha, q1_samples,q2_samples, p1_samples,p2_samples):

    scaled_observable = observable_flow_composed * scaling_factor(q1_samples, q2_samples, p1_samples, p2_samples)

    # sum over cn* Hn, correction summation
    correction = np.zeros_like(scaled_observable)

    for i in range(len(c_alpha)):
        k = basis_indices[i]
        H_values = H(k,q1_samples,q2_samples,p1_samples,p2_samples)

        for n in range(len(scaled_observable)):
            correction[n] += c_alpha[i][n] * H_values # c_alpha[i][n] is coefficient for basis function i at time step n

    corrected_values = scaled_observable - correction
    return np.mean(corrected_values, axis=1)    # final summation and mean computation
