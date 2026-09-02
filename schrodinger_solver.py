import numpy as np
from config import *

times = np.linspace(0.0, final_time, n_steps + 1)

# spatial discretization
grid_size = 8192
L = 60.0
dx = L / grid_size

x = -L / 2.0 + np.arange(grid_size) * dx

# Fourier discretization from LibreTexts
dk = 2.0 * np.pi / L
k = -np.pi * grid_size / L + np.arange(grid_size) * dk

# (-1)^m factor appearing in the centred DFT formula
alternating_sign = (-1.0) ** np.arange(grid_size)

# potential
def potential(x):
    return 1.0 - np.cos(x)

# initial Gaussian wavepacket
psi = (d1 / (np.pi * epsilon)) ** 0.25 * np.exp(
    -d1 * (x - q0) ** 2 / (2.0 * epsilon)
    + 1j * p0 * (x - q0) / epsilon
)

# numerical normalization
psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

# kinetic half-step multiplier
kinetic_multiplier = np.exp(-1j * epsilon * dt * k**2 / 4.0)

# potential full-step multiplier
potential_multiplier = np.exp(-1j * dt * potential(x) / epsilon)


# kinetic half-step
def kinetic_step(psi):

    psi_fourier = np.fft.fft(alternating_sign * psi)

    psi_fourier = kinetic_multiplier * psi_fourier

    return alternating_sign * np.fft.ifft(psi_fourier)


# potential full-step
def potential_step(psi):

    return potential_multiplier * psi


# position expectation
def position_expectation(psi):

    return np.real(np.sum(np.conj(psi) * x * psi) * dx)


# momentum operator
def momentum_expectation(psi):

    psi_fourier = np.fft.fft(alternating_sign * psi)

    p_psi_fourier = epsilon * k * psi_fourier

    p_psi = alternating_sign * np.fft.ifft(p_psi_fourier)

    return np.real(np.sum(np.conj(psi) * p_psi) * dx)


# kinetic energy
def kinetic_expectation(psi):

    psi_fourier = np.fft.fft(alternating_sign * psi)

    kinetic_psi_fourier = 0.5 * epsilon**2 * k**2 * psi_fourier

    kinetic_psi = alternating_sign * np.fft.ifft(kinetic_psi_fourier)

    return np.real(np.sum(np.conj(psi) * kinetic_psi) * dx)


# potential energy
def potential_expectation(psi):

    return np.real(np.sum(np.conj(psi) * potential(x) * psi) * dx)


# total energy
def total_expectation(psi):

    return kinetic_expectation(psi) + potential_expectation(psi)


# storage
position_qm = np.zeros(n_steps + 1)
momentum_qm = np.zeros(n_steps + 1)
kinetic_qm = np.zeros(n_steps + 1)
potential_qm = np.zeros(n_steps + 1)
total_qm = np.zeros(n_steps + 1)

# initial expectation values
position_qm[0] = position_expectation(psi)
momentum_qm[0] = momentum_expectation(psi)
kinetic_qm[0] = kinetic_expectation(psi)
potential_qm[0] = potential_expectation(psi)
total_qm[0] = total_expectation(psi)

# split-step Fourier evolution
for n in range(n_steps):

    psi = kinetic_step(psi)

    psi = potential_step(psi)

    psi = kinetic_step(psi)

    position_qm[n + 1] = position_expectation(psi)
    momentum_qm[n + 1] = momentum_expectation(psi)
    kinetic_qm[n + 1] = kinetic_expectation(psi)
    potential_qm[n + 1] = potential_expectation(psi)
    total_qm[n + 1] = total_expectation(psi)
