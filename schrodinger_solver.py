import numpy as np
from config import *

times = np.linspace(0.0, final_time, n_steps + 1)

# spatial grid
grid_size = 8192
x_min = -30.0
x_max = 30.0

x = np.linspace(x_min, x_max, grid_size, endpoint=False)
dx = x[1] - x[0]

# Fourier grid
k = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=dx)
p_grid = epsilon * k


# potential
def V(x):
    return 1.0 - np.cos(x)


# initial Gaussian wavepacket
psi = (d1 / (np.pi * epsilon)) ** 0.25 * np.exp(
    -d1 * (x - q0) ** 2 / (2.0 * epsilon)
    + 1j * p0 * (x - q0) / epsilon
)

# discrete normalization
psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)


# split-operator propagators
potential_propagator = np.exp(-1j * V(x) * dt / (2.0 * epsilon))
kinetic_propagator = np.exp(-1j * (p_grid ** 2 / 2.0) * dt / epsilon)


# expectation values
def position_expectation(psi):
    return np.real(np.sum(np.conj(psi) * x * psi) * dx)


def momentum_expectation(psi):
    psi_fourier = np.fft.fft(psi)
    derivative = np.fft.ifft(1j * k * psi_fourier)
    p_psi = -1j * epsilon * derivative

    return np.real(np.sum(np.conj(psi) * p_psi) * dx)


def kinetic_expectation(psi):
    psi_fourier = np.fft.fft(psi)
    kinetic_psi = np.fft.ifft(0.5 * p_grid ** 2 * psi_fourier)

    return np.real(np.sum(np.conj(psi) * kinetic_psi) * dx)


def potential_expectation(psi):
    return np.real(np.sum(np.abs(psi) ** 2 * V(x)) * dx)


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


print("starting grid Schrödinger solver")

for n in range(n_steps):

    # half potential step
    psi = potential_propagator * psi

    # full kinetic step in Fourier space
    psi_fourier = np.fft.fft(psi)
    psi_fourier = kinetic_propagator * psi_fourier
    psi = np.fft.ifft(psi_fourier)

    # half potential step
    psi = potential_propagator * psi

    # expectation values
    position_qm[n + 1] = position_expectation(psi)
    momentum_qm[n + 1] = momentum_expectation(psi)
    kinetic_qm[n + 1] = kinetic_expectation(psi)
    potential_qm[n + 1] = potential_expectation(psi)
    total_qm[n + 1] = total_expectation(psi)

print("finished grid Schrödinger solver")
