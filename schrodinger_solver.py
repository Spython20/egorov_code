import numpy as np
import qutip as qt
from config import *

times = np.linspace(0.0, final_time, n_steps + 1)
hilbert_size = 40

# creation and annihilation operators
a1 = qt.tensor(qt.destroy(hilbert_size), qt.qeye(hilbert_size))
a2 = qt.tensor(qt.qeye(hilbert_size), qt.destroy(hilbert_size))

adag1 = a1.dag()
adag2 = a2.dag()

# position and momentum operators
q1 = np.sqrt(epsilon / (2 * d1)) * (a1 + adag1)
p1 = -1j * np.sqrt(epsilon * d1 / 2) * (a1 - adag1)
q2 = np.sqrt(epsilon / (2 * d2)) * (a2 + adag2)
p2 = -1j * np.sqrt(epsilon * d2 / 2) * (a2 - adag2)

# energy operators
kinetic = 0.5 * (p1**2 + p2**2)
potential = 0.5 * (q1**2 + q2**2) + lamb * (q1**2 * q2 - (1.0 / 3.0) * q2**3)
total = kinetic + potential

# initial Gaussian wavepacket
alpha1 = np.sqrt(d1 / (2 * epsilon)) * q10 + 1j * p10 / np.sqrt(2 * epsilon * d1)
alpha2 = np.sqrt(d2 / (2 * epsilon)) * q20 + 1j * p20 / np.sqrt(2 * epsilon * d2)
psi0 = qt.tensor(qt.coherent(hilbert_size, alpha1), qt.coherent(hilbert_size, alpha2))

# QuTiP solves i psi_t = H psi, so divide the physical Hamiltonian by epsilon
result = qt.sesolve(total / epsilon, psi0, times, e_ops=[q1, q2, p1, p2, kinetic, potential, total])

position1_qm = np.real(result.expect[0])
position2_qm = np.real(result.expect[1])

momentum1_qm = np.real(result.expect[2])
momentum2_qm = np.real(result.expect[3])

kinetic_qm = np.real(result.expect[4])
potential_qm = np.real(result.expect[5])
total_qm = np.real(result.expect[6])
