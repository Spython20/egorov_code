import numpy as np
import qutip as qt
from config import *

times = np.linspace(0.0, final_time, n_steps + 1)
basis_size = 1000

# creation and annihilation operators
a = qt.destroy(basis_size)
adag = a.dag()

# position and momentum operators
q = np.sqrt(epsilon / (2 * d1)) * (a + adag)
p = -1j * np.sqrt(epsilon * d1 / 2) * (a - adag)

# energy operators
kinetic = 0.5 * p**2
potential = qt.qeye(basis_size) - q.cosm()
total = kinetic + potential

# initial Gaussian wavepacket
alpha = (
    np.sqrt(d1 / (2 * epsilon)) * q0
    + 1j * p0 / np.sqrt(2 * epsilon * d1)
)

psi0 = qt.coherent(basis_size, alpha)

# QuTiP solves i psi_t = H psi, so divide the physical Hamiltonian by epsilon
result = qt.sesolve(total / epsilon, psi0, times, e_ops=[q, p, kinetic, potential, total])

position_qm = np.real(result.expect[0])
momentum_qm = np.real(result.expect[1])
kinetic_qm = np.real(result.expect[2])
potential_qm = np.real(result.expect[3])
total_qm = np.real(result.expect[4])
