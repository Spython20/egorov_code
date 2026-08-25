import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import *
from schrodinger_solver import *
from egorov import *

output_directory = Path.cwd() / "phase_space_plots"
output_directory.mkdir(exist_ok=True)

def position_1_observable(q, p=None):
    return q[:, 0]

def position_2_observable(q, p=None):
    return q[:, 1]

def momentum_1_observable(q, p):
    return p[:, 0]

def momentum_2_observable(q, p):
    return p[:, 1]

def kinetic_energy_observable(q, p):
    return 0.5 * (p[:, 0]**2 + p[:, 1]**2)

def potential_energy_observable(q, p=None):
    q1 = q[:, 0]
    q2 = q[:, 1]
    return (
        0.5 * (q1**2 + q2**2) + lamb * (q1**2 * q2 - (1.0 / 3.0) * q2**3)
    )

def total_energy_observable(q, p):
    return (
        kinetic_energy_observable(q, p)
        + potential_energy_observable(q, p)
    )


rng = np.random.default_rng(seed=42)
q_samples, p_samples = MC_sample(N, rng)

times, position_1, _, _, _ = compute_expectation(
    position_1_observable,
    q_samples,
    p_samples,
    dt,
    n_steps
)

_, position_2, _, _, _ = compute_expectation(
    position_2_observable,
    q_samples,
    p_samples,
    dt,
    n_steps
)

_, momentum_1, _, _, _ = compute_expectation(
    momentum_1_observable,
    q_samples,
    p_samples,
    dt,
    n_steps
)

_, momentum_2, _, _, _ = compute_expectation(
    momentum_2_observable,
    q_samples,
    p_samples,
    dt,
    n_steps
)

_, kinetic_energy, _, _, _ = compute_expectation(
    kinetic_energy_observable,
    q_samples,
    p_samples,
    dt,
    n_steps
)

_, potential_energy, _, _, _ = compute_expectation(
    potential_energy_observable,
    q_samples,
    p_samples,
    dt,
    n_steps
)

_, total_energy, _, _, _ = compute_expectation(
    total_energy_observable,
    q_samples,
    p_samples,
    dt,
    n_steps
)


plt.figure()
plt.plot(times, position_1, label="Egorov")
plt.plot(times, position1_qm, "--", label="Schrodinger")
plt.xlabel("time")
plt.ylabel(r"$\langle q_1 \rangle$")
plt.title(r"Position expectation $\langle q_1 \rangle$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    output_directory / "position_q1.png",
    dpi=200
)
plt.close()

plt.figure()
plt.plot(times, position_2, label="Egorov")
plt.plot(times, position2_qm, "--", label="Schrodinger")
plt.xlabel("time")
plt.ylabel(r"$\langle q_2 \rangle$")
plt.title(r"Position expectation $\langle q_2 \rangle$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    output_directory / "position_q2.png",
    dpi=200
)
plt.close()

plt.figure()
plt.plot(times, momentum_1, label="Egorov")
plt.plot(times, momentum1_qm, "--", label="Schrodinger")
plt.xlabel("time")
plt.ylabel(r"$\langle p_1 \rangle$")
plt.title(r"Momentum expectation $\langle p_1 \rangle$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    output_directory / "momentum_p1.png",
    dpi=200
)
plt.close()

plt.figure()
plt.plot(times, momentum_2, label="Egorov")
plt.plot(times, momentum2_qm, "--", label="Schrodinger")
plt.xlabel("time")
plt.ylabel(r"$\langle p_2 \rangle$")
plt.title(r"Momentum expectation $\langle p_2 \rangle$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    output_directory / "momentum_p2.png",
    dpi=200
)
plt.close()

plt.figure(figsize=(9, 6))

plt.plot(
    times,
    kinetic_energy,
    label="Egorov kinetic"
)

plt.plot(
    times,
    kinetic_qm,
    "--",
    label="Schrodinger kinetic"
)

plt.plot(
    times,
    potential_energy,
    label="Egorov potential"
)

plt.plot(
    times,
    potential_qm,
    "--",
    label="Schrodinger potential"
)

plt.plot(
    times,
    total_energy,
    label="Egorov total"
)

plt.plot(
    times,
    total_qm,
    "--",
    label="Schrodinger total"
)

plt.xlabel("time")
plt.ylabel("energy expectation")
plt.title("Energy expectations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    output_directory / "energy.png",
    dpi=200
)
plt.close()
