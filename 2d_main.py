import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
from config import *
from egorov import *
from chorin import *

print("starting schrodinger solver")
from schrodinger_solver import times, position1_qm, position2_qm, momentum1_qm, momentum2_qm
print("finished schrodinger solver")

# output directory
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
    return 0.5 * (q1**2 + q2**2) + lamb * (q1**2 * q2 - (1.0 / 3.0) * q2**3)

def total_energy_observable(q, p):
    return kinetic_energy_observable(q, p) + potential_energy_observable(q, p)

qm_target_index = int(np.argmin(np.abs(times - target_time)))

schrodinger_q1 = position1_qm[qm_target_index]
schrodinger_q2 = position2_qm[qm_target_index]
schrodinger_p1 = momentum1_qm[qm_target_index]
schrodinger_p2 = momentum2_qm[qm_target_index]

schrodinger_vector = np.array([schrodinger_q1, schrodinger_q2, schrodinger_p1, schrodinger_p2])

egorov_q1_trials = []
egorov_q2_trials = []
egorov_p1_trials = []
egorov_p2_trials = []

chorin_q1_trials = []
chorin_q2_trials = []
chorin_p1_trials = []
chorin_p2_trials = []

for trial in range(number_of_trials):
    trial_num = trial + 1

    print("begin trial #" + str(trial_num))

    # initial monte carlo sampling
    initial_rng = np.random.default_rng(1000 + 2 * trial)
    q_initial, p_initial = MC_sample(N, initial_rng)

    print("computing initial q1 coefficients #" + str(trial_num))
    _, initial_q1_values = compute_observable(position_1_observable, q_initial, p_initial, dt, n_steps)
    q1_coefficients = chorin_step_one(initial_q1_values, q_initial, p_initial)
    del initial_q1_values

    print("computing initial q2 coefficients #" + str(trial_num))
    _, initial_q2_values = compute_observable(position_2_observable, q_initial, p_initial, dt, n_steps)
    q2_coefficients = chorin_step_one(initial_q2_values, q_initial, p_initial)
    del initial_q2_values

    print("computing initial p1 coefficients #" + str(trial_num))
    _, initial_p1_values = compute_observable(momentum_1_observable, q_initial, p_initial, dt, n_steps)
    p1_coefficients = chorin_step_one(initial_p1_values, q_initial, p_initial)
    del initial_p1_values

    print("computing initial p2 coefficients #" + str(trial_num))
    _, initial_p2_values = compute_observable(momentum_2_observable, q_initial, p_initial, dt, n_steps)
    p2_coefficients = chorin_step_one(initial_p2_values, q_initial, p_initial)
    del initial_p2_values

    # second independent sampling
    second_rng = np.random.default_rng(1001 + 2 * trial)
    q_second, p_second = MC_sample(N, second_rng)

    print("computing q1 estimates #" + str(trial_num))
    second_times, second_q1_values = compute_observable(position_1_observable, q_second, p_second, dt, n_steps)
    egorov_q1 = np.mean(second_q1_values, axis=1)
    chorin_q1 = chorin_step_two(second_q1_values, q1_coefficients, q_second, p_second)
    del second_q1_values

    print("computing q2 estimates #" + str(trial_num))
    _, second_q2_values = compute_observable(position_2_observable, q_second, p_second, dt, n_steps)
    egorov_q2 = np.mean(second_q2_values, axis=1)
    chorin_q2 = chorin_step_two(second_q2_values, q2_coefficients, q_second, p_second)
    del second_q2_values

    # p1
    print("computing p1 estimates #" + str(trial_num))
    _, second_p1_values = compute_observable(momentum_1_observable, q_second, p_second, dt, n_steps)
    egorov_p1 = np.mean(second_p1_values, axis=1)
    chorin_p1 = chorin_step_two(second_p1_values, p1_coefficients, q_second, p_second)
    del second_p1_values

    # p2
    print("computing p2 estimates #" + str(trial_num))
    _, second_p2_values = compute_observable(momentum_2_observable, q_second, p_second, dt, n_steps)
    egorov_p2 = np.mean(second_p2_values, axis=1)
    chorin_p2 = chorin_step_two(second_p2_values, p2_coefficients, q_second, p_second)
    del second_p2_values

    # target time index
    target_index = int(np.argmin(np.abs(second_times - target_time)))

    # store 4D phase-space point for this trial
    egorov_q1_trials.append(egorov_q1[target_index])
    egorov_q2_trials.append(egorov_q2[target_index])
    egorov_p1_trials.append(egorov_p1[target_index])
    egorov_p2_trials.append(egorov_p2[target_index])

    chorin_q1_trials.append(chorin_q1[target_index])
    chorin_q2_trials.append(chorin_q2[target_index])
    chorin_p1_trials.append(chorin_p1[target_index])
    chorin_p2_trials.append(chorin_p2[target_index])


# collect trials into R^4, (q1, q2, p1, p2)
egorov_trials = np.column_stack((egorov_q1_trials, egorov_q2_trials, egorov_p1_trials, egorov_p2_trials))
chorin_trials = np.column_stack((chorin_q1_trials, chorin_q2_trials, chorin_p1_trials, chorin_p2_trials))

# covariance 
egorov_covariance = np.cov(egorov_trials, rowvar=False, ddof=1)
chorin_covariance = np.cov(chorin_trials, rowvar=False, ddof=1)

# variance in each phase-space coordinate
egorov_coordinate_variances = np.diag(egorov_covariance)
chorin_coordinate_variances = np.diag(chorin_covariance)
coordinate_names = ["q1", "q2", "p1", "p2"]

for i, name in enumerate(coordinate_names):
    print(name)
    print("    Egorov variance:", egorov_coordinate_variances[i])
    print("    Chorin variance:", chorin_coordinate_variances[i])
    print("    Variance ratio:", chorin_coordinate_variances[i] / egorov_coordinate_variances[i])

# mean square error
egorov_coordinate_mse = np.mean((egorov_trials - schrodinger_vector)**2, axis=0)
chorin_coordinate_mse = np.mean((chorin_trials - schrodinger_vector)**2, axis=0)

for i, name in enumerate(coordinate_names):
    print(name)
    print("    Egorov MSE:", egorov_coordinate_mse[i])
    print("    Chorin MSE:", chorin_coordinate_mse[i])

plt.figure(figsize=(8, 6))

# Schrödinger reference
plt.scatter([schrodinger_q1], [schrodinger_p1], color="black", marker="*", s=220, label=f"Schrödinger at t={target_time}", zorder=5)

# base Egorov trials
plt.scatter(egorov_q1_trials, egorov_p1_trials, color="blue", marker="o", s=60, alpha=0.75, label="Base Egorov trials")

# Chorin variance-reduced trials
plt.scatter(chorin_q1_trials, chorin_p1_trials, color="red", marker="s", s=60, alpha=0.75, label="Chorin trials")

plt.xlabel(r"$\langle q_1 \rangle$")
plt.ylabel(r"$\langle p_1 \rangle$")
plt.title(f"Phase Space $(q_1,p_1)$ at t={target_time}")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = output_directory / f"phase_space_q1_p1_t_{target_time}.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {save_path}")

plt.figure(figsize=(8, 6))

# Schrödinger reference
plt.scatter([schrodinger_q2], [schrodinger_p2], color="black", marker="*", s=220, label=f"Schrödinger at t={target_time}", zorder=5)

# base Egorov trials
plt.scatter(egorov_q2_trials, egorov_p2_trials, color="blue", marker="o", s=60, alpha=0.75, label="Base Egorov trials")

# Chorin variance-reduced trials
plt.scatter(chorin_q2_trials, chorin_p2_trials, color="red", marker="s", s=60, alpha=0.75, label="Chorin trials")

plt.xlabel(r"$\langle q_2 \rangle$")
plt.ylabel(r"$\langle p_2 \rangle$")
plt.title(f"Phase Space $(q_2,p_2)$ at t={target_time}")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = output_directory / f"phase_space_q2_p2_t_{target_time}.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {save_path}")
