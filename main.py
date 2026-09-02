import matplotlib.pyplot as plt
import numpy as np
from config import *
from schrodinger_solver import times, position_qm, momentum_qm
from egorov import *
from chorin import *
from pathlib import Path

output_directory = Path.cwd() / "phase_space_plots"
output_directory.mkdir(exist_ok=True)

# observables
def position_observable(q, p=None):
    return q

def momentum_observable(q, p):
    return p

def kinetic_energy_observable(q, p):
    return 0.5 * p**2

def potential_energy_observable(q, p=None):
    return 1.0 - np.cos(q)

def total_energy_observable(q, p):
    return 0.5 * p**2 + 1.0 - np.cos(q)

# ratio between original Wigner density and scaled probability density
def scaling_factor(q, p):
    return 2.0 * np.exp(-z_origin_squared(q,p) / (2.0 * epsilon))

# Schrodinger point at target time
print("starting schrodinger solver")
qm_target_index = int(np.argmin(np.abs(times - target_time)))
schrodinger_q = position_qm[qm_target_index]
schrodinger_p = momentum_qm[qm_target_index]
print("finished schrodinger solver")

egorov_q_trials = []
egorov_p_trials = []
chorin_q_trials = []
chorin_p_trials = []

trial_num = 1

for trial in range(number_of_trials):
    print("begin trial #" + str(trial_num))

    # samples for Chorin coefficients from scaled distribution
    initial_rng = np.random.default_rng(1000 + 3 * trial)
    q_initial, p_initial = scaled_MC_sample(N, initial_rng)

    # computes observables for first Chorin sample
    print("computing q observable #" + str(trial_num))
    _, initial_position_values = compute_observable(position_observable, q_initial, p_initial, dt, n_steps)

    print("computing p observable #" + str(trial_num))
    _, initial_momentum_values = compute_observable(momentum_observable, q_initial, p_initial, dt, n_steps)

    # scale observables
    initial_scaling = scaling_factor(q_initial, p_initial)
    initial_position_values = initial_position_values * initial_scaling
    initial_momentum_values = initial_momentum_values * initial_scaling

    # first Chorin step
    print("computing initial chorin q #" + str(trial_num))
    position_coefficients = chorin_step_one(initial_position_values, q_initial, p_initial)

    print("computing initial chorin p #" + str(trial_num))
    momentum_coefficients = chorin_step_one(initial_momentum_values, q_initial, p_initial)

    # ordinary Egorov sampling
    egorov_rng = np.random.default_rng(1001 + 3 * trial)
    q_egorov, p_egorov = MC_sample(N, egorov_rng)

    print("computing q expectation values #" + str(trial_num))
    second_times, egorov_position, _, _, _ = compute_expectation(position_observable, q_egorov, p_egorov, dt, n_steps)

    print("computing p expectation values #" + str(trial_num))
    _, egorov_momentum, _, _, _ = compute_expectation(momentum_observable, q_egorov, p_egorov, dt, n_steps)

    # second independent scaled sample for Chorin
    chorin_rng = np.random.default_rng(1002 + 3 * trial)
    q_chorin, p_chorin = scaled_MC_sample(N, chorin_rng)

    print("computing second chorin q observable #" + str(trial_num))
    _, second_position_values = compute_observable(position_observable, q_chorin, p_chorin, dt, n_steps)

    print("computing second chorin p observable #" + str(trial_num))
    _, second_momentum_values = compute_observable(momentum_observable, q_chorin, p_chorin, dt, n_steps)

    # scale second Chorin observables
    second_scaling = scaling_factor(q_chorin, p_chorin)
    second_position_values = second_position_values * second_scaling
    second_momentum_values = second_momentum_values * second_scaling

    # second Chorin step
    print("computing second chorin q #" + str(trial_num))
    chorin_position = chorin_step_two(second_position_values, position_coefficients, q_chorin, p_chorin)

    print("computing second chorin p #" + str(trial_num))
    chorin_momentum = chorin_step_two(second_momentum_values, momentum_coefficients, q_chorin, p_chorin)

    # fix machine precision issues
    target_index = int(np.argmin(np.abs(second_times - target_time)))

    egorov_q_trials.append(egorov_position[target_index])
    egorov_p_trials.append(egorov_momentum[target_index])

    chorin_q_trials.append(chorin_position[target_index])
    chorin_p_trials.append(chorin_momentum[target_index])

    trial_num = trial_num + 1

# q variance
egorov_q_variance = np.var(egorov_q_trials, ddof=1)
chorin_q_variance = np.var(chorin_q_trials, ddof=1)
egorov_q_mse = np.mean((np.asarray(egorov_q_trials) - schrodinger_q) ** 2)
chorin_q_mse = np.mean((np.asarray(chorin_q_trials) - schrodinger_q) ** 2)

print("Egorov q variance:", egorov_q_variance)
print("Chorin q variance:", chorin_q_variance)
print("Variance ratio:", chorin_q_variance / egorov_q_variance) # < 1 means variance reduction
print("Egorov q Mean Squared Error: ", egorov_q_mse)
print("Chorin q Mean Squared Error: ", chorin_q_mse)

# p variance
egorov_p_variance = np.var(egorov_p_trials, ddof=1)
chorin_p_variance = np.var(chorin_p_trials, ddof=1)
egorov_p_mse = np.mean((np.asarray(egorov_p_trials) - schrodinger_p) ** 2)
chorin_p_mse = np.mean((np.asarray(chorin_p_trials) - schrodinger_p) ** 2)

print("Egorov p variance:", egorov_p_variance)
print("Chorin p variance:", chorin_p_variance)
print("Variance ratio:", chorin_p_variance / egorov_p_variance) # < 1 means variance reduction
print("Egorov p Mean Squared Error: ", egorov_p_mse)
print("Chorin p Mean Squared Error: ", chorin_p_mse)

plt.figure(figsize=(8, 6))

# Schrödinger point
plt.scatter([schrodinger_q], [schrodinger_p], color="black", marker="*", s=220, label=f"Schrödinger at t={target_time}", zorder=5)
plt.scatter(egorov_q_trials, egorov_p_trials, color="blue", marker="o", s=60, alpha=0.75, label="Base Egorov trials")
plt.scatter(chorin_q_trials, chorin_p_trials, color="red", marker="s", s=60, alpha=0.75, label="Chorin trials")
plt.xlabel(r"$\langle q \rangle$")
plt.ylabel(r"$\langle p \rangle$")
plt.title(f"Phase-Space Points at t={target_time}")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = output_directory / f"phase_space_points_t_{target_time}.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {save_path}")
