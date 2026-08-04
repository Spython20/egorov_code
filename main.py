import matplotlib.pyplot as plt
import numpy as np
from config import *
from schrodinger_solver import times, position_qm, momentum_qm
from egorov import MC_sample, exp_flow
from chorin import *
from pathlib import Path
import matplotlib.pyplot as plt

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

# evaluate Schrodinger at target time
target_index = int(np.argmin(np.abs(times - target_time)))
actual_target_time = times[target_index]

position_schrodinger = position_qm[target_index]
momentum_schrodinger = momentum_qm[target_index]

egorov_trials = []
chorin_trials = []
'''
for trial in range(number_of_trials):
    # Samples for first estimating coefficients c*
    rng = np.random.default_rng(1000 + 2 * trial)
    q_sample_one, p_sample_one = MC_sample(N, rng)

    pilot_times, _, pilot_observables, _, _, = exp_flow(position_observable, q_sample_one, p_sample_one, dt, n_steps)

    c_alpha = chorin_step_one(pilot_observables, q_sample_one, p_sample_one)

    # Second phase space sampling
    rng = np.random.default_rng(1001 + 2 * trial)
    q_sample_two, p_sample_two = MC_sample(N, rng)

    evaluation_times, egorov_estimates, evaluation_observables, _, _ = exp_flow(position_observable, q_sample_two, p_sample_two, dt, n_steps)

    chorin_estimates = chorin_step_two(evaluation_observables, c_alpha, q_sample_two, p_sample_two)

    target_index = int(np.argmin(np.abs(evaluation_times - target_time))) # remove machine precision errors

    egorov_trials.append(egorov_estimates[target_index])
    chorin_trials.append(chorin_estimates[target_index])

egorov_variance = np.var(egorov_trials, ddof=1)
chorin_variance = np.var(chorin_trials, ddof=1)

egorov_mse = np.mean(
    (np.asarray(egorov_trials) - position_schrodinger) ** 2
)

chorin_mse = np.mean(
    (np.asarray(chorin_trials) - position_schrodinger) ** 2
)

print("Egorov estimator variance:", egorov_variance)
print("Chorin estimator variance:", chorin_variance)
print("Variance ratio:", chorin_variance / egorov_variance) # < 1 means variance reduction
print("Egorov Error: ", egorov_mse)
print("Chorin Error: ", chorin_mse)
'''

from pathlib import Path

output_directory = Path.cwd() / "phase_space_target_points"
output_directory.mkdir(exist_ok=True)

# Schrödinger point at target time
qm_target_index = int(np.argmin(np.abs(times - target_time)))
schrodinger_q = position_qm[qm_target_index]
schrodinger_p = momentum_qm[qm_target_index]

egorov_q_trials = []
egorov_p_trials = []
chorin_q_trials = []
chorin_p_trials = []

for trial in range(number_of_trials):
    # Pilot samples for Chorin coefficients
    pilot_rng = np.random.default_rng(1000 + 2 * trial)
    q_pilot, p_pilot = MC_sample(N, pilot_rng)

    _, _, pilot_position_values, _, _ = exp_flow(
        position_observable,
        q_pilot,
        p_pilot,
        dt,
        n_steps,
    )

    _, _, pilot_momentum_values, _, _ = exp_flow(
        momentum_observable,
        q_pilot,
        p_pilot,
        dt,
        n_steps,
    )

    position_coefficients = chorin_step_one(
        pilot_position_values,
        q_pilot,
        p_pilot,
    )

    momentum_coefficients = chorin_step_one(
        pilot_momentum_values,
        q_pilot,
        p_pilot,
    )

    # Independent evaluation samples
    evaluation_rng = np.random.default_rng(1001 + 2 * trial)
    q_evaluation, p_evaluation = MC_sample(N, evaluation_rng)

    (
        evaluation_times,
        egorov_position,
        evaluation_position_values,
        _,
        _,
    ) = exp_flow(
        position_observable,
        q_evaluation,
        p_evaluation,
        dt,
        n_steps,
    )

    (
        _,
        egorov_momentum,
        evaluation_momentum_values,
        _,
        _,
    ) = exp_flow(
        momentum_observable,
        q_evaluation,
        p_evaluation,
        dt,
        n_steps,
    )

    chorin_position = chorin_step_two(
        evaluation_position_values,
        position_coefficients,
        q_evaluation,
        p_evaluation,
    )

    chorin_momentum = chorin_step_two(
        evaluation_momentum_values,
        momentum_coefficients,
        q_evaluation,
        p_evaluation,
    )

    target_index = int(np.argmin(np.abs(evaluation_times - target_time)))

    egorov_q_trials.append(egorov_position[target_index])
    egorov_p_trials.append(egorov_momentum[target_index])

    chorin_q_trials.append(chorin_position[target_index])
    chorin_p_trials.append(chorin_momentum[target_index])

plt.figure(figsize=(8, 6))

# Schrödinger point
plt.scatter(
    [schrodinger_q],
    [schrodinger_p],
    color="black",
    marker="*",
    s=220,
    label=f"Schrödinger at t={target_time}",
    zorder=5,
)

# Egorov trial points
plt.scatter(
    egorov_q_trials,
    egorov_p_trials,
    color="blue",
    marker="o",
    s=60,
    alpha=0.75,
    label="Base Egorov trials",
)

# Chorin trial points
plt.scatter(
    chorin_q_trials,
    chorin_p_trials,
    color="red",
    marker="s",
    s=60,
    alpha=0.75,
    label="Chorin trials",
)

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
