import numpy as np 
import matplotlib.pyplot as plt 
from config import * 
from schrodinger_solver import * 
from egorov import * 
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

rng = np.random.default_rng(seed=42)
q_samples, p_samples = MC_sample(N,rng) 

times, position,_, _, _ = compute_expectation(position_observable, q_samples, p_samples, dt, n_steps) 
_,  momentum,_,_,_ = compute_expectation(momentum_observable, q_samples, p_samples, dt, n_steps) 
_,kinetic_energy,_, _, _ = compute_expectation(kinetic_energy_observable, q_samples, p_samples, dt, n_steps) 
_, potential_energy,_, _, _ = compute_expectation(potential_energy_observable, q_samples, p_samples, dt,n_steps) 
_, total_energy,_, _, _ = compute_expectation(total_energy_observable, q_samples, p_samples, dt, n_steps) 

# visualization 
plt.plot(times, position, label="Egorov") 
plt.plot(times, position_qm, "--", label="Schrodinger") 
plt.xlabel("time") 
plt.ylabel("position expectation") 
plt.legend() 
plt.savefig('position.png') 
plt.cla() 

plt.plot(times, momentum, label="Egorov") 
plt.plot(times, momentum_qm, "--", label="Schrodinger")
plt.xlabel("time")
plt.ylabel("momentum expectation") 
plt.legend() 
plt.savefig('momentum.png') 
plt.cla() 

plt.plot(times, kinetic_energy, label="Egorov kinetic")
plt.plot(times, kinetic_qm, "--", label="Quantum kinetic") 
plt.plot(times, potential_energy, label="Egorov potential") 
plt.plot(times, potential_qm, "--", label="Quantum potential") 
plt.plot(times, total_energy, label="Egorov total") 
plt.plot(times, total_qm, "--", label="Quantum total") 
plt.xlabel("time") 
plt.ylabel("energy") 
plt.legend() 
plt.savefig('energy.png')
