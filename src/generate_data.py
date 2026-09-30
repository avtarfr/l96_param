"""Runs the truth simulation and saves datasets at different data volumes."""
import os
import numpy as np
import sys

current_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from l96_model import L96, RK2, RK4, EulerFwd, L96_eq1_xdot, integrate_L96_2t

time_steps = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]  # Different time steps for datasets
dt = 0.01  # Time step size
i = 0 # Index for the current time step
T = dt * time_steps[i] # Total simulation time
forcing = 18

spinup_steps = 1000  # Number of spin-up steps to reach a steady state
for i, steps in enumerate(time_steps):
    W = L96(K=8, J=32, F=forcing)  # Initialize the L96 model with K=8, J=32, and forcing F=18
    total_steps = spinup_steps + steps  # Total steps including spin-up
    T = dt * total_steps  # Total simulation time including spin-up
    X_true, _, _, xy_true = W.run(dt, T=T, store=True, return_coupling=True)
    X_true, xy_true = X_true[spinup_steps:, :].astype(np.float32), xy_true[spinup_steps:, :].astype(np.float32)  # Convert to float32 for PyTorch compatibility
    # store the datasets in a file
    np.savez(f"data/raw/L96_true_data_{steps}.npz", X_true=X_true, xy_true=xy_true)