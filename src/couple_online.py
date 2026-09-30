"""Plugs the trained NN back into the X-only integration and runs online."""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from l96_model import L96
from model_classes import FlexibleFCNN, GCM_network


steps = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]  # Different time steps for datasets
print(f"Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

spinup_steps = 1000  # Number of spin-up steps to reach a steady state
BLOWUP_THRESHOLD = 1e3

results = [] # To store results for each step

for step in steps:
    
    np.random.seed(14)
    torch.manual_seed(14)

    fcnn_relu_weights = torch.load(f"models/fcnn_model_relu_step_{step}.pth", weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    fcnn_tanh_weights = torch.load(f"models/fcnn_model_tanh_step_{step}.pth", weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    fcnn_model_relu = FlexibleFCNN(input_size=1, hidden_size=32, num_hidden_layers=3, out_size=1, act_fn=nn.ReLU).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    fcnn_model_tanh = FlexibleFCNN(input_size=1, hidden_size=32, num_hidden_layers=3, out_size=1, act_fn=nn.Tanh).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    fcnn_model_relu.load_state_dict(fcnn_relu_weights)
    fcnn_model_tanh.load_state_dict(fcnn_tanh_weights)

    print(f"Loaded models for step {step} on device {next(fcnn_model_relu.parameters()).device} and {next(fcnn_model_tanh.parameters()).device}.")

    T_test = 1000 # Testing for long horizons to see if the model can maintain stability and accuracy over extended periods. This is crucial for understanding the long-term behavior of the system under the influence of the trained neural network.
    forcing = 18
    dt = 0.01
    k = 8
    j = 32
    W = L96(K=k, J=j, F=forcing)

    total_steps = spinup_steps + int(T_test / dt)
    T_total = dt * total_steps
    X_full_raw, _, _ = W.run(dt, T=T_total)
    X_full = X_full_raw[spinup_steps:, :].astype(np.float32) # Convert to float32 for PyTorch compatibility

    init_conditions = X_full[0, :]
    nt = X_full.shape[0] - 1  # match reference run length exactly

    gcm_relu = GCM_network(F=forcing, network=fcnn_model_relu)
    Xnn_relu, t = gcm_relu(init_conditions, dt, nt, fcnn_model_relu)
    gcm_tanh = GCM_network(F=forcing, network=fcnn_model_tanh)
    Xnn_tanh, t = gcm_tanh(init_conditions, dt, nt, fcnn_model_tanh)

    true_mean, true_var = X_full.mean(), X_full.var()

    for name, Xnn in [("relu", Xnn_relu), ("tanh", Xnn_tanh)]:
        blew_up = bool(np.isnan(Xnn).any() or np.abs(Xnn).max() > BLOWUP_THRESHOLD)
        if blew_up:
            mean_diff, var_diff = np.nan, np.nan
        else:
            mean_diff = float(Xnn.mean() - true_mean)
            var_diff = float(Xnn.var() - true_var)
        results.append({
            "step": step,
            "model": name,
            "blew_up": blew_up,
            "mean_diff": mean_diff,
            "var_diff": var_diff,
            "max_abs_val": float(np.nanmax(np.abs(Xnn))),
        })
        print(f"  [{name}] step={step} blew_up={blew_up} "
              f"mean_diff={mean_diff} var_diff={var_diff}")


    k_plot = 0
    plt.figure(dpi=150)
    plt.plot(t, X_full[:, k_plot], "--", label="True Dynamics")
    plt.plot(t, Xnn_relu[:, k_plot], label="NN ReLU Dynamics")
    plt.plot(t, Xnn_tanh[:, k_plot], label="NN Tanh Dynamics")
    plt.legend()
    plt.title(f"Online Integration Comparison, X_{k_plot} (Step: {step})")
    plt.xlabel("Time")
    plt.ylabel("State Variable")
    plt.savefig(f"results/online_integration_step_{step}.png")
    plt.close()


os.makedirs("results", exist_ok=True)
with open("results/stability_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "model", "blew_up", "mean_diff", "var_diff", "max_abs_val"])
    writer.writeheader()
    writer.writerows(results)
print("\nSaved results/stability_table.csv")
