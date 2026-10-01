'''Evaluates offline error metrics'''

import csv
import numpy as np
import torch
import torch.nn as nn

from project_1_archive.src.model_classes import FlexibleFCNN

steps = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

for step in steps:
    data = np.load(f"data/raw/L96_true_data_{step}.npz")
    X_true = data["X_true"]
    xy_true = data["xy_true"]

    train_size = int(0.8 * step)
    X_true_test = X_true[train_size:, :]
    xy_true_test = xy_true[train_size:, :]

    X_test_flat = torch.from_numpy(np.reshape(X_true_test, -1)).float()
    y_test_flat = torch.from_numpy(np.reshape(xy_true_test, -1)).float()

    loss_fn = nn.MSELoss()

    for act_name, act_fn in [("relu", nn.ReLU), ("tanh", nn.Tanh)]:
        model = FlexibleFCNN(input_size=1, hidden_size=32, num_hidden_layers=3, out_size=1, act_fn=act_fn)
        weights = torch.load(f"models/fcnn_model_{act_name}_step_{step}.pth", weights_only=True, map_location=device)
        model.load_state_dict(weights)
        model.to(device).eval()

        with torch.no_grad():
            X_in = torch.unsqueeze(X_test_flat, 1).to(device)
            pred = torch.squeeze(model(X_in)).cpu()
            mse = loss_fn(pred, y_test_flat).item()

        results.append({"step": step, "model": act_name, "test_mse": mse})
        print(f"step={step} model={act_name} test_mse={mse:.6f}")

with open("results/offline_error_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "model", "test_mse"])
    writer.writeheader()
    writer.writerows(results)

print("\nSaved results/offline_error_table.csv")