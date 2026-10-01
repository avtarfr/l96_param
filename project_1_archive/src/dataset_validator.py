"""
Full validation suite for the L96 data-volume sweep.
Run from the notebooks/ or src/ directory (wherever L96 model + data/ are reachable).

Checks:
1. Spin-up sufficiency  -> full run w/ vline at cutoff, visual check transient is gone
2. Climatology convergence -> mean/var of X vs data volume, should flatten
3. Distribution shape -> histogram compare largest vs smallest dataset
4. Coupling term sanity -> mean(U_k) over time, should hover not drift
5. Basic shape/heatmap look -> your original per-dataset visual check

Generated with the help of Claude Sonnet 5 Medium-Thinking.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from project_1_archive.src.l96_model import L96

# ---- config, must match generate_data.py ----
time_steps = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
dt = 0.01
forcing = 18
spinup_steps = 1000
data_dir = "data/raw"


# =========================================================
# 1. SPIN-UP CHECK
# Run once at max length INCLUDING spinup, plot X over time
# with a vline at the spinup cutoff. Before the line dynamics
# should look like they're settling; after, uniformly chaotic.
# =========================================================
def check_spinup():
    W = L96(K=8, J=32, F=forcing)
    total_steps = spinup_steps + max(time_steps)
    T_total = dt * total_steps
    X_full, _, _, _ = W.run(dt, T=T_total, store=True, return_coupling=True)

    plt.figure(figsize=(12, 4))
    # plot first few k-indices to keep it readable
    for k in range(min(4, X_full.shape[1])):
        plt.plot(X_full[:, k], label=f"X_{k}", alpha=0.7)
    plt.axvline(spinup_steps, color="red", linestyle="--", label="spin-up cutoff")
    plt.title("Spin-up check: dynamics before/after cutoff")
    plt.xlabel("timestep")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Spin-up steps: {spinup_steps}")
    print("Visually confirm: no visible settling/trend after the red line.\n")


# =========================================================
# 2. CLIMATOLOGY CONVERGENCE
# mean(X) and var(X) per dataset, plotted vs data volume.
# Should flatten out as steps increases. Big systematic
# offset at small volumes (not just noise) = problem.
# =========================================================
def check_climatology_convergence():
    means, variances = [], []
    for step in time_steps:
        data = np.load(f"{data_dir}/L96_true_data_{step}.npz")
        X_true = data["X_true"]
        means.append(X_true.mean())
        variances.append(X_true.var())

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(time_steps, means, marker="o")
    axs[0].set_xscale("log")
    axs[0].set_title("Mean(X) vs data volume")
    axs[0].set_xlabel("time_steps")
    axs[0].axhline(means[-1], color="gray", linestyle="--", alpha=0.5, label="largest-volume value")
    axs[0].legend()

    axs[1].plot(time_steps, variances, marker="o")
    axs[1].set_xscale("log")
    axs[1].set_title("Var(X) vs data volume")
    axs[1].set_xlabel("time_steps")
    axs[1].axhline(variances[-1], color="gray", linestyle="--", alpha=0.5, label="largest-volume value")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    print("Check: do small-volume points scatter around the dashed line (fine, "
          "just sampling noise), or sit systematically off it (problem)?\n")


# =========================================================
# 3. DISTRIBUTION SHAPE CHECK
# Histogram of pooled X values, smallest vs largest dataset.
# Shapes should look similar (same attractor), smallest just
# noisier/coarser.
# =========================================================
def check_distribution_shape():
    small = np.load(f"{data_dir}/L96_true_data_{time_steps[0]}.npz")["X_true"].flatten()
    large = np.load(f"{data_dir}/L96_true_data_{time_steps[-1]}.npz")["X_true"].flatten()

    plt.figure(figsize=(8, 4))
    plt.hist(large, bins=50, density=True, alpha=0.5, label=f"{time_steps[-1]} steps")
    plt.hist(small, bins=50, density=True, alpha=0.5, label=f"{time_steps[0]} steps")
    plt.title("X value distribution: smallest vs largest dataset")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Check: similar shape/center, not shifted or clearly different.\n")


# =========================================================
# 4. COUPLING TERM (U_k) SANITY CHECK
# mean(xy_true) over time, largest dataset. Should hover
# around a roughly stable value with fluctuations, not drift.
# =========================================================
def check_coupling_term():
    data = np.load(f"{data_dir}/L96_true_data_{time_steps[-1]}.npz")
    xy_true = data["xy_true"]

    mean_over_k = xy_true.mean(axis=1)  # mean across k at each timestep

    plt.figure(figsize=(10, 4))
    plt.plot(mean_over_k)
    plt.title(f"Mean coupling term U_k over time ({time_steps[-1]} steps)")
    plt.xlabel("timestep")
    plt.ylabel("mean(U_k)")
    plt.tight_layout()
    plt.show()

    print(f"Overall mean(U_k): {xy_true.mean():.4f}, std: {xy_true.std():.4f}")
    print("Check: fluctuates around a stable level, no upward/downward drift, "
          "no NaNs, magnitude plausible given h/b/c.\n")


# =========================================================
# 5. SHAPE + HEATMAP LOOK (your original check)
# =========================================================
def check_shapes_and_heatmaps():
    for step in time_steps:
        data = np.load(f"{data_dir}/L96_true_data_{step}.npz")
        X_true = data["X_true"]
        xy_true = data["xy_true"]

        print(f"Dataset for {step} steps: X_true {X_true.shape}, xy_true {xy_true.shape}")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"X_true, {step} steps")
        plt.imshow(X_true.T, aspect="auto", cmap="viridis")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title(f"xy_true, {step} steps")
        plt.imshow(xy_true.T, aspect="auto", cmap="viridis")
        plt.colorbar()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=== 1. Spin-up check ===")
    check_spinup()

    print("=== 2. Climatology convergence ===")
    check_climatology_convergence()

    print("=== 3. Distribution shape ===")
    check_distribution_shape()

    print("=== 4. Coupling term sanity ===")
    check_coupling_term()

    print("=== 5. Shapes + heatmaps (all datasets) ===")
    check_shapes_and_heatmaps()