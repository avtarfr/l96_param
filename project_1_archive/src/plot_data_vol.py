'''Plot the errors against data volume for offline and online models. Run a spearman correlation test to see if the errors decrease with increasing data volume.'''
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

offline = pd.read_csv("results/offline_error_table.csv")
online = pd.read_csv("results/stability_table.csv")

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

for name, grp in offline.groupby("model"):
    grp = grp.sort_values("step")
    axs[0].plot(grp["step"], grp["test_mse"], marker="o", label=name)
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_title("Offline test MSE vs data volume")
axs[0].set_xlabel("time_steps")
axs[0].set_ylabel("test MSE")
axs[0].legend()

for name, grp in online.groupby("model"):
    grp = grp.sort_values("step")
    axs[1].plot(grp["step"], grp["mean_diff"].abs(), marker="o", label=name)
axs[1].set_xscale("log")
axs[1].set_title("Online |mean_diff| vs data volume")
axs[1].set_xlabel("time_steps")
axs[1].set_ylabel("|mean_diff|")
axs[1].legend()

for name, grp in online.groupby("model"):
    grp = grp.sort_values("step")
    axs[2].plot(grp["step"], grp["var_diff"].abs(), marker="o", label=name)
axs[2].set_xscale("log")
axs[2].set_title("Online |var_diff| vs data volume")
axs[2].set_xlabel("time_steps")
axs[2].set_ylabel("|var_diff|")
axs[2].legend()

plt.tight_layout()
plt.savefig("results/offline_vs_online.png", dpi=150)
print("Saved results/offline_vs_online.png")

merged = pd.merge(offline, online, on=["step", "model"])
merged.to_csv("results/merged_offline_online.csv", index=False)
print("Saved results/merged_offline_online.csv")
print(merged)

print("\nSpearman rank correlation: test_mse vs |mean_diff| or |var_diff| (per model)")
for name, grp in merged.groupby("model"):
    grp = grp.sort_values("step")
    rho_mean, p_mean = spearmanr(grp["test_mse"], grp["mean_diff"].abs())
    rho_var, p_var = spearmanr(grp["test_mse"], grp["var_diff"].abs())
    print(f"  {name}: mse vs |mean_diff|  rho={rho_mean:.3f} p={p_mean:.3f}")
    print(f"  {name}: mse vs |var_diff|   rho={rho_var:.3f} p={p_var:.3f}")
