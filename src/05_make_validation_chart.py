"""
Produces an annotated GBM-vs-Jump PD chart for the ground-truth validation
tickers (CVNA, AMC), marking known real-world distress/recovery events so the
model's PD path can be checked against what actually happened.

Run this AFTER 03_run_gbm_baseline.py and 04_run_jump_diffusion.py.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE2_DIR = "../output/phase2_jump"
OUT_PATH = "../output/phase2_jump/validation_cvna_amc.png"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, t in zip(axes, ["CVNA", "AMC"]):
    df = pd.read_csv(f"{PHASE2_DIR}/{t}_jump_vs_gbm.csv", parse_dates=["date"])
    ax.plot(df["date"], df["pd_gbm"] * 100, marker="o", markersize=3, label="GBM (Merton)")
    ax.plot(df["date"], df["pd_jump"] * 100, marker="s", markersize=3, label="Jump-Diffusion")
    ax.set_title(t)
    ax.set_ylabel("1-Year PD (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Real, publicly documented events -- not fitted to the model, added for
# context only.
axes[0].axvline(pd.Timestamp("2022-12-27"), color="red", linestyle="--", alpha=0.6)
axes[0].text(pd.Timestamp("2022-12-27"), 5, " price bottom\n -99%", fontsize=8, color="red")
axes[0].axvline(pd.Timestamp("2023-07-01"), color="green", linestyle="--", alpha=0.6)
axes[0].text(pd.Timestamp("2023-07-01"), 60, " debt\n restructuring\n announced", fontsize=8, color="green")

axes[1].axvline(pd.Timestamp("2023-12-01"), color="red", linestyle="--", alpha=0.6)
axes[1].text(pd.Timestamp("2023-12-01"), 5, " heavy dilution/\n debt-equity swaps", fontsize=8, color="red")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved validation chart to {OUT_PATH}")
