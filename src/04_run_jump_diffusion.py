"""
Phase 2: Merton Jump-Diffusion (1976) PD, built on top of the Phase 1 GBM
Merton calibration outputs (merton_output/{ticker}_merton_pd.csv).

Steps per ticker:
  1. Detect jump days in the DAILY EQUITY return series (threshold method).
  2. Estimate jump intensity (lambda) and equity-level jump size distribution
     (mu_J_equity, sigma_J_equity) from the flagged days.
  3. Convert equity-level jump size -> asset-level jump size using the same
     delta/elasticity scaling Merton uses to relate equity vol to asset vol:
        Omega = (A/E) * N(d1)      [elasticity of equity value w.r.t. asset value]
        asset jump size ~= equity jump size / Omega
  4. Decompose Phase 1's total implied asset variance (which, under a GBM
     assumption, silently absorbs jump variance into "diffusive" vol) into
     a genuine diffusive component and a jump component:
        sigma_A_gbm^2 (total) = sigma_diffusive^2 + lambda*(mu_J^2 + sigma_J^2)
  5. Simulate the terminal (T=1yr) asset value via Monte Carlo under
     GBM + compound Poisson lognormal jumps, and compute
     PD_jump = P(A_T < D).
  6. Compare PD_jump to Phase 1's closed-form GBM PD.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATA_DIRS = ["../data/raw", "../data/validation"]
PHASE1_DIR = "../output/phase1_gbm"
OUT_DIR = "../output/phase2_jump"
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "JNJ", "F", "CCL", "AAL", "WBD", "CVNA", "AMC", "SBH"]

R = 0.045
T = 1.0
JUMP_THRESHOLD = 4.0     # flag a day as a jump if |return| > 4x trailing realized vol
VOL_LOOKBACK = 60        # trailing window (days) for the realized-vol jump filter
N_SIMS = 200_000
MIN_JUMPS_REQUIRED = 4   # need a handful of detected jumps to fit mu_J/sigma_J reliably


def detect_jumps(log_ret):
    """Threshold jump detection: flag days where |return| exceeds N x trailing
    realized vol (excluding the most extreme 5% of days from the trailing vol
    calc, so a jump day doesn't inflate the very threshold used to detect it)."""
    roll_vol = log_ret.rolling(VOL_LOOKBACK).apply(
        lambda x: x[np.abs(x) < np.nanpercentile(np.abs(x), 95)].std(), raw=False
    )
    is_jump = np.abs(log_ret) > (JUMP_THRESHOLD * roll_vol)
    return is_jump.fillna(False), roll_vol


def estimate_equity_jump_params(prices):
    log_ret = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()
    is_jump, roll_vol = detect_jumps(log_ret)

    jump_days = log_ret[is_jump]
    n_days = len(log_ret)
    n_jumps = len(jump_days)

    if n_jumps < MIN_JUMPS_REQUIRED:
        return None  # not enough signal to fit a jump distribution reliably

    lam = n_jumps / n_days * 252  # jumps per year
    mu_J_eq = jump_days.mean()
    sigma_J_eq = jump_days.std()

    # diffusive-only equity vol, excluding jump days
    sigma_E_diffusive = log_ret[~is_jump].std() * np.sqrt(252)

    return {
        "lambda": lam, "mu_J_equity": mu_J_eq, "sigma_J_equity": sigma_J_eq,
        "n_jumps": n_jumps, "n_days": n_days,
        "sigma_E_diffusive": sigma_E_diffusive,
        "jump_dates": jump_days.index.tolist(),
    }


def compute_elasticity(A, sigma_A, D, r, T):
    """Omega = (A/E) * N(d1); requires E, which we get from Phase 1's csv too."""
    d1 = (np.log(A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    return norm.cdf(d1), d1


def simulate_jump_pd(A0, D, sigma_diffusive, lam, mu_J, sigma_J, r, T, n_sims=N_SIMS):
    rng = np.random.default_rng(42)
    k = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1  # jump compensator for martingale drift

    n_jump_draws = rng.poisson(lam * T, size=n_sims)
    max_jumps = max(n_jump_draws.max(), 1)

    # sum of jump log-sizes per path (vectorized: draw max_jumps per path, mask, sum)
    jump_matrix = rng.normal(mu_J, sigma_J, size=(n_sims, max_jumps))
    mask = np.arange(max_jumps)[None, :] < n_jump_draws[:, None]
    jump_sum = np.where(mask, jump_matrix, 0.0).sum(axis=1)

    diffusion_shock = rng.normal(0, 1, size=n_sims)
    log_AT = (np.log(A0)
              + (r - 0.5 * sigma_diffusive ** 2 - lam * k) * T
              + sigma_diffusive * np.sqrt(T) * diffusion_shock
              + jump_sum)

    A_T = np.exp(log_AT)
    pd_jump = np.mean(A_T < D)
    return pd_jump


def find_data_dir(ticker):
    for d in DATA_DIRS:
        if os.path.exists(f"{d}/{ticker}_prices.csv"):
            return d
    raise FileNotFoundError(f"No price data found for {ticker} in {DATA_DIRS}")


results_summary = []
comparison_data = {}

for ticker in TICKERS:
    print(f"\n--- {ticker} ---")
    phase1 = pd.read_csv(f"{PHASE1_DIR}/{ticker}_merton_pd.csv", parse_dates=["date"])
    if phase1.empty:
        print("  No Phase 1 data, skipping.")
        continue

    data_dir = find_data_dir(ticker)
    prices = pd.read_csv(f"{data_dir}/{ticker}_prices.csv")
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    prices = prices.set_index("Date")[["Close"]].sort_index()

    jump_params = estimate_equity_jump_params(prices)
    if jump_params is None:
        print("  Not enough detected jumps to calibrate a jump distribution — skipping jump model, GBM stands.")
        continue

    print(f"  Detected {jump_params['n_jumps']} jump days out of {jump_params['n_days']} "
          f"(lambda={jump_params['lambda']:.2f}/yr), "
          f"equity jump mean={jump_params['mu_J_equity']*100:.2f}%, "
          f"std={jump_params['sigma_J_equity']*100:.2f}%")

    # elasticity per month, then average to convert equity jump -> asset jump
    omegas = []
    for _, row in phase1.iterrows():
        omega, _ = compute_elasticity(row["asset_value"], row["sigma_A"], row["default_point"], R, T)
        omegas.append(omega)
    omega_avg = np.mean(omegas)

    mu_J_asset = jump_params["mu_J_equity"] / omega_avg
    sigma_J_asset = jump_params["sigma_J_equity"] / omega_avg
    lam = jump_params["lambda"]

    print(f"  Avg elasticity Omega={omega_avg:.3f} -> asset jump mean={mu_J_asset*100:.2f}%, "
          f"std={sigma_J_asset*100:.2f}%")

    jump_var_contribution = lam * (mu_J_asset ** 2 + sigma_J_asset ** 2)

    rows = []
    for _, row in phase1.iterrows():
        total_var = row["sigma_A"] ** 2
        diffusive_var = max(total_var - jump_var_contribution, (0.05) ** 2)  # floor
        sigma_diffusive = np.sqrt(diffusive_var)

        pd_jump = simulate_jump_pd(
            row["asset_value"], row["default_point"], sigma_diffusive,
            lam, mu_J_asset, sigma_J_asset, R, T
        )

        rows.append({
            "date": row["date"], "pd_gbm": row["pd_1yr"], "pd_jump": pd_jump,
            "sigma_A_total": row["sigma_A"], "sigma_diffusive": sigma_diffusive,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/{ticker}_jump_vs_gbm.csv", index=False)
    comparison_data[ticker] = df

    max_row = df.loc[df["pd_jump"].idxmax()]
    print(f"  Max jump-model PD: {max_row['pd_jump']*100:.4f}% on {max_row['date'].date()} "
          f"(GBM said {max_row['pd_gbm']*100:.4f}% that same date)")

    results_summary.append({
        "ticker": ticker, "lambda": lam, "mu_J_asset": mu_J_asset, "sigma_J_asset": sigma_J_asset,
        "omega_avg": omega_avg, "max_pd_gbm": df["pd_gbm"].max(), "max_pd_jump": df["pd_jump"].max(),
        "max_pd_jump_date": max_row["date"],
    })

summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(f"{OUT_DIR}/_summary.csv", index=False)
print("\n\n=== SUMMARY: GBM vs Jump-Diffusion max 1yr PD ===")
print(summary_df.to_string(index=False))

# --- Plot: GBM vs Jump PD for the most interesting tickers ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
plot_tickers = [t for t in ["CCL", "AAL", "WBD", "F"] if t in comparison_data]
for ax, t in zip(axes.flat, plot_tickers):
    df = comparison_data[t]
    ax.plot(df["date"], df["pd_gbm"] * 100, label="GBM (Merton)", marker="o", markersize=3)
    ax.plot(df["date"], df["pd_jump"] * 100, label="Jump-Diffusion", marker="s", markersize=3)
    ax.set_title(t)
    ax.set_ylabel("1-Year PD (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/gbm_vs_jump_comparison.png", dpi=150)
print(f"\nSaved comparison chart to {OUT_DIR}/gbm_vs_jump_comparison.png")
