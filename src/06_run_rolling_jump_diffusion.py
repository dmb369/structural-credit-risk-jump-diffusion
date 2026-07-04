"""
Phase 2b: Rolling (walk-forward) Jump-Diffusion PD.

Fixes a look-ahead bias in 04_run_jump_diffusion.py: that script estimates
ONE jump distribution (lambda, mu_J, sigma_J) from the ENTIRE 5-year equity
return sample, then applies it at every date -- including dates years before
some of those jumps actually happened. A firm's estimated 2022 jump risk
should not be informed by a jump that occurs in 2024.

This version re-estimates jump parameters at each month-end date using only
equity returns observed UP TO that date (rolling window, minimum history
required before producing an estimate). This is also what a real deployment
would have to do -- you only ever have the past to calibrate on.

Output: {ticker}_rolling_jump_vs_gbm.csv, comparable in structure to Phase 2's
output but with time-varying (lambda, mu_J, sigma_J) instead of one fixed set.
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
OUT_DIR = "../output/phase2b_rolling_jump"
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "JNJ", "F", "CCL", "AAL", "WBD", "CVNA", "AMC", "SBH"]

R = 0.045
T = 1.0
JUMP_THRESHOLD = 4.0
VOL_LOOKBACK = 60
N_SIMS = 100_000          # halved vs Phase 2 since this now runs per-date, not once per ticker
MIN_JUMPS_REQUIRED = 4
ROLLING_WINDOW_DAYS = 504  # ~2 trading years of history used to estimate jump params at each date
MIN_HISTORY_DAYS = 252     # need at least 1 year of history before producing any estimate


def find_data_dir(ticker):
    for d in DATA_DIRS:
        if os.path.exists(f"{d}/{ticker}_prices.csv"):
            return d
    raise FileNotFoundError(f"No price data found for {ticker} in {DATA_DIRS}")


def detect_jumps(log_ret):
    roll_vol = log_ret.rolling(VOL_LOOKBACK).apply(
        lambda x: x[np.abs(x) < np.nanpercentile(np.abs(x), 95)].std(), raw=False
    )
    is_jump = np.abs(log_ret) > (JUMP_THRESHOLD * roll_vol)
    return is_jump.fillna(False)


def estimate_jump_params_asof(log_ret_full, asof_date):
    """Estimate jump params using only data in
    (asof_date - ROLLING_WINDOW_DAYS, asof_date] -- i.e. no future information."""
    window = log_ret_full.loc[:asof_date].tail(ROLLING_WINDOW_DAYS)
    if len(window) < MIN_HISTORY_DAYS:
        return None

    is_jump = detect_jumps(window)
    jump_days = window[is_jump]
    n_jumps = len(jump_days)

    if n_jumps < MIN_JUMPS_REQUIRED:
        return {"lambda": 0.0, "mu_J_equity": 0.0, "sigma_J_equity": 0.0, "n_jumps": n_jumps}

    lam = n_jumps / len(window) * 252
    return {
        "lambda": lam,
        "mu_J_equity": jump_days.mean(),
        "sigma_J_equity": jump_days.std(),
        "n_jumps": n_jumps,
    }


def compute_elasticity(A, sigma_A, D, r, T):
    d1 = (np.log(A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    return norm.cdf(d1)


def simulate_jump_pd(A0, D, sigma_diffusive, lam, mu_J, sigma_J, r, T, n_sims=N_SIMS, seed=42):
    rng = np.random.default_rng(seed)
    if lam <= 0 or sigma_J <= 0:
        # no reliable jump signal yet at this date -> pure GBM path
        log_AT = (np.log(A0) + (r - 0.5 * sigma_diffusive ** 2) * T
                  + sigma_diffusive * np.sqrt(T) * rng.normal(0, 1, n_sims))
        return np.mean(np.exp(log_AT) < D)

    k = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1
    n_jump_draws = rng.poisson(lam * T, size=n_sims)
    max_jumps = max(n_jump_draws.max(), 1)
    jump_matrix = rng.normal(mu_J, sigma_J, size=(n_sims, max_jumps))
    mask = np.arange(max_jumps)[None, :] < n_jump_draws[:, None]
    jump_sum = np.where(mask, jump_matrix, 0.0).sum(axis=1)

    diffusion_shock = rng.normal(0, 1, size=n_sims)
    log_AT = (np.log(A0) + (r - 0.5 * sigma_diffusive ** 2 - lam * k) * T
              + sigma_diffusive * np.sqrt(T) * diffusion_shock + jump_sum)
    return np.mean(np.exp(log_AT) < D)


comparison_data = {}
param_history = []

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
    log_ret_full = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()

    rows = []
    n_estimated, n_gbm_only = 0, 0
    for _, row in phase1.iterrows():
        asof = row["date"]
        params = estimate_jump_params_asof(log_ret_full, asof)

        if params is None:
            continue  # not enough history yet at this date

        omega = compute_elasticity(row["asset_value"], row["sigma_A"], row["default_point"], R, T)
        omega = max(omega, 1e-3)

        lam = params["lambda"]
        mu_J_asset = params["mu_J_equity"] / omega
        sigma_J_asset = params["sigma_J_equity"] / omega

        total_var = row["sigma_A"] ** 2
        jump_var_contribution = lam * (mu_J_asset ** 2 + sigma_J_asset ** 2)
        diffusive_var = max(total_var - jump_var_contribution, 0.05 ** 2)
        sigma_diffusive = np.sqrt(diffusive_var)

        pd_jump = simulate_jump_pd(
            row["asset_value"], row["default_point"], sigma_diffusive,
            lam, mu_J_asset, sigma_J_asset, R, T
        )

        if lam > 0:
            n_estimated += 1
        else:
            n_gbm_only += 1

        rows.append({
            "date": asof, "pd_gbm": row["pd_1yr"], "pd_jump_rolling": pd_jump,
            "lambda": lam, "mu_J_asset": mu_J_asset, "sigma_J_asset": sigma_J_asset,
            "n_jumps_in_window": params["n_jumps"], "sigma_diffusive": sigma_diffusive,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  Not enough history to produce any rolling estimate, skipping.")
        continue

    df.to_csv(f"{OUT_DIR}/{ticker}_rolling_jump_vs_gbm.csv", index=False)
    comparison_data[ticker] = df
    print(f"  {len(df)} dates | {n_estimated} with an active rolling jump estimate, "
          f"{n_gbm_only} fell back to GBM-only (too few jumps in trailing window)")

    max_row = df.loc[df["pd_jump_rolling"].idxmax()]
    print(f"  Max rolling jump-model PD: {max_row['pd_jump_rolling']*100:.4f}% on {max_row['date'].date()} "
          f"(lambda at that date={max_row['lambda']:.2f}/yr)")

    param_history.append(df.assign(ticker=ticker))

if param_history:
    pd.concat(param_history, ignore_index=True).to_csv(f"{OUT_DIR}/_all_rolling_params.csv", index=False)

# --- Plot: rolling jump lambda over time, per ticker (shows jump risk is NOT static) ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
plot_tickers = [t for t in ["CCL", "AAL", "WBD", "CVNA"] if t in comparison_data]
for ax, t in zip(axes.flat, plot_tickers):
    df = comparison_data[t]
    ax2 = ax.twinx()
    ax.plot(df["date"], df["pd_gbm"] * 100, label="GBM PD", color="tab:blue", marker="o", markersize=3)
    ax.plot(df["date"], df["pd_jump_rolling"] * 100, label="Rolling Jump PD", color="tab:orange", marker="s", markersize=3)
    ax2.plot(df["date"], df["lambda"], label="Jump intensity (lambda)", color="gray", linestyle=":", alpha=0.7)
    ax.set_title(t)
    ax.set_ylabel("1-Year PD (%)")
    ax2.set_ylabel("lambda (jumps/yr)", color="gray")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rolling_jump_intensity.png", dpi=150)
print(f"\nSaved rolling jump intensity chart to {OUT_DIR}/rolling_jump_intensity.png")
