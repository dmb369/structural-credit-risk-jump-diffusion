"""
Phase 3: Regime-Conditional Rolling Jump-Diffusion.

Novelty over Phase 2b: jump intensity (lambda) is no longer a single pooled
number per trailing window. A 2-state Gaussian HMM is fit walk-forward on a
market-wide return proxy (equal-weighted average of all tickers' daily
returns) to classify each trailing day as "low-vol" or "high-vol" regime.
Jump days within the trailing window are then split by the regime active on
the day they occurred, giving a regime-specific lambda:

    lambda_current = (jumps that occurred while in the SAME regime as today)
                     / (trailing days spent in that regime) * 252

If there isn't enough regime-specific history, falls back to the pooled
(regime-blind) rolling estimate from 06_run_rolling_jump_diffusion.py.

Mu_J and sigma_J (jump size) are kept pooled across regimes -- conditioning
jump *intensity* on regime is the well-supported, low-parameter extension;
conditioning jump *size* on regime as well would need more data than this
window can support reliably.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIRS = ["../data/raw", "../data/validation"]
PHASE1_DIR = "../output/phase1_gbm"
OUT_DIR = "../output/phase3_regime_jump"
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "JNJ", "F", "CCL", "AAL", "WBD", "CVNA", "AMC", "SBH"]

R, T = 0.045, 1.0
JUMP_THRESHOLD, VOL_LOOKBACK = 4.0, 60
N_SIMS = 100_000
MIN_JUMPS_REQUIRED = 4
ROLLING_WINDOW_DAYS = 504
MIN_HISTORY_DAYS = 252
MIN_REGIME_JUMPS = 2  # need at least this many in-regime jumps to trust a regime-specific lambda


def find_data_dir(t):
    for d in DATA_DIRS:
        if os.path.exists(f"{d}/{t}_prices.csv"):
            return d
    raise FileNotFoundError(t)


def load_returns(t):
    p = pd.read_csv(f"{find_data_dir(t)}/{t}_prices.csv")
    p["Date"] = pd.to_datetime(p["Date"], utc=True).dt.tz_localize(None)
    p = p.set_index("Date")["Close"].sort_index()
    return np.log(p / p.shift(1)).dropna()


# --- Build market-wide proxy: equal-weighted average daily return across all tickers ---
all_rets = {t: load_returns(t) for t in TICKERS}
market_ret = pd.concat(all_rets, axis=1).mean(axis=1).dropna()
print(f"Market proxy: {len(market_ret)} days, {market_ret.index.min().date()} to {market_ret.index.max().date()}")


def classify_regime_walkforward(asof_date):
    """Fit a 2-state Gaussian HMM on the trailing window of market_ret up to
    (not including anything after) asof_date. Returns the in-sample state
    label for every day in that window plus which state is 'today's' regime."""
    window = market_ret.loc[:asof_date].tail(ROLLING_WINDOW_DAYS)
    if len(window) < MIN_HISTORY_DAYS:
        return None, None

    X = window.values.reshape(-1, 1)
    try:
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=0)
        model.fit(X)
        states = model.predict(X)
    except Exception:
        return None, None

    # label state with higher variance as "high-vol"; also check mean return
    # direction so regimes read as bull/bear, not just calm/stressed
    var0, var1 = model.covars_[0][0][0], model.covars_[1][0][0]
    mean0, mean1 = model.means_[0][0], model.means_[1][0]
    high_vol_state = 0 if var0 > var1 else 1
    bear_state = 0 if mean0 < mean1 else 1
    # a state counts as "high_vol" (the regime we condition jumps on) if it's
    # EITHER the higher-variance state OR the negative-mean state when they disagree
    if high_vol_state == bear_state:
        stress_state = high_vol_state
    else:
        # disagreement: prioritize the negative-mean (bear) state as the stress regime
        stress_state = bear_state
    regime_labels = pd.Series(np.where(states == stress_state, "high_vol", "low_vol"), index=window.index)
    today_regime = regime_labels.iloc[-1]
    return regime_labels, today_regime


def detect_jumps(log_ret):
    roll_vol = log_ret.rolling(VOL_LOOKBACK).apply(
        lambda x: x[np.abs(x) < np.nanpercentile(np.abs(x), 95)].std(), raw=False)
    return (np.abs(log_ret) > (JUMP_THRESHOLD * roll_vol)).fillna(False)


def compute_elasticity(A, sigma_A, D, r, T):
    d1 = (np.log(A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    return max(norm.cdf(d1), 1e-3)


def simulate_jump_pd(A0, D, sigma_diffusive, lam, mu_J, sigma_J, r, T, n_sims=N_SIMS, seed=42):
    rng = np.random.default_rng(seed)
    if lam <= 0 or sigma_J <= 0:
        log_AT = np.log(A0) + (r - 0.5 * sigma_diffusive ** 2) * T + sigma_diffusive * np.sqrt(T) * rng.normal(0, 1, n_sims)
        return np.mean(np.exp(log_AT) < D)
    k = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1
    n_jumps = rng.poisson(lam * T, size=n_sims)
    max_j = max(n_jumps.max(), 1)
    jm = rng.normal(mu_J, sigma_J, size=(n_sims, max_j))
    mask = np.arange(max_j)[None, :] < n_jumps[:, None]
    jump_sum = np.where(mask, jm, 0.0).sum(axis=1)
    log_AT = np.log(A0) + (r - 0.5 * sigma_diffusive ** 2 - lam * k) * T + sigma_diffusive * np.sqrt(T) * rng.normal(0, 1, n_sims) + jump_sum
    return np.mean(np.exp(log_AT) < D)


comparison = {}
for ticker in TICKERS:
    print(f"\n--- {ticker} ---")
    phase1 = pd.read_csv(f"{PHASE1_DIR}/{ticker}_merton_pd.csv", parse_dates=["date"])
    if phase1.empty:
        continue
    log_ret_full = all_rets[ticker]

    rows = []
    for _, row in phase1.iterrows():
        asof = row["date"]
        regime_labels, today_regime = classify_regime_walkforward(asof)
        if regime_labels is None:
            continue

        window_ret = log_ret_full.loc[:asof].tail(ROLLING_WINDOW_DAYS)
        if len(window_ret) < MIN_HISTORY_DAYS:
            continue
        is_jump = detect_jumps(window_ret)
        jump_days = window_ret[is_jump]

        # align jump days to regime labels (same trailing window / dates)
        jump_regimes = regime_labels.reindex(jump_days.index)
        in_regime_jumps = jump_days[jump_regimes == today_regime]
        days_in_regime = (regime_labels == today_regime).sum()

        if len(in_regime_jumps) >= MIN_REGIME_JUMPS and days_in_regime > 0:
            lam = len(in_regime_jumps) / days_in_regime * 252
            mu_J_eq, sigma_J_eq = jump_days.mean(), jump_days.std()  # pooled size, regime-specific rate
            source = "regime"
        elif len(jump_days) >= MIN_JUMPS_REQUIRED:
            lam = len(jump_days) / len(window_ret) * 252
            mu_J_eq, sigma_J_eq = jump_days.mean(), jump_days.std()
            source = "pooled_fallback"
        else:
            lam, mu_J_eq, sigma_J_eq, source = 0.0, 0.0, 0.0, "gbm_fallback"

        omega = compute_elasticity(row["asset_value"], row["sigma_A"], row["default_point"], R, T)
        mu_J_asset, sigma_J_asset = mu_J_eq / omega, sigma_J_eq / omega

        total_var = row["sigma_A"] ** 2
        diffusive_var = max(total_var - lam * (mu_J_asset ** 2 + sigma_J_asset ** 2), 0.05 ** 2)
        pd_regime = simulate_jump_pd(row["asset_value"], row["default_point"], np.sqrt(diffusive_var),
                                      lam, mu_J_asset, sigma_J_asset, R, T)

        rows.append({"date": asof, "pd_gbm": row["pd_1yr"], "pd_regime_jump": pd_regime,
                     "regime": today_regime, "lambda": lam, "lambda_source": source})

    df = pd.DataFrame(rows)
    if df.empty:
        print("  insufficient history, skipped")
        continue
    df.to_csv(f"{OUT_DIR}/{ticker}_regime_jump.csv", index=False)
    comparison[ticker] = df
    n_regime = (df["lambda_source"] == "regime").sum()
    print(f"  {len(df)} dates | {n_regime} used a regime-specific lambda | "
          f"max PD={df['pd_regime_jump'].max()*100:.2f}% on {df.loc[df['pd_regime_jump'].idxmax(),'date'].date()}")

# --- Plot: regime-conditional PD vs GBM for a few tickers, regime shaded ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, t in zip(axes.flat, [t for t in ["AAL", "WBD", "CVNA", "F"] if t in comparison]):
    df = comparison[t]
    ax.plot(df["date"], df["pd_gbm"] * 100, label="GBM", marker="o", markersize=3)
    ax.plot(df["date"], df["pd_regime_jump"] * 100, label="Regime-Conditional Jump", marker="s", markersize=3)
    for _, r in df.iterrows():
        if r["regime"] == "high_vol":
            ax.axvspan(r["date"], r["date"], color="red", alpha=0.05)
    ax.set_title(t)
    ax.set_ylabel("1-Year PD (%)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/regime_conditional_comparison.png", dpi=150)
print(f"\nSaved chart to {OUT_DIR}/regime_conditional_comparison.png")
