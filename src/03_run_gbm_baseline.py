"""
Run Merton GBM structural PD model across all tickers, through time.

For each ticker, at each month-end date, we compute:
  - E: market equity value = price * shares outstanding
  - sigma_E: trailing 1-year annualized equity volatility
  - D: default point (KMV convention) = Current Liabilities + 0.5 * Long-Term Debt
  - Calibrate (A, sigma_A) via Merton's simultaneous equations
  - Compute 1-year Distance-to-Default and PD

Output: one CSV of PD time series per ticker + a summary comparison chart.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from merton_gbm import calibrate_merton, probability_of_default

DATA_DIRS = ["../data/raw", "../data/validation"]
OUT_DIR = "../output/phase1_gbm"
import os
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "JNJ", "F", "CCL", "AAL", "WBD", "CVNA", "AMC", "SBH"]

R = 0.045   # risk-free rate assumption (approx current short-term UST yield)
T = 1.0     # 1-year PD horizon
VOL_WINDOW = 252  # trading days for trailing equity vol


def find_data_dir(ticker):
    """Ticker CSVs may live in data/raw (stable/moderate names) or
    data/validation (bankruptcy-candidate / distress-control names)."""
    for d in DATA_DIRS:
        if os.path.exists(f"{d}/{ticker}_prices.csv"):
            return d
    raise FileNotFoundError(f"No price data found for {ticker} in {DATA_DIRS}")


def load_ticker(ticker):
    data_dir = find_data_dir(ticker)
    prices = pd.read_csv(f"{data_dir}/{ticker}_prices.csv")
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    prices = prices.set_index("Date")[["Close"]].sort_index()

    bs = pd.read_csv(f"{data_dir}/{ticker}_balance_sheet_annual.csv", index_col=0)
    bs = bs.T
    bs.index = pd.to_datetime(bs.index)
    bs = bs.sort_index()

    shares = bs["Ordinary Shares Number"].astype(float)
    curr_liab = bs["Current Liabilities"].astype(float) if "Current Liabilities" in bs else None
    lt_debt = bs["Long Term Debt"].astype(float) if "Long Term Debt" in bs else None
    total_debt = bs["Total Debt"].astype(float)

    # KMV-style default point; fall back to Total Debt if fields are missing
    if curr_liab is not None and lt_debt is not None:
        default_point = curr_liab.fillna(0) + 0.5 * lt_debt.fillna(0)
        default_point = default_point.where(default_point > 0, total_debt)
    else:
        default_point = total_debt

    return prices, shares, default_point


def build_pd_series(ticker):
    prices, shares, default_point = load_ticker(ticker)

    # daily log returns for rolling vol
    log_ret = np.log(prices["Close"] / prices["Close"].shift(1))
    sigma_E_series = log_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)

    # forward-fill quarterly shares/debt onto the daily price index
    shares_daily = shares.reindex(prices.index, method="ffill")
    debt_daily = default_point.reindex(prices.index, method="ffill")

    # sample monthly (month-end) to keep calibration fast
    monthly_dates = prices.resample("ME").last().index
    monthly_dates = monthly_dates[monthly_dates.isin(prices.index) |
                                   (monthly_dates >= prices.index.min())]

    rows = []
    A_guess, sigA_guess = None, None
    for dt in monthly_dates:
        # nearest available price date at or before dt
        idx = prices.index[prices.index <= dt]
        if len(idx) == 0:
            continue
        d = idx[-1]

        E_val = prices.loc[d, "Close"] * shares_daily.loc[d]
        sigE_val = sigma_E_series.loc[d]
        D_val = debt_daily.loc[d]

        if pd.isna(E_val) or pd.isna(sigE_val) or pd.isna(D_val) or D_val <= 0 or sigE_val <= 0:
            continue

        A, sigA, converged = calibrate_merton(
            E_val, sigE_val, D_val, R, T, A_guess=A_guess, sigma_A_guess=sigA_guess
        )
        if not converged:
            continue

        A_guess, sigA_guess = A, sigA  # warm-start next iteration
        pd_val, dd_val = probability_of_default(A, sigA, D_val, R, T)

        rows.append({
            "date": d, "equity_value": E_val, "sigma_E": sigE_val,
            "default_point": D_val, "asset_value": A, "sigma_A": sigA,
            "distance_to_default": dd_val, "pd_1yr": pd_val,
        })

    return pd.DataFrame(rows)


all_results = {}
for t in TICKERS:
    print(f"Calibrating Merton model for {t}...")
    try:
        df = build_pd_series(t)
        df.to_csv(f"{OUT_DIR}/{t}_merton_pd.csv", index=False)
        all_results[t] = df
        if len(df):
            print(f"  {len(df)} monthly points | latest 1yr PD: {df['pd_1yr'].iloc[-1]*100:.4f}% "
                  f"| max 1yr PD: {df['pd_1yr'].max()*100:.4f}% on {df.loc[df['pd_1yr'].idxmax(),'date'].date()}")
        else:
            print("  No valid points produced.")
    except Exception as e:
        print(f"  ERROR: {e}")

# --- Plot: PD term structure over time for all tickers ---
fig, ax = plt.subplots(figsize=(11, 6))
for t, df in all_results.items():
    if len(df):
        ax.plot(df["date"], df["pd_1yr"] * 100, marker="o", markersize=2, label=t)

ax.set_yscale("log")
ax.set_ylabel("1-Year PD (%, log scale)")
ax.set_xlabel("Date")
ax.set_title("Merton (GBM) 1-Year Probability of Default — Monthly, 2022-2026")
ax.legend(ncol=4, fontsize=9)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/merton_pd_comparison.png", dpi=150)
print(f"\nSaved comparison chart to {OUT_DIR}/merton_pd_comparison.png")
