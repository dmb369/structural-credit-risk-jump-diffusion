"""
Phase 4: GARCH-Vol GBM baseline.

Novelty: replaces Phase 1's trailing-252-day REALIZED volatility (backward-
looking, reacts slowly, equal-weights a full year of history) with a
GARCH(1,1) FORECAST of forward volatility, fit walk-forward (only data up to
each date is used -- no look-ahead) at each month-end.

Motivation: GARCH captures volatility clustering (Engle 1982, Bollerslev
1986) -- a large recent shock raises the near-term variance forecast
immediately, rather than waiting for the shock to enter a 252-day average
and then persist there for a full year after conditions normalize. If this
is a real improvement, it should show up as EARLIER PD increases ahead of
distress and better proxy-label statistics, not just a different chart.

Everything else (default point, elasticity, calibration equations) is
IDENTICAL to Phase 1 -- this isolates the volatility-estimation method as
the only changed variable, so any metric difference is attributable to it.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")
import os

DATA_DIRS = ["../data/raw", "../data/validation"]
OUT_DIR = "../output/phase4_garch_gbm"
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "JNJ", "F", "CCL", "AAL", "WBD", "CVNA", "AMC", "SBH"]
R, T = 0.045, 1.0
GARCH_WINDOW_DAYS = 504   # trailing window used to fit GARCH at each date
MIN_HISTORY_DAYS = 252


def find_data_dir(t):
    for d in DATA_DIRS:
        if os.path.exists(f"{d}/{t}_prices.csv"):
            return d
    raise FileNotFoundError(t)


def _merton_eqs(x, E, sigma_E, D, r, T):
    A, sigma_A = x
    if A <= 0 or sigma_A <= 0:
        return [1e10, 1e10]
    d1 = (np.log(A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    return [A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E,
            norm.cdf(d1) * sigma_A * A - sigma_E * E]


def calibrate_merton(E, sigma_E, D, r, T, A_guess=None, sigma_A_guess=None):
    A_guess = A_guess or (E + D)
    sigma_A_guess = sigma_A_guess or (sigma_E * E / (E + D))
    sol, info, ier, msg = fsolve(_merton_eqs, x0=[A_guess, sigma_A_guess],
                                  args=(E, sigma_E, D, r, T), full_output=True)
    A, sigma_A = sol
    return A, sigma_A, (ier == 1 and A > 0 and sigma_A > 0)


def pd_from_merton(A, sigma_A, D, r, T):
    dd = (np.log(A / D) + (r - 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    return norm.cdf(-dd)


def garch_forecast_vol(log_ret_full, asof_date):
    """Fit GARCH(1,1) on trailing window up to asof_date (in %, arch's
    convention), forecast the average variance over the next 252 days,
    annualize, and return as a decimal volatility."""
    window = log_ret_full.loc[:asof_date].tail(GARCH_WINDOW_DAYS)
    if len(window) < MIN_HISTORY_DAYS:
        return None
    try:
        am = arch_model(window * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off")
        fc = res.forecast(horizon=252, reindex=False)
        avg_daily_var_pct2 = fc.variance.values[-1].mean()  # in (% return)^2 units
        annual_vol = np.sqrt(avg_daily_var_pct2 * 252) / 100  # back to decimal
        return float(annual_vol)
    except Exception:
        return None


def load_ticker(ticker):
    data_dir = find_data_dir(ticker)
    prices = pd.read_csv(f"{data_dir}/{ticker}_prices.csv")
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    prices = prices.set_index("Date")[["Close"]].sort_index()

    bs = pd.read_csv(f"{data_dir}/{ticker}_balance_sheet_annual.csv", index_col=0).T
    bs.index = pd.to_datetime(bs.index)
    bs = bs.sort_index()
    shares = bs["Ordinary Shares Number"].astype(float)
    curr_liab = bs["Current Liabilities"].astype(float) if "Current Liabilities" in bs else None
    lt_debt = bs["Long Term Debt"].astype(float) if "Long Term Debt" in bs else None
    total_debt = bs["Total Debt"].astype(float)
    if curr_liab is not None and lt_debt is not None:
        dp = curr_liab.fillna(0) + 0.5 * lt_debt.fillna(0)
        dp = dp.where(dp > 0, total_debt)
    else:
        dp = total_debt
    return prices, shares, dp


for ticker in TICKERS:
    print(f"\n--- {ticker} ---")
    prices, shares, default_point = load_ticker(ticker)
    log_ret_full = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()
    shares_daily = shares.reindex(prices.index, method="ffill")
    debt_daily = default_point.reindex(prices.index, method="ffill")

    monthly_dates = prices.resample("ME").last().index
    rows = []
    A_guess, sigA_guess = None, None
    for dt in monthly_dates:
        idx = prices.index[prices.index <= dt]
        if len(idx) == 0:
            continue
        d = idx[-1]
        E_val = prices.loc[d, "Close"] * shares_daily.loc[d]
        D_val = debt_daily.loc[d]
        if pd.isna(E_val) or pd.isna(D_val) or D_val <= 0:
            continue

        sigE_garch = garch_forecast_vol(log_ret_full, d)
        if sigE_garch is None or sigE_garch <= 0:
            continue

        A, sigA, converged = calibrate_merton(E_val, sigE_garch, D_val, R, T, A_guess, sigA_guess)
        if not converged:
            continue
        A_guess, sigA_guess = A, sigA
        pd_val = pd_from_merton(A, sigA, D_val, R, T)
        rows.append({"date": d, "equity_value": E_val, "sigma_E_garch": sigE_garch,
                     "default_point": D_val, "asset_value": A, "sigma_A": sigA, "pd_garch_gbm": pd_val})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/{ticker}_garch_gbm.csv", index=False)
    if len(df):
        print(f"  {len(df)} points | max PD={df['pd_garch_gbm'].max()*100:.2f}% on "
              f"{df.loc[df['pd_garch_gbm'].idxmax(),'date'].date()}")
    else:
        print("  no valid points")
