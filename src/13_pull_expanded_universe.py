"""
Pull an expanded, credit-quality-diversified universe (~100 tickers) to fix
two problems in the original 10-ticker evaluation:
  1. Only 1 ticker (AMC) drove almost all positive proxy-distress labels.
  2. Jump/regime models were data-starved (2-4 jump days per ticker).

List spans investment-grade -> high-yield -> historically distressed-but-
survived names, across sectors, so the eventual proxy-label evaluation has
more than one ticker contributing positive events.

Add any tickers you've manually mapped from
12_find_recent_bankruptcies.py's output to REAL_DEFAULT_TICKERS below --
those are the highest-value additions, since they're actual confirmed
defaults with known dates, not proxy labels.

Run locally: pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd
import os
import time

OUT_DIR = "../data/expanded"
os.makedirs(OUT_DIR, exist_ok=True)

INVESTMENT_GRADE = ["AAPL", "MSFT", "JNJ", "PG", "KO", "WMT", "HD", "COST",
                     "UNH", "V", "MA", "PEP", "MCD", "LIN", "TXN"]

HIGH_YIELD_CYCLICAL = ["F", "GM", "CCL", "RCL", "NCLH", "AAL", "UAL", "DAL",
                        "WBD", "PARAA", "LYFT", "UBER", "DKNG", "CHWY"]

MODERATE_LEVERAGE = ["T", "VZ", "PARA", "SIRI", "NWL", "KHC", "HAS", "M",
                      "KSS", "GPS", "BBWI", "DASH"]

HISTORICALLY_DISTRESSED_SURVIVED = ["CVNA", "AMC", "SBH", "BBBY", "GME",
                                     "OSTK", "UPST", "AFRM", "SPCE", "PLUG",
                                     "FUBO", "RIDE", "NKLA", "LCID", "RIVN"]

# Fill in with tickers you've manually confirmed as real, dated Chapter 11
# filers with pre-delisting price history still available (check each one --
# most delisted small/micro caps will fail, as RAD/WE/PRTY/YELL did).
REAL_DEFAULT_TICKERS = [
    # "XYZQ",  # example: add confirmed-mapped tickers from step 12 here
]

ALL_TICKERS = sorted(set(INVESTMENT_GRADE + HIGH_YIELD_CYCLICAL + MODERATE_LEVERAGE
                          + HISTORICALLY_DISTRESSED_SURVIVED + REAL_DEFAULT_TICKERS))

BALANCE_SHEET_FIELDS = ["Total Debt", "Total Liabilities Net Minority Interest",
                         "Total Assets", "Current Liabilities", "Long Term Debt"]

results = []
for i, ticker in enumerate(ALL_TICKERS):
    print(f"[{i+1}/{len(ALL_TICKERS)}] {ticker}...")
    row = {"ticker": ticker, "price_rows": 0, "annual_bs_available": False}
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5y")
        if not hist.empty:
            hist.to_csv(f"{OUT_DIR}/{ticker}_prices.csv")
            row["price_rows"] = len(hist)

        bs = tk.balance_sheet
        if bs is not None and not bs.empty:
            bs.to_csv(f"{OUT_DIR}/{ticker}_balance_sheet_annual.csv")
            row["annual_bs_available"] = any(f in bs.index for f in BALANCE_SHEET_FIELDS)

        qbs = tk.quarterly_balance_sheet
        if qbs is not None and not qbs.empty:
            qbs.to_csv(f"{OUT_DIR}/{ticker}_balance_sheet_quarterly.csv")
    except Exception as e:
        print(f"  ERROR: {e}")

    results.append(row)
    time.sleep(0.5)

summary = pd.DataFrame(results)
summary.to_csv(f"{OUT_DIR}/_expanded_summary.csv", index=False)
usable = summary[(summary["price_rows"] > 0) & (summary["annual_bs_available"])]
print(f"\n{len(usable)}/{len(ALL_TICKERS)} tickers came back fully usable "
      f"(price + balance sheet data).")
print("Zip the expanded/ folder and continue the pipeline with this larger universe.")
