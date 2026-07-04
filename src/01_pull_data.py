"""
Data pull script for structural credit risk project (Merton GBM vs. Jump-Diffusion PD).

Run this LOCALLY (not in a sandbox with restricted network access).
Requires: pip install yfinance pandas

What it does:
  1. Pulls 5 years of daily equity price history per ticker
  2. Pulls annual + quarterly balance sheet data (Total Debt, Total Liabilities,
     Total Assets — the inputs a Merton-style model needs)
  3. Saves everything to CSVs in ./credit_data/
  4. Prints a summary of what succeeded/failed, since delisted or thinly-covered
     tickers sometimes return incomplete data from yfinance

After running, upload the contents of ./credit_data/ back to Claude.
"""

import yfinance as yf
import pandas as pd
import os
import time

# --- Ticker universe -------------------------------------------------------
# Mix of stable investment-grade names (should show low, flat PD) and
# names that had real credit stress or distress (should show PD spiking
# around the known stress period — this is your validation set).
#
# NOTE: Some distressed/delisted names (SVB=SIVB, Credit Suisse=CS, WeWork=WE)
# may return incomplete or no data via yfinance once delisted. The script
# will flag these rather than fail silently — swap in alternatives if needed.

TICKERS = {
    # Stable / investment-grade (baseline, low PD expected)
    "AAPL": "Apple - stable large-cap",
    "MSFT": "Microsoft - stable large-cap",
    "JNJ":  "Johnson & Johnson - stable, low leverage",

    # Moderate leverage / cyclical (mid-range PD expected)
    "F":    "Ford - cyclical, moderate leverage",
    "CCL":  "Carnival Corp - high leverage, COVID-stressed then recovered",
    "AAL":  "American Airlines - high leverage, cyclical",

    # Known distress / stress events (PD should spike around known dates)
    "BBBY": "Bed Bath & Beyond - filed bankruptcy 2023 (may be delisted, check output)",
    "SIVB": "Silicon Valley Bank - collapsed March 2023 (likely delisted, check output)",
    "PARA": "Paramount - recent credit stress / rating pressure",
    "WBD":  "Warner Bros Discovery - high leverage post-merger",
}

OUT_DIR = "../data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

BALANCE_SHEET_FIELDS = [
    "Total Debt",
    "Total Liabilities Net Minority Interest",
    "Total Assets",
    "Common Stock Equity",
]

results = []

for ticker, note in TICKERS.items():
    print(f"\n--- {ticker} ({note}) ---")
    row = {"ticker": ticker, "note": note, "price_rows": 0,
           "annual_bs_available": False, "quarterly_bs_available": False}
    try:
        tk = yf.Ticker(ticker)

        # 1. Price history
        hist = tk.history(period="5y")
        if not hist.empty:
            hist.to_csv(f"{OUT_DIR}/{ticker}_prices.csv")
            row["price_rows"] = len(hist)
            print(f"  Prices: {len(hist)} daily rows saved.")
        else:
            print("  Prices: NO DATA returned.")

        # 2. Annual balance sheet
        bs = tk.balance_sheet
        if bs is not None and not bs.empty:
            bs.to_csv(f"{OUT_DIR}/{ticker}_balance_sheet_annual.csv")
            row["annual_bs_available"] = any(f in bs.index for f in BALANCE_SHEET_FIELDS)
            print(f"  Annual balance sheet: saved ({bs.shape[1]} periods).")
        else:
            print("  Annual balance sheet: NO DATA returned.")

        # 3. Quarterly balance sheet (more granular for tracking stress)
        qbs = tk.quarterly_balance_sheet
        if qbs is not None and not qbs.empty:
            qbs.to_csv(f"{OUT_DIR}/{ticker}_balance_sheet_quarterly.csv")
            row["quarterly_bs_available"] = any(f in qbs.index for f in BALANCE_SHEET_FIELDS)
            print(f"  Quarterly balance sheet: saved ({qbs.shape[1]} periods).")
        else:
            print("  Quarterly balance sheet: NO DATA returned.")

    except Exception as e:
        print(f"  ERROR: {e}")

    results.append(row)
    time.sleep(1)  # be polite to Yahoo's endpoint

# --- Summary -----------------------------------------------------------
summary = pd.DataFrame(results)
summary.to_csv(f"{OUT_DIR}/_pull_summary.csv", index=False)

print("\n\n=== SUMMARY ===")
print(summary.to_string(index=False))
print(f"\nAll files saved to ./{OUT_DIR}/")
print("Upload the CSVs (or the whole credit_data folder, zipped) back to Claude to continue.")
print("\nIf any ticker shows 0 price_rows or missing balance sheet data,")
print("it's likely delisted or thinly covered by yfinance — we can swap in a replacement.")
