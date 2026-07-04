"""
Ground-truth validation data pull.

Unlike the first pull (stable/moderately-stressed but surviving firms), this
targets companies that ACTUALLY filed for Chapter 11 / defaulted in recent
years. If our PD models are doing something real, PD should visibly rise in
the months BEFORE the actual filing date below.

Run this LOCALLY: pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd
import os
import time

OUT_DIR = "../data/validation"
os.makedirs(OUT_DIR, exist_ok=True)

# Real Chapter 11 filings / defaults with their actual filing dates.
# yfinance coverage for delisted names is inconsistent -- script will tell you
# what actually came back.
BANKRUPTCY_CASES = {
    "RAD":  {"note": "Rite Aid - filed Chapter 11", "filing_date": "2023-10-15"},
    "WE":   {"note": "WeWork - filed Chapter 11", "filing_date": "2023-11-06"},
    "PRTY": {"note": "Party City - filed Chapter 11 (1st time)", "filing_date": "2023-01-17"},
    "YELL": {"note": "Yellow Corp (trucking) - filed Chapter 11, liquidated", "filing_date": "2023-08-06"},
    "SBH":  {"note": "Sally Beauty - NOT bankrupt, control case (stable-ish)", "filing_date": None},
    "CVNA": {"note": "Carvana - severe distress 2022-23, survived (near-miss control case)", "filing_date": None},
    "AMC":  {"note": "AMC Entertainment - heavy leverage, survived (near-miss control case)", "filing_date": None},
}

BALANCE_SHEET_FIELDS = [
    "Total Debt", "Total Liabilities Net Minority Interest",
    "Total Assets", "Current Liabilities", "Long Term Debt",
]

results = []

for ticker, info in BANKRUPTCY_CASES.items():
    print(f"\n--- {ticker} ({info['note']}) ---")
    row = {"ticker": ticker, "note": info["note"], "filing_date": info["filing_date"],
           "price_rows": 0, "annual_bs_available": False, "quarterly_bs_available": False,
           "price_covers_filing_date": False}
    try:
        tk = yf.Ticker(ticker)

        # use max period to have the best shot at covering pre-delisting history
        hist = tk.history(period="max")
        if not hist.empty:
            hist.to_csv(f"{OUT_DIR}/{ticker}_prices.csv")
            row["price_rows"] = len(hist)
            last_date = hist.index[-1].tz_localize(None)
            print(f"  Prices: {len(hist)} rows, {hist.index[0].date()} to {hist.index[-1].date()}")

            if info["filing_date"]:
                filing_dt = pd.Timestamp(info["filing_date"])
                row["price_covers_filing_date"] = hist.index[0].tz_localize(None) <= filing_dt
                if row["price_covers_filing_date"]:
                    print(f"  Covers filing date {info['filing_date']} - GOOD, usable for validation.")
                else:
                    print(f"  Does NOT reach filing date {info['filing_date']} - data starts too late "
                          f"or stops too early, check manually.")
        else:
            print("  Prices: NO DATA returned (likely fully delisted/purged from Yahoo).")

        bs = tk.balance_sheet
        if bs is not None and not bs.empty:
            bs.to_csv(f"{OUT_DIR}/{ticker}_balance_sheet_annual.csv")
            row["annual_bs_available"] = any(f in bs.index for f in BALANCE_SHEET_FIELDS)
            print(f"  Annual balance sheet: saved ({bs.shape[1]} periods).")
        else:
            print("  Annual balance sheet: NO DATA (expected for delisted names).")

        qbs = tk.quarterly_balance_sheet
        if qbs is not None and not qbs.empty:
            qbs.to_csv(f"{OUT_DIR}/{ticker}_balance_sheet_quarterly.csv")
            row["quarterly_bs_available"] = any(f in qbs.index for f in BALANCE_SHEET_FIELDS)
            print(f"  Quarterly balance sheet: saved ({qbs.shape[1]} periods).")

    except Exception as e:
        print(f"  ERROR: {e}")

    results.append(row)
    time.sleep(1)

summary = pd.DataFrame(results)
summary.to_csv(f"{OUT_DIR}/_validation_summary.csv", index=False)
print("\n\n=== SUMMARY ===")
print(summary.to_string(index=False))
print(f"\nAll files saved to ./{OUT_DIR}/")
print("Zip this folder and upload it back — even partial coverage (price data only, no balance")
print("sheet, since delisted companies often lose their fundamentals data) is still useful:")
print("we can pull the LAST available balance sheet figures from before delisting via SEC filings")
print("if yfinance comes back empty, but try this first since it's fastest.")
