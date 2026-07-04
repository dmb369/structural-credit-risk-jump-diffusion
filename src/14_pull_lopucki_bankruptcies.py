"""
Pulls the free LoPucki BRD Cases table (real company names + filing dates,
~1,000 large public bankruptcies 1979-Dec 2022) and prepares it for ticker
matching.

MANUAL STEP FIRST: the download requires accepting terms on a form, so it
can't be scripted end-to-end. Go to:
    https://lopucki.law.ufl.edu/download_cases_table_terms.php
accept the terms, download the Cases table (CSV or Excel), and save it as
../data/bankruptcy_filings/lopucki_cases_table.csv

Then run this script to filter to recent-ish, ticker-matchable cases and
check which ones still have pre-delisting price history on Yahoo.

Run locally: pip install pandas yfinance
"""

import pandas as pd
import yfinance as yf
import os
import time

IN_PATH = "../data/bankruptcy_filings/lopucki_cases_table.csv"
OUT_DIR = "../data/bankruptcy_filings"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(IN_PATH):
    raise FileNotFoundError(
        f"{IN_PATH} not found. Download the Cases table manually from "
        "https://lopucki.law.ufl.edu/download_cases_table_terms.php first."
    )

df = pd.read_csv(IN_PATH, low_memory=False)
print(f"Loaded {len(df)} cases. Columns available: {list(df.columns)[:15]}...")

# Column names vary by BRD export version -- inspect and adjust these two
# lines to match your actual file (check df.columns for the real names,
# commonly something like 'NameCorp' and 'DateFiled').
NAME_COL = next((c for c in df.columns if "name" in c.lower()), df.columns[0])
DATE_COL = next((c for c in df.columns if "datefiled" in c.lower().replace(" ", "")
                  or "filingdate" in c.lower().replace(" ", "")), None)

print(f"Using name column: {NAME_COL}, date column: {DATE_COL}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
recent = df[df[DATE_COL] >= "2015-01-01"].copy()  # Yahoo more likely to still have these
recent = recent[[NAME_COL, DATE_COL]].dropna().sort_values(DATE_COL, ascending=False)
recent.to_csv(f"{OUT_DIR}/lopucki_recent_candidates.csv", index=False)
print(f"\n{len(recent)} bankruptcies filed since 2015 -- saved to "
      f"{OUT_DIR}/lopucki_recent_candidates.csv")
print("\nNext: manually look up the pre-bankruptcy ticker for each company name")
print("(a quick web search per name is fastest -- BRD doesn't include tickers).")
print("Then test each ticker below to see which still have Yahoo price history.\n")

# Once you've manually filled in tickers, list them here and this will test
# each one against yfinance in one pass:
CANDIDATE_TICKERS = [
    # "XYZ",  # Company Name Here, filed YYYY-MM-DD
]

if CANDIDATE_TICKERS:
    results = []
    for t in CANDIDATE_TICKERS:
        print(f"Testing {t}...")
        try:
            hist = yf.Ticker(t).history(period="max")
            results.append({"ticker": t, "price_rows": len(hist),
                             "usable": len(hist) > 252})
        except Exception as e:
            results.append({"ticker": t, "price_rows": 0, "usable": False})
        time.sleep(0.5)
    pd.DataFrame(results).to_csv(f"{OUT_DIR}/ticker_test_results.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))
else:
    print("Add tickers to CANDIDATE_TICKERS above once you've mapped some names, then rerun.")
