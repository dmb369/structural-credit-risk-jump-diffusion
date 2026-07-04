"""
Find real, recent Chapter 11 filings via SEC EDGAR full-text search.

Free, no API key needed. Searches 8-K filings for Item 1.03 ("Bankruptcy or
Receivership") language, which is the SEC's own required disclosure trigger
-- this is a real, dated, verifiable ground-truth event, not a guessed
ticker list.

Run locally: pip install requests pandas
"""

import requests
import pandas as pd
import time
import os

OUT_DIR = "../data/bankruptcy_filings"
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Research Project research@example.com"}  # SEC requires a User-Agent
SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22Item+1.03%22&forms=8-K&dateRange=custom"

# SEC full text search only indexes 2001+. We search year by year to stay
# under result-count limits and keep queries fast.
YEARS = list(range(2019, 2027))


def search_bankruptcies(year):
    """Query SEC EDGAR full-text search for 8-Ks mentioning Item 1.03 in a
    given year. Returns raw hits (company name, CIK, filing date, accession)."""
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": "\"Item 1.03\"",
        "forms": "8-K",
        "dateRange": "custom",
        "startdt": f"{year}-01-01",
        "enddt": f"{year}-12-31",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  {year}: request failed ({e})")
        return []

    hits = []
    for h in data.get("hits", {}).get("hits", {}).get("hits", []) if isinstance(data.get("hits", {}).get("hits"), dict) else data.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append({
            "company": src.get("display_names", [""])[0] if src.get("display_names") else "",
            "cik": src.get("ciks", [""])[0] if src.get("ciks") else "",
            "filing_date": src.get("file_date", ""),
            "form": src.get("root_form", ""),
            "accession": h.get("_id", ""),
        })
    return hits


all_hits = []
for year in YEARS:
    print(f"Searching {year}...")
    hits = search_bankruptcies(year)
    print(f"  {len(hits)} Item 1.03 8-K filings found")
    all_hits.extend(hits)
    time.sleep(0.3)  # be polite to SEC's rate limit (10 req/sec max, we stay well under)

df = pd.DataFrame(all_hits).drop_duplicates(subset=["cik", "filing_date"])
df.to_csv(f"{OUT_DIR}/sec_bankruptcy_filings.csv", index=False)
print(f"\nSaved {len(df)} unique bankruptcy-disclosure filings to "
      f"{OUT_DIR}/sec_bankruptcy_filings.csv")
print("\nNOTE: 'company' names here are as filed with the SEC and won't include")
print("ticker symbols directly. Next step: manually (or via a ticker-lookup API")
print("like OpenFIGI, free with registration) map the company names/CIKs in this")
print("file to their pre-bankruptcy ticker symbols, then feed those into")
print("13_pull_expanded_universe.py below.")
print(df.head(20).to_string(index=False))
