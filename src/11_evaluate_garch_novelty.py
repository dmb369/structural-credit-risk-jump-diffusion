"""
Isolated test: does GARCH-forecasted vol beat naive trailing-vol in the
Merton GBM baseline, on the SAME proxy-label evaluation used for Phase 2/3?

Only one variable changes (volatility estimation method) so any metric
difference is attributable to that change, not confounded by jumps/regimes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import os

DATA_DIRS = ["../data/raw", "../data/validation"]
GBM_DIR = "../output/phase1_gbm"
GARCH_DIR = "../output/phase4_garch_gbm"
OUT_DIR = "../output/evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "JNJ", "F", "CCL", "AAL", "WBD", "CVNA", "AMC", "SBH"]
FORWARD_HORIZON_DAYS, DISTRESS_THRESHOLD = 252, 0.40


def find_data_dir(t):
    for d in DATA_DIRS:
        if os.path.exists(f"{d}/{t}_prices.csv"):
            return d
    raise FileNotFoundError(t)


def build_forward_labels(ticker):
    p = pd.read_csv(f"{find_data_dir(ticker)}/{ticker}_prices.csv")
    p["Date"] = pd.to_datetime(p["Date"], utc=True).dt.tz_localize(None)
    p = p.set_index("Date")["Close"].sort_index()
    labels, dates = {}, p.index
    for i, d in enumerate(dates):
        j = i + FORWARD_HORIZON_DAYS
        if j >= len(dates):
            continue
        labels[d] = int(p.iloc[i:j + 1].min() <= DISTRESS_THRESHOLD * p.iloc[i])
    return pd.Series(labels)


rows = []
for ticker in TICKERS:
    gbm_path, garch_path = f"{GBM_DIR}/{ticker}_merton_pd.csv", f"{GARCH_DIR}/{ticker}_garch_gbm.csv"
    if not (os.path.exists(gbm_path) and os.path.exists(garch_path)):
        continue
    gbm = pd.read_csv(gbm_path, parse_dates=["date"])[["date", "pd_1yr"]]
    garch = pd.read_csv(garch_path, parse_dates=["date"])[["date", "pd_garch_gbm"]]
    merged = gbm.merge(garch, on="date", how="inner")
    labels = build_forward_labels(ticker)
    merged["label"] = merged["date"].map(lambda d: labels.get(d, np.nan))
    merged = merged.dropna(subset=["label"])
    merged["ticker"] = ticker
    rows.append(merged)

eval_df = pd.concat(rows, ignore_index=True)
eval_df.to_csv(f"{OUT_DIR}/garch_novelty_eval.csv", index=False)
print(f"Evaluation set: {len(eval_df)} obs, {eval_df['label'].sum():.0f} distressed")

EPS = 1e-6
y = eval_df["label"].values
p_naive = np.clip(eval_df["pd_1yr"].values, EPS, 1 - EPS)
p_garch = np.clip(eval_df["pd_garch_gbm"].values, EPS, 1 - EPS)

results = {}
for name, p in [("GBM (naive trailing vol)", p_naive), ("GBM (GARCH-forecast vol)", p_garch)]:
    results[name] = {
        "ROC AUC": roc_auc_score(y, p),
        "Brier Score": brier_score_loss(y, p),
        "Log Loss": log_loss(y, p, labels=[0, 1]),
    }

results_df = pd.DataFrame(results).T
results_df.to_csv(f"{OUT_DIR}/garch_novelty_metrics.csv")
print("\n=== Naive trailing-vol vs GARCH-forecast vol (same proxy label) ===")
print(results_df.to_string())

auc_delta = results["GBM (GARCH-forecast vol)"]["ROC AUC"] - results["GBM (naive trailing vol)"]["ROC AUC"]
brier_delta = results["GBM (GARCH-forecast vol)"]["Brier Score"] - results["GBM (naive trailing vol)"]["Brier Score"]
print(f"\nROC AUC change: {auc_delta:+.4f}  |  Brier Score change: {brier_delta:+.4f} (negative = improvement)")
