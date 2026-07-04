"""
Statistical model comparison: GBM vs Rolling Jump-Diffusion.

IMPORTANT CAVEAT (read before trusting these numbers):
This dataset contains no confirmed defaults (see METHODOLOGY.md / RESULTS.md
for why). Every metric below is therefore computed against a PROXY distress
label, not a real default label:

    label = 1  if the stock's price falls to <= 40% of its value on the
               PD-observation date, at any point in the following 12 months
    label = 0  otherwise

This is a common workaround when a licensed default database (Moody's/S&P/
Bloomberg) isn't available, but it measures "did the market later treat this
as a severe risk event", not "did the firm actually default". Treat these
metrics as a RELATIVE comparison between the two models under a consistent
proxy, not an absolute accuracy claim.

Metrics computed: ROC AUC, Brier score, log loss, a calibration
(reliability) plot, and Spearman rank correlation between average predicted
PD and realized distress severity per ticker.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATA_DIRS = ["../data/raw", "../data/validation"]
ROLLING_DIR = "../output/phase2b_rolling_jump"
REGIME_DIR = "../output/phase3_regime_jump"
OUT_DIR = "../output/evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "JNJ", "F", "CCL", "AAL", "WBD", "CVNA", "AMC", "SBH"]

FORWARD_HORIZON_DAYS = 252   # 1 year, matching the PD horizon T=1
DISTRESS_THRESHOLD = 0.40    # label=1 if price falls to <= 40% of its value on the obs date


def find_data_dir(ticker):
    for d in DATA_DIRS:
        if os.path.exists(f"{d}/{ticker}_prices.csv"):
            return d
    raise FileNotFoundError(f"No price data found for {ticker}")


def build_forward_labels(ticker):
    """For each date, label=1 if price drops to <=40% of that date's price
    within the next 12 months. Dates too close to the end of the sample
    (no full forward window available) are dropped -- not labeled 0 by
    default, since we genuinely don't know what would have happened."""
    data_dir = find_data_dir(ticker)
    prices = pd.read_csv(f"{data_dir}/{ticker}_prices.csv")
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    prices = prices.set_index("Date")["Close"].sort_index()

    labels = {}
    dates = prices.index
    for i, d in enumerate(dates):
        window_end_idx = i + FORWARD_HORIZON_DAYS
        if window_end_idx >= len(dates):
            continue  # not enough forward data -- censored, excluded rather than guessed
        forward_min = prices.iloc[i:window_end_idx + 1].min()
        labels[d] = int(forward_min <= DISTRESS_THRESHOLD * prices.iloc[i])
    return pd.Series(labels)


all_rows = []
for ticker in TICKERS:
    path = f"{ROLLING_DIR}/{ticker}_rolling_jump_vs_gbm.csv"
    rpath = f"{REGIME_DIR}/{ticker}_regime_jump.csv"
    if not os.path.exists(path) or not os.path.exists(rpath):
        continue
    df = pd.read_csv(path, parse_dates=["date"])
    rdf = pd.read_csv(rpath, parse_dates=["date"])[["date", "pd_regime_jump"]]
    df = df.merge(rdf, on="date", how="inner")
    labels = build_forward_labels(ticker)

    # match each PD-observation date to the nearest available forward-label date
    df["label"] = df["date"].map(lambda d: labels.get(d, np.nan))
    if df["label"].isna().all():
        # fall back to nearest-date matching if exact dates don't line up
        label_series = labels.reindex(labels.index.union(df["date"])).sort_index().ffill()
        df["label"] = df["date"].map(lambda d: label_series.get(d, np.nan))

    df = df.dropna(subset=["label"])
    df["ticker"] = ticker
    all_rows.append(df[["ticker", "date", "pd_gbm", "pd_jump_rolling", "pd_regime_jump", "label"]])

eval_df = pd.concat(all_rows, ignore_index=True)
eval_df.to_csv(f"{OUT_DIR}/evaluation_dataset.csv", index=False)

print(f"Evaluation set: {len(eval_df)} (ticker, date) observations, "
      f"{eval_df['label'].sum():.0f} labeled distressed ({eval_df['label'].mean()*100:.1f}%)")
print(eval_df.groupby("ticker")["label"].agg(["count", "sum"]).rename(columns={"sum": "distress_events"}))

EPS = 1e-6
y = eval_df["label"].values
p_gbm = np.clip(eval_df["pd_gbm"].values, EPS, 1 - EPS)
p_jump = np.clip(eval_df["pd_jump_rolling"].values, EPS, 1 - EPS)
p_regime = np.clip(eval_df["pd_regime_jump"].values, EPS, 1 - EPS)

metrics = {}
for name, p in [("GBM", p_gbm), ("Rolling Jump-Diffusion", p_jump), ("Regime-Conditional Jump", p_regime)]:
    metrics[name] = {
        "ROC AUC": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "Brier Score": brier_score_loss(y, p),
        "Log Loss": log_loss(y, p, labels=[0, 1]),
    }

metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(f"{OUT_DIR}/metrics_summary.csv")
print("\n=== Statistical comparison (proxy-label distress prediction) ===")
print(metrics_df.to_string())
print("\nLower Brier Score / Log Loss = better. Higher ROC AUC = better.")

# --- Calibration plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for p, label, color in [(p_gbm, "GBM", "tab:blue"), (p_jump, "Rolling Jump", "tab:orange"), (p_regime, "Regime-Jump", "tab:green")]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    axes[0].plot(mean_pred, frac_pos, marker="o", label=label, color=color)
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="perfect calibration")
axes[0].set_xlabel("Mean predicted PD (bin)")
axes[0].set_ylabel("Observed distress frequency (bin)")
axes[0].set_title("Calibration (reliability) curve")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# --- Ranking distressed firms: avg PD per ticker vs realized severity ---
severity = eval_df.groupby("ticker")["label"].mean().rename("distress_rate")
avg_pd_gbm = eval_df.groupby("ticker")["pd_gbm"].mean().rename("avg_pd_gbm")
avg_pd_jump = eval_df.groupby("ticker")["pd_jump_rolling"].mean().rename("avg_pd_jump")
rank_df = pd.concat([severity, avg_pd_gbm, avg_pd_jump], axis=1)
rank_df.to_csv(f"{OUT_DIR}/ticker_ranking.csv")

rho_gbm, p_gbm_corr = spearmanr(rank_df["avg_pd_gbm"], rank_df["distress_rate"])
rho_jump, p_jump_corr = spearmanr(rank_df["avg_pd_jump"], rank_df["distress_rate"])

print("\n=== Ranking distressed firms (Spearman rank correlation vs realized distress rate) ===")
print(rank_df.sort_values("distress_rate", ascending=False).to_string())
print(f"\nGBM avg-PD rank correlation:               rho={rho_gbm:.3f} (p={p_gbm_corr:.3f})")
print(f"Rolling Jump-Diffusion avg-PD rank correlation: rho={rho_jump:.3f} (p={p_jump_corr:.3f})")

axes[1].scatter(rank_df["avg_pd_gbm"] * 100, rank_df["distress_rate"] * 100, label=f"GBM (rho={rho_gbm:.2f})", color="tab:blue")
axes[1].scatter(rank_df["avg_pd_jump"] * 100, rank_df["distress_rate"] * 100, label=f"Jump (rho={rho_jump:.2f})", color="tab:orange")
for t in rank_df.index:
    axes[1].annotate(t, (rank_df.loc[t, "avg_pd_jump"] * 100, rank_df.loc[t, "distress_rate"] * 100), fontsize=7)
axes[1].set_xlabel("Average predicted 1yr PD (%)")
axes[1].set_ylabel("Realized distress rate (%, proxy label)")
axes[1].set_title("Ranking check: predicted PD vs realized distress")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/calibration_and_ranking.png", dpi=150)
print(f"\nSaved calibration + ranking chart to {OUT_DIR}/calibration_and_ranking.png")

with open(f"{OUT_DIR}/README_CAVEAT.txt", "w") as f:
    f.write(
        "These metrics use a PROXY distress label (price falling to <=40% of its\n"
        "value within 12 months), not a confirmed default label -- this dataset\n"
        "contains no confirmed defaults (see METHODOLOGY.md). Interpret as a\n"
        "relative comparison between the two models under a consistent proxy,\n"
        "not an absolute accuracy claim against true default risk.\n"
    )
