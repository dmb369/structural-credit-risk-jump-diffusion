# Structural Credit Risk PD: GBM Baseline vs. Jump-Diffusion Extension

A from-scratch implementation of Merton's (1974) structural probability-of-default
model, extended with Merton's (1976) jump-diffusion process to address the
well-documented "short-horizon PD collapse" problem of pure Brownian-motion
structural models.

See `METHODOLOGY.md` for the full technical approach, `RESULTS.md` for findings,
and `REFERENCES.md` for the academic grounding.

## Repo structure

```
credit-risk-pd/
├── README.md                  <- you are here
├── METHODOLOGY.md              <- full technical writeup
├── RESULTS.md                  <- findings, charts, validation
├── REFERENCES.md               <- academic references
├── requirements.txt
├── src/
│   ├── 01_pull_data.py                 pulls prices + balance sheets (main ticker set)
│   ├── 02_pull_validation_data.py      pulls ground-truth / distress-case tickers
│   ├── merton_gbm.py                   core Merton GBM calibration functions
│   ├── 03_run_gbm_baseline.py          Phase 1: GBM Merton PD, monthly time series
│   ├── 04_run_jump_diffusion.py        Phase 2: jump-diffusion extension + comparison
│   └── 05_make_validation_chart.py     annotated validation chart (CVNA / AMC)
├── data/
│   ├── raw/                    prices + balance sheets, stable/moderate-leverage tickers
│   └── validation/              prices + balance sheets, distress/near-default tickers
└── output/
    ├── phase1_gbm/              Phase 1 CSVs + comparison chart
    └── phase2_jump/             Phase 2 CSVs + comparison + validation charts
```

Sample data and output are already included in this repo (from the original run),
so you can inspect `RESULTS.md` and the charts in `output/` without running
anything. Instructions below are for reproducing or extending the analysis.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run (full pipeline, in order)

All scripts are run from inside `src/`.

### Step 1 — Pull market data
```bash
cd src
python3 01_pull_data.py
```
Pulls 5 years of daily prices + annual/quarterly balance sheets for the main
ticker set (stable large-caps, cyclicals, and moderately-levered names) via
`yfinance`. Saves to `../data/raw/`.

**Note:** `yfinance` has no route through this environment's sandbox — this
step must be run on your own machine with internet access. Some tickers may
return incomplete data if `yfinance`'s fundamentals coverage is thin; the
script prints a per-ticker summary so you can see what came back clean.

### Step 2 — Pull validation data
```bash
python3 02_pull_validation_data.py
```
Pulls tickers used for ground-truth validation: real Chapter 11 filers
(coverage-dependent — Yahoo purges price history for delisted small/mid-caps
within roughly a year or two, so these often come back empty) and
distress-but-survived control cases (e.g. Carvana's 2022–23 near-default).
Saves to `../data/validation/`.

### Step 3 — Run the GBM (Merton) baseline
```bash
python3 03_run_gbm_baseline.py
```
For each ticker, calibrates implied asset value and asset volatility monthly
via Merton's simultaneous equations, computes 1-year Distance-to-Default and
PD. Outputs `../output/phase1_gbm/{TICKER}_merton_pd.csv` and a comparison
chart.

### Step 4 — Run the jump-diffusion extension
```bash
python3 04_run_jump_diffusion.py
```
Detects jumps in each ticker's equity return series, converts them to
asset-level jump parameters, decomposes Phase 1's implied variance into
diffusive vs. jump components, and simulates the jump-diffusion PD via Monte
Carlo. Outputs `../output/phase2_jump/{TICKER}_jump_vs_gbm.csv` and a
comparison chart.

### Step 5 — Build the validation chart
```bash
python3 05_make_validation_chart.py
```
Produces an annotated PD chart for CVNA and AMC with real, publicly
documented event dates marked, so the model's PD path can be checked
against what actually happened.

## Adding more tickers

Add the ticker symbol to the `TICKERS` list at the top of `01_pull_data.py`
(or `02_pull_validation_data.py` for a distress/validation case), re-run step
1, then add the same ticker to the `TICKERS` list in both
`03_run_gbm_baseline.py` and `04_run_jump_diffusion.py` and re-run steps 3–5.

### Step 6 — Rolling (walk-forward) jump-diffusion
```bash
python3 06_run_rolling_jump_diffusion.py
```
Fixes a look-ahead bias in Step 4: re-estimates jump parameters at each date
using only trailing history, not the full 5-year sample. Outputs to
`../output/phase2b_rolling_jump/`.

### Step 7 — Regime-conditional jump-diffusion (novelty layer)
```bash
python3 08_run_regime_conditional_jump.py
```
Fits a walk-forward 2-state HMM on a market-wide return proxy and
conditions jump intensity on the detected bull/bear regime. Outputs to
`../output/phase3_regime_jump/`.

### Step 8 — Statistical evaluation (all 3 models)
```bash
python3 09_evaluate_all_models.py
```
Computes ROC AUC, Brier score, log loss, calibration curves, and ranking
correlation for GBM vs. rolling jump vs. regime-conditional jump, against a
proxy distress label. **Read the caveat printed at the top of the output
and in `output/evaluation/README_CAVEAT.txt` before citing these numbers.**

### Step 9 — GARCH-vol GBM (the measurable improvement)
```bash
python3 10_run_garch_gbm.py
python3 11_evaluate_garch_novelty.py
```
Isolated test: replaces Phase 1's trailing realized vol with a walk-forward
GARCH(1,1) forecast, holding everything else fixed, and evaluates against
the same proxy-label framework. This is the one change in the project shown
to measurably improve ROC AUC / Brier / log loss simultaneously.
