# Methodology

## 1. Overview

This project estimates 1-year probability of default (PD) for a set of
publicly traded firms using a structural credit risk framework: a firm
defaults when the value of its assets falls below a debt threshold. Since
asset value is not directly observable, it is inferred from what *is*
observable — the firm's equity price and balance sheet.

Two models are built and compared:

1. **Phase 1 — Merton (1974) GBM baseline.** Assets follow geometric Brownian
   motion. This is the textbook structural model and the standard baseline
   against which any extension should be judged.
2. **Phase 2 — Merton (1976) Jump-Diffusion extension.** Assets follow GBM
   plus a compound Poisson process of lognormally-distributed jumps, to
   address a well-documented weakness of the pure-GBM model at short
   horizons (see Section 4 and `REFERENCES.md`).

## 2. Phase 1: Merton GBM Baseline

### 2.1 Core equations

Equity is modeled as a European call option on the firm's assets, struck at
the debt threshold (the "default point"), following Black-Scholes-Merton
option pricing:

```
E = A * N(d1) - D * exp(-rT) * N(d2)          (1) equity as a call option
sigma_E * E = N(d1) * sigma_A * A             (2) Ito's lemma / delta relation

where:
d1 = [ln(A/D) + (r + 0.5*sigma_A^2)*T] / (sigma_A*sqrt(T))
d2 = d1 - sigma_A*sqrt(T)
```

`E` (market equity value) and `sigma_E` (equity volatility) are observable.
`A` (asset value) and `sigma_A` (asset volatility) are not. Equations (1) and
(2) form a system of 2 equations in 2 unknowns, solved numerically via
`scipy.optimize.fsolve` (see `src/merton_gbm.py::calibrate_merton`).

### 2.2 Distance-to-Default and PD

Once `A` and `sigma_A` are calibrated:

```
DD = [ln(A/D) + (r - 0.5*sigma_A^2)*T] / (sigma_A*sqrt(T))
PD = N(-DD)
```

### 2.3 Inputs, per ticker, per month

- **E (market equity value):** daily closing price x shares outstanding
  (`Ordinary Shares Number` from the balance sheet, forward-filled from
  annual reports onto the daily price index).
- **sigma_E:** trailing 252-trading-day annualized volatility of daily log
  equity returns.
- **D (default point):** KMV convention — `Current Liabilities + 0.5 x
  Long-Term Debt`, falling back to `Total Debt` if either field is missing.
  Sourced from **annual** balance sheet data (not quarterly) — `yfinance`'s
  free tier only exposes ~6 quarters of quarterly history, which would have
  truncated the usable time window to ~14 months. Annual data covers the
  full ~5-year window at the cost of within-year granularity, which is an
  acceptable trade-off since debt levels move far more slowly than equity
  prices.
- **r:** fixed at 4.5% (approximate short-term risk-free rate).
- **T:** 1 year.

Calibration is run monthly (month-end) per ticker, with each month's solve
warm-started from the previous month's solution for faster, more stable
convergence.

## 3. Phase 2: Jump-Diffusion Extension

### 3.1 Why a jump-diffusion model, and why not the exact paper it was inspired by

The original research target for this project was Aguilar, Pesci & James
(2021), which models the asset process as a pure-jump Lévy process (NegGamma
/ NegIG) with closed-form equity pricing via special functions. That paper
was not implemented directly: its closed forms depend on special-function
machinery that could not be reproduced here with full confidence, and an
unverified reproduction of someone else's closed-form derivation is worse
than a simpler, independently verifiable model. Merton's (1976)
Jump-Diffusion model was used instead — a different, older, well-documented
paper, but addressing the exact same underlying problem (GBM's thin tails
underestimate short-horizon default risk) via the same core mechanism
(adding jumps to the asset process). See `REFERENCES.md` for how these
relate.

### 3.2 Jump detection (on observable equity returns)

Since only equity returns are observable, jumps are first detected there:

- Trailing 60-day realized volatility is computed, **excluding the most
  extreme 5% of daily moves**, so an actual jump day doesn't inflate the
  very threshold used to detect it.
- A day is flagged as a jump if `|return| > 4 x trailing realized vol`.
- From flagged days: jump intensity `lambda` (jumps per year), and the mean
  and standard deviation of jump-day log returns (`mu_J_equity`,
  `sigma_J_equity`).
- A minimum of 4 detected jump days is required per ticker to fit a jump
  distribution; tickers with fewer are left on the GBM-only baseline.

### 3.3 Converting equity-level jumps to asset-level jumps

Equity jump size is not the same as asset jump size — equity is a levered,
convex claim on assets. The same delta/elasticity relationship Merton uses
to convert equity volatility to asset volatility (equation 2 above) is
reused here:

```
Omega = (A/E) * N(d1)          (elasticity of equity value w.r.t. asset value)
asset jump size ≈ equity jump size / Omega
```

`Omega` is computed at each Phase 1 calibration point and averaged per
ticker.

### 3.4 Variance decomposition

Phase 1's calibrated `sigma_A` is a **total** implied volatility — under a
pure-GBM assumption, it silently absorbs whatever jump variance is actually
present, inflating the "diffusive" estimate. This is corrected before
simulating:

```
sigma_diffusive^2 = max(sigma_A_total^2 - lambda*(mu_J_asset^2 + sigma_J_asset^2), floor)
```

(a small floor is applied to avoid degenerate near-zero volatility).

### 3.5 Simulating the jump-diffusion PD

For each month-end date, the terminal (T = 1 year) log-asset value is
simulated via Monte Carlo (200,000 paths):

```
log(A_T) = log(A_0) + (r - 0.5*sigma_diffusive^2 - lambda*k)*T
           + sigma_diffusive*sqrt(T)*Z
           + sum_{i=1}^{N} J_i

where:
  N ~ Poisson(lambda * T)
  J_i ~ Normal(mu_J_asset, sigma_J_asset^2)
  k = exp(mu_J_asset + 0.5*sigma_J_asset^2) - 1   (drift compensator)
  Z ~ Normal(0, 1)

PD_jump = P(A_T < D)  (estimated as the simulated exceedance frequency)
```

## 4. Known limitations

- **No confirmed default in the dataset.** `yfinance`'s free tier purges
  price history for delisted small/mid-caps, so genuine Chapter 11 cases
  (Rite Aid, WeWork, Party City, Yellow Corp) could not be pulled. Validation
  instead relies on Carvana (CVNA) — a company the market widely treated as
  a near-default candidate in 2022–23 but which survived via debt
  restructuring. See `RESULTS.md` for why this is still a meaningful test.
- **Fixed jump parameters per ticker.** Jump intensity/size are estimated
  once per ticker over the full sample, not conditioned on market regime.
  A regime-conditional extension (e.g. using an HMM regime classifier to let
  jump intensity vary by bull/bear/sideways state) is a natural next step,
  not yet implemented.
- **Equity-derived jump proxy.** Asset-level jump parameters are inferred
  from equity jumps via a linear elasticity scaling, which is an
  approximation, not an exact inversion.
- **Fixed risk-free rate and horizon.** `r = 4.5%`, `T = 1 year` are held
  constant across the whole sample rather than using the period-appropriate
  term structure.

## 5. Phase 3: Regime-Conditional Jump-Diffusion (novelty layer)

Phase 2b's rolling jump intensity is still regime-blind — it pools all jumps
in the trailing window regardless of the market environment they occurred
in. Phase 3 conditions jump intensity on market regime instead:

1. Build a market-wide return proxy (equal-weighted average of all 10
   tickers' daily returns).
2. At each date, fit a 2-state Gaussian HMM **walk-forward** (trailing
   window only, no look-ahead) on this proxy. States are labeled bull/bear
   using both variance (higher-vol state) and mean return direction
   (negative-mean state), so regimes read as genuine bull/bear rather than
   just calm/stressed.
3. Within the trailing window, split detected jump days by the regime
   active on the day they occurred. Jump intensity for "today" is computed
   using only the jumps that occurred in the SAME regime as today, not the
   full window — e.g. a ticker's 2024 jump risk is estimated from jumps that
   happened during 2024-like (bull or bear) conditions, not blended with
   unrelated-regime history.
4. Falls back to the pooled Phase 2b estimate when there isn't enough
   regime-specific history (fewer than 2 in-regime jump days).

This is the one piece of this project that isn't just an application of an
existing paper — conditioning a structural jump-diffusion credit model on a
separately-estimated market regime doesn't appear in the reviewed
literature (see REFERENCES.md).

**Result, stated honestly:** regime-conditioning shifts individual PD
readings (see RESULTS.md) but does **not** improve the proxy-label
statistical metrics over Phase 2b — ROC AUC/Brier/log loss are within noise
of each other across all three models. The value of this layer is
mechanistic (jump risk demonstrably varies by regime, and that variation is
now visible and interpretable) rather than a proven accuracy gain.

## 6. Phase 4: GARCH-Forecast Volatility (novelty layer that measurably improves results)

Phase 1-3 all use trailing 252-day REALIZED equity volatility as the input
to Merton's calibration -- a flat, equal-weighted average of the past year,
slow to react to a fresh shock and slow to forget an old one. Phase 4
replaces this single input with a GARCH(1,1) FORECAST of forward volatility
(Engle 1982; Bollerslev 1986), fit walk-forward at each date using only
trailing history (no look-ahead): the trailing 504-day return window is used
to fit GARCH(1,1), then the average forecasted daily variance over the next
252 days is annualized into `sigma_E`. Every other input (default point,
elasticity, calibration equations) is UNCHANGED from Phase 1 -- this isolates
volatility estimation as the only variable that differs, so any metric
change is attributable to it specifically.

**Result: this is the one change in the whole project that measurably
improves the proxy-label statistics, on all three metrics simultaneously**
(`src/11_evaluate_garch_novelty.py`):

| | ROC AUC | Brier Score | Log Loss |
|---|---|---|---|
| Naive trailing-vol GBM | 0.911 | 0.0547 | 0.157 |
| GARCH-forecast-vol GBM | **0.939** | **0.0440** | **0.126** |

The mechanism: GARCH's variance forecast responds to volatility clustering
immediately after a large move, rather than waiting for that move to enter
a 252-day average and then persisting there for a full year after
conditions normalize. This directly helps a proxy label built on *forward*
distress, since the model's risk signal now rises earlier and decays faster
in line with actual conditions.

This should still be described precisely: it is an application of a
well-established forecasting technique (not a new estimator), but it is the
one part of this project backed by a clean, isolated, measurable
improvement over the naive baseline -- unlike the jump-diffusion and
regime-conditioning layers, which change PD readings without a proven
statistical edge on this dataset.
