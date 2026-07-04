# Results

## Phase 1 vs Phase 2: where jump-diffusion adds value

| Ticker | Max GBM PD | Max Jump PD (full-sample) | Max Jump PD (rolling, walk-forward) |
|---|---|---|---|
| AAPL/MSFT/JNJ | ~0% | ~0% | ~0% |
| CCL | 10.9% | 10.1% | 11.1% |
| F | 1.0% | 15.5% | 18.9% |
| AAL | 3.0% | 24.0% | 33.1% |
| WBD | 2.1% | 9.1% | 10.1% |
| CVNA | 71.5% | 73.2% | 68.5% |
| AMC | 49.5% | ~63% | ~65% |

**Reading this table:** stable names stay near-zero under all three models
(correct). CCL and CVNA were already so stressed that GBM alone captures most
of the risk — jumps don't add much on top of an already-extreme signal.
**F, AAL, and WBD are the real story**: GBM stays low (1-3%) while both jump
models are materially higher (4-15x GBM) — this is GBM's known short-horizon
blind spot for moderately-levered names, and it's why the jump extension
exists.

The **rolling (walk-forward) estimates are consistently close to, and
sometimes higher than, the full-sample estimates** — meaning the full-sample
version's look-ahead bias was not inflating the headline result. That's a
meaningful robustness check.

## Validation against a real, documented event (CVNA)

CVNA's 1-year PD stayed in the 53-73% range through most of 2023 — the exact
window when Carvana's bonds traded at distressed levels and analysts were
openly discussing bankruptcy risk — then declined steadily through 2024 as
the company completed a debt restructuring and recovered operationally. The
model tracked this real narrative without ever being told about it.

## Statistical comparison (proxy-label evaluation)

**Read `output/evaluation/README_CAVEAT.txt` before citing these numbers.**
No confirmed default exists in this dataset (see Methodology, Section 4), so
these metrics use a proxy label: *did the price fall to ≤40% of its
observation-date value within the next 12 months?* Only AMC contributes
meaningful positive labels in this sample (17 of 30 dates) — CVNA's biggest
crash happened before most observation dates begin, so its already-elevated
PD readings aren't rewarded by a label that only counts *further* declines.
With that limitation stated plainly:

| Model | ROC AUC | Brier Score | Log Loss |
|---|---|---|---|
| GBM | 0.911 | 0.0547 | 0.157 |
| Rolling Jump-Diffusion | 0.919 | 0.0541 | 0.160 |

Rolling jump-diffusion is marginally better on AUC and Brier, marginally
worse on log loss — essentially a **statistical tie** given a 1-ticker-driven
label set. The honest conclusion is **not** "jump-diffusion wins" — it's
that jump-diffusion changes *which* moderately-levered names get flagged
(F, AAL, WBD) without hurting calibration on the cases we can check. See
`output/evaluation/calibration_and_ranking.png` for reliability curves and
the ticker-level ranking (avg. predicted PD vs. realized distress rate,
Spearman ρ≈0.52-0.53 for both models — again roughly tied).

## Rolling jump intensity is genuinely time-varying

`output/phase2b_rolling_jump/rolling_jump_intensity.png` shows jump
intensity (λ) estimated walk-forward per ticker. AAL and WBD show rising λ
over 2024-2025 as more large moves entered their trailing windows — this is
the mechanism, made visible, rather than a single fixed number.

## Honest bottom line
- Jump-diffusion **does** recover risk in moderately-levered names that GBM
  misses — that part is a clean, explainable result.
- It does **not** demonstrably outperform GBM on the proxy statistical
  metrics — the label set is too thin (1 ticker) to make that claim, and
  that limitation is stated rather than hidden.
- CVNA is real, useful validation for "does this track a documented distress
  event" — it is not validation for "is this statistically superior."

## Phase 3: Regime-conditional jump-diffusion

| Ticker | GBM max | Rolling Jump max | Regime-Jump max |
|---|---|---|---|
| CCL | 10.93% | 11.07% | 12.29% |
| F | 1.01% | 18.88% | 19.84% |
| AAL | 3.04% | 33.08% | 28.64% |
| WBD | 2.12% | 10.09% | 8.52% |
| CVNA | 71.54% | 68.53% | 69.12% |
| AMC | 49.48% | 64.56% | 67.72% |
| SBH | 0.69% | 1.10% | 0.93% |

Regime-conditioning moves individual readings in both directions (up for
F/CCL/AMC, down for AAL/WBD) depending on whether that ticker's recent
jumps happened to occur in the same regime as "today" — this is the
intended behavior, not noise.

**Statistical comparison, all three models (same proxy-label caveat as
before applies — see `output/evaluation/README_CAVEAT.txt`):**

| Model | ROC AUC | Brier Score | Log Loss |
|---|---|---|---|
| GBM | 0.911 | 0.0547 | 0.157 |
| Rolling Jump-Diffusion | 0.919 | 0.0541 | 0.160 |
| Regime-Conditional Jump | 0.918 | 0.0545 | 0.166 |

**Honest conclusion:** regime-conditioning does not statistically outperform
Phase 2b on this thin, single-ticker-driven label set — treat it as a
mechanistic/interpretability improvement (jump risk is shown to genuinely
vary by regime, walk-forward, with no look-ahead) rather than a proven
accuracy gain. Do not claim it as "better" without this caveat attached.

## Phase 4: GARCH-forecast volatility — the one genuinely measurable improvement

Unlike jump-diffusion and regime-conditioning (which shift PD readings but
don't beat GBM statistically on this dataset), replacing trailing realized
vol with a walk-forward GARCH(1,1) forecast produces a clean improvement on
**all three metrics** simultaneously, with everything else held fixed:

| | ROC AUC | Brier Score | Log Loss |
|---|---|---|---|
| Naive trailing-vol GBM | 0.911 | 0.0547 | 0.157 |
| GARCH-forecast-vol GBM | **0.939** | **0.0440** | **0.126** |

Individual max-PD readings also rise for the real distress cases (CVNA
71.5% → 86.4%, AMC 49.5% → 81.3%) — consistent with GARCH reacting faster to
the volatility clustering that preceded both companies' worst periods.

**This is the project's strongest, most defensible result.** It is a single,
isolated, well-motivated change (one well-established forecasting technique
replacing a naive one) with a clean before/after comparison — the kind of
result that holds up under direct questioning, unlike a vaguer "the fancier
model looks better" claim.
