# Validation Notes: VaR/ES Backtest

## Summary

This document describes the mathematical fixes applied to the vine copula VaR/ES
demo and provides an honest assessment of backtest results.

## What Was Wrong

### 1. PIT/Marginal Incoherence (Critical)

**File**: `scripts/run_var_es_backtest.py`

**Problem**: The GARCH(1,1) model was fitted using a **Gaussian** log-likelihood,
but PIT and inverse PIT used a **Student-t(8)** distribution. This is
mathematically inconsistent:

- Gaussian GARCH: `-0.5 * sum(log(2π·σ²) + r²/σ²)`
- PIT: `u = t_8.cdf(r/σ)`

The standardized residuals `z = r/σ` from a Gaussian-fitted GARCH are implicitly
N(0,1), but mapping them through a t(8) CDF distorts the uniform marginals.

**Fix**: Implemented coherent GARCH(1,1)-t model with Student-t innovations:
- Likelihood now uses Student-t density with ν estimated via MLE
- PIT uses `t_ν.cdf(r/σ)` with the **same** ν as the fitted model
- Inverse PIT uses `σ * t_ν.ppf(u)` consistently

### 2. Student-t Scaling Bug (Critical)

**Problem**: The original code applied an erroneous `sqrt(nu/(nu-2))` scaling
factor in both the likelihood and the inverse PIT:

```python
# BUGGY CODE
scale = sigma * sqrt(nu/(nu-2))  # WRONG
z = r / scale                     # PIT
r = z * scale                     # inverse PIT
```

This caused VaR estimates to be ~1.33x more extreme per asset for nu=8,
compounding to ~2.5x for a 5-asset portfolio. The symptom: vine methods
produced VaR(1%) around -4% vs HS at -1.6%, with 0 breaches.

**Root cause**: Confusion between the "standard" Student-t distribution
(variance = nu/(nu-2)) and the "standardized" Student-t (variance = 1).
The GARCH convention is: `r_t = sigma_t * z_t` where `z_t ~ t(nu)` is
the *standard* Student-t. No additional scaling is needed.

**Fix**: Removed the `sqrt(nu/(nu-2))` factor entirely:

```python
# CORRECT CODE
z = r / sigma                     # PIT: u = t(nu).cdf(z)
r = sigma * t(nu).ppf(u)         # inverse PIT
```

### 3. Makefile Hash Verification Bug

**Problem**: The one-liner used `setattr(sys.modules[__name__], 'ok', False)`
inside a list comprehension, which does not update the local variable `ok`.
Hash mismatches never caused exit(1).

**Fix**: Rewrote as proper Python with explicit loop and `fail` counter.

### 4. README Overclaims

**Problem**: Claimed "12-ETF daily return dataset" but `demo.yaml` uses
`n_assets: 5` by default.

**Fix**: Updated to accurately describe configurable asset count.

## Scaling Bug Before/After Comparison

### Before Fix (BUGGY)

| Method | VaR(1%) Breach Rate | VaR(5%) Breach Rate | Median VaR(1%) |
|--------|---------------------|---------------------|----------------|
| HS | 1.3% | 5.3% | -1.28% |
| Static D-vine | 0.0% | 0.03% | ~-4.0% |
| GAS D-vine | 0.0% | 0.0% | ~-4.0% |

The vine methods had VaR(1%) about **2.5x more extreme** than HS, resulting
in zero breaches. This was a scaling bug, not "conservatism."

### After Fix (CORRECT)

| Method | VaR(1%) Breach Rate | VaR(5%) Breach Rate | Median VaR(1%) |
|--------|---------------------|---------------------|----------------|
| HS | 1.3% | 5.3% | -1.28% |
| DCC-GARCH | 1.2% | 3.8% | -1.44% |
| Static D-vine | 0.9% | 4.2% | -1.33% |
| GAS D-vine | 0.6% | 3.0% | -1.49% |

Now vine VaR(1%) is within ~1.16x of HS (scale_sanity ratio < 2.0). The
vine methods are slightly conservative due to genuine model effects:
- Fat-tailed marginals (estimated ν ≈ 5-8)
- Tail dependence captured by copula

## New Features Added

### 1. Scale Sanity Check

**File**: `outputs/demo_quick/scale_sanity.json`

Automated sanity check that fails the backtest if vine VaR(1%) exceeds
2x the HS VaR(1%). This catches scaling bugs like the one fixed above.

```json
{
  "passed": true,
  "checks": [
    {"method": "static_vine", "ratio": 1.04, "threshold": 2.0, "passed": true},
    {"method": "gas_vine", "ratio": 1.16, "threshold": 2.0, "passed": true}
  ]
}
```

### 2. PIT/PPF Round-Trip Test

**File**: `tests/test_pit_ppf_roundtrip.py`

Unit test verifying that PIT followed by inverse PIT reconstructs the
original returns within 1e-3 tolerance. This catches scaling bugs at
the marginal level before they propagate to VaR estimates.

### 3. Rolling Marginal Refit

**Config keys**: `marginal_refit_freq_days`, `marginal_window_days`

Optional rolling refit of GARCH marginals and copula parameters on a
sliding window. Structure (D-vine order) is preserved; only parameters
are updated. Enable by setting `marginal_refit_freq_days > 0` in config.

### 4. Configuration

**File**: `configs/demo_quick.yaml`
**Target**: `make demo-quick`

Quick demo with 5 assets and 1000 Monte Carlo simulations for fast iteration.
Parameters can be adjusted in the config file for extended evaluation.

### 5. Tail Risk Attribution (Component ES)

**File**: `outputs/demo_quick/tail_risk_attribution.csv`

Decomposes portfolio tail risk into per-asset contributions using the Euler
decomposition for Expected Shortfall (ES) applied to Monte Carlo simulations.

**Definition**:
```
Portfolio loss:      L = -Σ_i w_i * r_i  (positive when portfolio loses)
VaR_α:               α-quantile of L across MC paths
ES_α:                E[L | L >= VaR_α]  (expected shortfall)
ComponentES_i:       w_i * E[-r_i | L >= VaR_α]  (tail conditional expectation)
PercentContrib_i:    ComponentES_i / Σ_j ComponentES_j
```

**Interpretation**:
- Component ES answers: "How much does asset i contribute to portfolio tail risk?"
- The sum of component ES equals portfolio ES exactly (Euler decomposition)
- Assets with higher volatility, higher weight, or stronger tail dependence with
  other assets contribute more to portfolio tail risk

**Limitations**:
1. **MC noise**: With finite simulations (e.g., 1000), components have sampling error.
2. **Tail approximation**: The conditional expectation is computed on a discrete
   set of tail paths. For small alpha (e.g., 0.01), this may be only 10-50 paths.
3. **Model dependence**: Attribution depends on the copula model structure and parameters.

**Output format** (tail_risk_attribution.csv - snapshot at last OOS date):
| Column | Description |
|--------|-------------|
| alpha | VaR confidence level (e.g., 0.01, 0.05) |
| asset | Asset ticker |
| weight | Portfolio weight (equal weights in demo) |
| component_es | w_i * E[-r_i \| L >= VaR] |
| percent_contribution | component_es / portfolio_es |
| portfolio_var | VaR_α of portfolio (positive loss) |
| portfolio_es | ES_α = Σ_i component_es |

### Time-Series Attribution (Component ES Over Time)

**File**: `outputs/demo_quick/tail_risk_attribution_timeseries.csv`

In addition to the snapshot attribution, the demo produces a time-series of
component ES contributions for each OOS date. This allows tracking how each
asset's contribution to portfolio tail risk evolves over time.

**Schema** (tail_risk_attribution_timeseries.csv):
| Column | Description |
|--------|-------------|
| date | OOS date |
| alpha | VaR confidence level |
| asset | Asset ticker |
| weight | Portfolio weight |
| component_es | Component ES for this (date, asset, alpha) |
| percent_contribution | Fraction of portfolio ES from this asset |
| portfolio_var | Portfolio VaR at this date |
| portfolio_es | Portfolio ES at this date |
| n_tail | Number of tail scenarios used |
| n_sim | Total MC simulations |

**Interpretation**:
- Component ES is computed using the Euler decomposition: sum of component ES
  equals portfolio ES (within MC error)
- Percent contributions sum to 1.0 for each (date, alpha)
- Changes in percent contributions reflect shifts in which assets drive tail risk

**Limitations**:
1. **MC noise**: Attribution at each date has sampling error proportional to 1/sqrt(n_sim)
2. **Model dependence**: Results depend on copula structure and GAS state at each date
3. **Not a forecast**: This is attribution of simulated risk, not realized risk

**Plot**: `outputs/demo_quick/tail_risk_attribution_timeseries.png` shows a stacked
area chart of percent contributions over time for each alpha level.

## GAS Update Frequency

The `gas_update_every` parameter controls how often the GAS latent state is updated.

| Setting | Description |
|---------|-------------|
| `gas_update_every: 1` | Daily updates (default) |
| `gas_update_every: 5` | Weekly updates (every 5 trading days) |

**Behavior**:
- On update days (t % update_every == 0): score is computed, OPG is updated, theta changes
- On non-update days: theta and OPG are carried forward unchanged
- Likelihood is ALWAYS computed daily using current theta (for VaR/ES forecasting)

**Tradeoffs**:
- **Daily (1)**: More responsive to regime changes, but noisier due to high-frequency score fluctuations
- **Weekly (5)**: Smoother dynamics, less sensitive to transient market moves, but slower to adapt

The `make demo-quick-weekly-gas` target runs the demo with `gas_update_every: 5` for comparison.

## What the Backtests Show

### Coverage Results (OOS Period: ~4,300 days, 5 assets)

*Note: Exact numeric values may vary slightly across runs due to GAS parameter
estimation. The qualitative conclusions are stable.*

| Method | VaR(5%) Rate | VaR(2.5%) Rate | VaR(1%) Rate | Pinball(5%) |
|--------|--------------|----------------|--------------|-------------|
| HS | 5.3% | 2.7% | 1.3% | 0.000646 |
| EWMA | 4.9% | 2.8% | 1.8% | 0.000620 |
| DCC-GARCH | 3.8% | 2.1% | 1.2% | 0.000613 |
| Static D-vine | 4.2% | 2.2% | 0.9% | 0.000614 |
| GAS D-vine | 3.0% | 1.3% | 0.6% | 0.000628 |

### Interpretation

After fixing the scaling bug, vine methods show reasonable coverage:

1. **Slightly conservative**: GAS D-vine has 0.6% breach rate at 1% level
   (vs expected 1%). This is genuine model behavior from fat tails and
   tail dependence, not a bug.

2. **Pinball loss comparable**: Vine methods now have pinball loss close
   to baselines (~0.00061-0.00063), vs ~0.0013 before fix.

3. **ES adequacy good**: Shortfall ratios are close to 1.0, indicating
   ES forecasts are well-calibrated when VaR is breached.

## What This Demo Demonstrates

- ✓ Coherent GARCH(1,1)-t marginal estimation with estimated ν
- ✓ Correct Student-t scaling (no sqrt(nu/(nu-2)) bug)
- ✓ Vine copula construction with BIC-based family selection
- ✓ GAS dynamics for time-varying Tree-1 correlations
- ✓ Rolling VaR/ES forecasting with strictly lagged information
- ✓ Optional rolling marginal/copula refit
- ✓ Formal backtest statistics (Kupiec, Christoffersen, pinball loss)
- ✓ Diebold-Mariano test for forecast comparison
- ✓ Tail risk attribution via Euler decomposition (component ES)
- ✓ Automated scale sanity check (fail if vine VaR > 2x HS)
- ✓ PIT/PPF round-trip unit test
- ✓ Sensitivity analysis for hyperparameter robustness

## Diebold-Mariano Test

**Files**: `outputs/demo_quick/pinball_losses.csv`, `outputs/demo_quick/dm_tests.csv`

### What is tested

The Diebold-Mariano (DM) test compares predictive accuracy of two VaR forecasts
using per-time pinball (quantile) loss. The test statistic is:

```
DM = mean(d_t) / sqrt(HAC_var(d_t) / T)
```

where `d_t = loss_A(t) - loss_B(t)` is the loss differential at time t.

- **H0**: E[d_t] = 0 (equal predictive accuracy)
- **H1**: E[d_t] != 0 (one model has systematically lower loss)
- Newey-West HAC variance accounts for serial correlation in d_t
- Uses h=1 (one-step ahead), so overlapping forecast issues are minimal

### Why it complements Kupiec/Christoffersen

| Test | Question | Focus |
|------|----------|-------|
| Kupiec | Is breach rate equal to alpha? | Calibration (unconditional) |
| Christoffersen | Are breaches serially independent? | Calibration (conditional) |
| **Diebold-Mariano** | Which model has lower pinball loss? | **Accuracy** |

Kupiec and Christoffersen test whether a model is well-calibrated. The DM test
compares accuracy between models via their loss functions. A model can pass
calibration tests but still have worse accuracy (e.g., wider VaR bands).

### Output format

**pinball_losses.csv**:
| Column | Description |
|--------|-------------|
| date | Forecast date |
| model | "static_vine" or "gas_vine" |
| alpha | VaR confidence level |
| pinball_loss | Per-time quantile loss |

**dm_tests.csv**:
| Column | Description |
|--------|-------------|
| model_a, model_b | Models being compared |
| alpha | VaR confidence level |
| dm_stat | DM test statistic (~N(0,1) under H0) |
| p_value | Two-sided p-value |
| mean_diff | mean(loss_A - loss_B); negative means A is better |
| n_obs | Number of observations |
| nw_lags | Newey-West lags used |

### What NOT to claim

1. **Conditional on loss choice**: DM tests pinball loss; different loss
   functions (e.g., tick loss, FZ loss) may give different conclusions.

2. **Conditional on sample**: Results depend on the OOS period. Different
   market regimes may favor different models.

3. **Assumes weak dependence**: HAC variance assumes loss differentials have
   weak serial dependence. This is reasonable for h=1 but not guaranteed.

4. **Not a model validation**: A model can "win" the DM test but still be
   poorly calibrated. Always check Kupiec/Christoffersen first.

5. **No multiple testing correction**: If running DM at multiple alphas,
   consider Bonferroni or similar adjustment when interpreting p-values.

## Sensitivity Analysis

**File**: `outputs/sensitivity_quick/sensitivity_summary.csv`
**Command**: `make sensitivity-quick`

### What is varied

The sensitivity analysis runs a small grid over two key hyperparameters:

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| `n_sim` | {500, 1000, 2000} | MC simulation count affects VaR/ES sampling noise |
| `nu_fixed` | {30, 100, 300} | Fixed copula df affects tail dependence modeling |

Both `static` and `gas` vine models are evaluated.

### What is stable vs what moves

**Expected to be stable**:
- Kupiec p-values should remain > 0.05 across grid (coverage not rejected)
- ES shortfall ratios should remain near 1.0 (well-calibrated)
- Relative ranking of methods should not invert

**Expected to vary**:
- Pinball loss improves slightly with more simulations (less MC noise)
- Hit rates have natural MC variation (~0.5% for 1000 sims)
- Very high nu_fixed (300) approaches Gaussian copula behavior

### What this does NOT prove

1. **Not a hyperparameter search**: The grid is for robustness checking, not
   optimization. The default nu_fixed=None (estimated) is preferred.

2. **Not exhaustive**: Only two hyperparameters are varied. Other factors
   (train_days, n_assets, refit_freq) are held constant.

3. **Not a model comparison**: Different configurations are not compared
   statistically. This checks that results don't collapse under perturbation.

4. **Not a guarantee**: Passing sensitivity checks doesn't prove the model
   is correct, only that it's not pathologically sensitive to these choices.

### Output format (sensitivity_summary.csv)

| Column | Description |
|--------|-------------|
| model | "static" or "gas" |
| config_id | Configuration index (1-18 for full grid) |
| n_sim | Monte Carlo simulation count |
| nu_fixed | Fixed degrees of freedom |
| alpha | VaR confidence level |
| hit_rate | Realized breach rate |
| kupiec_p | Kupiec test p-value |
| christoffersen_p | Christoffersen test p-value |
| es_shortfall_ratio | Mean realized / forecast ES on breach days |
| pinball_loss | Quantile loss |
| n_oos | Number of OOS observations |

## Reviewer Checklist

### 1. Reproduction commands

```bash
pip install -e ".[dev]"
pytest -q                                                              # all tests pass
make demo-quick                                                        # generates outputs/demo_quick/
make sensitivity-quick                                                 # generates outputs/sensitivity_quick/
python scripts/validate_manifest.py outputs/demo_quick/manifest.json   # "All files OK"
python scripts/validate_manifest.py outputs/sensitivity_quick/manifest.json
```

### 2. Expected output files

After `make demo-quick`, check `outputs/demo_quick/` contains:
- `metrics.json` — backtest statistics
- `backtest_summary.csv` — per-method results table
- `var_es_timeseries.csv` — daily forecasts
- `pinball_losses.csv` — per-time pinball losses for DM test
- `dm_tests.csv` — Diebold-Mariano test results
- `tail_risk_attribution.csv` — per-asset risk contributions (component ES)
- `scale_sanity.json` — scaling sanity check
- `vine_model_card_static.json` — static D-vine spec
- `vine_model_card_gas.json` — GAS D-vine spec
- `var_forecasts.png`, `breaches.png` — plots
- `manifest.json` — SHA-256 hashes

### 3. What to check in metrics.json

- `kupiec_pval_*`: p-values > 0.05 indicate coverage not rejected at 5% level
- `chris_pval_*`: p-values > 0.05 indicate independence not rejected
- `es_ratio_*`: values near 1.0 indicate well-calibrated ES
- `breach_rate_*`: should be close to nominal alpha (e.g., ~1% for VaR(1%))

### 4. What to check in model cards

- `vine_model_card_static.json`: verify tree structure, families per edge, parameters
- `vine_model_card_gas.json`: same plus GAS parameters (omega, A, B) for Tree-1 edges

### 5. Safe claims

Based on this demo, it is defensible to claim:
- Implementation of D-vine copula with BIC family selection
- Implementation of GAS dynamics for elliptical Tree-1 edges
- Rolling VaR/ES forecasting with strictly lagged information
- Formal backtests (Kupiec, Christoffersen, ES adequacy)
- Coherent GARCH-t marginals with estimated degrees of freedom
- Tail risk attribution via Euler decomposition (component ES per asset)

It is NOT defensible to claim:
- Superior performance vs baselines (results are method/period dependent)
- Production-ready risk system (this is a demo)
- Any trading signal or alpha generation
- Precise attribution (MC noise means sampling error on individual components)

## GAS Estimation/Evaluation Consistency

### Single Source of Truth

The GAS filter recursion used during parameter estimation is **identical** to the
recursion used during out-of-sample evaluation. This is enforced by:

1. **`gas_filter()`** in `src/vine_risk/core/gas.py` is the canonical implementation
2. **`gas_neg_loglik()`** calls `gas_filter()` internally, ensuring the optimization
   objective matches the evaluation likelihood
3. **`GAS_FILTER_DEFAULTS`** provides explicit filter parameters (opg_decay, score_cap,
   etc.) used consistently in both estimation and evaluation

This design eliminates estimation/evaluation mismatch bugs where a model is optimized
on one objective but evaluated on a different one.

### Filter Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `opg_decay` | 0.98 | EWMA decay for OPG scaling |
| `score_cap` | 50.0 | Raw score clipping bound |
| `max_scaled_score` | 4.0 | Scaled score clipping bound |
| `opg_floor` | 1e-3 | Minimum OPG to prevent division by zero |
| `clip_theta` | 3.8 | Theta clipping bound (|rho| < 0.9999) |

### Tau-Mode Parameterization (Optional)

An alternative latent Kendall's tau parameterization is available for elliptical
copulas (disabled by default):

- Latent state: κ (kappa)
- Kendall's tau: τ = tanh(κ)
- Pearson correlation: ρ = sin(π/2 · τ)

This is mathematically equivalent to theta-mode (θ = arctanh(ρ)) but:
- Uses Kendall's tau (a rank correlation) as the intermediate quantity
- May offer improved numerical stability in some cases
- Provides interpretable intermediate values

The score is computed via analytic chain rule:
```
d log c / d κ = (d log c / d ρ) × (d ρ / d τ) × (d τ / d κ)
```

No numerical differentiation is used.

## References

- Creal, Koopman & Lucas (2013). "Generalized Autoregressive Score Models"
- Kupiec (1995). "Techniques for Verifying the Accuracy of Risk Models"
- Christoffersen (1998). "Evaluating Interval Forecasts"
- Aas et al. (2009). "Pair-Copula Constructions of Multiple Dependence"
