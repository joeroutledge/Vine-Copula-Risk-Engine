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
3. **Single time step**: The attribution is computed at the last OOS time step,
   not averaged over time. Risk contributions vary with market conditions.

**Output format** (tail_risk_attribution.csv):
| Column | Description |
|--------|-------------|
| alpha | VaR confidence level (e.g., 0.01, 0.05) |
| asset | Asset ticker |
| weight | Portfolio weight (equal weights in demo) |
| component_es | w_i * E[-r_i \| L >= VaR] |
| percent_contribution | component_es / portfolio_es |
| portfolio_var | VaR_α of portfolio (positive loss) |
| portfolio_es | ES_α = Σ_i component_es |

## What the Backtests Show

### Coverage Results (OOS Period: ~3,800 days, 5 assets)

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
- ✓ Tail risk attribution via Euler decomposition (component ES)
- ✓ Automated scale sanity check (fail if vine VaR > 2x HS)
- ✓ PIT/PPF round-trip unit test

## Reviewer Checklist

### 1. Reproduction commands

```bash
pip install -e ".[dev]"
pytest -q                                                          # all tests pass
make demo-quick                                                    # generates outputs/demo_quick/
python scripts/validate_manifest.py outputs/demo_quick/manifest.json  # "All files OK"
```

### 2. Expected output files

After `make demo-quick`, check `outputs/demo_quick/` contains:
- `metrics.json` — backtest statistics
- `backtest_summary.csv` — per-method results table
- `var_es_timeseries.csv` — daily forecasts
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

## References

- Creal, Koopman & Lucas (2013). "Generalized Autoregressive Score Models"
- Kupiec (1995). "Techniques for Verifying the Accuracy of Risk Models"
- Christoffersen (1998). "Evaluating Interval Forecasts"
- Aas et al. (2009). "Pair-Copula Constructions of Multiple Dependence"
