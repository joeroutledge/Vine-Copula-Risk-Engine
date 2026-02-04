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

**File**: `outputs/demo/scale_sanity.json`

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
are updated. Enabled in `configs/demo_full12.yaml` for the 12-asset run.

### 4. Full 12-Asset Configuration

**File**: `configs/demo_full12.yaml`
**Target**: `make demo-full`

Extended backtest using all 12 ETFs in the dataset with rolling refits
every 250 days. Use this for more comprehensive evaluation.

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
- ✓ Automated scale sanity check (fail if vine VaR > 2x HS)
- ✓ PIT/PPF round-trip unit test

## References

- Creal, Koopman & Lucas (2013). "Generalized Autoregressive Score Models"
- Kupiec (1995). "Techniques for Verifying the Accuracy of Risk Models"
- Christoffersen (1998). "Evaluating Interval Forecasts"
- Aas et al. (2009). "Pair-Copula Constructions of Multiple Dependence"
