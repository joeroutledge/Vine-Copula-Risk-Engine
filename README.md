# vine_risk_xva_demo

[![CI](https://github.com/joeroutledge/Vine-Copula-Risk-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/joeroutledge/Vine-Copula-Risk-Engine/actions/workflows/ci.yml)

D-vine copula + score-driven (GAS) risk distribution engine for
portfolio VaR/ES forecasting, benchmarked against strong baselines
with formal statistical backtests.

## Quickstart

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run the demo pipeline
make demo-quick

# Verify output integrity
python scripts/validate_manifest.py outputs/demo_quick/manifest.json

# Run tests
pytest -q
```

## What this demo does

Runs a **rolling one-step-ahead VaR/ES backtest** on daily ETF
log-returns (5 of 12 available assets), comparing five methods:

| Method | Description |
|--------|-------------|
| Historical Simulation (HS) | Empirical quantile of a 250-day rolling window |
| EWMA Gaussian | Exponentially weighted variance, Gaussian quantile |
| DCC-GARCH | Dynamic Conditional Correlation with univariate GARCH(1,1) marginals |
| Static D-vine | D-vine copula with BIC family selection, time-invariant parameters |
| GAS D-vine | D-vine with score-driven Tree-1 dynamics, static higher trees |

### Vine Copula Specification

- **Vine type**: D-vine (not R-vine). Variable ordering determined by greedy Kendall's tau.
- **Family set (Static)**: {Gaussian, Student-t, Clayton, Gumbel, Frank} with rotations, selected via BIC per edge.
- **GAS dynamics**: Applied to Tree-1 edges with **elliptical families only** (gaussian, student-t). Non-elliptical edges (Clayton, Gumbel, Frank) remain static with their BIC-selected family. Higher trees always use static families.
- **Model cards**: Exported to `outputs/demo_quick/vine_model_card_static.json` and `vine_model_card_gas.json` for independent verification of trees, edges, families, and parameters.

### Design Choices

**1. Tree-1 only dynamics (not all trees)**

GAS dynamics are applied only to Tree-1 (unconditional) pair copulas. Rationale:
- Tree-1 edges directly model unconditional dependencies between adjacent assets
- Higher-tree edges model conditional dependencies (given intervening assets)
- Computational cost scales linearly with number of dynamic edges
- Empirically, Tree-1 correlations are more important for tail risk

**2. Elliptical-only GAS (gaussian/student, not Archimedean)**

GAS dynamics require a well-defined score function based on the Fisher-z transformation (arctanh of correlation). This is only meaningful for elliptical copulas:
- **Gaussian/Student-t**: Correlation parameter rho maps cleanly to Fisher-z theta = arctanh(rho)
- **Clayton/Gumbel/Frank**: Dependence is parameterized by theta with different interpretation (not correlation). Forcing Student-t would misspecify tail dependence patterns.

When BIC selects a non-elliptical family for a Tree-1 edge, that edge remains static to respect the data-driven family selection.

## Baselines

- **HS**: Non-parametric, no distributional assumption.
- **EWMA Gaussian**: Industry-standard (RiskMetrics-style).
- **DCC-GARCH**: Engle (2002); multivariate volatility benchmark.

## Backtests reported

- **Kupiec (1995)** unconditional coverage test (LR ~ chi-sq(1))
- **Christoffersen (1998)** independence test (LR ~ chi-sq(1))
- **ES adequacy**: mean realized shortfall / mean forecast ES on breach days

All forecasts use **strictly lagged** information. No lookahead.

## What this repo explicitly does NOT do

- No trading strategy, no portfolio signal layering, no Sharpe claims.
- No regime gating or exogenous signal conditioning.
- No options pricing or vol surface modeling.
- No performance marketing of any kind.

## Outputs (in `outputs/demo_quick/`)

| File | Contents |
|------|----------|
| `metrics.json` | Per-method backtest statistics |
| `var_es_timeseries.csv` | Daily VaR/ES forecasts + realized returns |
| `backtest_summary.csv` | Breach counts, Kupiec/Christoffersen p-values, ES ratio |
| `scale_sanity.json` | Sanity check: vine VaR must not exceed 2x HS VaR |
| `vine_model_card_static.json` | Full D-vine specification: trees, edges, families, params |
| `vine_model_card_gas.json` | GAS D-vine specification with Tree-1 GAS parameters |
| `var_forecasts.png` | VaR forecast comparison (OOS period) |
| `breaches.png` | Breach indicator with rolling breach rate for GAS D-vine |
| `manifest.json` | SHA-256 hashes for all produced files |

## Release artifacts

To attach a reproducible archive to a GitHub Release:

1. Run the pipeline: `make demo-quick`
2. The pipeline creates `outputs/demo_quick/Archive01.zip` containing all outputs
3. Create a GitHub Release and attach `Archive01.zip`
4. Reviewers can verify integrity:
   ```bash
   unzip Archive01.zip -d outputs/demo_quick_verify/
   python scripts/validate_manifest.py outputs/demo_quick_verify/manifest.json
   ```

## Data

`data/public_returns.csv` — daily log-returns for 12 liquid US ETFs
(AGG, EEM, EFA, GLD, IEF, IWM, LQD, QQQ, SPY, TLT, VNQ, XLF),
5327 trading days starting 2004-11-19. Source: Yahoo Finance (public).

## Dependencies

- Python >= 3.10
- numpy, pandas, scipy, pyvinecopulib, pyyaml, matplotlib

## References

- Creal, Koopman & Lucas (2013). *Generalized Autoregressive Score Models
  with Applications.* JBES, 28(5), 777--795.
- Engle (2002). *Dynamic Conditional Correlation.* JBES, 20(3), 339--350.
- Kupiec (1995). *Techniques for Verifying the Accuracy of Risk Measurement
  Models.* Journal of Derivatives, 3(2), 73--84.
- Christoffersen (1998). *Evaluating Interval Forecasts.* International
  Economic Review, 39(4), 841--862.
- Demarta & McNeil (2005). *The t Copula and Related Copulas.* International
  Statistical Review, 73(1), 111--129.
