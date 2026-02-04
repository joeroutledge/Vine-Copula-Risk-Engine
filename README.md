# vine_risk_xva_demo

D-vine copula + score-driven (GAS) risk distribution engine for
portfolio VaR/ES forecasting, benchmarked against strong baselines
with formal statistical backtests.

## What this demo does

Runs a **rolling one-step-ahead VaR/ES backtest** on daily ETF
log-returns (default: 5 of 12 available assets, configurable via
`n_assets` in `configs/demo.yaml`), comparing five methods:

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
- **Model cards**: Exported to `outputs/<run>/vine_model_card_static.json` and `vine_model_card_gas.json` for independent verification of trees, edges, families, and parameters.

### Design Choices

**1. Tree-1 only dynamics (not all trees)**

GAS dynamics are applied only to Tree-1 (unconditional) pair copulas. Rationale:
- Tree-1 edges directly model unconditional dependencies between adjacent assets
- Higher-tree edges model conditional dependencies (given intervening assets)
- Computational cost scales linearly with number of dynamic edges
- Empirically, Tree-1 correlations are more important for tail risk

**2. Elliptical-only GAS (gaussian/student, not Archimedean)**

GAS dynamics require a well-defined score function based on the Fisher-z transformation (arctanh of correlation). This is only meaningful for elliptical copulas:
- **Gaussian/Student-t**: Correlation parameter ρ maps cleanly to Fisher-z θ = arctanh(ρ)
- **Clayton/Gumbel/Frank**: Dependence is parameterized by θ with different interpretation (not correlation). Forcing Student-t would misspecify tail dependence patterns.

When BIC selects a non-elliptical family for a Tree-1 edge, that edge remains static to respect the data-driven family selection.

**3. Future work: Family-specific GAS**

Extending GAS to non-elliptical families would require:
- Deriving the copula score function for each family
- Implementing family-specific parameter transformations
- Validating stationarity and information bounds per family

This is a research direction beyond the scope of this demo.

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

## Reproduce

```bash
pip install -e ".[dev]"
make demo       # Runs 5-asset VaR/ES backtest (fast, ~2 min)
make demo-full  # Runs 12-asset backtest with rolling refits (~10 min)
make test       # pytest -q
```

### Rolling Refit (optional)

When `marginal_refit_freq_days > 0` in config, the model periodically refits:
- GARCH-t marginal parameters on a rolling window
- Copula pair-copula parameters (structure preserved)

Enabled by default in `configs/demo_full12.yaml` (refit every 250 days, window=1000).

### Outputs (in `outputs/demo/`)

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

### Verify artifact integrity

```bash
python -c "
import json, hashlib, pathlib
m = json.load(open('outputs/demo/manifest.json'))
for k, v in m['files'].items():
    h = hashlib.sha256(open(f'outputs/demo/{k}', 'rb').read()).hexdigest()
    print(f'{k}: {\"OK\" if h == v else \"MISMATCH\"}')"
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
