# Model Validation Report

**Vine Copula VaR/ES Risk Engine**

Version: 1.0
Date: 2026-02

---

## A. Purpose and Intended Use

This engine produces **one-step-ahead VaR and ES forecasts** for a portfolio of assets using vine copula models with optional score-driven (GAS) dynamics.

**Intended use cases:**
- Market risk VaR/ES forecasting for regulatory or internal reporting
- Model comparison and backtesting research
- Tail risk attribution (component ES) for portfolio monitoring

**Not intended for:**
- XVA calculations (CVA, DVA, FVA, MVA)
- Trading signal generation
- High-frequency or intraday forecasting

---

## B. Model Specification

### B.1 Marginal Models

Each asset's return series is modeled with univariate GARCH(1,1) with Student-t innovations:

```
r_t = σ_t · ε_t,    ε_t ~ t(ν)
σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
```

Parameters (ω, α, β, ν) are estimated via MLE on the training window. Degrees of freedom ν are constrained to [2.5, 50].

### B.2 Probability Integral Transform (PIT)

Standardized residuals z_t = r_t / σ_t are transformed to uniform marginals:

```
u_t = F_ν(z_t)
```

where F_ν is the Student-t CDF with estimated ν.

### B.3 D-Vine Copula

A D-vine copula is fitted to the PIT uniforms U = (u¹, u², ..., uᵈ). Key specifications:

| Component | Specification |
|-----------|---------------|
| Vine type | D-vine (not R-vine) |
| Ordering | Greedy Kendall's tau on **training data only** |
| Family set | {Gaussian, Student-t, Clayton, Gumbel, Frank} + rotations |
| Selection | BIC per edge |

### B.4 GAS Dynamics (Optional)

For Tree-1 edges with **elliptical families only** (Gaussian or Student-t), time-varying correlation is modeled via GAS:

```
θ_{t+1} = ω + A·s̃_t + B·θ_t
ρ_t = tanh(θ_t)
```

where s̃_t is the scaled score (Fisher information-weighted gradient of the log-likelihood).

**Scope restrictions:**
- GAS dynamics apply to Tree-1 only (unconditional pairs)
- Non-elliptical Tree-1 edges (Clayton, Gumbel, Frank) remain static
- Higher trees are always static

**Rationale:** The Fisher-z score is only well-defined for elliptical copulas where θ = arctanh(ρ). Forcing GAS on Archimedean families would misspecify tail dependence.

### B.5 Recalibration Schedule

| Component | Default behavior |
|-----------|------------------|
| Vine ordering | Fixed at train time (no rolling refit) |
| Pair-copula families | Fixed at train time |
| Pair-copula parameters | Optionally refitted on rolling window |
| GARCH parameters | Optionally refitted on rolling window |
| GAS dynamics (ω, A, B) | Fixed at train time |

Recalibration frequency is configurable via `refit_freq` and `refit_window` in the config.

---

## C. Leakage Controls

### C.1 Structure Selection

**Control:** Vine ordering is computed from PIT uniforms of the **training period only**.

**Implementation:** `StaticDVineModel.compute_order_from_data(U_train)` is called before model instantiation, and the resulting `fixed_order` is passed to both static and GAS models.

**Verification:** Model cards include the field:
```json
"order_source": "train_only_fixed"
```

If this field shows `"inferred_from_full_u"`, OOS leakage has occurred.

### C.2 Parameter Estimation

All copula and GAS parameters are estimated on `U_train` (indices 0 to train_end-1). Out-of-sample evaluation starts at index train_end.

### C.3 Forecast Lag

All forecasts use **strictly lagged information**:
- VaR/ES for day t uses GARCH volatility σ_t computed from returns through day t-1
- Copula state at time t uses information through day t-1

---

## D. Reproducibility and Governance

### D.1 Determinism Contract

The engine achieves **full determinism** via:

1. **NumPy seeding**: `np.random.seed(seed)` at start of each run
2. **pyvinecopulib seeding**: `simulate(n, seeds=[...])` with deterministic seeds

**Verification:** Model cards and `metrics.json` include:
```json
"determinism": {
  "determinism_mode": "strict",
  "numpy_seeded": true,
  "pyvinecopulib_seeded": true
}
```

### D.2 Manifest Integrity

All output files are hashed (SHA-256) and recorded in `manifest.json`. Validation:

```bash
python scripts/validate_manifest.py outputs/demo_quick/manifest.json
```

### D.3 CI Verification

GitHub Actions CI runs:
1. Full test suite (201 tests)
2. Demo pipeline execution
3. Manifest validation
4. Determinism check (two identical runs produce identical numeric outputs)
5. Order source validation (confirms no OOS leakage)

---

## E. Backtesting Suite

### E.1 Unconditional Coverage (Kupiec 1995)

Tests whether the observed breach rate matches the nominal α level.

```
H₀: P(breach) = α
LR = 2·[log L(p̂) - log L(α)] ~ χ²(1)
```

**Interpretation:** p-value < 0.05 suggests miscalibrated VaR level.

**Caveat:** Does not detect clustering of breaches.

### E.2 Independence (Christoffersen 1998)

Tests whether breaches are serially independent (no clustering).

```
H₀: P(breach_t | breach_{t-1}) = P(breach_t | no_breach_{t-1})
LR ~ χ²(1)
```

**Interpretation:** p-value < 0.05 suggests breach clustering, indicating volatility timing issues.

### E.3 ES Adequacy

Ratio of mean realized shortfall to mean forecast ES on breach days:

```
ES_ratio = mean(-r_t | breach_t) / mean(ES_t | breach_t)
```

**Interpretation:**
- Ratio ≈ 1.0: ES forecasts are well-calibrated
- Ratio > 1.0: ES underestimates tail severity
- Ratio < 1.0: ES overestimates tail severity

### E.4 Pinball Loss

Quantile-specific loss function for forecast comparison:

```
L_α(r, q) = (α - 1{r < q})·(r - q)
```

Lower average pinball loss indicates better quantile forecasts.

### E.5 Diebold-Mariano Test

Tests whether two models have equal predictive accuracy:

```
H₀: E[L_A - L_B] = 0
DM = mean(d_t) / se(d_t) ~ N(0,1)
```

where d_t = L_A,t - L_B,t.

**Caveat:** Assumes stationarity of loss differentials. Long-run variance estimation uses Newey-West with automatic lag selection.

---

## F. Sign Conventions

### F.1 Dual Export

Risk metrics are exported in **both** loss-space and return-space:

| Metric | Loss-space | Return-space |
|--------|------------|--------------|
| VaR | `portfolio_var` | `portfolio_var_return` |
| ES | `portfolio_es` | `portfolio_es_return` |
| Component ES | `component_es` | `component_es_return` |

**Conversion:** `return-space = -loss-space`

### F.2 Interpretation

- **Loss-space** (positive = loss): Natural for risk limit monitoring
- **Return-space** (negative = loss): Natural for P&L analysis

### F.3 Euler Decomposition

The sum property holds in both spaces:
```
sum(component_es) = portfolio_es         [loss-space]
sum(component_es_return) = portfolio_es_return  [return-space]
```

---

## G. Limitations and Monitoring

### G.1 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Fixed family selection | May miss regime changes in tail dependence | Monitor fit statistics; consider periodic reselection |
| Elliptical-only GAS | Cannot capture time-varying asymmetric tail dependence | Keep non-elliptical edges static; documented in model card |
| One-step-ahead only | Not suitable for multi-day VaR without scaling assumptions | Use multi-day simulation if needed |
| Equal weights assumed | Demo uses 1/d weights | Production: pass actual portfolio weights |
| No transaction costs | Clean backtest; no slippage | Add if used for trading research |

### G.2 Monitoring Recommendations

| Metric | Threshold | Action |
|--------|-----------|--------|
| Kupiec p-value | < 0.01 | Investigate model miscalibration |
| Christoffersen p-value | < 0.01 | Investigate volatility timing |
| ES ratio | > 1.5 or < 0.7 | Review tail distribution fit |
| DM p-value (vs baseline) | > 0.20 | Consider simpler model |
| Scale sanity | VaR > 2× HS | Flag potential estimation error |

### G.3 Model Risk

- **Estimation uncertainty:** GAS parameters estimated on finite training window may overfit
- **Copula misspecification:** D-vine structure may not capture true dependence; consider R-vine if cross-validation justifies
- **Marginal misspecification:** GARCH-t may miss leverage effects or long memory; consider GJR-GARCH or FIGARCH if diagnostics fail

---

## References

- Christoffersen, P. (1998). Evaluating Interval Forecasts. *International Economic Review*, 39(4), 841-862.
- Creal, D., Koopman, S.J., & Lucas, A. (2013). Generalized Autoregressive Score Models with Applications. *JBES*, 28(5), 777-795.
- Demarta, S., & McNeil, A.J. (2005). The t Copula and Related Copulas. *International Statistical Review*, 73(1), 111-129.
- Diebold, F.X., & Mariano, R.S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*, 3(2), 73-84.
- Tasche, D. (2000). Risk Contributions and Performance Measurement. Working Paper, Technische Universität München.
