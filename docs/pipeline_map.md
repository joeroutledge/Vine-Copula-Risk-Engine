# Pipeline Map

## Entry point

```
scripts/run_var_es_backtest.py --config configs/demo.yaml
```

## Data flow

```
data/public_returns.csv
    |
    v
[1] fit_garch_pit()             -> PIT uniforms U (pd.DataFrame)
    |                               garch_info (per-asset sigma, nu)
    |
    +--[2] historical_simulation_var_es()  -> HS VaR/ES arrays
    |
    +--[3] ewma_gaussian_var_es()          -> EWMA VaR/ES arrays
    |
    +--[4] DCCGARCHModel.fit() + rolling sim -> DCC VaR/ES arrays
    |
    +--[5a] StaticDVineModel.fit()         -> Static vine VaR/ES
    |       vine_copula_var_es()
    |
    +--[5b] GASDVineModel.fit()            -> GAS vine VaR/ES
    |       vine_copula_var_es()
    |
    v
[6] kupiec_test(), christoffersen_test(), es_adequacy()
    |
    v
outputs/demo/
    metrics.json
    var_es_timeseries.csv
    backtest_summary.csv
    var_forecasts.png
    breaches.png
    manifest.json
```

## Module dependency graph

```
scripts/run_var_es_backtest.py
  -> vine_risk.benchmarks.static_vine.StaticDVineModel
       -> vine_risk.core.copulas
       -> vine_risk.core.tail_dependence
  -> vine_risk.benchmarks.gas_vine.GASDVineModel
       -> vine_risk.benchmarks.static_vine
       -> vine_risk.core.gas
            -> vine_risk.core.copulas
       -> vine_risk.core.tail_dependence
  -> vine_risk.benchmarks.dcc_garch.DCCGARCHModel
       (self-contained: numpy, scipy only)
  -> vine_risk.core.copulas (clip01)
  -> vine_risk.core.tail_dependence
```

## Artifacts

| File | Producer | Consumer |
|------|----------|----------|
| `var_es_timeseries.csv` | `run_var_es_backtest.py` | Downstream analysis |
| `backtest_summary.csv` | `run_var_es_backtest.py` | README claims |
| `metrics.json` | `run_var_es_backtest.py` | Verification scripts |
| `manifest.json` | `run_var_es_backtest.py` | `make demo` hash check |
| `var_forecasts.png` | `run_var_es_backtest.py` | Visual inspection |
| `breaches.png` | `run_var_es_backtest.py` | Visual inspection |

## Reproduce

```bash
make demo    # runs pipeline end-to-end
make test    # pytest -q
```
