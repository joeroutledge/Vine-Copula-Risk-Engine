# Claims Ledger

This document verifies all claims made in README.md and documentation against actual observed behavior.

**Audit Date**: 2026-02-08
**Audit Environment**: Python 3.13.11, macOS Darwin 24.6.0, pyvinecopulib 0.7.5

---

## Verified Claims

| # | Claim | Evidence Command | Evidence File/Value | Status |
|---|-------|------------------|---------------------|--------|
| 1 | Tests pass | `pytest -q` | `213 passed, 3 skipped` | OK |
| 2 | `make demo-quick` produces outputs | `make demo-quick` | `outputs/demo_quick/` (13 files) | OK |
| 3 | Manifest validation passes | `python scripts/validate_manifest.py outputs/demo_quick/manifest.json` | `OK: All files valid` | OK |
| 4 | `make demo-quick-weekly-gas` produces outputs | `make demo-quick-weekly-gas` | `outputs/demo_quick_weekly_gas/` (13 files) | OK |
| 5 | `make sensitivity-quick` produces outputs | `make sensitivity-quick` | `outputs/sensitivity_quick/sensitivity_summary.csv` (36 rows) | OK |
| 6 | Determinism: two runs produce identical artifacts | `diff /tmp/demo_run1/*.csv outputs/demo_quick/*.csv` | All key files identical except timestamps | OK |
| 7 | `metrics.json` contains `gas_update_every` | `python -c "import json; print(json.load(open('outputs/demo_quick/metrics.json'))['gas_update_every'])"` | `1` | OK |
| 8 | `metrics.json` contains `determinism` block | Inspect `outputs/demo_quick/metrics.json` | `{"determinism_mode": "strict", "numpy_seeded": true, "pyvinecopulib_seeded": true}` | OK |
| 9 | `vine_model_card_gas.json` contains `order_source` | `python -c "import json; print(json.load(open('outputs/demo_quick/vine_model_card_gas.json'))['order_source'])"` | `train_only_fixed` | OK |
| 10 | Data file has 12 assets, 5327 days | `wc -l data/public_returns.csv; head -1 data/public_returns.csv` | 5328 lines (1 header + 5327 rows), 12 columns | OK |
| 11 | Date range: 2004-11-19 to 2026-01-23 | `head -2 data/public_returns.csv; tail -1 data/public_returns.csv` | First date: 2004-11-19, Last date: 2026-01-23 | OK |
| 12 | `backtest_summary.csv` contains Kupiec/Christoffersen p-values | Inspect `outputs/demo_quick/backtest_summary.csv` | Columns: `kupiec_pvalue`, `chris_pvalue` | OK |
| 13 | `dm_tests.csv` contains Diebold-Mariano results | Inspect `outputs/demo_quick/dm_tests.csv` | Contains `dm_stat`, `dm_pvalue` columns | OK |
| 14 | `tail_risk_attribution.csv` contains component ES | Inspect `outputs/demo_quick/tail_risk_attribution.csv` | Contains `component_es`, `component_es_return` columns | OK |
| 15 | `scale_sanity.json` contains sanity check | `cat outputs/demo_quick/scale_sanity.json` | `{"passed": true, ...}` | OK |
| 16 | GAS dynamics are truly dynamic OOS | `tests/test_gas_vine_oos_dynamic.py::TestOOSDynamicsNotFrozen::test_rhos_differ_across_oos` | Test passes | OK |
| 17 | Cadence invariance with t_offset | `tests/test_gas_update_every_segment_invariance.py::TestCadenceInvariance::test_split_equals_full_update_every_5` | Test passes | OK |
| 18 | `sensitivity_meta.json` records grid mode | `cat outputs/sensitivity_quick/sensitivity_meta.json` | `{"grid_mode": "FULL", "n_configs": 18, ...}` | OK |

---

## Verification Commands (Runbook)

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Run tests
pytest -q

# 3. Run demo-quick
make demo-quick
python scripts/validate_manifest.py outputs/demo_quick/manifest.json

# 4. Run demo-quick-weekly-gas
make demo-quick-weekly-gas
python scripts/validate_manifest.py outputs/demo_quick_weekly_gas/manifest.json

# 5. Run sensitivity-quick
make sensitivity-quick
python scripts/validate_manifest.py outputs/sensitivity_quick/manifest.json

# 6. Determinism check
python scripts/run_var_es_backtest.py --config configs/demo_quick.yaml --out-dir /tmp/demo_run1
python scripts/run_var_es_backtest.py --config configs/demo_quick.yaml --out-dir /tmp/demo_run2
diff /tmp/demo_run1/backtest_summary.csv /tmp/demo_run2/backtest_summary.csv  # Should be identical
diff /tmp/demo_run1/dm_tests.csv /tmp/demo_run2/dm_tests.csv  # Should be identical
```

---

## Environment Used for Verification

```
Python: 3.13.11
OS: Darwin 24.6.0 (macOS)
pyvinecopulib: 0.7.5
numpy: 2.4.0
pandas: 2.3.3
scipy: 1.16.3
pytest: 9.0.2
```
