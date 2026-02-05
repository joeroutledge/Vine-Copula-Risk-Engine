.PHONY: demo-quick demo-quick-weekly-gas sensitivity-quick test clean install validate-manifest

install:
	pip install -e ".[dev]"

demo-quick:
	@echo "=== vine_risk_xva_demo: Quick demo (reduced simulations) ==="
	python scripts/run_var_es_backtest.py --config configs/demo_quick.yaml
	@echo ""
	@echo "=== Verifying manifest hashes ==="
	python scripts/validate_manifest.py outputs/demo_quick/
	@echo "=== Done (quick demo) ==="

demo-quick-weekly-gas:
	@echo "=== vine_risk_xva_demo: Quick demo with weekly GAS updates ==="
	python scripts/run_var_es_backtest.py --config configs/demo_quick_weekly_gas.yaml
	@echo ""
	@echo "=== Verifying manifest hashes ==="
	python scripts/validate_manifest.py outputs/demo_quick_weekly_gas/
	@echo "=== Done (weekly GAS demo) ==="

sensitivity-quick:
	@echo "=== vine_risk_xva_demo: Sensitivity analysis ==="
	python scripts/run_sensitivity.py
	@echo ""
	@echo "=== Verifying manifest hashes ==="
	python scripts/validate_manifest.py outputs/sensitivity_quick/
	@echo "=== Done (sensitivity analysis) ==="

validate-manifest:
	@echo "Validating demo_quick manifest..."
	python scripts/validate_manifest.py outputs/demo_quick/

test:
	python -m pytest tests/ -q

clean:
	rm -rf outputs/demo_quick/*
	rm -rf outputs/demo_quick_weekly_gas/*
	rm -rf outputs/sensitivity_quick/*
