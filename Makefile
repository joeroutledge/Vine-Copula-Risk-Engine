.PHONY: demo demo-quick demo-full test clean install validate-manifest

install:
	pip install -e ".[dev]"

demo:
	@echo "=== vine_risk_xva_demo: Running VaR/ES backtest pipeline ==="
	python scripts/run_var_es_backtest.py --config configs/demo.yaml
	@echo ""
	@echo "=== Verifying manifest hashes ==="
	python scripts/validate_manifest.py outputs/demo/
	@echo "=== Done ==="

demo-quick:
	@echo "=== vine_risk_xva_demo: Quick demo (reduced simulations) ==="
	python scripts/run_var_es_backtest.py --config configs/demo_quick.yaml
	@echo ""
	@echo "=== Verifying manifest hashes ==="
	python scripts/validate_manifest.py outputs/demo_quick/
	@echo "=== Done (quick demo) ==="

demo-full:
	@echo "=== vine_risk_xva_demo: Running FULL 12-asset VaR/ES backtest ==="
	python scripts/run_var_es_backtest.py --config configs/demo_full12.yaml
	@echo ""
	@echo "=== Verifying manifest hashes ==="
	python scripts/validate_manifest.py outputs/demo_full/
	@echo "=== Done (full 12-asset backtest) ==="

validate-manifest:
	@echo "Validating demo manifest..."
	python scripts/validate_manifest.py outputs/demo/

test:
	python -m pytest tests/ -q

clean:
	rm -rf outputs/demo/*
	rm -rf outputs/demo_quick/*
	rm -rf outputs/demo_full/*
