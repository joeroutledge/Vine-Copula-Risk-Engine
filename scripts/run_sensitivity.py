#!/usr/bin/env python3
"""
Sensitivity Analysis: VaR/ES stability across hyperparameter grid.

This script runs a small grid over key hyperparameters to assess the stability
of VaR/ES forecasts and backtest metrics. This is a robustness check, NOT a
hyperparameter search or optimization.

Grid:
- n_sim: {500, 1000, 2000}  (Monte Carlo simulation count)
- nu_fixed: {30, 100, 300}  (Fixed degrees of freedom for Student-t copula)
- model: {static, gas}

Output:
- outputs/sensitivity_quick/sensitivity_summary.csv
- outputs/sensitivity_quick/manifest.json

Usage:
    python scripts/run_sensitivity.py

    # Tiny mode for CI testing (reduced grid):
    SENSITIVITY_TINY=1 python scripts/run_sensitivity.py
"""

import hashlib
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vine_risk.benchmarks.static_vine import StaticDVineModel
from vine_risk.benchmarks.gas_vine import GASDVineModel
from vine_risk.core.copulas import clip01

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "seed": 42,
    "data_file": "data/public_returns.csv",
    "train_days": 1000,
    "n_assets": 5,
    "alphas": [0.01, 0.05],
    "out_dir": "outputs/sensitivity_quick",
}

# Full grid
FULL_GRID = {
    "n_sim": [500, 1000, 2000],
    "nu_fixed": [30, 100, 300],
    "model": ["static", "gas"],
}

# Tiny grid for CI testing
TINY_GRID = {
    "n_sim": [500],
    "nu_fixed": [100],
    "model": ["gas"],
}


# ---------------------------------------------------------------------------
# Backtest statistics (copied from run_var_es_backtest.py for independence)
# ---------------------------------------------------------------------------

def kupiec_test(hits: np.ndarray, alpha: float) -> Dict:
    """Kupiec (1995) unconditional coverage test."""
    n = len(hits)
    n1 = int(np.sum(hits))
    n0 = n - n1
    pi_hat = n1 / n if n > 0 else 0

    if pi_hat == 0 or pi_hat == 1:
        return {"statistic": np.nan, "p_value": np.nan, "n_hits": n1,
                "hit_rate": pi_hat}

    ll_unrestricted = n1 * np.log(pi_hat) + n0 * np.log(1 - pi_hat)
    ll_restricted = n1 * np.log(alpha) + n0 * np.log(1 - alpha)
    lr = 2 * (ll_unrestricted - ll_restricted)

    return {
        "statistic": float(lr),
        "p_value": float(1 - chi2.cdf(lr, 1)),
        "n_hits": int(n1),
        "hit_rate": float(pi_hat),
    }


def christoffersen_test(hits: np.ndarray) -> Dict:
    """Christoffersen (1998) independence test."""
    n = len(hits)
    if n < 2:
        return {"statistic": np.nan, "p_value": np.nan}

    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        if hits[t - 1] == 0 and hits[t] == 0:
            n00 += 1
        elif hits[t - 1] == 0 and hits[t] == 1:
            n01 += 1
        elif hits[t - 1] == 1 and hits[t] == 0:
            n10 += 1
        else:
            n11 += 1

    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n, 1)

    if pi == 0 or pi == 1 or pi01 == 0 or pi01 == 1:
        return {"statistic": np.nan, "p_value": np.nan}

    ll_restricted = 0.0
    if n00 + n01 > 0 and 0 < pi < 1:
        ll_restricted += (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)

    ll_unrestricted = 0.0
    if n00 > 0 and pi01 < 1:
        ll_unrestricted += n00 * np.log(1 - pi01)
    if n01 > 0 and pi01 > 0:
        ll_unrestricted += n01 * np.log(pi01)
    if n10 > 0 and pi11 < 1:
        ll_unrestricted += n10 * np.log(1 - pi11)
    if n11 > 0 and pi11 > 0:
        ll_unrestricted += n11 * np.log(pi11)

    lr = max(2 * (ll_unrestricted - ll_restricted), 0.0)

    return {"statistic": float(lr), "p_value": float(1 - chi2.cdf(lr, 1))}


def es_adequacy(realized: np.ndarray, var_forecast: np.ndarray,
                es_forecast: np.ndarray) -> Dict:
    """ES adequacy: mean shortfall ratio on breach days."""
    valid = ~np.isnan(var_forecast) & ~np.isnan(es_forecast)
    r = realized[valid]
    v = var_forecast[valid]
    e = es_forecast[valid]

    breaches = r < v
    n_breaches = int(np.sum(breaches))

    if n_breaches == 0:
        return {"mean_shortfall_ratio": np.nan}

    realized_shortfall = np.mean(r[breaches])
    forecast_es = np.mean(e[breaches])
    ratio = realized_shortfall / forecast_es if forecast_es != 0 else np.nan

    return {"mean_shortfall_ratio": float(ratio)}


def pinball_loss(realized: np.ndarray, var_forecast: np.ndarray,
                 alpha: float) -> float:
    """Quantile (pinball) loss for VaR."""
    valid = ~np.isnan(var_forecast)
    r = realized[valid]
    q = var_forecast[valid]
    if len(r) == 0:
        return np.nan
    diff = r - q
    loss = np.where(diff >= 0, alpha * diff, (1 - alpha) * (-diff))
    return float(np.mean(loss))


# ---------------------------------------------------------------------------
# GARCH-t marginal fitting (simplified from run_var_es_backtest.py)
# ---------------------------------------------------------------------------

def fit_garch_pit(returns: pd.DataFrame, train_end: int) -> Tuple[pd.DataFrame, Dict]:
    """Fit GARCH(1,1)-t marginals and compute PIT."""
    from scipy.stats import t as tdist
    from scipy.optimize import minimize as sp_minimize
    from scipy.special import gammaln

    U = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    garch_info = {}

    for col in returns.columns:
        r = returns[col].values
        n = len(r)
        var0 = np.var(r[:min(50, n)])

        r_train = r[:train_end]
        n_train = len(r_train)

        def neg_ll(params):
            omega, alpha, beta, log_nu_m2 = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            nu = 2.0 + np.exp(log_nu_m2)
            if nu > 50.0:
                return 1e10
            sig2 = np.empty(n_train)
            sig2[0] = var0
            for t in range(1, n_train):
                sig2[t] = omega + alpha * r_train[t - 1] ** 2 + beta * sig2[t - 1]
                sig2[t] = max(sig2[t], 1e-10)
            z2 = r_train ** 2 / sig2
            ll = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
                  - 0.5 * np.log(np.pi * nu)
                  - 0.5 * np.sum(np.log(sig2))
                  - 0.5 * (nu + 1.0) * np.sum(np.log1p(z2 / nu)))
            return -ll

        x0 = [var0 * 0.05, 0.08, 0.90, np.log(6.0)]
        bounds = [(1e-10, None), (0, 0.5), (0, 0.999), (np.log(2.0), np.log(48.0))]
        res = sp_minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
        if res.success:
            omega, alpha, beta, log_nu_m2 = res.x
            nu_hat = 2.0 + np.exp(log_nu_m2)
        else:
            omega, alpha, beta = var0 * 0.05, 0.08, 0.90
            nu_hat = 8.0

        nu_hat = float(np.clip(nu_hat, 4.0, 50.0))

        sig2 = np.empty(n)
        sig2[0] = var0
        for t in range(1, n):
            sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
            sig2[t] = max(sig2[t], 1e-10)
        sigma = np.sqrt(sig2)

        z = r / np.maximum(sigma, 1e-10)
        u = tdist.cdf(z, nu_hat)
        u = np.clip(u, 1e-8, 1 - 1e-8)

        U[col] = u
        garch_info[col] = {
            "omega": omega, "alpha": alpha, "beta": beta,
            "sigma": sigma, "nu": nu_hat,
        }

    return U, garch_info


# ---------------------------------------------------------------------------
# Vine VaR/ES forecasting (simplified)
# ---------------------------------------------------------------------------

def vine_var_es_rolling(
    model,
    garch_info: Dict,
    returns: pd.DataFrame,
    weights: np.ndarray,
    train_end: int,
    n_sim: int,
    alphas: List[float],
    seed: int,
) -> Dict[str, np.ndarray]:
    """Roll forward vine copula for VaR/ES forecasts."""
    from scipy.stats import t as tdist

    n = len(returns)
    cols = list(returns.columns)
    result = {f"var_{a}": np.full(n, np.nan) for a in alphas}
    result.update({f"es_{a}": np.full(n, np.nan) for a in alphas})

    for t in range(train_end, n):
        np.random.seed(seed + t)
        U_sim = model.simulate(n_sim=n_sim, t_idx=t)

        r_sim = np.empty_like(U_sim)
        for j, col in enumerate(cols):
            order_j = model.order[j] if j < len(model.order) else j
            asset = model.assets[order_j]
            info = garch_info[asset]
            sig_t = info["sigma"][min(t, len(info["sigma"]) - 1)]
            nu = info["nu"]
            z = tdist.ppf(clip01(U_sim[:, j]), nu)
            r_sim[:, j] = z * sig_t

        r_aligned = np.empty((n_sim, len(cols)))
        for j in range(len(cols)):
            orig_idx = model.order[j]
            r_aligned[:, orig_idx] = r_sim[:, j]

        port_sim = r_aligned @ weights

        for a in alphas:
            var_a = np.quantile(port_sim, a)
            tail = port_sim[port_sim <= var_a]
            es_a = float(np.mean(tail)) if len(tail) > 0 else var_a
            result[f"var_{a}"][t] = var_a
            result[f"es_{a}"][t] = es_a

    return result


# ---------------------------------------------------------------------------
# Main sensitivity loop
# ---------------------------------------------------------------------------

def run_sensitivity_grid(grid: Dict, cfg: Dict) -> pd.DataFrame:
    """Run sensitivity analysis over the specified grid."""
    seed = cfg["seed"]
    np.random.seed(seed)

    # Load data
    returns = pd.read_csv(cfg["data_file"], index_col=0, parse_dates=True)
    asset_cols = list(returns.columns[:cfg["n_assets"]])
    returns = returns[asset_cols]
    n = len(returns)
    train_end = min(cfg["train_days"], n - 100)

    d = len(asset_cols)
    weights = np.ones(d) / d
    port_ret = returns.values @ weights

    # Fit GARCH marginals (shared across all configs)
    print("Fitting GARCH marginals...", flush=True)
    U, garch_info = fit_garch_pit(returns, train_end)
    U_train = U.iloc[:train_end]

    alphas = cfg["alphas"]
    results = []
    config_id = 0

    # Generate grid combinations
    models = grid["model"]
    n_sims = grid["n_sim"]
    nu_fixeds = grid["nu_fixed"]

    total_configs = len(models) * len(n_sims) * len(nu_fixeds)
    print(f"Running {total_configs} configurations...", flush=True)

    for model_type, n_sim, nu_fixed in product(models, n_sims, nu_fixeds):
        config_id += 1
        print(f"  [{config_id}/{total_configs}] model={model_type}, "
              f"n_sim={n_sim}, nu_fixed={nu_fixed}", flush=True)

        # Fit model with this nu_fixed
        if model_type == "static":
            model = StaticDVineModel(U, nu_fixed=float(nu_fixed))
            model.fit(U_train)
            model_seed = seed
        else:  # gas
            model = GASDVineModel(U, nu_fixed=float(nu_fixed))
            model.fit(U_train, fit_gas=True)
            model_seed = seed + 1000

        # Run VaR/ES forecast
        vine_results = vine_var_es_rolling(
            model, garch_info, returns, weights,
            train_end, n_sim, alphas, model_seed
        )

        # Compute backtest metrics for each alpha
        for a in alphas:
            var_arr = vine_results[f"var_{a}"]
            es_arr = vine_results[f"es_{a}"]

            valid_mask = ~np.isnan(var_arr)
            oos_mask = np.zeros(n, dtype=bool)
            oos_mask[train_end:] = True
            mask = valid_mask & oos_mask

            r_oos = port_ret[mask]
            v_oos = var_arr[mask]
            e_oos = es_arr[mask]

            hits = (r_oos < v_oos).astype(int)

            kup = kupiec_test(hits, a)
            chris = christoffersen_test(hits)
            es_adeq = es_adequacy(r_oos, v_oos, e_oos)
            pl = pinball_loss(r_oos, v_oos, a)

            results.append({
                "model": model_type,
                "config_id": config_id,
                "n_sim": n_sim,
                "nu_fixed": nu_fixed,
                "alpha": a,
                "hit_rate": kup["hit_rate"],
                "kupiec_p": kup["p_value"],
                "christoffersen_p": chris["p_value"],
                "es_shortfall_ratio": es_adeq["mean_shortfall_ratio"],
                "pinball_loss": pl,
                "n_oos": len(r_oos),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(out_dir: Path) -> dict:
    """Build manifest with SHA256 hashes."""
    manifest = {
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {}
    }
    for p in sorted(out_dir.iterdir()):
        if p.name == "manifest.json":
            continue
        if p.is_file():
            manifest["files"][p.name] = sha256_file(p)
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check for tiny mode (for CI testing)
    tiny_mode = os.environ.get("SENSITIVITY_TINY", "0") == "1"
    grid = TINY_GRID if tiny_mode else FULL_GRID

    cfg = DEFAULT_CONFIG.copy()
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  SENSITIVITY ANALYSIS — Vine Copula Risk Engine")
    print("=" * 64)
    print(f"  Mode:         {'TINY (CI)' if tiny_mode else 'FULL'}")
    print(f"  Grid:         {grid}")
    print(f"  Seed:         {cfg['seed']}")
    print(f"  Train days:   {cfg['train_days']}")
    print(f"  n_assets:     {cfg['n_assets']}")
    print(f"  Alphas:       {cfg['alphas']}")
    print()

    # Run sensitivity grid
    df = run_sensitivity_grid(grid, cfg)

    # Save results
    csv_path = out_dir / "sensitivity_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows)")

    # Build and save manifest
    manifest = build_manifest(out_dir)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {manifest_path}")

    print()
    print("=" * 64)
    print("  DONE.  Sensitivity analysis complete.")
    print("=" * 64)

    # Print summary statistics
    print("\nSummary by model:")
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        print(f"\n  {model}:")
        for alpha in cfg["alphas"]:
            alpha_df = model_df[model_df["alpha"] == alpha]
            hr_mean = alpha_df["hit_rate"].mean()
            hr_std = alpha_df["hit_rate"].std()
            kp_mean = alpha_df["kupiec_p"].mean()
            print(f"    alpha={alpha}: hit_rate={hr_mean:.4f}+/-{hr_std:.4f}, "
                  f"kupiec_p={kp_mean:.3f}")


if __name__ == "__main__":
    main()
