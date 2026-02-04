#!/usr/bin/env python3
"""
Rolling VaR/ES Backtest — Vine Copula Risk Distribution Engine Demo.

This script runs a rolling one-step-ahead VaR/ES forecasting backtest
comparing four methods:

1. Historical Simulation (HS):      Empirical quantiles of rolling window
2. EWMA Gaussian:                   Parametric VaR from EWMA covariance
3. DCC-GARCH:                       Dynamic conditional correlation GARCH
4. GAS D-vine copula:               Score-driven vine copula (this project)

Additionally, the static D-vine (no dynamics) serves as a sub-baseline.

Backtests applied:
- Kupiec (1995) unconditional coverage test
- Christoffersen (1998) independence test
- ES adequacy: mean shortfall ratio

All forecasts use strictly lagged information (no lookahead).

Usage:
    python scripts/run_var_es_backtest.py --config configs/demo.yaml

Output (in outputs/demo/):
    metrics.json             Summary statistics and test results
    var_es_timeseries.csv    Daily VaR/ES forecasts + realized returns
    backtest_summary.csv     Breach counts and test p-values per method
    var_forecasts.png        VaR forecast comparison plot
    breaches.png             Breach indicator plot
    manifest.json            SHA256 hashes of all produced files
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, kendalltau

# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vine_risk.benchmarks.static_vine import StaticDVineModel
from vine_risk.benchmarks.gas_vine import GASDVineModel
from vine_risk.benchmarks.dcc_garch import DCCGARCHModel
from vine_risk.core.copulas import clip01
from vine_risk.core.tail_dependence import lower_tail_dependence_t

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ===========================================================================
# MARGINALS: simple GARCH(1,1) PIT
# ===========================================================================

def fit_garch_pit(
    returns: pd.DataFrame,
    train_end: int,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fit univariate GARCH(1,1) with Student-t innovations to each series
    on the training window, then compute probability integral transform
    (PIT) to get uniforms using the FITTED distribution.

    Convention (standard in arch library and literature):
    - r_t = sigma_t * z_t where z_t ~ t(nu) (standard Student-t)
    - sigma_t is the scale parameter, NOT the conditional std
    - Conditional variance = sigma_t^2 * nu/(nu-2) for nu > 2
    - PIT: u = t(nu).cdf(r / sigma)
    - Inverse PIT: r = sigma * t(nu).ppf(u)

    Returns
    -------
    U : pd.DataFrame  (same shape as returns)
        PIT-uniform residuals in (0, 1)
    garch_info : dict
        Per-asset GARCH parameters and sigma paths (for PPF inversion).
    """
    from scipy.stats import t as tdist
    from scipy.optimize import minimize as sp_minimize
    from scipy.special import gammaln

    U = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    garch_info = {}

    for col in returns.columns:
        r = returns[col].values
        n = len(r)
        var0 = np.var(r[:min(50, n)])

        # ---------- fit on training window ----------
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
            # Student-t log-likelihood: r_t = sigma_t * z_t, z_t ~ t(nu)
            # f(r|sigma,nu) = f_t(r/sigma; nu) / sigma
            # log f = log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5*log(pi*nu)
            #         - log(sigma) - (nu+1)/2 * log(1 + (r/sigma)^2 / nu)
            z2 = r_train ** 2 / sig2
            ll = (
                gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
                - 0.5 * np.log(np.pi * nu)
                - 0.5 * np.sum(np.log(sig2))
                - 0.5 * (nu + 1.0) * np.sum(np.log1p(z2 / nu))
            )
            return -ll

        x0 = [var0 * 0.05, 0.08, 0.90, np.log(6.0)]  # nu_init ~ 8
        # Bound nu in [4, 50] for finite kurtosis
        bounds = [(1e-10, None), (0, 0.5), (0, 0.999), (np.log(2.0), np.log(48.0))]
        res = sp_minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
        if res.success:
            omega, alpha, beta, log_nu_m2 = res.x
            nu_hat = 2.0 + np.exp(log_nu_m2)
        else:
            omega, alpha, beta = var0 * 0.05, 0.08, 0.90
            nu_hat = 8.0

        nu_hat = float(np.clip(nu_hat, 4.0, 50.0))

        # ---------- full-sample sigma path (causal) ----------
        sig2 = np.empty(n)
        sig2[0] = var0
        for t in range(1, n):
            sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
            sig2[t] = max(sig2[t], 1e-10)
        sigma = np.sqrt(sig2)

        # ---------- PIT: u = t(nu).cdf(r / sigma) ----------
        # No extra scaling factor - sigma is scale parameter for standard t(nu)
        z = r / np.maximum(sigma, 1e-10)
        u = tdist.cdf(z, nu_hat)
        u = np.clip(u, 1e-8, 1 - 1e-8)

        U[col] = u
        garch_info[col] = {
            "omega": omega, "alpha": alpha, "beta": beta,
            "sigma": sigma, "nu": nu_hat,
        }

    return U, garch_info


# ===========================================================================
# BASELINE: Historical Simulation VaR/ES
# ===========================================================================

def historical_simulation_var_es(
    portfolio_returns: np.ndarray,
    window: int,
    alphas: Tuple[float, ...],
) -> Dict[str, np.ndarray]:
    """
    Rolling historical simulation.  At each t, VaR/ES are the empirical
    quantiles of portfolio returns in [t-window, t-1].
    """
    n = len(portfolio_returns)
    result = {f"var_{a}": np.full(n, np.nan) for a in alphas}
    result.update({f"es_{a}": np.full(n, np.nan) for a in alphas})

    for t in range(window, n):
        hist = portfolio_returns[t - window: t]
        for a in alphas:
            var_a = np.quantile(hist, a)
            tail = hist[hist <= var_a]
            es_a = float(np.mean(tail)) if len(tail) > 0 else var_a
            result[f"var_{a}"][t] = var_a
            result[f"es_{a}"][t] = es_a

    return result


# ===========================================================================
# BASELINE: EWMA Gaussian VaR/ES
# ===========================================================================

def ewma_gaussian_var_es(
    portfolio_returns: np.ndarray,
    halflife: int,
    alphas: Tuple[float, ...],
) -> Dict[str, np.ndarray]:
    """
    EWMA covariance -> Gaussian VaR/ES.

    sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}
    VaR_alpha = z_alpha * sigma_t   (z_alpha = norm.ppf(alpha))
    ES_alpha  = -sigma_t * phi(z_alpha) / alpha
    """
    n = len(portfolio_returns)
    lam = 1 - np.log(2) / halflife  # convert halflife to decay

    sig2 = np.empty(n)
    sig2[0] = np.var(portfolio_returns[:min(50, n)])

    for t in range(1, n):
        sig2[t] = lam * sig2[t - 1] + (1 - lam) * portfolio_returns[t - 1] ** 2
        sig2[t] = max(sig2[t], 1e-12)

    sigma = np.sqrt(sig2)

    result = {}
    for a in alphas:
        z_a = norm.ppf(a)
        result[f"var_{a}"] = z_a * sigma           # negative number
        result[f"es_{a}"] = -sigma * norm.pdf(z_a) / a  # negative number

    return result


# ===========================================================================
# DCC-GARCH VaR/ES (rolling)
# ===========================================================================

def dcc_garch_var_es(
    returns_train: pd.DataFrame,
    returns_full: pd.DataFrame,
    weights: np.ndarray,
    train_end: int,
    n_sim: int,
    alphas: Tuple[float, ...],
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Fit DCC-GARCH on training, then CAUSALLY filter sigma + correlations
    on the full sample, and do rolling MC simulation for OOS VaR/ES.

    The GARCH parameters are estimated on training data only.
    The GARCH recursion and DCC recursion are then run causally on the full
    sample (each step uses only past data).
    """
    n = len(returns_full)
    d = returns_full.shape[1]
    result = {f"var_{a}": np.full(n, np.nan) for a in alphas}
    result.update({f"es_{a}": np.full(n, np.nan) for a in alphas})

    # Step 1: Fit GARCH params on training data
    model = DCCGARCHModel(returns_full)
    model.fit(returns_train)

    # Step 2: Run GARCH sigma recursion on FULL sample using fitted params
    sigma_full = pd.DataFrame(index=returns_full.index, columns=returns_full.columns,
                              dtype=float)
    z_full = pd.DataFrame(index=returns_full.index, columns=returns_full.columns,
                          dtype=float)

    for col in returns_full.columns:
        r = returns_full[col].values
        params = model.garch_params[col]
        omega, alpha_g, beta_g = params["omega"], params["alpha"], params["beta"]
        var0 = np.var(r[:min(50, n)])

        sig2 = np.empty(n)
        sig2[0] = var0
        for t in range(1, n):
            sig2[t] = omega + alpha_g * r[t - 1] ** 2 + beta_g * sig2[t - 1]
            sig2[t] = max(sig2[t], 1e-10)

        sigma_full[col] = np.sqrt(sig2)
        z_full[col] = r / np.maximum(np.sqrt(sig2), 1e-10)

    # Step 3: Fit DCC on training z, then filter on full z
    z_train = z_full.iloc[:train_end].values
    dcc_params = model._fit_dcc(z_train)
    model.dcc_params = dcc_params

    # Filter DCC correlations on full sample (causal recursion)
    R_path = model.filter_correlations(z_full.values)

    rng = np.random.RandomState(seed)

    for t in range(train_end, n):
        R_t = R_path[t]
        sig_t = sigma_full.iloc[t].values

        Z = rng.standard_normal((n_sim, d))
        try:
            L = np.linalg.cholesky(R_t)
            Z_corr = Z @ L.T
        except np.linalg.LinAlgError:
            Z_corr = Z
        r_sim = Z_corr * sig_t
        port_sim = r_sim @ weights

        for a in alphas:
            var_a = np.quantile(port_sim, a)
            tail = port_sim[port_sim <= var_a]
            es_a = float(np.mean(tail)) if len(tail) > 0 else var_a
            result[f"var_{a}"][t] = var_a
            result[f"es_{a}"][t] = es_a

    return result


# ===========================================================================
# VINE COPULA VaR/ES (rolling)
# ===========================================================================

def vine_copula_var_es(
    model,
    garch_info: Dict,
    returns: pd.DataFrame,
    weights: np.ndarray,
    train_end: int,
    n_sim: int,
    alphas: Tuple[float, ...],
    seed: int,
    label: str = "vine",
    refit_freq: int = 0,
    refit_window: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    Roll forward the vine copula model for one-step-ahead VaR/ES.

    At each t >= train_end:
    1. Simulate U from vine copula at time t (using data up to t-1)
    2. Invert marginals: r_i = t_nu^{-1}(u_i) * sigma_i(t)
    3. Portfolio return = w' @ r
    4. VaR/ES from empirical distribution of simulated portfolio returns

    Parameters
    ----------
    refit_freq : int
        Days between marginal/copula refits. 0 = no refit (default).
    refit_window : int
        Rolling window size for refit (default 1000 days).
    """
    from scipy.stats import t as tdist

    n = len(returns)
    cols = list(returns.columns)
    result = {f"var_{a}": np.full(n, np.nan) for a in alphas}
    result.update({f"es_{a}": np.full(n, np.nan) for a in alphas})

    rng = np.random.RandomState(seed)

    # Track current garch_info (may be updated on refit)
    current_garch_info = garch_info.copy()
    last_refit = train_end

    for t in range(train_end, n):
        # Check if refit is needed
        if refit_freq > 0 and (t - last_refit) >= refit_freq:
            # Rolling window for refit
            window_start = max(0, t - refit_window)
            window_end = t

            # Refit GARCH marginals
            _, new_garch_info = fit_garch_pit(returns.iloc[:t], window_end)

            # CRITICAL: Extend sigma by one step for forecasting at time t.
            # fit_garch_pit returns sigma[0:t] computed from returns[0:t].
            # sigma[t-1] uses r[t-2], sig2[t-2]. But we need sigma[t] which
            # requires r[t-1], sig2[t-1]. Compute the one-step-ahead forecast.
            for col in returns.columns:
                info = new_garch_info[col]
                sigma = info["sigma"]
                if len(sigma) < len(returns):
                    r_prev = returns[col].values[t - 1]  # r[t-1]
                    sig2_prev = sigma[-1] ** 2  # sig2[t-1]
                    sig2_t = info["omega"] + info["alpha"] * r_prev**2 + info["beta"] * sig2_prev
                    sig_t = np.sqrt(max(sig2_t, 1e-10))
                    # Extend sigma array
                    info["sigma"] = np.append(sigma, sig_t)

            current_garch_info = new_garch_info

            # Recompute PIT uniforms for the window
            U_window = pd.DataFrame(index=returns.index[window_start:window_end],
                                     columns=returns.columns, dtype=float)
            for col in returns.columns:
                info = new_garch_info[col]
                r = returns[col].values[window_start:window_end]
                sigma = info["sigma"][window_start:window_end]
                nu = info["nu"]
                z = r / np.maximum(sigma, 1e-10)
                u = tdist.cdf(z, nu)
                U_window[col] = np.clip(u, 1e-8, 1 - 1e-8)

            # Refit copula parameters (keep structure)
            model.refit_params(U_window)
            last_refit = t

        # Set seed for reproducibility per time step
        np.random.seed(seed + t)

        # Simulate from copula at time t
        U_sim = model.simulate(n_sim=n_sim, t_idx=t)

        # Invert marginals: r = sigma * t(nu).ppf(u)
        # No extra scaling factor - consistent with PIT
        r_sim = np.empty_like(U_sim)
        for j, col in enumerate(cols):
            order_j = model.order[j] if j < len(model.order) else j
            asset = model.assets[order_j]
            info = current_garch_info[asset]
            sig_t = info["sigma"][min(t, len(info["sigma"]) - 1)]
            nu = info["nu"]
            z = tdist.ppf(clip01(U_sim[:, j]), nu)
            r_sim[:, j] = z * sig_t

        # Reorder to match original asset order for weight multiplication
        # model.order maps: column j of U_sim corresponds to model.assets[model.order[j]]
        # We need portfolio return = sum_i w_i * r_i
        # Build return vector aligned to original asset ordering
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


# ===========================================================================
# FORMAL BACKTEST STATISTICS
# ===========================================================================

def kupiec_test(hits: np.ndarray, alpha: float) -> Dict:
    """
    Kupiec (1995) unconditional coverage test.

    H0: true breach probability = alpha
    Test statistic: LR_uc = 2 * [log L(pi_hat) - log L(alpha)]
    Asymptotically chi^2(1).
    """
    n = len(hits)
    n1 = int(np.sum(hits))
    n0 = n - n1
    pi_hat = n1 / n if n > 0 else 0

    if pi_hat == 0 or pi_hat == 1:
        return {"statistic": np.nan, "p_value": np.nan, "n_hits": n1,
                "hit_rate": pi_hat, "expected_rate": alpha}

    ll_unrestricted = n1 * np.log(pi_hat) + n0 * np.log(1 - pi_hat)
    ll_restricted = n1 * np.log(alpha) + n0 * np.log(1 - alpha)
    lr = 2 * (ll_unrestricted - ll_restricted)

    return {
        "statistic": float(lr),
        "p_value": float(1 - chi2.cdf(lr, 1)),
        "n_hits": int(n1),
        "hit_rate": float(pi_hat),
        "expected_rate": alpha,
    }


def christoffersen_test(hits: np.ndarray) -> Dict:
    """
    Christoffersen (1998) independence test.

    Tests whether VaR breaches are serially independent.
    H0: P(hit_t | hit_{t-1}=i) does not depend on i
    Test statistic: LR_ind ~ chi^2(1)
    """
    n = len(hits)
    if n < 2:
        return {"statistic": np.nan, "p_value": np.nan}

    # Transition counts
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

    # Transition probabilities
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n, 1)

    if pi == 0 or pi == 1 or pi01 == 0 or pi01 == 1:
        return {"statistic": np.nan, "p_value": np.nan,
                "n00": n00, "n01": n01, "n10": n10, "n11": n11}

    # Log-likelihoods
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

    lr = 2 * (ll_unrestricted - ll_restricted)
    lr = max(lr, 0.0)

    return {
        "statistic": float(lr),
        "p_value": float(1 - chi2.cdf(lr, 1)),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
    }


def es_adequacy(
    realized: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
) -> Dict:
    """
    ES adequacy test: compare realized shortfall to forecast ES on breach days.

    If the model is correct, the average realized loss on breach days should
    approximate the forecast ES.
    """
    valid = ~np.isnan(var_forecast) & ~np.isnan(es_forecast)
    r = realized[valid]
    v = var_forecast[valid]
    e = es_forecast[valid]

    breaches = r < v
    n_breaches = int(np.sum(breaches))

    if n_breaches == 0:
        return {"n_breaches": 0, "mean_shortfall_ratio": np.nan}

    realized_shortfall = np.mean(r[breaches])
    forecast_es = np.mean(e[breaches])

    ratio = realized_shortfall / forecast_es if forecast_es != 0 else np.nan

    return {
        "n_breaches": n_breaches,
        "mean_realized_shortfall": float(realized_shortfall),
        "mean_forecast_es": float(forecast_es),
        "mean_shortfall_ratio": float(ratio),
    }


def pinball_loss(
    realized: np.ndarray,
    var_forecast: np.ndarray,
    alpha: float,
) -> float:
    """
    Quantile (pinball) loss for VaR forecast evaluation.

    L_alpha(r, q) = alpha * (r - q)      if r >= q
                  = (1 - alpha) * (q - r) if r < q

    Lower is better. A well-calibrated VaR minimizes this loss.
    """
    valid = ~np.isnan(var_forecast)
    r = realized[valid]
    q = var_forecast[valid]
    if len(r) == 0:
        return np.nan
    diff = r - q
    loss = np.where(diff >= 0, alpha * diff, (1 - alpha) * (-diff))
    return float(np.mean(loss))


# ===========================================================================
# PLOTTING
# ===========================================================================

def format_alpha_pct(alpha: float) -> str:
    """
    Format alpha as percentage string with minimal decimal places.

    Examples:
        0.01  -> "1%"
        0.025 -> "2.5%"
        0.05  -> "5%"
    """
    pct = alpha * 100
    if pct == int(pct):
        return f"{int(pct)}%"
    else:
        # Format with 1 decimal, strip trailing zeros
        s = f"{pct:.1f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return f"{s}%"


def plot_var_forecasts(
    dates: pd.DatetimeIndex,
    realized: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    alpha: float,
    train_end: int,
    output_path: Path,
    refit_freq: int = 0,
    refit_window: int = 0,
):
    """Plot VaR forecasts vs realized returns."""
    if not HAS_MPL:
        return

    oos_dates = dates[train_end:]
    oos_realized = realized[train_end:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(oos_dates, oos_realized, color="gray", alpha=0.5, linewidth=0.5,
            label="Realized return")

    colors = {"hs": "#1f77b4", "ewma": "#ff7f0e", "dcc": "#2ca02c",
              "gas_vine": "#d62728", "static_vine": "#9467bd"}

    alpha_str = format_alpha_pct(alpha)
    for method, var_arr in forecasts.items():
        oos_var = var_arr[train_end:]
        ax.plot(oos_dates, oos_var, label=f"{method} VaR({alpha_str})",
                color=colors.get(method, "black"), linewidth=0.8)

    # Add refit date annotations if rolling refit is enabled
    if refit_freq > 0:
        n_oos = len(oos_dates)
        for t in range(refit_freq, n_oos, refit_freq):
            ax.axvline(oos_dates[t], color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
        # Add annotation note
        ax.text(0.02, 0.98, f"Refit every {refit_freq} days, window={refit_window}",
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(f"VaR({alpha_str}) Forecast Comparison (OOS)")
    ax.set_ylabel("Return")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_breaches(
    dates: pd.DatetimeIndex,
    realized: np.ndarray,
    var_forecast: np.ndarray,
    train_end: int,
    method: str,
    alpha: float,
    output_path: Path,
    rolling_window: int = 250,
    refit_freq: int = 0,
    refit_window: int = 0,
):
    """Plot breach indicator for one method with rolling breach rate."""
    if not HAS_MPL:
        return

    oos = slice(train_end, None)
    oos_dates = dates[oos]
    r = realized[oos]
    v = var_forecast[oos]
    valid = ~np.isnan(v)
    breaches = np.zeros(len(r), dtype=bool)
    breaches[valid] = r[valid] < v[valid]

    alpha_str = format_alpha_pct(alpha)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(oos_dates, r, color="gray", alpha=0.5, linewidth=0.5, label="Realized")
    ax1.plot(oos_dates, v, color="red", linewidth=0.8, label=f"VaR({alpha_str})")
    ax1.scatter(oos_dates[breaches], r[breaches], color="red", s=12, zorder=5,
                label="Breach")
    ax1.set_ylabel("Return")
    ax1.set_title(f"{method}: VaR({alpha_str}) Breaches")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Add refit date annotations if rolling refit is enabled
    n_oos = len(oos_dates)
    if refit_freq > 0:
        for t in range(refit_freq, n_oos, refit_freq):
            ax1.axvline(oos_dates[t], color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
            ax2.axvline(oos_dates[t], color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
        # Add annotation note
        ax1.text(0.02, 0.98, f"Refit every {refit_freq} days, window={refit_window}",
                 transform=ax1.transAxes, fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Rolling breach rate (avoids misleading early cumulative spike)
    breach_series = pd.Series(breaches.astype(float), index=oos_dates)
    rolling_rate = breach_series.rolling(window=rolling_window, min_periods=rolling_window).mean()

    ax2.plot(oos_dates, rolling_rate, color="red", linewidth=1,
             label=f"Rolling {rolling_window}-day")
    ax2.axhline(alpha, color="black", linestyle="--", linewidth=0.8,
                label=f"Target {alpha_str}")
    ax2.set_ylabel("Breach rate")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, max(alpha * 3, 0.1))  # Cap y-axis for readability
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# SHA-256 manifest
# ===========================================================================

# Files to exclude from manifest (junk, temporary, archives)
MANIFEST_EXCLUDE_PATTERNS = [
    ".DS_Store",     # macOS filesystem junk
    "*.zip",         # archives
    "*.tar.gz",      # archives
    "*.tgz",         # archives
    "__pycache__",   # Python cache
    "*.pyc",         # compiled Python
    ".gitkeep",      # git placeholder
]


def _is_excluded(filename: str) -> bool:
    """Check if filename matches any exclusion pattern."""
    import fnmatch
    for pattern in MANIFEST_EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(out_dir: Path) -> dict:
    """Build manifest with SHA256 hashes, excluding junk files."""
    manifest = {"produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": {}}
    for p in sorted(out_dir.iterdir()):
        if p.name == "manifest.json":
            continue
        if _is_excluded(p.name):
            continue
        if p.is_file():
            manifest["files"][p.name] = sha256_file(p)
    return manifest


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rolling VaR/ES backtest: vine copula vs baselines")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.get("out_dir", "outputs/demo"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    data_file = cfg.get("data_file", "data/public_returns.csv")
    train_days = cfg.get("train_days", 1000)
    hs_window = cfg.get("hs_window", 250)
    ewma_halflife = cfg.get("ewma_halflife", 20)
    n_sim = cfg.get("n_sim", 5000)
    alphas = tuple(cfg.get("alphas", [0.01, 0.025, 0.05]))
    n_assets = cfg.get("n_assets", 5)  # use a subset for speed
    refit_freq = cfg.get("marginal_refit_freq_days", 0)
    refit_window = cfg.get("marginal_window_days", 1000)

    print("=" * 64)
    print("  ROLLING VaR/ES BACKTEST — Vine Copula Risk Engine Demo")
    print("=" * 64)
    print(f"  Config:       {args.config}")
    print(f"  Seed:         {seed}")
    print(f"  Train days:   {train_days}")
    print(f"  n_sim:        {n_sim}")
    print(f"  Alphas:       {alphas}")
    print(f"  n_assets:     {n_assets}")
    if refit_freq > 0:
        print(f"  Refit freq:   {refit_freq} days")
        print(f"  Refit window: {refit_window} days")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    returns = pd.read_csv(data_file, index_col=0, parse_dates=True)
    # Use a subset of assets for tractability
    asset_cols = list(returns.columns[:n_assets])
    returns = returns[asset_cols]
    print(f"Data: {returns.shape[0]} days x {returns.shape[1]} assets")
    print(f"Assets: {asset_cols}")
    print(f"Period: {returns.index[0].date()} to {returns.index[-1].date()}")

    n = len(returns)
    train_end = min(train_days, n - 100)
    print(f"Train: 0..{train_end-1}  |  OOS: {train_end}..{n-1}  "
          f"({n - train_end} days)")
    print()

    # Equal-weight portfolio
    d = len(asset_cols)
    weights = np.ones(d) / d
    port_ret = (returns.values @ weights)

    # ------------------------------------------------------------------
    # Step 1: Fit GARCH marginals + PIT
    # ------------------------------------------------------------------
    print("[1/5] Fitting GARCH(1,1)-t marginals + PIT ...", flush=True)
    U, garch_info = fit_garch_pit(returns, train_end)
    nu_values = {col: f"{info['nu']:.2f}" for col, info in garch_info.items()}
    print(f"       PIT uniforms: shape {U.shape}")
    print(f"       Estimated df: {nu_values}")

    # ------------------------------------------------------------------
    # Step 2: Historical Simulation baseline
    # ------------------------------------------------------------------
    print("[2/5] Historical Simulation baseline ...", flush=True)
    hs_results = historical_simulation_var_es(port_ret, hs_window, alphas)

    # ------------------------------------------------------------------
    # Step 3: EWMA Gaussian baseline
    # ------------------------------------------------------------------
    print("[3/5] EWMA Gaussian baseline ...", flush=True)
    ewma_results = ewma_gaussian_var_es(port_ret, ewma_halflife, alphas)

    # ------------------------------------------------------------------
    # Step 4: DCC-GARCH benchmark
    # ------------------------------------------------------------------
    print("[4/5] DCC-GARCH benchmark ...", flush=True)
    dcc_results = dcc_garch_var_es(
        returns.iloc[:train_end], returns, weights, train_end, n_sim, alphas, seed)

    # ------------------------------------------------------------------
    # Step 5: Vine copula models
    # ------------------------------------------------------------------
    print("[5/5] Vine copula models ...", flush=True)
    U_train = U.iloc[:train_end]

    # Static D-vine (nu estimated per-edge via MLE)
    print("       Fitting Static D-vine ...", flush=True)
    static_model = StaticDVineModel(U, nu_fixed=None)
    static_model.fit(U_train)
    if refit_freq > 0:
        print(f"       Rolling refit enabled: freq={refit_freq}, window={refit_window}")
    static_results = vine_copula_var_es(
        static_model, garch_info, returns, weights, train_end, n_sim, alphas,
        seed, label="static_vine", refit_freq=refit_freq, refit_window=refit_window)

    # GAS D-vine (nu estimated per-edge via MLE)
    print("       Fitting GAS D-vine ...", flush=True)
    gas_model = GASDVineModel(U, nu_fixed=None)
    gas_model.fit(U_train, fit_gas=True)
    gas_results = vine_copula_var_es(
        gas_model, garch_info, returns, weights, train_end, n_sim, alphas,
        seed + 1000, label="gas_vine", refit_freq=refit_freq, refit_window=refit_window)

    # Print GAS diagnostics
    for edge, info in gas_model.tree1_models.items():
        p = info["params"]
        print(f"       Edge {edge}: omega={p.omega:.4f}  A={p.A:.4f}  "
              f"B_eff={p.B_effective:.4f}")

    # ------------------------------------------------------------------
    # Export vine model cards
    # ------------------------------------------------------------------
    static_card = static_model.get_model_card()
    with open(out_dir / "vine_model_card_static.json", "w") as f:
        json.dump(static_card, f, indent=2)
    print(f"       Exported: vine_model_card_static.json ({static_card['total_pair_copulas']} pair copulas)")

    gas_card = gas_model.get_model_card()
    with open(out_dir / "vine_model_card_gas.json", "w") as f:
        json.dump(gas_card, f, indent=2)
    print(f"       Exported: vine_model_card_gas.json ({gas_card['total_pair_copulas']} pair copulas)")

    print()

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    methods = {
        "hs": hs_results,
        "ewma": ewma_results,
        "dcc": dcc_results,
        "static_vine": static_results,
        "gas_vine": gas_results,
    }

    # ------------------------------------------------------------------
    # Build var_es_timeseries.csv
    # ------------------------------------------------------------------
    ts_df = pd.DataFrame({"date": returns.index, "realized_return": port_ret})
    for method, res in methods.items():
        for a in alphas:
            ts_df[f"{method}_var_{a}"] = res[f"var_{a}"]
            ts_df[f"{method}_es_{a}"] = res[f"es_{a}"]
    ts_df.set_index("date", inplace=True)
    ts_df.to_csv(out_dir / "var_es_timeseries.csv")
    print(f"Saved: var_es_timeseries.csv ({len(ts_df)} rows)")

    # ------------------------------------------------------------------
    # Run formal backtests
    # ------------------------------------------------------------------
    print()
    print("-" * 64)
    print("  BACKTEST RESULTS")
    print("-" * 64)

    backtest_rows = []
    metrics_all = {}

    for method, res in methods.items():
        metrics_all[method] = {}
        for a in alphas:
            var_arr = res[f"var_{a}"]
            es_arr = res[f"es_{a}"]

            # Only OOS
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
            ql = pinball_loss(r_oos, v_oos, a)

            row = {
                "method": method, "alpha": a,
                "n_obs": len(r_oos), "n_breaches": kup["n_hits"],
                "hit_rate": kup["hit_rate"], "expected_rate": a,
                "kupiec_stat": kup["statistic"], "kupiec_pval": kup["p_value"],
                "chris_stat": chris["statistic"], "chris_pval": chris["p_value"],
                "es_shortfall_ratio": es_adeq["mean_shortfall_ratio"],
                "pinball_loss": ql,
            }
            backtest_rows.append(row)
            metrics_all[method][str(a)] = row

    backtest_df = pd.DataFrame(backtest_rows)
    backtest_df.to_csv(out_dir / "backtest_summary.csv", index=False)
    print(f"Saved: backtest_summary.csv")

    # Print summary table
    print()
    hdr = f"{'Method':<14} {'Alpha':>6} {'Hits':>5} {'Rate':>7} {'Kupiec p':>9} {'Chris p':>9} {'ES ratio':>9} {'Pinball':>9}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in backtest_df.iterrows():
        print(f"{row['method']:<14} {row['alpha']:>6.3f} "
              f"{row['n_breaches']:>5.0f} {row['hit_rate']:>7.3f} "
              f"{row['kupiec_pval']:>9.3f} {row['chris_pval']:>9.3f} "
              f"{row['es_shortfall_ratio']:>9.3f} "
              f"{row['pinball_loss']:>9.6f}")
    print()

    # ------------------------------------------------------------------
    # Scale sanity check: vine VaR should not exceed 2x HS VaR
    # ------------------------------------------------------------------
    scale_sanity = {"passed": True, "checks": []}
    alpha_check = 0.01  # Check at 1% level

    # Compute median VaR for each method in OOS period
    oos_slice = slice(train_end, n)
    hs_var_oos = hs_results[f"var_{alpha_check}"][oos_slice]
    hs_median = float(np.nanmedian(hs_var_oos))

    for vine_method in ["static_vine", "gas_vine"]:
        vine_var_oos = methods[vine_method][f"var_{alpha_check}"][oos_slice]
        vine_median = float(np.nanmedian(vine_var_oos))

        # VaR is negative, so ratio = vine/hs; if vine is more extreme, ratio > 1
        ratio = vine_median / hs_median if hs_median != 0 else np.nan

        check = {
            "method": vine_method,
            "alpha": alpha_check,
            "vine_median_var": vine_median,
            "hs_median_var": hs_median,
            "ratio": ratio,
            "threshold": 2.0,
            "passed": abs(ratio) <= 2.0 if not np.isnan(ratio) else False,
        }
        scale_sanity["checks"].append(check)

        if not check["passed"]:
            scale_sanity["passed"] = False
            print(f"WARNING: {vine_method} VaR({alpha_check}) is {abs(ratio):.2f}x HS "
                  f"(threshold: 2.0x)")

    with open(out_dir / "scale_sanity.json", "w") as f:
        json.dump(scale_sanity, f, indent=2)
    print(f"Saved: scale_sanity.json (passed={scale_sanity['passed']})")

    # ------------------------------------------------------------------
    # metrics.json
    # ------------------------------------------------------------------
    metrics_json = {
        "seed": seed,
        "train_days": train_end,
        "oos_days": n - train_end,
        "n_assets": d,
        "assets": asset_cols,
        "alphas": list(alphas),
        "n_sim": n_sim,
        "methods": metrics_all,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    print(f"Saved: metrics.json")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    primary_alpha = alphas[1] if len(alphas) > 1 else alphas[0]  # 0.025

    var_forecasts_for_plot = {m: res[f"var_{primary_alpha}"] for m, res in methods.items()}
    plot_var_forecasts(returns.index, port_ret, var_forecasts_for_plot,
                       primary_alpha, train_end, out_dir / "var_forecasts.png",
                       refit_freq=refit_freq, refit_window=refit_window)
    print(f"Saved: var_forecasts.png")

    plot_breaches(returns.index, port_ret,
                  gas_results[f"var_{primary_alpha}"],
                  train_end, "gas_vine", primary_alpha,
                  out_dir / "breaches.png",
                  refit_freq=refit_freq, refit_window=refit_window)
    print(f"Saved: breaches.png")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest = build_manifest(out_dir)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: manifest.json")

    print()
    print("=" * 64)
    print("  DONE.  All artifacts in:", out_dir)
    print("=" * 64)

    # Exit with error if scale sanity check failed
    if not scale_sanity["passed"]:
        print("\nERROR: Scale sanity check failed. Vine VaR exceeds 2x HS VaR.")
        print("       This indicates a potential scaling bug in PIT/PPF.")
        sys.exit(1)


if __name__ == "__main__":
    main()
