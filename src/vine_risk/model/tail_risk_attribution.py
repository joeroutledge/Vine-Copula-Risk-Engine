"""
Tail Risk Attribution: Component ES via Monte Carlo Tail Conditional Expectation.

This module computes risk contributions (component ES) for portfolio assets
using the Euler decomposition principle applied to Monte Carlo simulations.

Definitions
-----------
Let simulated asset returns at time t be r_{t,i}^{(k)} for path k and asset i,
with portfolio weights w_i.

Portfolio loss: L_t^{(k)} = -sum_i w_i * r_{t,i}^{(k)}
VaR_alpha(t): alpha-quantile of {L_t^{(k)}} across paths
ComponentES_i(t) = w_i * E[-r_{t,i} | L_t >= VaR_alpha(t)]
PercentContribution_i(t) = ComponentES_i(t) / sum_j ComponentES_j(t)

Euler Decomposition for ES
--------------------------
The formula ComponentES_i = w_i * E[-r_i | L >= VaR] is the Euler decomposition
for Expected Shortfall (ES). This is mathematically correct because:

- ES (the tail conditional expectation) IS subadditive and coherent
- sum_i ComponentES_i = E[L | L >= VaR] = ES (portfolio ES)
- Percent contributions sum to exactly 100%

The decomposition answers "which assets contribute to tail risk?" and components
scale proportionally with portfolio ES.

References
----------
- Tasche, D. (2000). "Risk contributions and performance measurement."
- McNeil, Frey & Embrechts (2005). "Quantitative Risk Management."
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List


def compute_component_es(
    sim_returns: Union[np.ndarray, pd.DataFrame],
    weights: np.ndarray,
    alpha: float,
    asset_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute component ES using Monte Carlo tail conditional expectation.

    Parameters
    ----------
    sim_returns : np.ndarray or pd.DataFrame, shape (n_sim, n_assets)
        Simulated asset returns for a single time step.
        Each row is one Monte Carlo path, each column is one asset.
    weights : np.ndarray, shape (n_assets,)
        Portfolio weights. Will be normalized to sum to 1 if not already.
    alpha : float
        VaR confidence level, e.g., 0.01 for 1% VaR.
        Must be in (0, 0.5] (left tail).
    asset_names : list of str, optional
        Asset labels. If None and sim_returns is a DataFrame, uses column names.
        Otherwise uses "Asset_0", "Asset_1", etc.

    Returns
    -------
    pd.DataFrame
        Columns: alpha, asset, weight, component_es, percent_contribution,
                 portfolio_var, portfolio_es
        One row per asset.

    Raises
    ------
    ValueError
        If alpha out of bounds, shapes incompatible, or empty tail set.

    Notes
    -----
    - ComponentES_i = w_i * E[-r_i | L >= VaR_alpha]
      where L = -sum_j w_j * r_j (portfolio loss, positive when portfolio loses).
    - The sum of component ES equals portfolio ES (Euler decomposition).
    - Percent contributions sum to exactly 100%.
    - For attribution purposes, this answers "which assets drive tail risk?"
    """
    # Convert DataFrame to array if needed
    if isinstance(sim_returns, pd.DataFrame):
        if asset_names is None:
            asset_names = list(sim_returns.columns)
        sim_returns = sim_returns.values

    sim_returns = np.asarray(sim_returns, dtype=np.float64)

    # Validate inputs
    if sim_returns.ndim != 2:
        raise ValueError(f"sim_returns must be 2D, got shape {sim_returns.shape}")

    n_sim, n_assets = sim_returns.shape

    if n_sim == 0:
        raise ValueError("sim_returns has no rows (n_sim=0)")

    weights = np.asarray(weights, dtype=np.float64).flatten()
    if len(weights) != n_assets:
        raise ValueError(
            f"weights length ({len(weights)}) != n_assets ({n_assets})"
        )

    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum == 0:
        raise ValueError("weights sum to zero")
    weights = weights / w_sum

    # Validate alpha
    if not (0 < alpha <= 0.5):
        raise ValueError(f"alpha must be in (0, 0.5], got {alpha}")

    # Asset names
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(n_assets)]
    if len(asset_names) != n_assets:
        raise ValueError(
            f"asset_names length ({len(asset_names)}) != n_assets ({n_assets})"
        )

    # Check for NaNs
    if np.any(np.isnan(sim_returns)):
        raise ValueError("sim_returns contains NaN values")

    # Compute portfolio loss: L = -sum_i w_i * r_i (positive when losing)
    portfolio_returns = sim_returns @ weights  # shape (n_sim,)
    portfolio_loss = -portfolio_returns  # positive loss

    # VaR_alpha is the alpha-quantile of loss distribution
    # Since loss is positive when portfolio loses, VaR_alpha is positive
    var_alpha_loss = np.quantile(portfolio_loss, 1 - alpha)

    # Identify tail paths: L >= VaR_alpha (worst alpha fraction)
    # Use >= to include the boundary; for continuous dist this is measure-zero
    tail_mask = portfolio_loss >= var_alpha_loss
    n_tail = np.sum(tail_mask)

    if n_tail == 0:
        raise ValueError(
            f"Empty tail set at alpha={alpha} with n_sim={n_sim}. "
            "Increase n_sim or check for degenerate simulations."
        )

    # Extract tail returns (negative returns in the tail)
    tail_returns = sim_returns[tail_mask, :]  # shape (n_tail, n_assets)

    # Component ES: w_i * E[-r_i | L >= VaR]
    # Note: -r_i is the asset loss contribution
    mean_neg_returns_tail = -np.mean(tail_returns, axis=0)  # E[-r_i | tail]
    component_es = weights * mean_neg_returns_tail  # shape (n_assets,)

    # Portfolio ES = sum of component ES (Euler decomposition)
    portfolio_es = np.sum(component_es)

    # Portfolio VaR (expressed as positive loss)
    portfolio_var = var_alpha_loss

    # Percent contribution
    if portfolio_es != 0:
        percent_contribution = component_es / portfolio_es
    else:
        percent_contribution = np.full(n_assets, np.nan)

    # Build output DataFrame
    df = pd.DataFrame({
        "alpha": [alpha] * n_assets,
        "asset": asset_names,
        "weight": weights,
        "component_es": component_es,
        "percent_contribution": percent_contribution,
        "portfolio_var": [portfolio_var] * n_assets,
        "portfolio_es": [portfolio_es] * n_assets,
    })

    return df
