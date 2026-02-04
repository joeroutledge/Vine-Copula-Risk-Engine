"""
Risk Metrics: VaR and CVaR (Expected Shortfall) Computation.

Stage 3: Risk metrics for scenario-based reporting.

SIGN CONVENTION:
- P&L (pnl): Positive = gain, Negative = loss
- VaR: Reported as a POSITIVE number representing the loss amount at the given confidence level
       VaR(95%) = 2% means there's a 5% chance of losing 2% or more
- CVaR: Average loss in the tail, also reported as POSITIVE
        CVaR(95%) = 3% means the average loss in the worst 5% of scenarios is 3%

For consistency with industry convention:
- VaR(alpha) answers: "What is the maximum loss we won't exceed with probability (1-alpha)?"
- CVaR(alpha) answers: "What is the expected loss given we're in the worst alpha fraction?"

Both are LOSS measures: larger values indicate higher risk.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def var_from_pnl(
    pnl: Union[np.ndarray, pd.Series],
    alpha: float,
    method: str = 'linear',
) -> float:
    """
    Compute Value-at-Risk from P&L distribution.

    VaR(alpha) is the (alpha)-quantile of the LOSS distribution,
    i.e., the -(1-alpha)-quantile of the P&L distribution.

    Parameters
    ----------
    pnl : array-like
        P&L values (positive = gain, negative = loss).
    alpha : float
        Confidence level, e.g., 0.05 for VaR(5%) or 0.01 for VaR(1%).
        This is the probability of exceeding the VaR loss.
    method : str
        Interpolation method for percentile ('linear', 'lower', 'higher', 'nearest').

    Returns
    -------
    float
        VaR as a POSITIVE number representing the loss threshold.
        E.g., VaR=0.02 means 2% loss is the threshold.

    Examples
    --------
    >>> pnl = np.array([0.01, 0.02, -0.01, -0.03, -0.05, 0.03, 0.0, -0.02, 0.01, -0.01])
    >>> var_from_pnl(pnl, alpha=0.05)  # 5% VaR
    0.05  # (approximately, depending on method)

    Notes
    -----
    VaR at alpha=0.05 means: "There is a 5% chance of losing more than VaR."
    Equivalently, we compute the alpha-quantile of losses = -pnl.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    pnl_arr = np.asarray(pnl).flatten()
    if len(pnl_arr) == 0:
        raise ValueError("pnl array is empty")

    # VaR = alpha-quantile of losses = -pnl
    # This is equivalent to -(1-alpha)-quantile of pnl
    losses = -pnl_arr
    var = np.percentile(losses, (1 - alpha) * 100, method=method)

    return float(var)


def cvar_from_pnl(
    pnl: Union[np.ndarray, pd.Series],
    alpha: float,
) -> float:
    """
    Compute Conditional Value-at-Risk (Expected Shortfall) from P&L distribution.

    CVaR(alpha) is the expected loss given that loss exceeds VaR(alpha).
    Also known as Expected Shortfall (ES) or Average Value-at-Risk (AVaR).

    Parameters
    ----------
    pnl : array-like
        P&L values (positive = gain, negative = loss).
    alpha : float
        Confidence level, e.g., 0.05 for CVaR(5%) or 0.01 for CVaR(1%).
        This is the probability of being in the tail.

    Returns
    -------
    float
        CVaR as a POSITIVE number representing the expected tail loss.
        E.g., CVaR=0.03 means average loss in the tail is 3%.

    Examples
    --------
    >>> pnl = np.array([0.01, 0.02, -0.01, -0.03, -0.05, 0.03, 0.0, -0.02, 0.01, -0.01])
    >>> cvar_from_pnl(pnl, alpha=0.05)  # 5% CVaR
    0.05  # (approximately - average of worst 5% of scenarios)

    Notes
    -----
    CVaR at alpha=0.05 means: "The average loss in the worst 5% of scenarios."
    This is computed as the mean of losses in the alpha-tail.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    pnl_arr = np.asarray(pnl).flatten()
    if len(pnl_arr) == 0:
        raise ValueError("pnl array is empty")

    # CVaR = mean of losses exceeding VaR
    # We take the alpha fraction of worst outcomes
    losses = -pnl_arr  # Convert to losses (positive = bad)
    n_tail = max(1, int(np.ceil(len(losses) * alpha)))

    # Sort losses in descending order (worst first)
    sorted_losses = np.sort(losses)[::-1]
    tail_losses = sorted_losses[:n_tail]

    return float(np.mean(tail_losses))


def compute_risk_metrics(
    pnl: Union[np.ndarray, pd.Series],
    alphas: Optional[Tuple[float, ...]] = None,
) -> dict:
    """
    Compute comprehensive risk metrics from P&L distribution.

    Parameters
    ----------
    pnl : array-like
        P&L values (positive = gain, negative = loss).
    alphas : Tuple[float, ...], optional
        Alpha levels for VaR/CVaR. Default: (0.01, 0.05).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'mean_pnl': Mean P&L
        - 'std_pnl': Standard deviation of P&L
        - 'min_pnl': Minimum P&L (worst scenario)
        - 'max_pnl': Maximum P&L (best scenario)
        - 'var_{alpha}': VaR at each alpha level
        - 'cvar_{alpha}': CVaR at each alpha level
        - 'count': Number of observations
    """
    if alphas is None:
        alphas = (0.01, 0.05)

    pnl_arr = np.asarray(pnl).flatten()

    metrics = {
        'mean_pnl': float(np.mean(pnl_arr)),
        'std_pnl': float(np.std(pnl_arr)),
        'min_pnl': float(np.min(pnl_arr)),
        'max_pnl': float(np.max(pnl_arr)),
        'count': len(pnl_arr),
    }

    for alpha in alphas:
        alpha_pct = int(alpha * 100)
        metrics[f'var_{alpha_pct}pct'] = var_from_pnl(pnl_arr, alpha)
        metrics[f'cvar_{alpha_pct}pct'] = cvar_from_pnl(pnl_arr, alpha)

    return metrics


def compute_scenario_risk_metrics(
    portfolio_pnl: np.ndarray,
    dates: pd.DatetimeIndex,
    alphas: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """
    Compute risk metrics for each date from scenario P&L.

    Parameters
    ----------
    portfolio_pnl : np.ndarray
        2D array of shape (n_dates, n_scenarios).
    dates : pd.DatetimeIndex
        Dates corresponding to each row.
    alphas : Tuple[float, ...], optional
        Alpha levels for VaR/CVaR.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per date and columns for each metric.
    """
    if alphas is None:
        alphas = (0.01, 0.05)

    rows = []
    for i, date in enumerate(dates):
        pnl = portfolio_pnl[i, :]
        metrics = compute_risk_metrics(pnl, alphas)
        metrics['date'] = date
        rows.append(metrics)

    df = pd.DataFrame(rows)

    # Reorder columns
    cols = ['date', 'mean_pnl', 'std_pnl', 'min_pnl', 'max_pnl']
    for alpha in alphas:
        alpha_pct = int(alpha * 100)
        cols.extend([f'var_{alpha_pct}pct', f'cvar_{alpha_pct}pct'])
    cols.append('count')

    return df[cols]
