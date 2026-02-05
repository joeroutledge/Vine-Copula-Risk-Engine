"""
Diebold-Mariano Test for Comparing Forecast Accuracy.

Implements the Diebold-Mariano (1995) test for comparing predictive accuracy
of two forecasts using a loss differential series with Newey-West HAC standard
errors to account for serial correlation in the loss differentials.

References
----------
- Diebold, F.X. and Mariano, R.S. (1995). "Comparing Predictive Accuracy."
  Journal of Business & Economic Statistics, 13(3), 253-263.
- Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  Econometrica, 55(3), 703-708.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional


def newey_west_variance(x: np.ndarray, lags: int) -> float:
    """
    Compute Newey-West HAC variance estimator for a series.

    Parameters
    ----------
    x : np.ndarray
        Time series (assumed zero-mean for variance computation).
    lags : int
        Number of lags for HAC estimator (bandwidth).

    Returns
    -------
    float
        HAC variance estimate.

    Notes
    -----
    Uses Bartlett kernel weights: w_j = 1 - j/(lags+1) for j = 1, ..., lags.
    """
    n = len(x)
    if n == 0:
        return np.nan

    # Demean
    x_centered = x - np.mean(x)

    # Variance (gamma_0)
    gamma_0 = np.sum(x_centered ** 2) / n

    # Autocovariances with Bartlett weights
    weighted_sum = 0.0
    for j in range(1, lags + 1):
        if j >= n:
            break
        # Autocovariance at lag j
        gamma_j = np.sum(x_centered[j:] * x_centered[:-j]) / n
        # Bartlett kernel weight
        weight = 1.0 - j / (lags + 1)
        # Add twice (symmetric)
        weighted_sum += 2.0 * weight * gamma_j

    # HAC variance
    hac_var = gamma_0 + weighted_sum

    # Ensure non-negative (numerical issues can cause small negatives)
    return max(hac_var, 0.0)


def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
    nw_lags: Optional[int] = None,
) -> Dict:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[loss_a - loss_b] = 0 against H1: E[loss_a - loss_b] != 0.

    Parameters
    ----------
    loss_a : np.ndarray
        Loss series for model A.
    loss_b : np.ndarray
        Loss series for model B (same length as loss_a).
    h : int, default=1
        Forecast horizon. For h=1 (one-step ahead), overlapping forecast
        issues are minimal. Documented for clarity.
    nw_lags : int or None, default=None
        Number of lags for Newey-West HAC variance. If None, uses
        floor(T^(1/3)) as a common rule of thumb.

    Returns
    -------
    dict
        stat : float
            DM test statistic (asymptotically N(0,1) under H0).
        p_value : float
            Two-sided p-value from asymptotic normal distribution.
        n_obs : int
            Number of observations used.
        nw_lags : int
            Number of Newey-West lags used.
        mean_diff : float
            Mean of (loss_a - loss_b). Negative means A has lower loss.
        h : int
            Forecast horizon used.

    Raises
    ------
    ValueError
        If arrays have different lengths, contain all NaNs, or have
        zero variance in the differential (constant series).

    Notes
    -----
    - The test statistic is: DM = mean(d_t) / sqrt(HAC_var(d_t) / T)
      where d_t = loss_a_t - loss_b_t.
    - A significantly negative statistic suggests model A is more accurate
      (lower loss). A significantly positive statistic suggests model B
      is more accurate.
    - For h=1, serial correlation in d_t is typically mild, but HAC
      variance is still used for robustness.
    - If the loss differential has zero variance (constant), the test
      is degenerate and returns p_value=1.0 with stat=0.0.
    """
    loss_a = np.asarray(loss_a, dtype=np.float64)
    loss_b = np.asarray(loss_b, dtype=np.float64)

    if loss_a.shape != loss_b.shape:
        raise ValueError(
            f"loss_a and loss_b must have same shape: "
            f"{loss_a.shape} vs {loss_b.shape}"
        )

    # Handle NaNs by removing paired observations
    valid_mask = ~np.isnan(loss_a) & ~np.isnan(loss_b)
    loss_a = loss_a[valid_mask]
    loss_b = loss_b[valid_mask]

    n = len(loss_a)
    if n == 0:
        raise ValueError("No valid observations after removing NaNs")

    # Loss differential
    d = loss_a - loss_b
    mean_diff = float(np.mean(d))

    # Newey-West lags
    if nw_lags is None:
        nw_lags = int(np.floor(n ** (1 / 3)))
    nw_lags = max(0, min(nw_lags, n - 1))  # Ensure valid range

    # HAC variance of the mean
    hac_var = newey_west_variance(d, nw_lags)

    # Variance of the mean
    var_mean = hac_var / n

    # Handle zero/near-zero variance (constant or near-constant differential)
    # Use a relative tolerance based on the magnitude of the differential
    var_threshold = 1e-20 * (1 + abs(mean_diff) ** 2)
    if var_mean <= var_threshold or np.isnan(var_mean):
        # Degenerate case: no variability in differential
        # If mean_diff is essentially zero, models are identical
        # Otherwise, we can't compute a valid test statistic
        return {
            "stat": 0.0 if abs(mean_diff) < 1e-15 else np.nan,
            "p_value": 1.0,
            "n_obs": n,
            "nw_lags": nw_lags,
            "mean_diff": mean_diff,
            "h": h,
        }

    # DM statistic
    se_mean = np.sqrt(var_mean)
    dm_stat = mean_diff / se_mean

    # Two-sided p-value from asymptotic normal
    p_value = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))

    return {
        "stat": float(dm_stat),
        "p_value": float(p_value),
        "n_obs": n,
        "nw_lags": nw_lags,
        "mean_diff": mean_diff,
        "h": h,
    }
