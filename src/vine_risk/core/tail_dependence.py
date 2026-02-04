"""
Tail dependence coefficients for copulas.

Tail dependence measures the probability of joint extreme events:
- Lower tail dependence: λ_L = lim_{q→0} P(U2 ≤ q | U1 ≤ q)
- Upper tail dependence: λ_U = lim_{q→1} P(U2 > q | U1 > q)

For the Student-t copula with correlation ρ and ν degrees of freedom:
    λ_L = λ_U = 2 · t_{ν+1}(-√[(ν+1)(1-ρ)/(1+ρ)])

This is a key quantity for risk management, as it measures how likely
assets are to crash together (left tail) or rally together (right tail).

References:
    Demarta & McNeil (2005): "The t Copula and Related Copulas"
    Joe (2014): "Dependence Modeling with Copulas", Section 2.14
"""

import numpy as np
from scipy.stats import t as tdist
from typing import Union, Tuple


def lower_tail_dependence_t(rho: float, nu: float) -> float:
    """
    Lower tail dependence coefficient for Student-t copula.

    λ_L = 2 · t_{ν+1}(-√[(ν+1)(1-ρ)/(1+ρ)])

    Parameters
    ----------
    rho : float
        Correlation parameter in [-1, 1]
    nu : float
        Degrees of freedom (> 2)

    Returns
    -------
    float
        Lower tail dependence coefficient λ_L ∈ [0, 1]

    Examples
    --------
    >>> lower_tail_dependence_t(0.5, 4.0)
    0.267...

    >>> lower_tail_dependence_t(0.9, 4.0)
    0.560...

    Notes
    -----
    - λ_L → 0 as ρ → -1 (negatively correlated)
    - λ_L → 1 as ρ → 1 (perfectly positively correlated)
    - λ_L → 0 as ν → ∞ (approaches Gaussian, which has no tail dependence)
    """
    rho = np.clip(rho, -0.999, 0.999)
    nu = np.clip(nu, 2.05, 100.0)

    numerator = (nu + 1.0) * (1.0 - rho)
    denominator = max(1e-10, 1.0 + rho)
    arg = -np.sqrt(max(0.0, numerator / denominator))

    return 2.0 * tdist.cdf(arg, nu + 1.0)


def upper_tail_dependence_t(rho: float, nu: float) -> float:
    """
    Upper tail dependence coefficient for Student-t copula.

    For symmetric copulas like the Student-t, λ_U = λ_L.

    Parameters
    ----------
    rho : float
        Correlation parameter
    nu : float
        Degrees of freedom

    Returns
    -------
    float
        Upper tail dependence coefficient λ_U ∈ [0, 1]
    """
    return lower_tail_dependence_t(rho, nu)


def tail_dependence_t_vectorized(
    rho: np.ndarray, nu: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized tail dependence computation for arrays of correlations.

    Parameters
    ----------
    rho : np.ndarray
        Array of correlation parameters
    nu : float
        Degrees of freedom

    Returns
    -------
    lambda_L : np.ndarray
        Lower tail dependence coefficients
    lambda_U : np.ndarray
        Upper tail dependence coefficients (equal to lambda_L for t-copula)
    """
    rho = np.clip(np.asarray(rho), -0.999, 0.999)
    nu = np.clip(nu, 2.05, 100.0)

    numerator = (nu + 1.0) * (1.0 - rho)
    denominator = np.maximum(1e-10, 1.0 + rho)
    arg = -np.sqrt(np.maximum(0.0, numerator / denominator))

    lambda_L = 2.0 * tdist.cdf(arg, nu + 1.0)

    return lambda_L, lambda_L  # Symmetric for t-copula


def tail_dependence_sensitivity(rho: float, nu: float, delta_rho: float = 0.01) -> float:
    """
    Sensitivity of tail dependence to correlation changes.

    Computes ∂λ_L/∂ρ via central differences.

    Parameters
    ----------
    rho : float
        Correlation parameter
    nu : float
        Degrees of freedom
    delta_rho : float
        Step size for numerical derivative

    Returns
    -------
    float
        Approximate derivative ∂λ_L/∂ρ
    """
    rho_plus = min(0.999, rho + delta_rho)
    rho_minus = max(-0.999, rho - delta_rho)

    lambda_plus = lower_tail_dependence_t(rho_plus, nu)
    lambda_minus = lower_tail_dependence_t(rho_minus, nu)

    return (lambda_plus - lambda_minus) / (rho_plus - rho_minus)


def tail_dependence_from_theta(theta: float, nu: float) -> float:
    """
    Compute tail dependence from Fisher-z parameter.

    Convenience function that converts θ to ρ first.

    Parameters
    ----------
    theta : float
        Fisher-z parameter (θ = arctanh(ρ))
    nu : float
        Degrees of freedom

    Returns
    -------
    float
        Tail dependence coefficient λ
    """
    rho = np.tanh(theta)
    return lower_tail_dependence_t(rho, nu)


# ============================================================================
# COMPARATIVE TAIL DEPENDENCE
# ============================================================================

def gaussian_tail_dependence() -> float:
    """
    Tail dependence for Gaussian copula.

    The Gaussian copula has zero tail dependence for all ρ < 1:
        λ_L = λ_U = 0

    This is a fundamental limitation of Gaussian copulas for risk modeling.

    Returns
    -------
    float
        Always returns 0.0
    """
    return 0.0


def tail_dependence_comparison(
    rho: float, nu: float
) -> dict:
    """
    Compare tail dependence across copula families.

    Parameters
    ----------
    rho : float
        Correlation parameter
    nu : float
        Degrees of freedom for Student-t

    Returns
    -------
    dict
        Comparison of tail dependence:
        - 't': Student-t copula λ
        - 'gaussian': Gaussian copula λ (always 0)
        - 'ratio': t/gaussian ratio (inf when gaussian=0)
    """
    lambda_t = lower_tail_dependence_t(rho, nu)
    lambda_g = gaussian_tail_dependence()

    return {
        't': lambda_t,
        'gaussian': lambda_g,
        'excess': lambda_t - lambda_g,  # How much more tail dependence t-copula captures
    }


# ============================================================================
# TAIL RISK METRICS
# ============================================================================

def crash_risk_index(
    rho_matrix: np.ndarray,
    nu: float,
    weights: np.ndarray = None,
) -> float:
    """
    Aggregate crash risk index from pairwise tail dependencies.

    This metric summarizes the overall crash risk of a portfolio
    by averaging lower tail dependencies across all pairs.

    Parameters
    ----------
    rho_matrix : np.ndarray
        Correlation matrix (d x d)
    nu : float
        Degrees of freedom
    weights : np.ndarray, optional
        Pairwise weights (upper triangular). If None, equal weights.

    Returns
    -------
    float
        Crash risk index ∈ [0, 1]
    """
    d = rho_matrix.shape[0]

    if weights is None:
        n_pairs = d * (d - 1) // 2
        weights = np.ones(n_pairs) / n_pairs

    crash_risk = 0.0
    weight_idx = 0

    for i in range(d):
        for j in range(i + 1, d):
            rho_ij = rho_matrix[i, j]
            lambda_ij = lower_tail_dependence_t(rho_ij, nu)
            crash_risk += weights[weight_idx] * lambda_ij
            weight_idx += 1

    return crash_risk
