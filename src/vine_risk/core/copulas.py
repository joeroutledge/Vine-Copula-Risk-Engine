"""
Student-t copula density and conditional distribution functions.

This module provides the mathematical primitives for bivariate Student-t copulas:
- Log-density computation (analytically exact)
- h-functions (conditional CDF)
- Inverse h-functions (for simulation via Rosenblatt transform)

Mathematical References:
- Demarta & McNeil (2005): "The t Copula and Related Copulas"
- Joe (2014): "Dependence Modeling with Copulas", Chapter 4
"""

import numpy as np
from scipy.stats import t as tdist
from scipy.special import gammaln
from scipy.optimize import brentq
from typing import Union

# ============================================================================
# NUMERICAL CONSTANTS
# ============================================================================

SAFE_RHO_MAX = 0.995   # Maximum correlation to avoid numerical instability
MIN_DENOM = 1e-6       # Floor for denominators to prevent division by zero
EPS = 1e-8             # Small epsilon for clipping uniform variates


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clip01(u: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Clip values to the open unit interval (eps, 1-eps).

    This prevents numerical issues when computing quantiles at boundaries.

    Parameters
    ----------
    u : np.ndarray
        Values to clip
    eps : float
        Margin from 0 and 1

    Returns
    -------
    np.ndarray
        Clipped values in (eps, 1-eps)
    """
    return np.clip(u, eps, 1.0 - eps)


def atanh_safe(rho: float, max_rho: float = SAFE_RHO_MAX) -> float:
    """
    Safe arctanh transformation for correlation to Fisher-z space.

    The Fisher-z transformation θ = arctanh(ρ) maps correlations from
    [-1, 1] to the real line, providing an unconstrained parameterization
    for optimization.

    Parameters
    ----------
    rho : float
        Correlation coefficient
    max_rho : float
        Maximum absolute correlation (for numerical stability)

    Returns
    -------
    float
        Fisher-z transformed value θ = arctanh(ρ)
    """
    return np.arctanh(np.clip(rho, -max_rho, max_rho))


def tanh_safe(theta: float, max_rho: float = SAFE_RHO_MAX) -> float:
    """
    Safe inverse Fisher-z transformation from link space to correlation.

    Parameters
    ----------
    theta : float
        Fisher-z parameter
    max_rho : float
        Maximum absolute correlation (for clipping output)

    Returns
    -------
    float
        Correlation ρ = tanh(θ), clipped to [-max_rho, max_rho]
    """
    return np.clip(np.tanh(theta), -max_rho, max_rho)


# ============================================================================
# STUDENT-T COPULA DENSITY
# ============================================================================

def log_student_t_copula_density(z1: float, z2: float, rho: float, nu: float) -> float:
    """
    Exact log-density of bivariate Student-t copula at t-quantiles (z1, z2).

    The copula density is:
        c(u1, u2) = f_t^(2)(z1, z2; ρ, ν) / [f_t^(1)(z1; ν) · f_t^(1)(z2; ν)]

    where z_i = t_ν^{-1}(u_i) are the t-quantiles.

    This implementation uses numerically stable formulations:
    - log1p for small arguments
    - Proper handling of the quadratic form

    Parameters
    ----------
    z1 : float
        First t-quantile (z1 = t_ν^{-1}(u1))
    z2 : float
        Second t-quantile (z2 = t_ν^{-1}(u2))
    rho : float
        Correlation parameter in [-1, 1]
    nu : float
        Degrees of freedom (> 2 for finite variance)

    Returns
    -------
    float
        Log copula density log c(u1, u2)

    References
    ----------
    Demarta & McNeil (2005), Equation (2.1)
    """
    rho = np.clip(rho, -SAFE_RHO_MAX, SAFE_RHO_MAX)
    nu = np.clip(nu, 2.05, 100.0)

    one_minus_rho2 = max(MIN_DENOM, 1.0 - rho * rho)
    quadratic_form = z1 * z1 - 2.0 * rho * z1 * z2 + z2 * z2
    q = quadratic_form / one_minus_rho2

    # Numerically stable log(1 + q/nu)
    if q < nu:
        log1p_q_nu = np.log1p(q / nu)
    else:
        log1p_q_nu = np.log(q / nu) + np.log1p(nu / q)

    # Bivariate t-density (log scale)
    log_biv = (
        gammaln((nu + 2.0) / 2.0)
        - gammaln(nu / 2.0)
        - np.log(np.pi * nu)
        - 0.5 * np.log(one_minus_rho2)
        - 0.5 * (nu + 2.0) * log1p_q_nu
    )

    # Univariate t-density (log scale)
    def log_univ(z: float) -> float:
        s = z * z / nu
        if s < 1.0:
            l1p = np.log1p(s)
        else:
            l1p = np.log(s) + np.log1p(1.0 / s)
        return (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(np.pi * nu)
            - 0.5 * (nu + 1.0) * l1p
        )

    # Copula density = bivariate / (marginal1 * marginal2)
    return log_biv - log_univ(z1) - log_univ(z2)


def t_copula_logpdf_terms(
    z1: float, z2: float, rho: float, nu: float
) -> tuple[float, float, float]:
    """
    Return core quantities for bivariate Student-t copula log-density.

    Useful for computing derivatives and information quantities.

    Parameters
    ----------
    z1, z2 : float
        Quantiles from the t_ν distribution
    rho : float
        Linear correlation parameter
    nu : float
        Degrees of freedom

    Returns
    -------
    q : float
        Quadratic form (z1² - 2ρz1z2 + z2²)/(1-ρ²)
    denom : float
        1 - ρ²
    log_c : float
        Copula log-density (core terms, marginals omitted)
    """
    rho = float(np.clip(rho, -0.99, 0.99))
    nu = float(np.clip(nu, 2.1, 50.0))
    denom = 1.0 - rho * rho
    q = (z1 * z1 - 2.0 * rho * z1 * z2 + z2 * z2) / denom
    log_c = -0.5 * np.log(denom) - 0.5 * (nu + 2.0) * np.log(1.0 + q / nu)
    return q, denom, log_c


# ============================================================================
# H-FUNCTIONS (CONDITIONAL CDF)
# ============================================================================

def student_t_h_function(
    u1: np.ndarray, u2: np.ndarray, rho: float, nu: float
) -> np.ndarray:
    """
    Compute h-function (conditional CDF) for Student-t copula.

    The h-function is defined as:
        h(u2|u1) = P(U2 ≤ u2 | U1 = u1) = C_{2|1}(u2|u1)

    For the Student-t copula with correlation ρ and ν degrees of freedom:
        h(u2|u1) = t_{ν+1}((z2 - ρz1) / √[(1-ρ²)(ν+z1²)/(ν+1)])

    where z_i = t_ν^{-1}(u_i).

    Parameters
    ----------
    u1 : np.ndarray
        Conditioning variable values in (0, 1)
    u2 : np.ndarray
        Variable to condition on, values in (0, 1)
    rho : float
        Copula correlation parameter
    nu : float
        Degrees of freedom

    Returns
    -------
    np.ndarray
        h(u2|u1) values in (0, 1)

    References
    ----------
    Aas et al. (2009), Equation (10)
    """
    u1 = clip01(np.asarray(u1))
    u2 = clip01(np.asarray(u2))

    z1 = tdist.ppf(u1, nu)
    z2 = tdist.ppf(u2, nu)

    numerator = z2 - rho * z1
    denominator = np.sqrt((1 - rho**2) * (nu + z1**2) / (nu + 1))
    arg = numerator / np.maximum(denominator, MIN_DENOM)

    return tdist.cdf(arg, nu + 1)


def student_t_h_inverse(
    u_cond: np.ndarray,
    u_given: np.ndarray,
    rho: float,
    nu: float,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Inverse h-function for Student-t copula.

    Solves h(u2|u1) = u_cond for u2, which is needed for the inverse
    Rosenblatt transform (simulation from the copula).

    Uses Brent's method for root finding since no closed form exists.

    Parameters
    ----------
    u_cond : np.ndarray
        Target h-function values
    u_given : np.ndarray
        Conditioning variable values
    rho : float
        Copula correlation parameter
    nu : float
        Degrees of freedom
    tol : float
        Tolerance for root finding

    Returns
    -------
    np.ndarray
        u2 values satisfying h(u2|u1) = u_cond
    """
    u_given = clip01(np.asarray(u_given))
    u_cond = clip01(np.asarray(u_cond))
    result = np.zeros_like(u_cond)

    for i in range(len(u_cond)):

        def equation(u2: float) -> float:
            return (
                student_t_h_function(
                    np.array([u_given[i]]), np.array([u2]), rho, nu
                )[0]
                - u_cond[i]
            )

        try:
            result[i] = brentq(equation, EPS, 1 - EPS, xtol=tol)
        except ValueError:
            # If root finding fails, use the target as approximation
            result[i] = u_cond[i]

    return result


# ============================================================================
# VECTORIZED OPERATIONS
# ============================================================================

def log_student_t_copula_density_vectorized(
    z1: np.ndarray, z2: np.ndarray, rho: float, nu: float
) -> np.ndarray:
    """
    Vectorized log-density of bivariate Student-t copula.

    Parameters
    ----------
    z1, z2 : np.ndarray
        Arrays of t-quantiles
    rho : float
        Correlation parameter
    nu : float
        Degrees of freedom

    Returns
    -------
    np.ndarray
        Log copula densities for each pair
    """
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)
    rho = np.clip(rho, -SAFE_RHO_MAX, SAFE_RHO_MAX)
    nu = np.clip(nu, 2.05, 100.0)

    one_minus_rho2 = max(MIN_DENOM, 1.0 - rho * rho)
    quadratic_form = z1 * z1 - 2.0 * rho * z1 * z2 + z2 * z2
    q = quadratic_form / one_minus_rho2

    # Bivariate log-density
    log_biv = (
        gammaln((nu + 2.0) / 2.0)
        - gammaln(nu / 2.0)
        - np.log(np.pi * nu)
        - 0.5 * np.log(one_minus_rho2)
        - 0.5 * (nu + 2.0) * np.log1p(q / nu)
    )

    # Univariate log-densities
    def log_univ_vec(z: np.ndarray) -> np.ndarray:
        return (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(np.pi * nu)
            - 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
        )

    return log_biv - log_univ_vec(z1) - log_univ_vec(z2)
