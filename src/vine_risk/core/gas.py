"""
Generalized Autoregressive Score (GAS) dynamics for copula parameters.

This module implements GAS dynamics following Creal, Koopman & Lucas (2013):
- Analytical score computation for Student-t copulas
- Score scaling via OPG (outer product of gradients)
- GAS recursion

Mathematical Specification:
    theta_{t+1} = omega + A * s_tilde_t + B * theta_t

where:
    - theta_t is the Fisher-z transformed correlation (theta = arctanh(rho))
    - s_tilde_t is the scaled score: s_t / sqrt(I_t)
    - s_t is the analytical score: d log c / d theta
    - I_t is the Fisher information (approximated via OPG)

References:
    Creal, D., Koopman, S.J., & Lucas, A. (2013).
    "Generalized Autoregressive Score Models with Applications"
    Journal of Applied Econometrics, 28(5), 777-795.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from vine_risk.core.copulas import (
    t_copula_logpdf_terms,
    SAFE_RHO_MAX,
    MIN_DENOM,
)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GASParameters:
    """
    Parameters for GAS(1,1) recursion in Fisher-z (link) space.

    Attributes
    ----------
    omega : float
        Location/intercept parameter. Controls the unconditional mean
        of theta_t when the process is stationary.
    A : float
        Score sensitivity parameter. Controls how strongly the parameter
        responds to new information (innovations).
    B : float
        Autoregressive parameter (raw, pre-tanh transformation).
        The effective persistence is tanh(B) in (-1, 1).
    """
    omega: float = 0.0
    A: float = 0.1
    B: float = 2.0  # tanh(2.0) ~ 0.964

    @property
    def B_effective(self) -> float:
        """Effective persistence parameter: tanh(B) in (-1, 1)."""
        return np.tanh(self.B)

    @property
    def is_stationary(self) -> bool:
        """Check if the GAS process is stationary (|B_eff| < 1)."""
        return abs(self.B_effective) < 1.0

    def unconditional_mean(self) -> float:
        """
        Compute unconditional mean of theta_t under stationarity.

        E[theta] = omega / (1 - B_eff)  when |B_eff| < 1
        """
        if not self.is_stationary:
            return np.nan
        return self.omega / (1.0 - self.B_effective)


# ============================================================================
# SCORE FUNCTIONS
# ============================================================================

def dlogc_drho_t(z1: float, z2: float, rho: float, nu: float) -> float:
    """
    Derivative of log t-copula density with respect to correlation rho.

    This is the score function in the original (rho) parameterization:
        d log c / d rho

    Parameters
    ----------
    z1, z2 : float
        t-quantiles (z_i = t_nu^{-1}(u_i))
    rho : float
        Correlation parameter
    nu : float
        Degrees of freedom

    Returns
    -------
    float
        Score d log c / d rho
    """
    rho = float(np.clip(rho, -0.99, 0.99))
    nu = float(np.clip(nu, 2.1, 50.0))

    q, denom, _ = t_copula_logpdf_terms(z1, z2, rho, nu)

    # Derivative of quadratic form q w.r.t. rho
    num = -2.0 * z1 * z2 * denom + 2.0 * rho * (
        z1 * z1 - 2.0 * rho * z1 * z2 + z2 * z2
    )
    dq_drho = num / (denom * denom)

    # Score: d log c / d rho = rho/(1-rho^2) - (nu+2)/(2(nu+q)) * dq/drho
    return (rho / denom) - ((nu + 2.0) / (2.0 * (nu + q))) * dq_drho


def gas_score_theta_t(z1: float, z2: float, theta: float, nu: float) -> float:
    """
    GAS score in Fisher-z space (theta = arctanh(rho)) for bivariate t-copula.

    The score in theta-space is obtained via the chain rule:
        s_theta = (d log c / d rho) * (1 - rho^2)

    Parameters
    ----------
    z1, z2 : float
        t-quantiles
    theta : float
        Fisher-z parameter (theta = arctanh(rho))
    nu : float
        Degrees of freedom

    Returns
    -------
    float
        Score d log c / d theta
    """
    rho = np.tanh(theta)
    dtheta_to_rho = 1.0 - rho * rho  # Jacobian: d rho / d theta = 1 - rho^2
    return dlogc_drho_t(z1, z2, rho, nu) * dtheta_to_rho


def fisher_information_theta_t(rho: float, nu: float) -> float:
    """
    Fisher information for theta in the Student-t copula.

    The Fisher information in rho-space is:
        I_rho = (nu + 2) / [(nu + 3)(1 - rho^2)]

    Converting to theta-space:
        I_theta = I_rho * (d rho / d theta)^2 = I_rho * (1 - rho^2)^2
    """
    rho = np.clip(rho, -SAFE_RHO_MAX, SAFE_RHO_MAX)
    one_minus_rho2 = 1.0 - rho * rho

    info_rho = (nu + 2.0) / ((nu + 3.0) * one_minus_rho2)
    info_theta = one_minus_rho2 ** 2 * info_rho

    return max(1e-8, info_theta)


def scale_opg(
    score: float,
    opg: float,
    score_cap: float = 25.0,
    opg_floor: float = 1e-3,
) -> float:
    """
    Scale score by OPG (outer product of gradients) estimate.

    The scaled score is: s_tilde_t = s_t / sqrt(I_t)
    """
    score = np.clip(score, -score_cap, score_cap)
    opg = max(opg, opg_floor)
    return score / np.sqrt(opg)


# ============================================================================
# GAS FILTERING
# ============================================================================

@dataclass
class GASFilterState:
    """
    State container for GAS filter continuation.

    Allows resuming GAS filtering from a saved state for true OOS evaluation.

    Attributes
    ----------
    theta : float
        Current theta value
    opg : float
        Current OPG (outer product of gradients) estimate
    """
    theta: float
    opg: float


def compute_score_and_info(
    z1: float,
    z2: float,
    theta: float,
    nu: float,
    opg_floor: float = 1e-3,
) -> Tuple[float, float]:
    """Compute score and Fisher information for Student-t copula."""
    rho = np.tanh(theta)
    rho = np.clip(rho, -SAFE_RHO_MAX, SAFE_RHO_MAX)

    score_theta = gas_score_theta_t(z1, z2, theta, nu)

    if not np.isfinite(score_theta):
        score_theta = 0.0

    info_theta = fisher_information_theta_t(rho, nu)

    return score_theta, max(opg_floor, info_theta)


def gas_filter(
    z1: np.ndarray,
    z2: np.ndarray,
    params: GASParameters,
    nu: float,
    theta_init: float = 0.0,
    opg_init: float = 1.0,
    score_cap: float = 50.0,
    opg_decay: float = 0.98,
    opg_floor: float = 1e-3,
    max_scaled_score: float = 4.0,
    clip_theta: float = 3.8,
    return_final_state: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[GASFilterState]]:
    """
    Run GAS filter over a time series.

    This is the SINGLE SOURCE OF TRUTH for GAS recursion. Used for both
    in-sample fitting and out-of-sample evaluation.

    Recursion Equation (predict-then-update convention):
    -------------------------------------------------------
    At time t:
    1. Use theta_t to evaluate LL: ll_t = log_copula(u_t | theta_t)
    2. Compute score: s_t = d log_copula / d theta at (u_t, theta_t)
    3. Update for next step: theta_{t+1} = omega + A * s_tilde_t + B * theta_t

    Parameters
    ----------
    z1, z2 : np.ndarray
        Arrays of t-quantiles
    params : GASParameters
        GAS model parameters
    nu : float
        Degrees of freedom
    theta_init : float
        Initial theta value (theta_0 for first observation)
    opg_init : float
        Initial OPG value (for continuing from saved state)
    score_cap : float
        Cap for raw scores
    opg_decay : float
        EWMA decay for OPG
    opg_floor : float
        Floor for OPG
    max_scaled_score : float
        Cap for scaled scores
    clip_theta : float
        Clipping bound for theta
    return_final_state : bool
        If True, return the final state for OOS continuation

    Returns
    -------
    theta_path : np.ndarray
        Filtered theta values (theta_t used for evaluating ll_t)
    ll_path : np.ndarray
        Log-likelihood contributions
    rho_path : np.ndarray
        Correlation values (rho = tanh(theta))
    final_state : GASFilterState or None
        Final state for continuation (if return_final_state=True)
    """
    from vine_risk.core.copulas import log_student_t_copula_density

    n = len(z1)

    theta_path = np.zeros(n)
    ll_path = np.zeros(n)
    rho_path = np.zeros(n)

    theta = theta_init
    opg = opg_init
    B_eff = params.B_effective

    for t in range(n):
        # Record current theta BEFORE update (predict-then-update)
        theta_path[t] = theta
        rho = np.tanh(theta)
        rho_path[t] = rho

        # Log-likelihood at current theta
        ll_path[t] = log_student_t_copula_density(z1[t], z2[t], rho, nu)

        # Score (computed after observing u_t)
        score, info = compute_score_and_info(z1[t], z2[t], theta, nu, opg_floor)
        score = np.clip(score, -score_cap, score_cap)

        # OPG update (EWMA of squared scores)
        opg = opg_decay * opg + (1.0 - opg_decay) * score**2

        # Scaled score
        scaled_score = score / np.sqrt(max(opg, opg_floor))
        scaled_score = np.clip(scaled_score, -max_scaled_score, max_scaled_score)

        # GAS recursion: update for NEXT time step
        theta = (
            params.omega
            + params.A * scaled_score
            + B_eff * theta_path[t]
        )
        theta = np.clip(theta, -clip_theta, clip_theta)

    final_state = GASFilterState(theta=theta, opg=opg) if return_final_state else None
    return theta_path, ll_path, rho_path, final_state
