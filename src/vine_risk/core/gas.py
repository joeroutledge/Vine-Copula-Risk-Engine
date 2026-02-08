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


# Default filter kwargs for consistency between estimation and evaluation
GAS_FILTER_DEFAULTS = {
    "score_cap": 50.0,
    "opg_decay": 0.98,
    "opg_floor": 1e-3,
    "max_scaled_score": 4.0,
    "clip_theta": 3.8,
    "update_every": 1,  # Update GAS state every K observations (1=daily, 5=weekly)
}


def gas_filter(
    z1: np.ndarray,
    z2: np.ndarray,
    params: GASParameters,
    nu: float,
    theta_init: float = 0.0,
    opg_init: float = 1.0,
    score_cap: float = GAS_FILTER_DEFAULTS["score_cap"],
    opg_decay: float = GAS_FILTER_DEFAULTS["opg_decay"],
    opg_floor: float = GAS_FILTER_DEFAULTS["opg_floor"],
    max_scaled_score: float = GAS_FILTER_DEFAULTS["max_scaled_score"],
    clip_theta: float = GAS_FILTER_DEFAULTS["clip_theta"],
    update_every: int = GAS_FILTER_DEFAULTS["update_every"],
    t_offset: int = 0,
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
    update_every : int
        Update GAS state every K observations. Default 1 (daily update).
        Setting update_every=5 produces weekly updates. On non-update days,
        both theta and opg are carried forward unchanged. This reduces
        noise from high-frequency score fluctuations.
    t_offset : int
        Global time offset for update schedule. The update condition is
        ((t_offset + t) % update_every) == 0. This ensures that when
        filtering is split into segments (e.g., train then OOS), the
        weekly cadence is preserved across the boundary. Default 0.
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

    Notes
    -----
    When update_every > 1, the GAS state is only updated on days where
    ((t_offset + t) % update_every == 0). On other days:
    - Likelihood is still computed using current theta (for VaR/ES)
    - Score is NOT computed
    - OPG is NOT updated
    - Theta is carried forward unchanged

    This is useful for reducing noise from high-frequency score fluctuations
    while still producing daily VaR/ES forecasts.

    The t_offset parameter ensures cadence invariance across segmentation:
    if you split filtering at time k, running gas_filter on z[:k] with
    t_offset=0, then gas_filter on z[k:] with t_offset=k (and theta_init,
    opg_init from the first segment's final state), the concatenated
    theta_path will be identical to running gas_filter on the full z
    with t_offset=0.
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

        # Log-likelihood at current theta (always computed)
        ll_path[t] = log_student_t_copula_density(z1[t], z2[t], rho, nu)

        # Only update state on update days (using global index for cadence invariance)
        if (t_offset + t) % update_every == 0:
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
        # On non-update days, theta and opg are carried forward unchanged

    final_state = GASFilterState(theta=theta, opg=opg) if return_final_state else None
    return theta_path, ll_path, rho_path, final_state


# ============================================================================
# GAS NEGATIVE LOG-LIKELIHOOD (FOR ESTIMATION)
# ============================================================================

def gas_neg_loglik(
    z1: np.ndarray,
    z2: np.ndarray,
    omega: float,
    A: float,
    B: float,
    nu: float,
    theta_init: float = 0.0,
    opg_init: float = 1.0,
    **gas_filter_kwargs,
) -> float:
    """
    Compute negative log-likelihood for GAS model estimation.

    This function is the SINGLE SOURCE OF TRUTH for GAS estimation.
    It calls gas_filter() with the same defaults used for evaluation,
    ensuring mathematical consistency between fitting and forecasting.

    Parameters
    ----------
    z1, z2 : np.ndarray
        Arrays of t-quantiles
    omega, A, B : float
        GAS parameters (B is raw, pre-tanh)
    nu : float
        Degrees of freedom for t-copula
    theta_init : float
        Initial theta value
    opg_init : float
        Initial OPG value
    **gas_filter_kwargs
        Additional kwargs passed to gas_filter. If not provided,
        uses GAS_FILTER_DEFAULTS for consistency.

    Returns
    -------
    float
        Negative sum of log-likelihoods (for minimization)

    Notes
    -----
    By using gas_filter() internally, this ensures that the recursion
    used during estimation is IDENTICAL to the recursion used during
    out-of-sample evaluation. This eliminates estimation/evaluation
    mismatch bugs.
    """
    params = GASParameters(omega=omega, A=A, B=B)

    # Merge with defaults: explicit kwargs override defaults
    kwargs = {**GAS_FILTER_DEFAULTS, **gas_filter_kwargs}

    _, ll_path, _, _ = gas_filter(
        z1, z2, params, nu,
        theta_init=theta_init,
        opg_init=opg_init,
        return_final_state=False,
        **kwargs,
    )

    return -np.sum(ll_path)


# ============================================================================
# TAU-MODE GAS (LATENT KENDALL'S TAU PARAMETERIZATION)
# ============================================================================

def kendall_tau_to_rho(tau: float) -> float:
    """
    Convert Kendall's tau to Pearson rho for elliptical copulas.

    For elliptical copulas (Gaussian, Student-t):
        rho = sin(pi/2 * tau)

    This is an exact relationship, not an approximation.

    Parameters
    ----------
    tau : float
        Kendall's tau in (-1, 1)

    Returns
    -------
    float
        Pearson correlation rho
    """
    return np.sin(np.pi / 2.0 * tau)


def rho_to_kendall_tau(rho: float) -> float:
    """
    Convert Pearson rho to Kendall's tau for elliptical copulas.

    For elliptical copulas:
        tau = 2/pi * arcsin(rho)

    Parameters
    ----------
    rho : float
        Pearson correlation in (-1, 1)

    Returns
    -------
    float
        Kendall's tau
    """
    return 2.0 / np.pi * np.arcsin(rho)


def gas_score_kappa_t(
    z1: float,
    z2: float,
    kappa: float,
    nu: float,
) -> float:
    """
    GAS score in latent Kendall's tau space (kappa) for bivariate t-copula.

    The latent state is kappa, mapped to tau = tanh(kappa), then to
    rho = sin(pi/2 * tau). The score is computed via chain rule:

        d log c / d kappa = (d log c / d rho) * (d rho / d tau) * (d tau / d kappa)

    where:
        d tau / d kappa = 1 - tanh(kappa)^2 = sech(kappa)^2
        d rho / d tau = (pi/2) * cos(pi/2 * tau)

    Parameters
    ----------
    z1, z2 : float
        t-quantiles
    kappa : float
        Latent state (kappa = arctanh(tau))
    nu : float
        Degrees of freedom

    Returns
    -------
    float
        Score d log c / d kappa

    Notes
    -----
    This parameterization has interpretability advantages:
    - kappa is unconstrained on the real line
    - tau = tanh(kappa) is Kendall's tau, a rank correlation
    - The chain to rho uses the exact elliptical relationship
    """
    # kappa -> tau -> rho
    tau = np.tanh(kappa)
    tau = np.clip(tau, -0.99, 0.99)  # Safety clip
    rho = kendall_tau_to_rho(tau)
    rho = np.clip(rho, -0.995, 0.995)

    # Score in rho space
    score_rho = dlogc_drho_t(z1, z2, rho, nu)

    # Chain rule derivatives
    # d tau / d kappa = sech(kappa)^2 = 1 - tanh(kappa)^2
    dtau_dkappa = 1.0 - tau * tau

    # d rho / d tau = (pi/2) * cos(pi/2 * tau)
    drho_dtau = (np.pi / 2.0) * np.cos(np.pi / 2.0 * tau)

    # Full chain rule
    return score_rho * drho_dtau * dtau_dkappa


def gas_filter_tau_mode(
    z1: np.ndarray,
    z2: np.ndarray,
    params: GASParameters,
    nu: float,
    kappa_init: float = 0.0,
    opg_init: float = 1.0,
    score_cap: float = GAS_FILTER_DEFAULTS["score_cap"],
    opg_decay: float = GAS_FILTER_DEFAULTS["opg_decay"],
    opg_floor: float = GAS_FILTER_DEFAULTS["opg_floor"],
    max_scaled_score: float = GAS_FILTER_DEFAULTS["max_scaled_score"],
    clip_kappa: float = 3.8,
    update_every: int = GAS_FILTER_DEFAULTS["update_every"],
    t_offset: int = 0,
    return_final_state: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[GASFilterState]]:
    """
    GAS filter with latent Kendall's tau parameterization.

    The latent state is kappa, mapped to:
        tau = tanh(kappa)       (Kendall's tau)
        rho = sin(pi/2 * tau)   (Pearson correlation)

    This parameterization is equivalent to theta-mode for elliptical
    copulas but provides better interpretability (tau is a rank correlation)
    and may offer numerical stability benefits.

    Parameters
    ----------
    z1, z2 : np.ndarray
        Arrays of t-quantiles
    params : GASParameters
        GAS model parameters (omega, A, B interpreted in kappa-space)
    nu : float
        Degrees of freedom
    kappa_init : float
        Initial kappa value
    opg_init : float
        Initial OPG value
    score_cap, opg_decay, opg_floor, max_scaled_score, clip_kappa : float
        Filter tuning parameters (same semantics as theta-mode)
    update_every : int
        Update GAS state every K observations. Default 1 (daily).
    t_offset : int
        Global time offset for update schedule (same semantics as theta-mode).
    return_final_state : bool
        If True, return final state for OOS continuation

    Returns
    -------
    kappa_path : np.ndarray
        Filtered kappa values
    tau_path : np.ndarray
        Kendall's tau path (tau = tanh(kappa))
    rho_path : np.ndarray
        Correlation path (rho = sin(pi/2 * tau))
    ll_path : np.ndarray
        Log-likelihood contributions
    final_state : GASFilterState or None
        Final state for continuation

    Notes
    -----
    The returned final_state stores (kappa, opg) for consistency with
    the GASFilterState dataclass. When resuming, use kappa_init=state.theta.
    """
    from vine_risk.core.copulas import log_student_t_copula_density

    n = len(z1)

    kappa_path = np.zeros(n)
    tau_path = np.zeros(n)
    rho_path = np.zeros(n)
    ll_path = np.zeros(n)

    kappa = kappa_init
    opg = opg_init
    B_eff = params.B_effective

    for t in range(n):
        # Record current state
        kappa_path[t] = kappa
        tau = np.tanh(kappa)
        tau_path[t] = tau
        rho = kendall_tau_to_rho(tau)
        rho = np.clip(rho, -0.995, 0.995)
        rho_path[t] = rho

        # Log-likelihood at current rho (always computed)
        ll_path[t] = log_student_t_copula_density(z1[t], z2[t], rho, nu)

        # Only update state on update days (using global index for cadence invariance)
        if (t_offset + t) % update_every == 0:
            # Score in kappa space
            score = gas_score_kappa_t(z1[t], z2[t], kappa, nu)
            if not np.isfinite(score):
                score = 0.0
            score = np.clip(score, -score_cap, score_cap)

            # OPG update
            opg = opg_decay * opg + (1.0 - opg_decay) * score**2

            # Scaled score
            scaled_score = score / np.sqrt(max(opg, opg_floor))
            scaled_score = np.clip(scaled_score, -max_scaled_score, max_scaled_score)

            # GAS recursion
            kappa = params.omega + params.A * scaled_score + B_eff * kappa_path[t]
            kappa = np.clip(kappa, -clip_kappa, clip_kappa)
        # On non-update days, kappa and opg are carried forward unchanged

    final_state = GASFilterState(theta=kappa, opg=opg) if return_final_state else None
    return kappa_path, tau_path, rho_path, ll_path, final_state


def gas_neg_loglik_tau_mode(
    z1: np.ndarray,
    z2: np.ndarray,
    omega: float,
    A: float,
    B: float,
    nu: float,
    kappa_init: float = 0.0,
    opg_init: float = 1.0,
    **gas_filter_kwargs,
) -> float:
    """
    Negative log-likelihood for GAS model in tau-mode.

    Parameters
    ----------
    z1, z2 : np.ndarray
        Arrays of t-quantiles
    omega, A, B : float
        GAS parameters in kappa-space
    nu : float
        Degrees of freedom
    kappa_init : float
        Initial kappa value
    opg_init : float
        Initial OPG value
    **gas_filter_kwargs
        Additional kwargs for gas_filter_tau_mode

    Returns
    -------
    float
        Negative sum of log-likelihoods
    """
    params = GASParameters(omega=omega, A=A, B=B)

    kwargs = {**GAS_FILTER_DEFAULTS, **gas_filter_kwargs}
    # Rename clip_theta to clip_kappa for tau mode
    if "clip_theta" in kwargs:
        kwargs["clip_kappa"] = kwargs.pop("clip_theta")

    _, _, _, ll_path, _ = gas_filter_tau_mode(
        z1, z2, params, nu,
        kappa_init=kappa_init,
        opg_init=opg_init,
        return_final_state=False,
        **kwargs,
    )

    return -np.sum(ll_path)
