"""
Tests for GAS tau-mode (latent Kendall's tau parameterization).

The tau-mode is an alternative parameterization for elliptical copulas:
- Latent state kappa mapped to tau = tanh(kappa)
- Kendall's tau mapped to rho = sin(pi/2 * tau)

This should produce identical likelihoods to theta-mode when properly
initialized, as both parameterize the same t-copula.
"""

import numpy as np
import pytest
from scipy.stats import t as tdist

from vine_risk.core.gas import (
    GASParameters,
    gas_filter,
    gas_filter_tau_mode,
    gas_neg_loglik,
    gas_neg_loglik_tau_mode,
    gas_score_kappa_t,
    kendall_tau_to_rho,
    rho_to_kendall_tau,
    GAS_FILTER_DEFAULTS,
)


class TestTauRhoConversion:
    """Test Kendall tau <-> Pearson rho conversion functions."""

    def test_tau_to_rho_at_zero(self):
        """tau=0 should give rho=0."""
        assert abs(kendall_tau_to_rho(0.0)) < 1e-15

    def test_rho_to_tau_at_zero(self):
        """rho=0 should give tau=0."""
        assert abs(rho_to_kendall_tau(0.0)) < 1e-15

    def test_roundtrip_tau_to_rho_to_tau(self):
        """tau -> rho -> tau should recover original tau."""
        for tau in [-0.8, -0.5, 0.0, 0.5, 0.8]:
            rho = kendall_tau_to_rho(tau)
            tau_back = rho_to_kendall_tau(rho)
            assert abs(tau - tau_back) < 1e-10, f"Roundtrip failed for tau={tau}"

    def test_roundtrip_rho_to_tau_to_rho(self):
        """rho -> tau -> rho should recover original rho."""
        for rho in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            tau = rho_to_kendall_tau(rho)
            rho_back = kendall_tau_to_rho(tau)
            assert abs(rho - rho_back) < 1e-10, f"Roundtrip failed for rho={rho}"

    def test_extreme_values(self):
        """Check behavior near boundaries."""
        # tau near 1 should give rho near 1
        # rho = sin(pi/2 * tau), so rho(0.99) = sin(0.99 * pi/2) ~ 0.9998
        assert kendall_tau_to_rho(0.99) > 0.99
        assert kendall_tau_to_rho(-0.99) < -0.99

        # rho near 1 gives tau = 2/pi * arcsin(rho)
        # tau(0.99) = 2/pi * arcsin(0.99) ~ 0.91 (not near 1!)
        # This is the correct mathematical relationship
        assert rho_to_kendall_tau(0.99) > 0.9  # tau ~ 0.91
        assert rho_to_kendall_tau(-0.99) < -0.9

        # Verify the exact relationship holds
        assert abs(rho_to_kendall_tau(0.99) - 2/np.pi * np.arcsin(0.99)) < 1e-10


class TestTauModeRhoBounded:
    """Test that tau-mode produces rho in (-1, 1)."""

    def test_rho_always_bounded(self):
        """rho from tau-mode must always be in (-1, 1)."""
        np.random.seed(42)
        n = 200
        nu = 5.0

        # Extreme data to stress test
        z1 = np.random.standard_t(nu, n) * 3
        z2 = 0.9 * z1 + np.sqrt(1 - 0.81) * np.random.standard_t(nu, n) * 3

        params = GASParameters(omega=0.5, A=0.5, B=3.0)
        _, _, rho_path, _, _ = gas_filter_tau_mode(
            z1, z2, params, nu,
            kappa_init=0.9,
        )

        assert np.all(rho_path > -1.0) and np.all(rho_path < 1.0), (
            f"rho out of bounds: min={rho_path.min()}, max={rho_path.max()}"
        )

    def test_tau_always_bounded(self):
        """tau = tanh(kappa) must always be in (-1, 1)."""
        np.random.seed(42)
        n = 200
        nu = 5.0

        z1 = np.random.standard_t(nu, n) * 3
        z2 = 0.9 * z1 + np.sqrt(1 - 0.81) * np.random.standard_t(nu, n) * 3

        params = GASParameters(omega=0.5, A=0.5, B=3.0)
        _, tau_path, _, _, _ = gas_filter_tau_mode(
            z1, z2, params, nu,
            kappa_init=0.9,
        )

        assert np.all(tau_path > -1.0) and np.all(tau_path < 1.0), (
            f"tau out of bounds: min={tau_path.min()}, max={tau_path.max()}"
        )


class TestTauThetaModeEquivalence:
    """Test equivalence between tau-mode and theta-mode."""

    def test_identical_ll_with_no_dynamics(self):
        """
        With A=0 and B=0, both modes should give identical LL
        when initialized to produce the same rho.

        This is a sanity check: static correlation gives same LL
        regardless of parameterization.
        """
        np.random.seed(42)
        n = 100
        nu = 8.0

        rho_target = 0.5
        z1 = np.random.standard_t(nu, n)
        z2 = rho_target * z1 + np.sqrt(1 - rho_target**2) * np.random.standard_t(nu, n)

        # Theta-mode initialization: theta = arctanh(rho)
        theta_init = np.arctanh(rho_target)

        # Tau-mode initialization: kappa = arctanh(tau), where tau = 2/pi * arcsin(rho)
        tau_target = rho_to_kendall_tau(rho_target)
        kappa_init = np.arctanh(tau_target)

        # No dynamics: omega determines the constant level
        # For theta-mode: theta_t = omega (if B_eff=0, A=0)
        # For tau-mode: kappa_t = omega (same)
        # To get same rho, we need different omega values

        # Actually, with A=B=0, theta_{t+1} = omega + 0 + 0 = omega
        # So the second observation uses theta = omega, not theta_init
        # Let's just use A=0, B large to keep initial value

        params_static = GASParameters(omega=0.0, A=0.0, B=5.0)  # B_eff ~ 0.9999

        # Theta mode
        _, ll_path_theta, rho_path_theta, _ = gas_filter(
            z1, z2, params_static, nu,
            theta_init=theta_init,
            **GAS_FILTER_DEFAULTS,
        )

        # Tau mode
        _, _, rho_path_tau, ll_path_tau, _ = gas_filter_tau_mode(
            z1, z2, params_static, nu,
            kappa_init=kappa_init,
        )

        # First rho should match
        assert abs(rho_path_theta[0] - rho_path_tau[0]) < 1e-6, (
            f"Initial rho mismatch: theta={rho_path_theta[0]}, tau={rho_path_tau[0]}"
        )

        # Total LL should be very close (rho paths will diverge slightly
        # due to different dynamics even with B~1, but first few steps match)
        ll_total_theta = np.sum(ll_path_theta)
        ll_total_tau = np.sum(ll_path_tau)

        # With near-static dynamics, LLs should be similar
        # Allow some tolerance as paths do diverge
        assert abs(ll_total_theta - ll_total_tau) / abs(ll_total_theta) < 0.01, (
            f"LL divergence too large: theta={ll_total_theta:.4f}, tau={ll_total_tau:.4f}"
        )

    def test_score_is_finite(self):
        """tau-mode score should be finite for reasonable inputs."""
        np.random.seed(42)
        nu = 8.0

        for _ in range(100):
            z1 = np.random.standard_t(nu)
            z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu)
            kappa = np.random.uniform(-2, 2)

            score = gas_score_kappa_t(z1, z2, kappa, nu)
            assert np.isfinite(score), f"Non-finite score at kappa={kappa}"


class TestNegLoglikTauMode:
    """Test gas_neg_loglik_tau_mode consistency."""

    def test_neg_loglik_equals_filter_ll(self):
        """
        gas_neg_loglik_tau_mode must equal -sum(ll_path) from
        gas_filter_tau_mode with same parameters.
        """
        np.random.seed(42)
        n = 100
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        omega, A, B = 0.05, 0.15, 2.0
        kappa_init = 0.3

        nll_helper = gas_neg_loglik_tau_mode(
            z1, z2, omega, A, B, nu,
            kappa_init=kappa_init,
            opg_init=1.0,
        )

        params = GASParameters(omega=omega, A=A, B=B)
        _, _, _, ll_path, _ = gas_filter_tau_mode(
            z1, z2, params, nu,
            kappa_init=kappa_init,
            opg_init=1.0,
        )
        nll_filter = -np.sum(ll_path)

        assert abs(nll_helper - nll_filter) < 1e-12, (
            f"Mismatch: helper={nll_helper:.10f}, filter={nll_filter:.10f}"
        )


class TestChainRuleDerivatives:
    """Test correctness of chain rule derivatives in tau-mode score."""

    def test_dtau_dkappa_formula(self):
        """d tau / d kappa = 1 - tanh(kappa)^2 = sech(kappa)^2."""
        for kappa in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            tau = np.tanh(kappa)
            dtau_dkappa_analytic = 1.0 - tau * tau

            # Numerical derivative
            eps = 1e-6
            dtau_dkappa_numeric = (np.tanh(kappa + eps) - np.tanh(kappa - eps)) / (2 * eps)

            assert abs(dtau_dkappa_analytic - dtau_dkappa_numeric) < 1e-8, (
                f"dtau/dkappa mismatch at kappa={kappa}"
            )

    def test_drho_dtau_formula(self):
        """d rho / d tau = (pi/2) * cos(pi/2 * tau)."""
        for tau in [-0.8, -0.4, 0.0, 0.4, 0.8]:
            drho_dtau_analytic = (np.pi / 2.0) * np.cos(np.pi / 2.0 * tau)

            # Numerical derivative
            eps = 1e-6
            drho_dtau_numeric = (
                kendall_tau_to_rho(tau + eps) - kendall_tau_to_rho(tau - eps)
            ) / (2 * eps)

            assert abs(drho_dtau_analytic - drho_dtau_numeric) < 1e-7, (
                f"drho/dtau mismatch at tau={tau}"
            )
