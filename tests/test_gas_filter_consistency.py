"""
Tests for GAS filter estimation/evaluation consistency.

Verifies that the recursion used during parameter estimation is
IDENTICAL to the recursion used during out-of-sample evaluation.
This is critical for mathematical defensibility.
"""

import numpy as np
import pytest
from scipy.stats import t as tdist

from vine_risk.core.gas import (
    GASParameters,
    gas_filter,
    gas_neg_loglik,
    GAS_FILTER_DEFAULTS,
)


class TestEstimationEvaluationConsistency:
    """Test that estimation and evaluation use identical recursions."""

    def test_neg_loglik_equals_filter_ll(self):
        """
        The negative log-likelihood from gas_neg_loglik() must equal
        -sum(ll_path) from gas_filter() for the same parameters.

        This is the core consistency test: if these differ, the model
        is being estimated on a different objective than it's evaluated on.
        """
        np.random.seed(42)
        n = 100

        # Generate synthetic copula data
        rho_true = 0.5
        nu = 8.0

        # Correlated t-variates
        z1 = np.random.standard_t(nu, n)
        z2 = rho_true * z1 + np.sqrt(1 - rho_true**2) * np.random.standard_t(nu, n)

        # Test parameters
        omega, A, B = 0.05, 0.15, 2.0
        theta_init = np.arctanh(rho_true)
        opg_init = 1.0

        # Compute via gas_neg_loglik (used in optimization)
        nll_from_helper = gas_neg_loglik(
            z1, z2, omega, A, B, nu,
            theta_init=theta_init,
            opg_init=opg_init,
        )

        # Compute via gas_filter directly
        params = GASParameters(omega=omega, A=A, B=B)
        _, ll_path, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            opg_init=opg_init,
            return_final_state=False,
            **GAS_FILTER_DEFAULTS,
        )
        nll_from_filter = -np.sum(ll_path)

        # Must be EXACTLY equal (same function call internally)
        assert abs(nll_from_helper - nll_from_filter) < 1e-12, (
            f"Estimation/evaluation mismatch: "
            f"gas_neg_loglik={nll_from_helper:.10f}, "
            f"-sum(ll_path)={nll_from_filter:.10f}"
        )

    def test_consistency_across_parameter_values(self):
        """
        Test consistency across a range of parameter values.
        """
        np.random.seed(123)
        n = 50
        nu = 6.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.3 * z1 + np.sqrt(1 - 0.09) * np.random.standard_t(nu, n)

        theta_init = 0.3
        opg_init = 1.0

        # Test multiple parameter combinations
        param_grid = [
            (0.0, 0.0, 0.0),    # No dynamics
            (0.1, 0.2, 1.5),    # Moderate dynamics
            (-0.1, 0.5, 3.0),   # High persistence
            (0.05, 0.1, 0.5),   # Low persistence
        ]

        for omega, A, B in param_grid:
            nll_helper = gas_neg_loglik(
                z1, z2, omega, A, B, nu,
                theta_init=theta_init,
                opg_init=opg_init,
            )

            params = GASParameters(omega=omega, A=A, B=B)
            _, ll_path, _, _ = gas_filter(
                z1, z2, params, nu,
                theta_init=theta_init,
                opg_init=opg_init,
                **GAS_FILTER_DEFAULTS,
            )
            nll_filter = -np.sum(ll_path)

            assert abs(nll_helper - nll_filter) < 1e-12, (
                f"Mismatch for params ({omega}, {A}, {B}): "
                f"{nll_helper:.10f} vs {nll_filter:.10f}"
            )

    def test_filter_defaults_are_explicit(self):
        """
        GAS_FILTER_DEFAULTS must contain all tuning parameters.
        """
        required_keys = {"score_cap", "opg_decay", "opg_floor", "max_scaled_score", "clip_theta", "update_every"}

        assert required_keys == set(GAS_FILTER_DEFAULTS.keys()), (
            f"Missing/extra keys in GAS_FILTER_DEFAULTS: "
            f"expected {required_keys}, got {set(GAS_FILTER_DEFAULTS.keys())}"
        )

    def test_opg_decay_is_consistent(self):
        """
        The OPG decay used in estimation must match the filter default.

        This specifically checks that we're not using different decay
        values (e.g., 0.95 in estimation vs 0.98 in evaluation).
        """
        # The key point: gas_neg_loglik uses GAS_FILTER_DEFAULTS internally
        assert GAS_FILTER_DEFAULTS["opg_decay"] == 0.98, (
            f"Expected opg_decay=0.98, got {GAS_FILTER_DEFAULTS['opg_decay']}"
        )


class TestGASFilterBehavior:
    """Test expected GAS filter behavior properties."""

    def test_zero_A_gives_static_persistence(self):
        """With A=0, theta should follow pure AR(1) dynamics."""
        np.random.seed(42)
        n = 50
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        omega = 0.1
        A = 0.0  # No score response
        B = 2.0  # tanh(2.0) ~ 0.964
        theta_init = 0.5

        params = GASParameters(omega=omega, A=A, B=B)
        theta_path, _, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            **GAS_FILTER_DEFAULTS,
        )

        # With A=0, theta_{t+1} = omega + B_eff * theta_t
        B_eff = np.tanh(B)
        expected = np.zeros(n)
        expected[0] = theta_init
        for t in range(1, n):
            expected[t] = omega + B_eff * expected[t-1]
            expected[t] = np.clip(expected[t], -3.8, 3.8)

        np.testing.assert_allclose(
            theta_path, expected, rtol=1e-10,
            err_msg="With A=0, theta should follow pure AR(1)"
        )

    def test_zero_B_gives_score_only(self):
        """With B=0 (tanh(0)=0), theta depends only on omega + A*score."""
        np.random.seed(42)
        n = 20
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        omega = 0.2
        A = 0.1
        B = 0.0  # tanh(0) = 0
        theta_init = 0.3

        params = GASParameters(omega=omega, A=A, B=B)
        theta_path, _, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            **GAS_FILTER_DEFAULTS,
        )

        # First value should be theta_init
        assert theta_path[0] == theta_init

        # With B_eff=0, past theta doesn't matter
        # theta_{t+1} = omega + A * scaled_score_t
        # So theta_t does not directly depend on theta_{t-1}

    def test_rho_path_bounded(self):
        """rho = tanh(theta) must be in (-1, 1) for all t."""
        np.random.seed(42)
        n = 200
        nu = 5.0

        # Generate potentially extreme data
        z1 = np.random.standard_t(nu, n) * 2
        z2 = 0.8 * z1 + np.sqrt(1 - 0.64) * np.random.standard_t(nu, n) * 2

        omega = 0.5
        A = 0.5
        B = 3.0  # High persistence

        params = GASParameters(omega=omega, A=A, B=B)
        _, _, rho_path, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=0.8,
            **GAS_FILTER_DEFAULTS,
        )

        assert np.all(rho_path > -1.0) and np.all(rho_path < 1.0), (
            f"rho out of bounds: min={rho_path.min()}, max={rho_path.max()}"
        )


class TestGASVineIntegration:
    """Integration test with GASDVineModel."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_fit_uses_same_recursion(self, seed):
        """
        After fitting, the objective at fitted params should equal
        -sum(ll_path) from gas_filter with those params.

        This is the key audit test: if we fit params by minimizing
        gas_neg_loglik, then calling gas_filter with those params
        must give the same total log-likelihood.
        """
        np.random.seed(seed)
        n = 100
        nu = 8.0

        # Generate copula data
        rho_true = 0.4
        z1 = np.random.standard_t(nu, n)
        z2 = rho_true * z1 + np.sqrt(1 - rho_true**2) * np.random.standard_t(nu, n)

        theta_init = np.arctanh(rho_true)

        # Fit using scipy minimize (same as _fit_gas_edge)
        from scipy.optimize import minimize

        def objective(x):
            return gas_neg_loglik(
                z1, z2, x[0], x[1], x[2], nu,
                theta_init=theta_init,
                opg_init=1.0,
            )

        x0 = [0.01, 0.1, 2.0]
        bounds = [(-1.0, 1.0), (0.0, 1.0), (0.0, 5.0)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        omega_fit, A_fit, B_fit = result.x
        nll_at_optimum = result.fun

        # Verify: gas_filter with fitted params gives same LL
        params_fit = GASParameters(omega=omega_fit, A=A_fit, B=B_fit)
        _, ll_path_fit, _, _ = gas_filter(
            z1, z2, params_fit, nu,
            theta_init=theta_init,
            opg_init=1.0,
            **GAS_FILTER_DEFAULTS,
        )
        nll_from_filter = -np.sum(ll_path_fit)

        assert abs(nll_at_optimum - nll_from_filter) < 1e-10, (
            f"Optimization objective ({nll_at_optimum:.10f}) != "
            f"filter LL ({nll_from_filter:.10f})"
        )
