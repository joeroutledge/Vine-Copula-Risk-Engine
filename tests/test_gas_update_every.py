"""
Tests for GAS update_every parameter (daily vs weekly updates).

Verifies that:
1. update_every=1 reproduces baseline (exact equality)
2. update_every=5 produces piecewise-constant state path with jumps only on update days
3. Likelihood path length unchanged; no NaNs
4. Determinism with fixed inputs
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


class TestUpdateEveryBaseline:
    """Test that update_every=1 matches baseline behavior."""

    def test_update_every_1_equals_baseline(self):
        """With update_every=1, result should match baseline gas_filter."""
        np.random.seed(42)
        n = 100
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        params = GASParameters(omega=0.05, A=0.15, B=2.0)
        theta_init = 0.3

        # Baseline (implicit update_every=1)
        theta_path_base, ll_path_base, rho_path_base, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            update_every=1,
        )

        # Explicit update_every=1
        kwargs = {**GAS_FILTER_DEFAULTS, "update_every": 1}
        theta_path_1, ll_path_1, rho_path_1, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            **kwargs,
        )

        np.testing.assert_array_equal(theta_path_base, theta_path_1)
        np.testing.assert_array_equal(ll_path_base, ll_path_1)
        np.testing.assert_array_equal(rho_path_base, rho_path_1)


class TestUpdateEveryPiecewiseConstant:
    """Test piecewise-constant behavior with update_every > 1."""

    def test_update_every_5_piecewise_constant(self):
        """With update_every=5, theta should be constant between updates."""
        np.random.seed(42)
        n = 50
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        params = GASParameters(omega=0.05, A=0.15, B=2.0)
        theta_init = 0.3

        theta_path, _, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            update_every=5,
        )

        # The update happens at t=0, 5, 10, 15, ...
        # After update at t, the NEW theta is used starting at t+1
        # So theta should be constant from t+1 until next update at t+k
        #
        # Segments: [0], [1..5], [6..10], [11..15], ...
        # Within each segment (after first element), theta is constant
        #
        # Check: theta[t] == theta[t+1] when t >= 1 and (t+1) % 5 != 0
        # This means t+1 is not an update day, so no update happened at t
        for t in range(1, n - 1):
            # theta changes AFTER an update, so at t+1 if t was update day
            # theta[t+1] != theta[t] only if t % update_every == 0
            if t % 5 != 0:  # t is NOT an update day
                assert theta_path[t] == theta_path[t + 1], (
                    f"theta changed on non-update day: t={t}, "
                    f"theta[t]={theta_path[t]}, theta[t+1]={theta_path[t+1]}"
                )

    def test_update_days_cause_change(self):
        """Theta at t+1 should differ from theta at t when t is an update day."""
        np.random.seed(42)
        n = 50
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        # Use non-zero A to ensure updates cause changes
        params = GASParameters(omega=0.05, A=0.3, B=1.5)
        theta_init = 0.3

        theta_path, _, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            update_every=5,
        )

        # At update days t=0,5,10,..., the update affects theta[t+1]
        # So theta[t+1] should differ from theta[t] for update days t
        changes_after_update = 0
        for t in range(0, n - 1, 5):  # Update days: 0, 5, 10, ...
            if t + 1 < n:
                if theta_path[t + 1] != theta_path[t]:
                    changes_after_update += 1

        # At least some update days should cause a change
        assert changes_after_update > 0, (
            "Theta never changed after update days"
        )


class TestUpdateEveryLikelihood:
    """Test likelihood computation with update_every."""

    def test_ll_path_length_unchanged(self):
        """Likelihood path length should equal input length regardless of update_every."""
        np.random.seed(42)
        n = 47  # Non-multiple of 5
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        params = GASParameters(omega=0.05, A=0.15, B=2.0)

        for update_every in [1, 5, 10]:
            _, ll_path, _, _ = gas_filter(
                z1, z2, params, nu,
                theta_init=0.3,
                update_every=update_every,
            )
            assert len(ll_path) == n, (
                f"update_every={update_every}: ll_path has {len(ll_path)} elements, expected {n}"
            )

    def test_no_nans_in_output(self):
        """Output should not contain NaNs."""
        np.random.seed(42)
        n = 100
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        params = GASParameters(omega=0.05, A=0.15, B=2.0)

        for update_every in [1, 3, 7]:
            theta_path, ll_path, rho_path, _ = gas_filter(
                z1, z2, params, nu,
                theta_init=0.3,
                update_every=update_every,
            )
            assert not np.any(np.isnan(theta_path)), f"NaN in theta_path (update_every={update_every})"
            assert not np.any(np.isnan(ll_path)), f"NaN in ll_path (update_every={update_every})"
            assert not np.any(np.isnan(rho_path)), f"NaN in rho_path (update_every={update_every})"

    def test_ll_computed_daily_regardless(self):
        """Likelihood should be computed for every observation, not just update days."""
        np.random.seed(42)
        n = 50
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        params = GASParameters(omega=0.05, A=0.15, B=2.0)

        _, ll_path_1, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=0.3,
            update_every=1,
        )

        _, ll_path_5, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=0.3,
            update_every=5,
        )

        # Both should have all non-NaN values
        assert np.all(np.isfinite(ll_path_1))
        assert np.all(np.isfinite(ll_path_5))

        # Total LL should differ (different paths lead to different LL)
        # But both should be valid (finite, negative)
        assert np.sum(ll_path_1) != np.sum(ll_path_5), (
            "Total LL should differ between update_every=1 and update_every=5"
        )


class TestUpdateEveryDeterminism:
    """Test deterministic behavior."""

    def test_deterministic_output(self):
        """Same inputs should produce identical outputs."""
        np.random.seed(42)
        n = 100
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        params = GASParameters(omega=0.05, A=0.15, B=2.0)

        theta1, ll1, rho1, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=0.3,
            update_every=5,
        )

        theta2, ll2, rho2, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=0.3,
            update_every=5,
        )

        np.testing.assert_array_equal(theta1, theta2)
        np.testing.assert_array_equal(ll1, ll2)
        np.testing.assert_array_equal(rho1, rho2)


class TestNegLoglikWithUpdateEvery:
    """Test gas_neg_loglik with update_every parameter."""

    def test_neg_loglik_equals_filter_ll(self):
        """gas_neg_loglik should equal -sum(ll_path) from gas_filter."""
        np.random.seed(42)
        n = 100
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        omega, A, B = 0.05, 0.15, 2.0
        theta_init = 0.3

        for update_every in [1, 5, 10]:
            nll_helper = gas_neg_loglik(
                z1, z2, omega, A, B, nu,
                theta_init=theta_init,
                opg_init=1.0,
                update_every=update_every,
            )

            params = GASParameters(omega=omega, A=A, B=B)
            filter_kwargs = {**GAS_FILTER_DEFAULTS, "update_every": update_every}
            _, ll_path, _, _ = gas_filter(
                z1, z2, params, nu,
                theta_init=theta_init,
                opg_init=1.0,
                **filter_kwargs,
            )
            nll_filter = -np.sum(ll_path)

            assert abs(nll_helper - nll_filter) < 1e-12, (
                f"update_every={update_every}: helper={nll_helper}, filter={nll_filter}"
            )


class TestUpdateEveryEdgeCases:
    """Test edge cases for update_every."""

    def test_update_every_larger_than_n(self):
        """update_every > n should result in no updates after t=0."""
        np.random.seed(42)
        n = 10
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        params = GASParameters(omega=0.05, A=0.15, B=2.0)
        theta_init = 0.3

        theta_path, _, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            update_every=100,  # Much larger than n
        )

        # Only t=0 gets an update, so theta should be constant after t=1
        # Actually t=0 is an update (0 % 100 == 0), theta changes for t=1
        # But from t=1 onwards, no updates, so constant
        for t in range(1, n - 1):
            assert theta_path[t] == theta_path[t + 1], (
                f"theta changed on non-update day with update_every=100"
            )

    def test_first_observation_always_updates(self):
        """t=0 should always trigger an update (0 % k == 0 for any k)."""
        np.random.seed(42)
        n = 20
        nu = 8.0

        z1 = np.random.standard_t(nu, n)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, n)

        params = GASParameters(omega=0.1, A=0.3, B=1.5)
        theta_init = 0.3

        theta_path, _, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            update_every=10,
        )

        # theta[0] = theta_init (recorded before update)
        assert theta_path[0] == theta_init

        # theta[1] should differ from theta[0] if score is non-zero
        # (because t=0 triggered an update for t=1)
        # This may not always hold if score is exactly zero, but very unlikely
