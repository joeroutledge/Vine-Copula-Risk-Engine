"""
Tests for GAS filter cadence invariance across segmentation.

Verifies that when filtering is split into segments (e.g., train then OOS),
the weekly update cadence is preserved across the boundary. This is critical
for methodological defensibility: the update schedule should not depend on
how the series is segmented for computational convenience.

The t_offset parameter in gas_filter() ensures this invariance.
"""

import numpy as np
import pytest
from scipy.stats import t as tdist

from vine_risk.core.gas import (
    GASParameters,
    gas_filter,
    GAS_FILTER_DEFAULTS,
)


class TestCadenceInvariance:
    """Test that split filtering equals full filtering."""

    def test_split_equals_full_update_every_5(self):
        """
        Splitting filtering at an arbitrary point k should produce
        identical theta_path when t_offset is used correctly.

        This is the core cadence invariance proof.
        """
        np.random.seed(42)
        N = 80
        nu = 8.0
        update_every = 5

        # Generate synthetic data
        z1 = np.random.standard_t(nu, N)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, N)

        params = GASParameters(omega=0.05, A=0.15, B=2.0)
        theta_init = 0.3

        # Full run (baseline)
        theta_full, ll_full, rho_full, state_full = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            opg_init=1.0,
            update_every=update_every,
            t_offset=0,
            return_final_state=True,
        )

        # Split at k=47 (not a multiple of 5, to stress test)
        k = 47

        # Segment 1: indices 0..k-1
        theta_seg1, ll_seg1, rho_seg1, state_k = gas_filter(
            z1[:k], z2[:k], params, nu,
            theta_init=theta_init,
            opg_init=1.0,
            update_every=update_every,
            t_offset=0,
            return_final_state=True,
        )

        # Segment 2: indices k..N-1, with t_offset=k to preserve cadence
        theta_seg2, ll_seg2, rho_seg2, state_final = gas_filter(
            z1[k:], z2[k:], params, nu,
            theta_init=state_k.theta,
            opg_init=state_k.opg,
            update_every=update_every,
            t_offset=k,  # Critical: global index offset
            return_final_state=True,
        )

        # Concatenate
        theta_concat = np.concatenate([theta_seg1, theta_seg2])
        ll_concat = np.concatenate([ll_seg1, ll_seg2])
        rho_concat = np.concatenate([rho_seg1, rho_seg2])

        # Must match exactly (or very tight tolerance)
        np.testing.assert_array_almost_equal(
            theta_full, theta_concat, decimal=12,
            err_msg="theta_path differs between full and split filtering"
        )
        np.testing.assert_array_almost_equal(
            ll_full, ll_concat, decimal=12,
            err_msg="ll_path differs between full and split filtering"
        )
        np.testing.assert_array_almost_equal(
            rho_full, rho_concat, decimal=12,
            err_msg="rho_path differs between full and split filtering"
        )

        # Final states should also match
        np.testing.assert_almost_equal(
            state_full.theta, state_final.theta, decimal=12,
            err_msg="Final theta differs"
        )
        np.testing.assert_almost_equal(
            state_full.opg, state_final.opg, decimal=12,
            err_msg="Final opg differs"
        )

    def test_split_at_multiple_of_update_every(self):
        """Split exactly at an update day boundary."""
        np.random.seed(123)
        N = 100
        nu = 8.0
        update_every = 5

        z1 = np.random.standard_t(nu, N)
        z2 = 0.4 * z1 + np.sqrt(0.84) * np.random.standard_t(nu, N)

        params = GASParameters(omega=0.02, A=0.1, B=2.5)
        theta_init = 0.2

        # Full run
        theta_full, _, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            update_every=update_every,
            t_offset=0,
        )

        # Split at k=50 (multiple of 5)
        k = 50

        theta_seg1, _, _, state_k = gas_filter(
            z1[:k], z2[:k], params, nu,
            theta_init=theta_init,
            update_every=update_every,
            t_offset=0,
            return_final_state=True,
        )

        theta_seg2, _, _, _ = gas_filter(
            z1[k:], z2[k:], params, nu,
            theta_init=state_k.theta,
            opg_init=state_k.opg,
            update_every=update_every,
            t_offset=k,
        )

        theta_concat = np.concatenate([theta_seg1, theta_seg2])

        np.testing.assert_array_almost_equal(
            theta_full, theta_concat, decimal=12
        )

    def test_daily_update_is_invariant_to_offset(self):
        """With update_every=1, t_offset should have no effect on theta."""
        np.random.seed(456)
        N = 50
        nu = 8.0

        z1 = np.random.standard_t(nu, N)
        z2 = 0.6 * z1 + np.sqrt(0.64) * np.random.standard_t(nu, N)

        params = GASParameters(omega=0.05, A=0.2, B=1.5)

        # Run with different t_offsets (all should be identical)
        theta_0, _, _, _ = gas_filter(
            z1, z2, params, nu,
            update_every=1,
            t_offset=0,
        )
        theta_10, _, _, _ = gas_filter(
            z1, z2, params, nu,
            update_every=1,
            t_offset=10,
        )
        theta_99, _, _, _ = gas_filter(
            z1, z2, params, nu,
            update_every=1,
            t_offset=99,
        )

        np.testing.assert_array_equal(theta_0, theta_10)
        np.testing.assert_array_equal(theta_0, theta_99)

    def test_wrong_offset_breaks_invariance(self):
        """
        Using wrong t_offset should produce DIFFERENT results.
        This is a sanity check that the offset actually matters.
        """
        np.random.seed(789)
        N = 80
        nu = 8.0
        update_every = 5

        z1 = np.random.standard_t(nu, N)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, N)

        params = GASParameters(omega=0.05, A=0.15, B=2.0)
        theta_init = 0.3
        k = 47

        # Full run
        theta_full, _, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            update_every=update_every,
            t_offset=0,
        )

        # Split WITHOUT using t_offset (wrong!)
        theta_seg1, _, _, state_k = gas_filter(
            z1[:k], z2[:k], params, nu,
            theta_init=theta_init,
            update_every=update_every,
            t_offset=0,
            return_final_state=True,
        )

        theta_seg2_wrong, _, _, _ = gas_filter(
            z1[k:], z2[k:], params, nu,
            theta_init=state_k.theta,
            opg_init=state_k.opg,
            update_every=update_every,
            t_offset=0,  # Wrong! Should be k
        )

        theta_concat_wrong = np.concatenate([theta_seg1, theta_seg2_wrong])

        # Should NOT match (within numerical tolerance)
        # The difference should be non-trivial
        max_diff = np.max(np.abs(theta_full - theta_concat_wrong))
        assert max_diff > 0.01, (
            f"Expected significant difference with wrong offset, got max_diff={max_diff}"
        )


class TestUpdateDayAlignment:
    """Test that update days are correctly aligned with global index."""

    def test_update_days_align_globally(self):
        """
        Update days should be at global indices 0, 5, 10, 15, ...
        regardless of segment boundaries.
        """
        np.random.seed(42)
        N = 50
        nu = 8.0
        update_every = 5

        z1 = np.random.standard_t(nu, N)
        z2 = 0.5 * z1 + np.sqrt(0.75) * np.random.standard_t(nu, N)

        params = GASParameters(omega=0.1, A=0.3, B=1.5)
        theta_init = 0.3

        # Full run
        theta_full, _, _, _ = gas_filter(
            z1, z2, params, nu,
            theta_init=theta_init,
            update_every=update_every,
        )

        # With update_every=5, theta should be piecewise constant
        # between update days. Check that theta changes only at
        # indices 0, 5, 10, 15, ... (after the update is applied for next day)
        for t in range(1, N - 1):
            if t % update_every != 0:
                # Not an update day: theta should not have changed from t to t+1
                # Actually, theta changes AFTER update day, so theta[t] == theta[t+1]
                # when t is NOT an update day
                assert theta_full[t] == theta_full[t + 1], (
                    f"theta changed on non-update day t={t}"
                )
