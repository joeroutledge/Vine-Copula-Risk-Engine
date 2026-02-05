"""
Tests for Diebold-Mariano test implementation.

Test cases:
1. Symmetry: swapping A/B flips stat and mean_diff signs; p_value unchanged.
2. Synthetic power: constructed differential should reject in correct direction.
3. Constant differential (zero variance) handled gracefully.
"""

import numpy as np
import pytest

from vine_risk.model.dm_test import dm_test, newey_west_variance


class TestDMSymmetry:
    """Test that DM test has correct symmetry properties."""

    def test_swap_flips_sign(self):
        """Swapping A and B should flip stat and mean_diff signs."""
        np.random.seed(42)
        n = 500

        loss_a = np.random.normal(0.01, 0.005, n)
        loss_b = np.random.normal(0.012, 0.005, n)

        result_ab = dm_test(loss_a, loss_b)
        result_ba = dm_test(loss_b, loss_a)

        # Stat should flip sign
        assert abs(result_ab["stat"] + result_ba["stat"]) < 1e-10, (
            f"Stats should be negatives: {result_ab['stat']} vs {result_ba['stat']}"
        )

        # Mean diff should flip sign
        assert abs(result_ab["mean_diff"] + result_ba["mean_diff"]) < 1e-10, (
            f"Mean diffs should be negatives: {result_ab['mean_diff']} vs {result_ba['mean_diff']}"
        )

        # P-value should be unchanged
        assert abs(result_ab["p_value"] - result_ba["p_value"]) < 1e-10, (
            f"P-values should match: {result_ab['p_value']} vs {result_ba['p_value']}"
        )

    def test_identical_losses_zero_stat(self):
        """Identical losses should give stat near zero."""
        np.random.seed(123)
        n = 500

        loss = np.random.normal(0.01, 0.005, n)

        result = dm_test(loss, loss.copy())

        # With identical losses, mean_diff should be exactly zero
        assert abs(result["mean_diff"]) < 1e-15, (
            f"Mean diff should be zero for identical losses: {result['mean_diff']}"
        )

        # P-value should be 1 (no evidence against H0)
        assert result["p_value"] == 1.0, (
            f"P-value should be 1.0 for identical losses: {result['p_value']}"
        )


class TestDMPower:
    """Test that DM test has power to detect differences."""

    def test_detects_lower_loss(self):
        """Should detect when model A has systematically lower loss."""
        np.random.seed(42)
        n = 1000

        # Model A has lower loss on average (independent noise)
        loss_a = np.random.normal(0.008, 0.003, n)  # mean 0.008
        loss_b = np.random.normal(0.010, 0.003, n)  # mean 0.010 (worse)

        result = dm_test(loss_a, loss_b)

        # mean_diff should be negative (A - B < 0 means A is better)
        assert result["mean_diff"] < 0, (
            f"Mean diff should be negative when A has lower loss: {result['mean_diff']}"
        )

        # DM stat should be significantly negative
        assert result["stat"] < -2.0, (
            f"DM stat should be significantly negative: {result['stat']}"
        )

        # P-value should be small (reject H0)
        assert result["p_value"] < 0.05, (
            f"P-value should be small when difference is clear: {result['p_value']}"
        )

    def test_detects_higher_loss(self):
        """Should detect when model A has systematically higher loss."""
        np.random.seed(42)
        n = 1000

        # Model A has higher loss on average (independent noise)
        loss_a = np.random.normal(0.010, 0.003, n)  # mean 0.010 (worse)
        loss_b = np.random.normal(0.008, 0.003, n)  # mean 0.008

        result = dm_test(loss_a, loss_b)

        # mean_diff should be positive (A - B > 0 means B is better)
        assert result["mean_diff"] > 0, (
            f"Mean diff should be positive when A has higher loss: {result['mean_diff']}"
        )

        # DM stat should be significantly positive
        assert result["stat"] > 2.0, (
            f"DM stat should be significantly positive: {result['stat']}"
        )

        # P-value should be small
        assert result["p_value"] < 0.05, (
            f"P-value should be small when difference is clear: {result['p_value']}"
        )

    def test_no_rejection_when_equal(self):
        """Should not reject when losses are drawn from same distribution."""
        np.random.seed(42)
        n = 500

        # Same distribution, no systematic difference
        loss_a = np.random.normal(0.01, 0.005, n)
        loss_b = np.random.normal(0.01, 0.005, n)

        result = dm_test(loss_a, loss_b)

        # P-value should typically be > 0.05 (not reject)
        # Note: this is a probabilistic test, but with same distribution
        # we expect high p-value most of the time
        # We use a very lenient threshold to avoid flaky test
        assert result["p_value"] > 0.001, (
            f"P-value should not be extremely small when no difference: {result['p_value']}"
        )


class TestDMConstantDifferential:
    """Test handling of constant differential (zero variance)."""

    def test_constant_zero_differential(self):
        """Constant zero differential should return p_value=1."""
        n = 100
        loss_a = np.ones(n) * 0.01
        loss_b = np.ones(n) * 0.01  # Same constant

        result = dm_test(loss_a, loss_b)

        assert result["p_value"] == 1.0, (
            f"P-value should be 1.0 for constant zero differential: {result['p_value']}"
        )
        assert result["mean_diff"] == 0.0, (
            f"Mean diff should be 0 for identical constants: {result['mean_diff']}"
        )

    def test_constant_nonzero_differential(self):
        """Constant nonzero differential should return p_value=1 (degenerate)."""
        n = 100
        loss_a = np.ones(n) * 0.01
        loss_b = np.ones(n) * 0.02  # Constant difference

        result = dm_test(loss_a, loss_b)

        # Zero variance in differential means we can't compute valid test
        # Should return p_value=1 to indicate no test conclusion
        assert result["p_value"] == 1.0, (
            f"P-value should be 1.0 for constant differential: {result['p_value']}"
        )

        # Mean diff should still be computed correctly
        assert abs(result["mean_diff"] - (-0.01)) < 1e-10, (
            f"Mean diff should be -0.01: {result['mean_diff']}"
        )


class TestDMEdgeCases:
    """Test edge cases and error handling."""

    def test_mismatched_lengths(self):
        """Should raise ValueError for mismatched array lengths."""
        loss_a = np.random.normal(0, 1, 100)
        loss_b = np.random.normal(0, 1, 50)

        with pytest.raises(ValueError, match="same shape"):
            dm_test(loss_a, loss_b)

    def test_handles_nans(self):
        """Should handle NaN values by removing paired observations."""
        np.random.seed(42)
        n = 100

        loss_a = np.random.normal(0.01, 0.005, n)
        loss_b = np.random.normal(0.012, 0.005, n)

        # Introduce some NaNs
        loss_a[10] = np.nan
        loss_b[20] = np.nan

        result = dm_test(loss_a, loss_b)

        # Should have n - 2 observations (two NaN positions)
        assert result["n_obs"] == n - 2, (
            f"Should have {n-2} obs after removing NaNs: {result['n_obs']}"
        )

    def test_all_nans_raises(self):
        """Should raise ValueError if all observations are NaN."""
        loss_a = np.array([np.nan, np.nan, np.nan])
        loss_b = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="No valid observations"):
            dm_test(loss_a, loss_b)

    def test_nw_lags_default(self):
        """Should use floor(T^(1/3)) as default for NW lags."""
        np.random.seed(42)
        n = 1000  # T^(1/3) = 10

        loss_a = np.random.normal(0.01, 0.005, n)
        loss_b = np.random.normal(0.012, 0.005, n)

        result = dm_test(loss_a, loss_b)

        expected_lags = int(np.floor(n ** (1 / 3)))
        assert result["nw_lags"] == expected_lags, (
            f"NW lags should be {expected_lags}: {result['nw_lags']}"
        )

    def test_custom_nw_lags(self):
        """Should use custom NW lags when specified."""
        np.random.seed(42)
        n = 500

        loss_a = np.random.normal(0.01, 0.005, n)
        loss_b = np.random.normal(0.012, 0.005, n)

        result = dm_test(loss_a, loss_b, nw_lags=5)

        assert result["nw_lags"] == 5, (
            f"NW lags should be 5: {result['nw_lags']}"
        )

    def test_horizon_recorded(self):
        """Should record the forecast horizon in output."""
        np.random.seed(42)
        n = 100

        loss_a = np.random.normal(0.01, 0.005, n)
        loss_b = np.random.normal(0.012, 0.005, n)

        result = dm_test(loss_a, loss_b, h=1)

        assert result["h"] == 1, f"Horizon should be 1: {result['h']}"


class TestNeweyWestVariance:
    """Test the Newey-West variance estimator."""

    def test_zero_lags_equals_sample_variance(self):
        """With zero lags, should equal sample variance."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)

        nw_var = newey_west_variance(x, lags=0)
        sample_var = np.var(x, ddof=0)

        assert abs(nw_var - sample_var) < 1e-10, (
            f"Zero-lag NW var should equal sample var: {nw_var} vs {sample_var}"
        )

    def test_positive_for_correlated_series(self):
        """Should be positive for correlated series."""
        np.random.seed(42)
        n = 500

        # Create autocorrelated series
        x = np.zeros(n)
        x[0] = np.random.normal()
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + np.random.normal()

        nw_var = newey_west_variance(x, lags=10)

        assert nw_var > 0, f"NW variance should be positive: {nw_var}"
