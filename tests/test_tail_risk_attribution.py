"""
Tests for tail risk attribution (component ES) computation.

Test cases:
1. Sum of component ES equals portfolio ES (Euler decomposition)
2. Symmetric case: equal weights + identical return series -> equal contributions
3. Determinism: fixed seed produces identical output
"""

import numpy as np
import pandas as pd
import pytest

from vine_risk.model.tail_risk_attribution import compute_component_es


class TestComponentESSum:
    """Test that sum of component ES equals portfolio ES (Euler decomposition).

    The formula ComponentES_i = w_i * E[-r_i | L >= VaR] is the Euler decomposition
    for Expected Shortfall (ES). The sum of components equals portfolio ES, which is
    greater than VaR for typical loss distributions. This is because:
    - VaR = quantile of loss distribution
    - ES = E[L | L >= VaR] > VaR

    The decomposition is exact for ES (a coherent risk measure).
    """

    def test_sum_equals_portfolio_es(self):
        """Sum of component ES should equal portfolio ES exactly."""
        np.random.seed(42)
        n_sim = 5000
        n_assets = 5

        # Generate correlated returns (make it realistic with some correlation)
        cov = np.eye(n_assets) * 0.0004  # ~2% daily vol
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    cov[i, j] = 0.0002  # positive correlation

        mean = np.zeros(n_assets)
        sim_returns = np.random.multivariate_normal(mean, cov, size=n_sim)

        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        portfolio_es = df["portfolio_es"].iloc[0]
        sum_component = df["component_es"].sum()

        # Sum of components should equal portfolio ES exactly (Euler decomposition)
        assert abs(sum_component - portfolio_es) < 1e-10, (
            f"Sum of component ES ({sum_component:.10f}) should equal "
            f"portfolio ES ({portfolio_es:.10f})"
        )

    def test_es_greater_than_var(self):
        """Portfolio ES should exceed portfolio VaR."""
        np.random.seed(42)
        n_sim = 5000
        n_assets = 5

        cov = np.eye(n_assets) * 0.0004
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    cov[i, j] = 0.0002

        mean = np.zeros(n_assets)
        sim_returns = np.random.multivariate_normal(mean, cov, size=n_sim)

        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        portfolio_var = df["portfolio_var"].iloc[0]
        portfolio_es = df["portfolio_es"].iloc[0]

        # ES should exceed VaR
        assert portfolio_es >= portfolio_var, (
            f"Portfolio ES ({portfolio_es:.6f}) should be >= "
            f"portfolio VaR ({portfolio_var:.6f})"
        )

        # And the ratio ES/VaR should be reasonable (typically 1.1-1.5 for normal-ish)
        ratio = portfolio_es / portfolio_var
        assert 1.0 <= ratio <= 2.0, (
            f"ES/VaR ratio {ratio:.2f} outside expected range [1.0, 2.0]"
        )

    def test_percent_contributions_sum_to_one(self):
        """Percent contributions should sum to 100%."""
        np.random.seed(123)
        n_sim = 10000
        n_assets = 3

        sim_returns = np.random.normal(0, 0.02, size=(n_sim, n_assets))
        weights = np.array([0.5, 0.3, 0.2])
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        pct_sum = df["percent_contribution"].sum()
        assert abs(pct_sum - 1.0) < 1e-10, (
            f"Percent contributions sum to {pct_sum:.6f}, expected 1.0"
        )


class TestSymmetricCase:
    """Test symmetric inputs produce equal contributions."""

    def test_equal_weights_identical_returns(self):
        """Equal weights + identical marginal distributions -> equal contributions."""
        np.random.seed(999)
        n_sim = 10000
        n_assets = 4

        # All assets have identical distribution (independent draws from same dist)
        base_returns = np.random.standard_t(df=5, size=n_sim) * 0.02
        sim_returns = np.column_stack([
            np.random.permutation(base_returns) for _ in range(n_assets)
        ])

        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        contributions = df["percent_contribution"].values

        # All contributions should be approximately equal (1/n_assets each)
        expected = 1.0 / n_assets
        for i, c in enumerate(contributions):
            assert abs(c - expected) < 0.05, (
                f"Asset {i} contribution {c:.3f} differs from expected "
                f"{expected:.3f} by more than 5%"
            )

    def test_two_assets_symmetric(self):
        """Two identical assets with equal weights -> 50/50 contribution."""
        np.random.seed(42)
        n_sim = 10000

        # Same distribution for both
        returns_1 = np.random.normal(-0.001, 0.02, n_sim)
        returns_2 = np.random.normal(-0.001, 0.02, n_sim)
        sim_returns = np.column_stack([returns_1, returns_2])

        weights = np.array([0.5, 0.5])
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        c1 = df["percent_contribution"].iloc[0]
        c2 = df["percent_contribution"].iloc[1]

        # Should be close to 50/50
        assert abs(c1 - 0.5) < 0.10, f"Asset 1 contribution {c1:.3f} not near 50%"
        assert abs(c2 - 0.5) < 0.10, f"Asset 2 contribution {c2:.3f} not near 50%"


class TestDeterminism:
    """Test that identical inputs produce identical outputs."""

    def test_deterministic_output(self):
        """Same input data should produce identical DataFrame output."""
        np.random.seed(42)
        n_sim = 1000
        n_assets = 3

        sim_returns = np.random.normal(0, 0.02, size=(n_sim, n_assets))
        weights = np.array([0.4, 0.35, 0.25])
        alpha = 0.05
        asset_names = ["A", "B", "C"]

        # Run twice with identical inputs
        df1 = compute_component_es(sim_returns, weights, alpha, asset_names)
        df2 = compute_component_es(sim_returns, weights, alpha, asset_names)

        # Should be exactly identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_fixed_seed_csv_consistency(self):
        """Fixed seed should produce consistent results across calls."""
        def generate_attribution():
            np.random.seed(12345)
            sim_returns = np.random.normal(0, 0.02, size=(2000, 4))
            weights = np.ones(4) / 4
            alpha = 0.01
            return compute_component_es(sim_returns, weights, alpha)

        df1 = generate_attribution()
        df2 = generate_attribution()

        # Component ES values should match exactly
        np.testing.assert_array_equal(
            df1["component_es"].values,
            df2["component_es"].values
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_alpha_too_high(self):
        """Alpha > 0.5 should raise ValueError."""
        sim_returns = np.random.normal(0, 0.02, size=(1000, 3))
        weights = np.ones(3) / 3

        with pytest.raises(ValueError, match="alpha must be in"):
            compute_component_es(sim_returns, weights, alpha=0.6)

    def test_invalid_alpha_zero(self):
        """Alpha = 0 should raise ValueError."""
        sim_returns = np.random.normal(0, 0.02, size=(1000, 3))
        weights = np.ones(3) / 3

        with pytest.raises(ValueError, match="alpha must be in"):
            compute_component_es(sim_returns, weights, alpha=0.0)

    def test_mismatched_weights_length(self):
        """Weights length != n_assets should raise ValueError."""
        sim_returns = np.random.normal(0, 0.02, size=(1000, 3))
        weights = np.ones(5) / 5  # Wrong length

        with pytest.raises(ValueError, match="weights length"):
            compute_component_es(sim_returns, weights, alpha=0.05)

    def test_nan_in_returns(self):
        """NaN in returns should raise ValueError."""
        sim_returns = np.random.normal(0, 0.02, size=(1000, 3))
        sim_returns[500, 1] = np.nan
        weights = np.ones(3) / 3

        with pytest.raises(ValueError, match="contains NaN"):
            compute_component_es(sim_returns, weights, alpha=0.05)

    def test_dataframe_input(self):
        """Should accept DataFrame and extract column names."""
        np.random.seed(42)
        n_sim = 1000
        df_input = pd.DataFrame({
            "SPY": np.random.normal(0, 0.02, n_sim),
            "AGG": np.random.normal(0, 0.01, n_sim),
        })
        weights = np.array([0.6, 0.4])

        result = compute_component_es(df_input, weights, alpha=0.05)

        assert list(result["asset"]) == ["SPY", "AGG"]

    def test_weights_normalized(self):
        """Non-normalized weights should be normalized internally."""
        np.random.seed(42)
        sim_returns = np.random.normal(0, 0.02, size=(1000, 2))

        # Weights that don't sum to 1
        weights_unnorm = np.array([2.0, 3.0])

        df = compute_component_es(sim_returns, weights_unnorm, alpha=0.05)

        # Output weights should be normalized
        assert abs(df["weight"].sum() - 1.0) < 1e-10


class TestContributionInterpretation:
    """Test that contributions have sensible interpretation."""

    def test_high_vol_asset_higher_contribution(self):
        """Asset with higher volatility should have higher risk contribution."""
        np.random.seed(42)
        n_sim = 10000

        # Asset 0: low vol, Asset 1: high vol
        low_vol = np.random.normal(0, 0.01, n_sim)
        high_vol = np.random.normal(0, 0.04, n_sim)
        sim_returns = np.column_stack([low_vol, high_vol])

        weights = np.array([0.5, 0.5])
        df = compute_component_es(sim_returns, weights, alpha=0.05)

        c_low = df["percent_contribution"].iloc[0]
        c_high = df["percent_contribution"].iloc[1]

        # High vol asset should contribute more to tail risk
        assert c_high > c_low, (
            f"High vol asset contribution ({c_high:.3f}) should exceed "
            f"low vol asset ({c_low:.3f})"
        )

    def test_larger_weight_higher_contribution(self):
        """Asset with larger weight should have higher contribution (same vol)."""
        np.random.seed(42)
        n_sim = 10000

        # Same volatility for both
        sim_returns = np.random.normal(0, 0.02, size=(n_sim, 2))

        # Different weights
        weights = np.array([0.8, 0.2])
        df = compute_component_es(sim_returns, weights, alpha=0.05)

        c_large = df["percent_contribution"].iloc[0]
        c_small = df["percent_contribution"].iloc[1]

        # Larger weight asset should contribute more
        assert c_large > c_small, (
            f"Large weight asset contribution ({c_large:.3f}) should exceed "
            f"small weight asset ({c_small:.3f})"
        )
