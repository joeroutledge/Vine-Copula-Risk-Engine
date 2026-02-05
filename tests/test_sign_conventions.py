"""
Tests for VaR/ES sign conventions.

Verifies that:
1. Return-space equals negative of loss-space for all metrics
2. Euler decomposition holds in both spaces
3. Percent contributions are invariant to sign convention
"""

import numpy as np
import pandas as pd
import pytest

from vine_risk.model.tail_risk_attribution import (
    compute_component_es,
    compute_component_es_single,
    compute_component_es_timeseries,
)


class TestSignConventionEquality:
    """Test that return-space = -loss-space."""

    def test_portfolio_var_sign_relation(self):
        """portfolio_var_return = -portfolio_var."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        # All rows should have same portfolio values
        var_loss = df["portfolio_var"].iloc[0]
        var_return = df["portfolio_var_return"].iloc[0]

        np.testing.assert_almost_equal(
            var_return, -var_loss,
            decimal=12,
            err_msg="portfolio_var_return should equal -portfolio_var"
        )

    def test_portfolio_es_sign_relation(self):
        """portfolio_es_return = -portfolio_es."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        es_loss = df["portfolio_es"].iloc[0]
        es_return = df["portfolio_es_return"].iloc[0]

        np.testing.assert_almost_equal(
            es_return, -es_loss,
            decimal=12,
            err_msg="portfolio_es_return should equal -portfolio_es"
        )

    def test_component_es_sign_relation(self):
        """component_es_return = -component_es for each asset."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        np.testing.assert_array_almost_equal(
            df["component_es_return"].values,
            -df["component_es"].values,
            decimal=12,
            err_msg="component_es_return should equal -component_es"
        )


class TestEulerDecompositionBothSpaces:
    """Test Euler decomposition in both loss and return space."""

    def test_euler_sum_loss_space(self):
        """sum(component_es) = portfolio_es in loss-space."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        sum_ces = df["component_es"].sum()
        portfolio_es = df["portfolio_es"].iloc[0]

        np.testing.assert_almost_equal(
            sum_ces, portfolio_es,
            decimal=10,
            err_msg="sum(component_es) should equal portfolio_es (loss-space)"
        )

    def test_euler_sum_return_space(self):
        """sum(component_es_return) = portfolio_es_return in return-space."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        sum_ces_return = df["component_es_return"].sum()
        portfolio_es_return = df["portfolio_es_return"].iloc[0]

        np.testing.assert_almost_equal(
            sum_ces_return, portfolio_es_return,
            decimal=10,
            err_msg="sum(component_es_return) should equal portfolio_es_return"
        )

    def test_euler_both_spaces_consistent(self):
        """
        If sum(component_es) = portfolio_es,
        then sum(-component_es) = -portfolio_es,
        so sum(component_es_return) = portfolio_es_return.
        """
        np.random.seed(42)
        n_sim, n_assets = 1000, 5
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.01

        df = compute_component_es(sim_returns, weights, alpha)

        # Loss-space sum
        sum_loss = df["component_es"].sum()
        portfolio_loss = df["portfolio_es"].iloc[0]

        # Return-space sum
        sum_return = df["component_es_return"].sum()
        portfolio_return = df["portfolio_es_return"].iloc[0]

        # Both should hold
        np.testing.assert_almost_equal(sum_loss, portfolio_loss, decimal=10)
        np.testing.assert_almost_equal(sum_return, portfolio_return, decimal=10)

        # And return = -loss
        np.testing.assert_almost_equal(sum_return, -sum_loss, decimal=10)


class TestPercentContributionInvariance:
    """Test that percent contributions are sign-invariant."""

    def test_percent_contributions_sum_to_one(self):
        """Percent contributions sum to 1.0."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        sum_pct = df["percent_contribution"].sum()

        np.testing.assert_almost_equal(
            sum_pct, 1.0,
            decimal=10,
            err_msg="percent_contribution should sum to 1.0"
        )

    def test_percent_contributions_are_ratios(self):
        """percent_contribution = component_es / portfolio_es."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        expected_pct = df["component_es"] / df["portfolio_es"]

        np.testing.assert_array_almost_equal(
            df["percent_contribution"].values,
            expected_pct.values,
            decimal=10,
            err_msg="percent_contribution should equal component_es/portfolio_es"
        )


class TestSingleTimeStepSignConventions:
    """Test compute_component_es_single sign conventions."""

    def test_single_step_sign_relations(self):
        """All sign relations hold for single-step computation."""
        np.random.seed(42)
        n_sim, n_assets = 500, 4
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        result = compute_component_es_single(sim_returns, weights, alpha)

        # portfolio_var_return = -portfolio_var
        np.testing.assert_almost_equal(
            result["portfolio_var_return"],
            -result["portfolio_var"],
            decimal=12
        )

        # portfolio_es_return = -portfolio_es
        np.testing.assert_almost_equal(
            result["portfolio_es_return"],
            -result["portfolio_es"],
            decimal=12
        )

        # component_es_return = -component_es
        np.testing.assert_array_almost_equal(
            result["component_es_return"],
            -result["component_es"],
            decimal=12
        )

    def test_single_step_euler_both_spaces(self):
        """Euler decomposition holds in both spaces for single-step."""
        np.random.seed(42)
        n_sim, n_assets = 500, 4
        sim_returns = np.random.normal(-0.01, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        result = compute_component_es_single(sim_returns, weights, alpha)

        # Loss-space
        np.testing.assert_almost_equal(
            np.sum(result["component_es"]),
            result["portfolio_es"],
            decimal=10
        )

        # Return-space
        np.testing.assert_almost_equal(
            np.sum(result["component_es_return"]),
            result["portfolio_es_return"],
            decimal=10
        )


class TestTimeSeriesSignConventions:
    """Test compute_component_es_timeseries sign conventions."""

    def test_timeseries_includes_return_columns(self):
        """Time-series output includes return-space columns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        n_sim, n_assets = 100, 3

        sim_returns_by_date = {
            date: np.random.normal(-0.01, 0.02, (n_sim, n_assets))
            for date in dates
        }
        weights = np.ones(n_assets) / n_assets
        alphas = [0.05]
        asset_names = ["A", "B", "C"]

        df = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        expected_cols = {
            "component_es", "component_es_return",
            "portfolio_var", "portfolio_var_return",
            "portfolio_es", "portfolio_es_return",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_timeseries_sign_relations(self):
        """Sign relations hold for each row in time-series."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        n_sim, n_assets = 100, 3

        sim_returns_by_date = {
            date: np.random.normal(-0.01, 0.02, (n_sim, n_assets))
            for date in dates
        }
        weights = np.ones(n_assets) / n_assets
        alphas = [0.01, 0.05]
        asset_names = ["A", "B", "C"]

        df = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        # component_es_return = -component_es
        np.testing.assert_array_almost_equal(
            df["component_es_return"].values,
            -df["component_es"].values,
            decimal=12
        )

        # portfolio_var_return = -portfolio_var
        np.testing.assert_array_almost_equal(
            df["portfolio_var_return"].values,
            -df["portfolio_var"].values,
            decimal=12
        )

        # portfolio_es_return = -portfolio_es
        np.testing.assert_array_almost_equal(
            df["portfolio_es_return"].values,
            -df["portfolio_es"].values,
            decimal=12
        )

    def test_timeseries_euler_per_date_alpha(self):
        """Euler decomposition holds for each (date, alpha) in time-series."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        n_sim, n_assets = 200, 4

        sim_returns_by_date = {
            date: np.random.normal(-0.01, 0.02, (n_sim, n_assets))
            for date in dates
        }
        weights = np.ones(n_assets) / n_assets
        alphas = [0.01, 0.05]
        asset_names = ["A", "B", "C", "D"]

        df = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        for (date, alpha), group in df.groupby(["date", "alpha"]):
            # Loss-space
            sum_ces = group["component_es"].sum()
            portfolio_es = group["portfolio_es"].iloc[0]
            np.testing.assert_almost_equal(sum_ces, portfolio_es, decimal=10)

            # Return-space
            sum_ces_return = group["component_es_return"].sum()
            portfolio_es_return = group["portfolio_es_return"].iloc[0]
            np.testing.assert_almost_equal(sum_ces_return, portfolio_es_return, decimal=10)


class TestLossVsReturnInterpretation:
    """Test that loss and return space have correct interpretations."""

    def test_loss_space_positive_for_losses(self):
        """In loss-space, portfolio_var and portfolio_es are positive for loss events."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5

        # Create data with clear negative returns (losses)
        sim_returns = np.random.normal(-0.05, 0.01, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        # When portfolio loses money, loss-space values should be positive
        assert df["portfolio_var"].iloc[0] > 0, (
            "portfolio_var should be positive when portfolio has losses"
        )
        assert df["portfolio_es"].iloc[0] > 0, (
            "portfolio_es should be positive when portfolio has losses"
        )

    def test_return_space_negative_for_losses(self):
        """In return-space, values are negative for loss events."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5

        # Create data with clear negative returns (losses)
        sim_returns = np.random.normal(-0.05, 0.01, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        df = compute_component_es(sim_returns, weights, alpha)

        # When portfolio loses money, return-space values should be negative
        assert df["portfolio_var_return"].iloc[0] < 0, (
            "portfolio_var_return should be negative when portfolio has losses"
        )
        assert df["portfolio_es_return"].iloc[0] < 0, (
            "portfolio_es_return should be negative when portfolio has losses"
        )
