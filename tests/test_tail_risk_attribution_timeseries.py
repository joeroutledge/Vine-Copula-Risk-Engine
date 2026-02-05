"""
Tests for time-series tail risk attribution (Component ES over time).

Verifies that:
1. Sum of component_es equals portfolio_es within tolerance
2. Percent contributions sum to 1.0 within tolerance
3. Corner cases (tiny n_sim) are handled gracefully
4. Output is deterministic given fixed inputs
"""

import numpy as np
import pandas as pd
import pytest

from vine_risk.model.tail_risk_attribution import (
    compute_component_es_single,
    compute_component_es_timeseries,
)


class TestComponentESSingle:
    """Tests for single time-step component ES."""

    def test_sum_equals_portfolio_es(self):
        """Sum of component ES must equal portfolio ES."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5

        sim_returns = np.random.normal(0, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        result = compute_component_es_single(sim_returns, weights, alpha)

        sum_component_es = np.sum(result["component_es"])
        portfolio_es = result["portfolio_es"]

        # Should be exactly equal (Euler decomposition property)
        assert abs(sum_component_es - portfolio_es) < 1e-10, (
            f"Sum of component ES ({sum_component_es}) != portfolio ES ({portfolio_es})"
        )

    def test_percent_contributions_sum_to_one(self):
        """Percent contributions must sum to 1.0."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5

        sim_returns = np.random.normal(0, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.01

        result = compute_component_es_single(sim_returns, weights, alpha)

        sum_pct = np.sum(result["percent_contribution"])

        assert abs(sum_pct - 1.0) < 1e-10, (
            f"Percent contributions sum to {sum_pct}, expected 1.0"
        )

    def test_handles_tiny_nsim(self):
        """Should handle tiny n_sim by guaranteeing at least 1 tail obs."""
        np.random.seed(42)
        n_sim, n_assets = 10, 3  # Very small n_sim

        sim_returns = np.random.normal(0, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.01  # Would need 100+ sims normally

        result = compute_component_es_single(sim_returns, weights, alpha)

        # Should not fail and n_tail should be at least 1
        assert result["n_tail"] >= 1, "n_tail should be at least 1"
        assert not np.isnan(result["portfolio_es"]), "portfolio_es should not be NaN"

    def test_portfolio_es_geq_portfolio_var(self):
        """Portfolio ES should be >= portfolio VaR (for loss distribution)."""
        np.random.seed(42)
        n_sim, n_assets = 1000, 5

        sim_returns = np.random.normal(0, 0.02, (n_sim, n_assets))
        weights = np.ones(n_assets) / n_assets
        alpha = 0.05

        result = compute_component_es_single(sim_returns, weights, alpha)

        # ES is expected tail loss, VaR is the threshold
        # ES >= VaR for the loss distribution
        assert result["portfolio_es"] >= result["portfolio_var"], (
            f"ES ({result['portfolio_es']}) < VaR ({result['portfolio_var']})"
        )


class TestComponentESTimeseries:
    """Tests for time-series component ES computation."""

    def test_deterministic_output(self):
        """Output should be deterministic given fixed inputs."""
        np.random.seed(42)

        # Create fixed simulations
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        n_sim, n_assets = 100, 3

        sim_returns_by_date = {
            date: np.random.normal(0, 0.02, (n_sim, n_assets))
            for date in dates
        }
        weights = np.ones(n_assets) / n_assets
        alphas = [0.01, 0.05]
        asset_names = ["A", "B", "C"]

        result1 = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        result2 = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        pd.testing.assert_frame_equal(result1, result2)

    def test_correct_schema(self):
        """Output should have correct columns."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        n_sim, n_assets = 100, 2

        sim_returns_by_date = {
            date: np.random.normal(0, 0.02, (n_sim, n_assets))
            for date in dates
        }
        weights = np.array([0.5, 0.5])
        alphas = [0.05]
        asset_names = ["X", "Y"]

        result = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        expected_cols = {
            "date", "alpha", "asset", "weight", "component_es",
            "percent_contribution", "portfolio_var", "portfolio_es",
            # Return-space equivalents (sign conventions)
            "component_es_return", "portfolio_var_return", "portfolio_es_return",
            "n_tail", "n_sim"
        }
        assert set(result.columns) == expected_cols

    def test_correct_row_count(self):
        """Should have n_dates * n_alphas * n_assets rows."""
        np.random.seed(42)

        n_dates, n_sim, n_assets = 5, 100, 3
        dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")

        sim_returns_by_date = {
            date: np.random.normal(0, 0.02, (n_sim, n_assets))
            for date in dates
        }
        weights = np.ones(n_assets) / n_assets
        alphas = [0.01, 0.05]
        asset_names = ["A", "B", "C"]

        result = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        expected_rows = n_dates * len(alphas) * n_assets
        assert len(result) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(result)}"
        )

    def test_sum_equals_portfolio_es_for_each_date_alpha(self):
        """For each (date, alpha), sum of component ES should equal portfolio ES."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        n_sim, n_assets = 200, 4

        sim_returns_by_date = {
            date: np.random.normal(0, 0.02, (n_sim, n_assets))
            for date in dates
        }
        weights = np.ones(n_assets) / n_assets
        alphas = [0.01, 0.05]
        asset_names = ["A", "B", "C", "D"]

        result = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        for (date, alpha), group in result.groupby(["date", "alpha"]):
            sum_ces = group["component_es"].sum()
            portfolio_es = group["portfolio_es"].iloc[0]

            assert abs(sum_ces - portfolio_es) < 1e-10, (
                f"At {date}, alpha={alpha}: sum(CES)={sum_ces} != ES={portfolio_es}"
            )

    def test_percent_contributions_sum_to_one_for_each_date_alpha(self):
        """For each (date, alpha), percent contributions should sum to 1."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        n_sim, n_assets = 200, 4

        sim_returns_by_date = {
            date: np.random.normal(0, 0.02, (n_sim, n_assets))
            for date in dates
        }
        weights = np.ones(n_assets) / n_assets
        alphas = [0.01, 0.05]
        asset_names = ["A", "B", "C", "D"]

        result = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        for (date, alpha), group in result.groupby(["date", "alpha"]):
            sum_pct = group["percent_contribution"].sum()

            assert abs(sum_pct - 1.0) < 1e-10, (
                f"At {date}, alpha={alpha}: percent sum={sum_pct} != 1.0"
            )

    def test_handles_missing_dates(self):
        """Should gracefully handle None or empty entries."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        n_sim, n_assets = 100, 2

        sim_returns_by_date = {
            dates[0]: np.random.normal(0, 0.02, (n_sim, n_assets)),
            dates[1]: None,  # Missing
            dates[2]: np.random.normal(0, 0.02, (n_sim, n_assets)),
        }
        weights = np.array([0.5, 0.5])
        alphas = [0.05]
        asset_names = ["X", "Y"]

        result = compute_component_es_timeseries(
            sim_returns_by_date, weights, alphas, asset_names, n_sim
        )

        # Should only have rows for dates[0] and dates[2]
        unique_dates = result["date"].unique()
        assert len(unique_dates) == 2
        assert dates[1] not in unique_dates
