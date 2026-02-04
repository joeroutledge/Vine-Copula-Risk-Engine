"""
Tests for plot label consistency.

Verifies that alpha labels in plots match configured values and that
backtest_summary.csv contains expected alphas.
"""

import sys
import pathlib
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from run_var_es_backtest import format_alpha_pct


class TestAlphaFormatting:
    """Test that alpha values are formatted correctly."""

    def test_format_whole_percent(self):
        """Whole percentages should not have decimals."""
        assert format_alpha_pct(0.01) == "1%"
        assert format_alpha_pct(0.05) == "5%"
        assert format_alpha_pct(0.10) == "10%"

    def test_format_fractional_percent(self):
        """Fractional percentages should show one decimal."""
        assert format_alpha_pct(0.025) == "2.5%"
        assert format_alpha_pct(0.075) == "7.5%"
        assert format_alpha_pct(0.015) == "1.5%"

    def test_format_not_rounded_incorrectly(self):
        """0.025 must NOT become '2%' or '3%'."""
        result = format_alpha_pct(0.025)
        assert result != "2%", "0.025 must not round to 2%"
        assert result != "3%", "0.025 must not round to 3%"
        assert "2.5" in result, f"0.025 should format as 2.5%, got {result}"


class TestBacktestSummaryAlphas:
    """Test that backtest_summary.csv contains expected alphas."""

    @pytest.fixture
    def backtest_summary(self):
        """Load backtest_summary.csv if it exists."""
        summary_path = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "demo" / "backtest_summary.csv"
        if not summary_path.exists():
            pytest.skip("backtest_summary.csv not found - run 'make demo' first")
        return pd.read_csv(summary_path)

    def test_summary_has_standard_alphas(self, backtest_summary):
        """Summary should contain the standard demo alphas."""
        alphas_in_summary = set(backtest_summary['alpha'].unique())

        # Standard demo alphas
        expected_alphas = {0.01, 0.025, 0.05}

        assert expected_alphas.issubset(alphas_in_summary), \
            f"Expected alphas {expected_alphas} not found in summary. Found: {alphas_in_summary}"

    def test_plotted_alpha_matches_config(self, backtest_summary):
        """The primary plotted alpha (0.025) must be in the summary."""
        primary_alpha = 0.025
        alphas_in_summary = backtest_summary['alpha'].unique()

        assert primary_alpha in alphas_in_summary, \
            f"Plotted alpha {primary_alpha} not in backtest_summary.csv alphas: {list(alphas_in_summary)}"

    def test_all_methods_have_all_alphas(self, backtest_summary):
        """Each method should have results for all configured alphas."""
        methods = backtest_summary['method'].unique()
        alphas = backtest_summary['alpha'].unique()

        for method in methods:
            method_alphas = backtest_summary[backtest_summary['method'] == method]['alpha'].unique()
            assert set(method_alphas) == set(alphas), \
                f"Method {method} missing some alphas. Has {set(method_alphas)}, expected {set(alphas)}"
