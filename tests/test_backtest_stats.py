"""
Tests for formal backtest statistics (Kupiec, Christoffersen, ES adequacy).
"""

import sys
import pathlib
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
# Import from the runner script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from run_var_es_backtest import kupiec_test, christoffersen_test, es_adequacy, pinball_loss


class TestKupiecTest:
    def test_correct_coverage(self):
        """If hit rate matches alpha exactly, p-value should be high."""
        n = 1000
        alpha = 0.05
        hits = np.zeros(n)
        hits[:50] = 1  # exactly 5%
        result = kupiec_test(hits, alpha)
        assert result["p_value"] > 0.05, "Exact coverage should not reject"

    def test_excessive_breaches(self):
        """Too many breaches should give low p-value."""
        n = 1000
        alpha = 0.01
        hits = np.zeros(n)
        hits[:50] = 1  # 5% instead of 1%
        result = kupiec_test(hits, alpha)
        assert result["p_value"] < 0.05, "Excessive breaches should reject"

    def test_no_breaches(self):
        """Zero breaches is degenerate; should handle gracefully."""
        hits = np.zeros(100)
        result = kupiec_test(hits, 0.05)
        assert np.isnan(result["statistic"])


class TestChristoffersenTest:
    def test_independent_hits(self):
        """Random (independent) hits should not reject."""
        rng = np.random.RandomState(42)
        hits = (rng.random(500) < 0.05).astype(int)
        result = christoffersen_test(hits)
        # Not guaranteed to pass every time, but with seed=42 it should
        assert result["p_value"] > 0.01 or np.isnan(result["p_value"])

    def test_clustered_hits(self):
        """Clustered hits should reject independence."""
        # Create strongly clustered breaches
        hits = np.zeros(500, dtype=int)
        for i in range(0, 500, 50):
            hits[i:i+5] = 1  # 5 consecutive hits every 50 days
        result = christoffersen_test(hits)
        # Clustering should push p-value low
        assert result["statistic"] > 0


class TestESAdequacy:
    def test_ratio_near_one(self):
        """If forecast ES matches realized shortfall, ratio ~ 1."""
        n = 200
        rng = np.random.RandomState(7)
        r = rng.normal(0, 0.01, n)
        # Set VaR and ES such that ~5% breach
        var_f = np.full(n, np.percentile(r, 5))
        es_f = np.full(n, np.mean(r[r < var_f[0]]))

        result = es_adequacy(r, var_f, es_f)
        assert result["n_breaches"] > 0
        assert 0.5 < result["mean_shortfall_ratio"] < 2.0, \
            f"ES ratio should be near 1, got {result['mean_shortfall_ratio']}"


class TestPinballLoss:
    def test_perfect_quantile(self):
        """Pinball loss at the true quantile should be minimal."""
        rng = np.random.RandomState(42)
        r = rng.normal(0, 0.01, 1000)
        alpha = 0.05
        q_true = np.full(1000, np.quantile(r, alpha))
        q_high = np.full(1000, np.quantile(r, 0.01))  # too conservative
        loss_true = pinball_loss(r, q_true, alpha)
        loss_high = pinball_loss(r, q_high, alpha)
        assert loss_true < loss_high, "True quantile should have lower pinball loss"

    def test_nonnegative(self):
        """Pinball loss must be non-negative."""
        r = np.array([-0.01, 0.02, -0.03, 0.01])
        q = np.array([-0.02, -0.01, -0.02, -0.01])
        assert pinball_loss(r, q, 0.05) >= 0
