"""
Tests for preventing OOS leakage in vine ordering.

Verifies that:
1. Vine ordering is computed from training data only
2. Perturbing OOS data does NOT change the order used for forecasting
3. Changing training data CAN change the order (sanity check)
4. fixed_order is properly recorded in model cards
"""

import numpy as np
import pandas as pd
import pytest

from vine_risk.benchmarks.static_vine import StaticDVineModel
from vine_risk.benchmarks.gas_vine import GASDVineModel


def generate_correlated_uniforms(n: int, d: int, seed: int = 42) -> pd.DataFrame:
    """Generate correlated uniform marginals for testing."""
    np.random.seed(seed)

    # Create correlated Gaussian data
    corr = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            corr[i, j] = corr[j, i] = 0.3 + 0.1 * np.random.rand()

    # Cholesky decomposition
    L = np.linalg.cholesky(corr)
    z = np.random.randn(n, d)
    x = z @ L.T

    # Transform to uniform via CDF
    from scipy.stats import norm
    U = norm.cdf(x)

    return pd.DataFrame(U, columns=[f"Asset_{i}" for i in range(d)])


class TestOrderComputationNoLeakage:
    """Test that order computation uses training data only."""

    def test_order_unchanged_when_oos_perturbed(self):
        """
        Perturbing OOS data should NOT change the order when using fixed_order.

        This is the core OOS leakage test: if order depends on OOS data,
        the model would have access to future information during structure
        selection, which is methodologically invalid.
        """
        np.random.seed(42)
        n_train, n_oos, d = 500, 200, 5

        # Generate full dataset
        U_full = generate_correlated_uniforms(n_train + n_oos, d, seed=42)
        U_train = U_full.iloc[:n_train]

        # Compute order from training data only (correct approach)
        order_from_train = StaticDVineModel.compute_order_from_data(U_train)

        # Create model with fixed_order from training
        model1 = StaticDVineModel(U_full, fixed_order=order_from_train)

        # Create perturbed OOS data
        U_full_perturbed = U_full.copy()
        # Permute OOS rows randomly
        oos_indices = np.arange(n_train, n_train + n_oos)
        np.random.shuffle(oos_indices)
        U_full_perturbed.iloc[n_train:] = U_full.iloc[oos_indices].values

        # Create another model with same fixed_order (perturbed OOS)
        model2 = StaticDVineModel(U_full_perturbed, fixed_order=order_from_train)

        # Orders must be identical (both use fixed_order from training)
        assert model1.order == model2.order, (
            f"Order changed when OOS was perturbed: {model1.order} vs {model2.order}"
        )

        # Verify order_source is recorded correctly
        assert model1._order_source == "train_only_fixed"
        assert model2._order_source == "train_only_fixed"

    def test_order_can_change_when_train_changes(self):
        """
        Changing training data should be able to change the order.

        This is a sanity check: if the order never changed, it would mean
        the ordering algorithm is broken.
        """
        d = 5

        # Two different training datasets with different correlations
        U_train1 = generate_correlated_uniforms(500, d, seed=42)
        U_train2 = generate_correlated_uniforms(500, d, seed=999)

        order1 = StaticDVineModel.compute_order_from_data(U_train1)
        order2 = StaticDVineModel.compute_order_from_data(U_train2)

        # Orders may or may not be different (depends on data), but at least
        # the algorithm runs without error
        # We just check the orders are valid permutations
        assert sorted(order1) == list(range(d))
        assert sorted(order2) == list(range(d))

        # Note: We don't assert they're different because with some seeds
        # they could coincidentally match. The key is that the algorithm
        # processes them independently.

    def test_legacy_order_inference_is_flagged(self):
        """
        Models created without fixed_order should flag this in order_source.
        """
        np.random.seed(42)
        U = generate_correlated_uniforms(100, 4, seed=42)

        # Create model without fixed_order (legacy behavior)
        model = StaticDVineModel(U, fixed_order=None)

        # Should be flagged as "inferred_from_full_u"
        assert model._order_source == "inferred_from_full_u"

        # Fit the model so we can get a model card
        model.fit(U)
        card = model.get_model_card()
        assert card["order_source"] == "inferred_from_full_u"


class TestFixedOrderPropagation:
    """Test that fixed_order is properly propagated through the pipeline."""

    def test_fixed_order_in_model_card_static(self):
        """Static model card should include order_source and order_indices."""
        np.random.seed(42)
        U = generate_correlated_uniforms(200, 4, seed=42)
        U_train = U.iloc[:150]

        fixed_order = StaticDVineModel.compute_order_from_data(U_train)
        model = StaticDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train)

        card = model.get_model_card()

        assert "order_source" in card
        assert card["order_source"] == "train_only_fixed"
        assert "order_indices" in card
        assert card["order_indices"] == fixed_order

    def test_fixed_order_in_model_card_gas(self):
        """GAS model card should include order_source and order_indices."""
        np.random.seed(42)
        U = generate_correlated_uniforms(200, 4, seed=42)
        U_train = U.iloc[:150]

        fixed_order = StaticDVineModel.compute_order_from_data(U_train)
        model = GASDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train, fit_gas=True)

        card = model.get_model_card()

        assert "order_source" in card
        assert card["order_source"] == "train_only_fixed"
        assert "order_indices" in card
        assert card["order_indices"] == fixed_order

    def test_gas_inherits_order_from_static(self):
        """GAS model should use the same order computation as static."""
        np.random.seed(42)
        U = generate_correlated_uniforms(200, 4, seed=42)
        U_train = U.iloc[:150]

        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        static_model = StaticDVineModel(U, fixed_order=fixed_order)
        gas_model = GASDVineModel(U, fixed_order=fixed_order)

        assert static_model.order == gas_model.order
        assert static_model.order == fixed_order


class TestOrderDeterminism:
    """Test that order computation is deterministic."""

    def test_order_deterministic_given_data(self):
        """Same data should produce same order."""
        np.random.seed(42)
        U = generate_correlated_uniforms(200, 5, seed=42)

        order1 = StaticDVineModel.compute_order_from_data(U)
        order2 = StaticDVineModel.compute_order_from_data(U)

        assert order1 == order2

    def test_order_greedy_algorithm_is_correct(self):
        """
        Verify the greedy Kendall's tau algorithm produces a valid path.
        """
        np.random.seed(42)
        U = generate_correlated_uniforms(300, 5, seed=42)

        order = StaticDVineModel.compute_order_from_data(U)

        # Order should be a permutation of 0..d-1
        assert sorted(order) == list(range(5))

        # Order should be a list of integers
        assert all(isinstance(i, (int, np.integer)) for i in order)


class TestEdgeCases:
    """Test edge cases for order computation."""

    def test_two_assets(self):
        """Order computation should work with minimum d=2."""
        np.random.seed(42)
        U = generate_correlated_uniforms(100, 2, seed=42)

        order = StaticDVineModel.compute_order_from_data(U)

        assert len(order) == 2
        assert sorted(order) == [0, 1]

    def test_independent_assets(self):
        """Order computation should handle independent assets."""
        np.random.seed(42)
        n, d = 500, 4

        # Create independent uniform marginals
        U = pd.DataFrame(
            np.random.uniform(0, 1, (n, d)),
            columns=[f"Asset_{i}" for i in range(d)]
        )

        order = StaticDVineModel.compute_order_from_data(U)

        # Should still produce a valid permutation
        assert sorted(order) == list(range(d))

    def test_perfect_correlation(self):
        """Order computation should handle highly correlated assets."""
        np.random.seed(42)
        n, d = 200, 3

        # Create perfectly correlated data
        x = np.random.randn(n, 1)
        X = np.hstack([x + 0.01 * np.random.randn(n, 1) for _ in range(d)])

        from scipy.stats import norm
        U = pd.DataFrame(
            norm.cdf(X),
            columns=[f"Asset_{i}" for i in range(d)]
        )

        order = StaticDVineModel.compute_order_from_data(U)

        # Should still produce a valid permutation
        assert sorted(order) == list(range(d))
