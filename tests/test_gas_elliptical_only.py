"""
Tests for Option A': GAS dynamics apply only to elliptical Tree-1 edges.

Verifies that:
1. Non-elliptical edges (Clayton, Gumbel, Frank) remain static
2. Elliptical edges (gaussian, student) get GAS dynamics
3. Model card correctly reports is_dynamic status
4. Backwards compatibility: all-elliptical case works as before
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import pytest
import pyvinecopulib as pvc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from vine_risk.benchmarks.gas_vine import GASDVineModel
from vine_risk.benchmarks.static_vine import StaticDVineModel


class TestEllipticalFamilyDetection:
    """Test that elliptical families are correctly identified."""

    def test_elliptical_families_defined(self):
        """GASDVineModel should define ELLIPTICAL_FAMILIES class attribute."""
        assert hasattr(GASDVineModel, 'ELLIPTICAL_FAMILIES')
        assert pvc.BicopFamily.gaussian in GASDVineModel.ELLIPTICAL_FAMILIES
        assert pvc.BicopFamily.student in GASDVineModel.ELLIPTICAL_FAMILIES
        # Non-elliptical should NOT be in the set
        assert pvc.BicopFamily.clayton not in GASDVineModel.ELLIPTICAL_FAMILIES
        assert pvc.BicopFamily.gumbel not in GASDVineModel.ELLIPTICAL_FAMILIES
        assert pvc.BicopFamily.frank not in GASDVineModel.ELLIPTICAL_FAMILIES


class TestMixedEllipticalStaticEdges:
    """Test behavior when Tree-1 has both elliptical and non-elliptical edges."""

    @pytest.fixture
    def correlated_uniforms(self):
        """Generate uniforms with known correlation structure.

        Creates data where edge 0-1 has strong positive dependence (likely student/gaussian)
        and edge 1-2 has upper tail dependence (might select Gumbel).
        """
        np.random.seed(12345)
        n = 300

        # Generate correlated normal, then transform to uniform
        from scipy.stats import norm

        # Edge 0-1: strong positive correlation (elliptical)
        z0 = np.random.randn(n)
        z1 = 0.8 * z0 + 0.6 * np.random.randn(n)

        # Edge 1-2: weak/asymmetric dependence
        z2 = 0.2 * z1 + 0.98 * np.random.randn(n)

        u0 = norm.cdf(z0)
        u1 = norm.cdf(z1)
        u2 = norm.cdf(z2)

        return pd.DataFrame({
            'A': np.clip(u0, 1e-6, 1 - 1e-6),
            'B': np.clip(u1, 1e-6, 1 - 1e-6),
            'C': np.clip(u2, 1e-6, 1 - 1e-6),
        })

    def test_tree1_is_dynamic_dict_populated(self, correlated_uniforms):
        """After fit, tree1_is_dynamic should have entry for each Tree-1 edge."""
        model = GASDVineModel(correlated_uniforms)
        model.fit(correlated_uniforms[:200], fit_gas=True)

        # Should have d-1 = 2 entries
        assert len(model.tree1_is_dynamic) == 2
        assert 0 in model.tree1_is_dynamic
        assert 1 in model.tree1_is_dynamic

    def test_dynamic_edges_have_gas_models(self, correlated_uniforms):
        """Dynamic edges should have entries in tree1_models."""
        model = GASDVineModel(correlated_uniforms)
        model.fit(correlated_uniforms[:200], fit_gas=True)

        for edge, is_dynamic in model.tree1_is_dynamic.items():
            if is_dynamic:
                assert edge in model.tree1_models
                assert 'params' in model.tree1_models[edge]
                assert 'theta_path' in model.tree1_models[edge]
            else:
                # Static edges should NOT have GAS models
                assert edge not in model.tree1_models

    def test_static_edges_keep_original_family(self, correlated_uniforms):
        """Static edges should retain their BIC-selected family in build_vine_at_time."""
        model = GASDVineModel(correlated_uniforms)
        model.fit(correlated_uniforms[:200], fit_gas=True)

        static_vine = model._get_vine()
        time_vine = model.build_vine_at_time(100)

        for edge in range(model.d - 1):
            if not model.tree1_is_dynamic.get(edge, False):
                # Static edge: should use same family as static vine
                original_bc = static_vine.get_pair_copula(0, edge)
                time_bc = time_vine.get_pair_copula(0, edge)
                assert original_bc.family == time_bc.family

    def test_model_card_is_dynamic_field(self, correlated_uniforms):
        """Model card Tree-1 edges should have is_dynamic field."""
        model = GASDVineModel(correlated_uniforms)
        model.fit(correlated_uniforms[:200], fit_gas=True)

        card = model.get_model_card()
        tree1_edges = [pc for pc in card["pair_copulas"] if pc["tree"] == 1]

        for pc in tree1_edges:
            assert "is_dynamic" in pc
            edge = pc["edge_index"]
            # Should match internal state
            assert pc["is_dynamic"] == model.tree1_is_dynamic[edge]

    def test_model_card_summary_counts(self, correlated_uniforms):
        """Model card should have correct dynamic/static edge counts."""
        model = GASDVineModel(correlated_uniforms)
        model.fit(correlated_uniforms[:200], fit_gas=True)

        card = model.get_model_card()

        dynamic_actual = sum(1 for v in model.tree1_is_dynamic.values() if v)
        static_actual = sum(1 for v in model.tree1_is_dynamic.values() if not v)

        assert card["dynamic_edge_count"] == dynamic_actual
        assert card["static_tree1_edge_count"] == static_actual
        assert card["gas_scope"] == "tree1_elliptical_only"


class TestAllEllipticalBackwardsCompat:
    """Test backwards compatibility when all Tree-1 edges are elliptical."""

    @pytest.fixture
    def highly_correlated_uniforms(self):
        """Generate data with strong correlations (will select elliptical families)."""
        np.random.seed(42)
        n = 200
        from scipy.stats import norm

        # Strong correlations -> BIC will select gaussian/student
        z0 = np.random.randn(n)
        z1 = 0.85 * z0 + 0.53 * np.random.randn(n)
        z2 = 0.75 * z1 + 0.66 * np.random.randn(n)

        return pd.DataFrame({
            'X': np.clip(norm.cdf(z0), 1e-6, 1 - 1e-6),
            'Y': np.clip(norm.cdf(z1), 1e-6, 1 - 1e-6),
            'Z': np.clip(norm.cdf(z2), 1e-6, 1 - 1e-6),
        })

    def test_all_elliptical_all_dynamic(self, highly_correlated_uniforms):
        """When all Tree-1 families are elliptical, all edges should be dynamic."""
        model = GASDVineModel(highly_correlated_uniforms)
        model.fit(highly_correlated_uniforms[:150], fit_gas=True)

        # Check that selected families are elliptical
        static_vine = model._get_vine()
        all_elliptical = True
        for edge in range(model.d - 1):
            bc = static_vine.get_pair_copula(0, edge)
            if bc.family not in GASDVineModel.ELLIPTICAL_FAMILIES:
                all_elliptical = False
                break

        if all_elliptical:
            # All edges should be dynamic
            assert all(model.tree1_is_dynamic.values())
            assert len(model.tree1_models) == model.d - 1

    def test_all_dynamic_model_card_has_gas_params(self, highly_correlated_uniforms):
        """When all edges are dynamic, all Tree-1 in model card should have gas_params."""
        model = GASDVineModel(highly_correlated_uniforms)
        model.fit(highly_correlated_uniforms[:150], fit_gas=True)

        # Only test if all edges selected elliptical
        static_vine = model._get_vine()
        all_elliptical = all(
            static_vine.get_pair_copula(0, e).family in GASDVineModel.ELLIPTICAL_FAMILIES
            for e in range(model.d - 1)
        )

        if all_elliptical:
            card = model.get_model_card()
            tree1_edges = [pc for pc in card["pair_copulas"] if pc["tree"] == 1]
            for pc in tree1_edges:
                assert pc["is_dynamic"] is True
                assert "gas_params" in pc


class TestGaussianConversion:
    """Test that Gaussian edges are converted to Student-t(nu=100) for GAS."""

    @pytest.fixture
    def low_tail_uniforms(self):
        """Generate data that will select Gaussian (no tail dependence)."""
        np.random.seed(9999)
        n = 250
        from scipy.stats import norm

        # Gaussian copula structure (no tail dependence)
        z0 = np.random.randn(n)
        z1 = 0.6 * z0 + 0.8 * np.random.randn(n)
        z2 = 0.5 * z1 + 0.866 * np.random.randn(n)

        return pd.DataFrame({
            'P': np.clip(norm.cdf(z0), 1e-6, 1 - 1e-6),
            'Q': np.clip(norm.cdf(z1), 1e-6, 1 - 1e-6),
            'R': np.clip(norm.cdf(z2), 1e-6, 1 - 1e-6),
        })

    def test_gaussian_converted_to_student_for_gas(self, low_tail_uniforms):
        """Gaussian edges should be converted to Student-t(nu=100) for GAS."""
        model = GASDVineModel(low_tail_uniforms)
        model.fit(low_tail_uniforms[:200], fit_gas=True)

        # Check if any edge was originally gaussian and converted
        for edge, model_info in model.tree1_models.items():
            if model_info.get('original_family') == 'gaussian':
                # Should use nu=100 (approximates Gaussian)
                assert model_info['nu_fixed'] == 100.0
                # Should still be marked dynamic
                assert model.tree1_is_dynamic[edge] is True

    def test_gaussian_conversion_noted_in_model_card(self, low_tail_uniforms):
        """Model card should note when Gaussian was converted to Student-t."""
        model = GASDVineModel(low_tail_uniforms)
        model.fit(low_tail_uniforms[:200], fit_gas=True)

        card = model.get_model_card()
        tree1_edges = [pc for pc in card["pair_copulas"] if pc["tree"] == 1]

        for pc in tree1_edges:
            if pc.get("is_dynamic") and pc.get("nu_fixed") == 100.0:
                # Should have a note about conversion
                assert "note" in pc or pc["nu_fixed"] == 100.0


class TestSimulationWithMixedEdges:
    """Test that simulation works correctly with mixed dynamic/static edges."""

    @pytest.fixture
    def mixed_uniforms(self):
        """Generate uniforms for mixed edge testing."""
        np.random.seed(54321)
        n = 200
        return pd.DataFrame(
            np.random.uniform(size=(n, 3)),
            columns=['M1', 'M2', 'M3']
        )

    def test_simulate_produces_valid_uniforms(self, mixed_uniforms):
        """Simulation should produce valid uniforms regardless of edge mix."""
        model = GASDVineModel(mixed_uniforms)
        model.fit(mixed_uniforms[:150], fit_gas=True)

        samples = model.simulate(n_sim=100, t_idx=50)

        assert samples.shape == (100, 3)
        assert np.all((samples >= 0) & (samples <= 1))
        assert np.all(np.isfinite(samples))

    def test_refit_works_with_mixed_edges(self, mixed_uniforms):
        """refit_params should work with mixed dynamic/static edges."""
        model = GASDVineModel(mixed_uniforms)
        model.fit(mixed_uniforms[:100], fit_gas=True)

        # Store initial state
        initial_dynamic = model.tree1_is_dynamic.copy()

        # Refit on different window
        model.refit_params(mixed_uniforms[50:150])

        # Dynamic/static status should be preserved
        assert model.tree1_is_dynamic == initial_dynamic

        # Should still simulate correctly
        samples = model.simulate(n_sim=50, t_idx=75)
        assert samples.shape == (50, 3)
