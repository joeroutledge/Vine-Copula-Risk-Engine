"""
Tests for pyvinecopulib API compatibility.

Ensures the code works across different pyvinecopulib versions.
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from vine_risk.core.pvc_compat import (
    make_dvine_structure, fit_vine_from_data, build_vine_from_pair_copulas,
    make_bicop_student, get_pvc_version_info
)
from vine_risk.benchmarks.static_vine import StaticDVineModel
from vine_risk.benchmarks.gas_vine import GASDVineModel
import pyvinecopulib as pvc


class TestPvcCompat:
    """Test pyvinecopulib compatibility wrappers."""

    def test_version_info(self):
        """Should report version and feature flags."""
        info = get_pvc_version_info()
        assert "version" in info
        assert isinstance(info["has_DVineStructure"], bool)
        assert isinstance(info["has_from_data"], bool)
        assert isinstance(info["has_from_structure"], bool)

    def test_make_dvine_structure(self):
        """Should create a valid D-vine structure."""
        struct = make_dvine_structure(4)
        assert struct.dim == 4

        # With explicit order
        struct2 = make_dvine_structure(3, [1, 2, 3])
        assert struct2.dim == 3

    def test_make_bicop_student(self):
        """Should create a valid Student-t bivariate copula."""
        bc = make_bicop_student(0.5, 8.0)
        assert bc.family == pvc.BicopFamily.student

        # Parameter clipping
        bc_extreme = make_bicop_student(0.999, 1.5)  # Should clip
        params = bc_extreme.parameters.flatten()
        assert abs(params[0]) <= 0.99
        assert params[1] >= 2.1

    def test_fit_vine_from_data(self):
        """Should fit a vine copula from synthetic data."""
        np.random.seed(42)
        data = np.random.uniform(size=(200, 3))

        struct = make_dvine_structure(3)
        controls = pvc.FitControlsVinecop(
            family_set=[pvc.BicopFamily.gaussian],
        )

        vine = fit_vine_from_data(data, struct, controls)
        assert vine.dim == 3
        assert vine.trunc_lvl >= 1

    def test_build_vine_from_pair_copulas(self):
        """Should build a vine from pair copulas."""
        struct = make_dvine_structure(3)

        # Create pair copulas for a 3-dim D-vine (2 trees)
        pcs = [
            [make_bicop_student(0.5, 8.0), make_bicop_student(0.3, 8.0)],  # Tree 1
            [make_bicop_student(0.2, 8.0)],  # Tree 2
        ]

        vine = build_vine_from_pair_copulas(struct, pcs)
        assert vine.dim == 3

        # Should be able to simulate
        samples = vine.simulate(100)
        assert samples.shape == (100, 3)


class TestModelsWithCompat:
    """Test that models work with compat wrappers."""

    @pytest.fixture
    def synthetic_uniforms(self):
        """Generate synthetic uniform data."""
        np.random.seed(42)
        n = 200
        d = 3
        return pd.DataFrame(
            np.random.uniform(size=(n, d)),
            columns=['A', 'B', 'C']
        )

    def test_static_model_fit_no_attribute_error(self, synthetic_uniforms):
        """StaticDVineModel.fit should not raise AttributeError."""
        model = StaticDVineModel(synthetic_uniforms)
        # Should not raise
        model.fit(synthetic_uniforms[:100])
        assert model._vine is not None
        assert model._vine.dim == 3

    def test_gas_model_fit_no_attribute_error(self, synthetic_uniforms):
        """GASDVineModel.fit should not raise AttributeError."""
        model = GASDVineModel(synthetic_uniforms)
        # Should not raise
        model.fit(synthetic_uniforms[:100], fit_gas=True)
        assert model._vine is not None
        # GAS models only for elliptical edges; may be 0, 1, or 2
        assert len(model.tree1_is_dynamic) == 2  # 3-1 = 2 edges tracked

    def test_static_model_simulate(self, synthetic_uniforms):
        """StaticDVineModel.simulate should work."""
        model = StaticDVineModel(synthetic_uniforms)
        model.fit(synthetic_uniforms[:100])

        samples = model.simulate(n_sim=50)
        assert samples.shape == (50, 3)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_gas_model_simulate(self, synthetic_uniforms):
        """GASDVineModel.simulate should work."""
        model = GASDVineModel(synthetic_uniforms)
        model.fit(synthetic_uniforms[:100], fit_gas=True)

        samples = model.simulate(n_sim=50, t_idx=50)
        assert samples.shape == (50, 3)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_refit_params_no_error(self, synthetic_uniforms):
        """refit_params should not raise AttributeError."""
        model = StaticDVineModel(synthetic_uniforms)
        model.fit(synthetic_uniforms[:100])

        # Refit on different window
        model.refit_params(synthetic_uniforms[50:150])
        assert model._vine is not None

    def test_gas_model_card_elliptical_only_dynamics(self, synthetic_uniforms):
        """GAS model card: only elliptical Tree-1 edges have GAS dynamics.

        Option A' design: GAS dynamics apply only to elliptical families
        (gaussian, student-t) where Fisher-z transformation is well-defined.
        Non-elliptical edges (Clayton, Gumbel, Frank) remain static.
        """
        model = GASDVineModel(synthetic_uniforms)
        model.fit(synthetic_uniforms[:100], fit_gas=True)

        card = model.get_model_card()

        # Check new scope indicators
        assert card["gas_scope"] == "tree1_elliptical_only"
        assert "dynamic_edge_count" in card
        assert "static_tree1_edge_count" in card
        assert card["dynamic_edge_count"] + card["static_tree1_edge_count"] == 2

        # All Tree-1 edges should have is_dynamic field
        tree1_edges = [pc for pc in card["pair_copulas"] if pc["tree"] == 1]
        assert len(tree1_edges) == 2  # 3-dim vine has 2 Tree-1 edges

        for pc in tree1_edges:
            assert "is_dynamic" in pc

            if pc["is_dynamic"]:
                # Dynamic edges must be Student-t with GAS params
                assert pc["family"] == "student"
                assert pc["rotation"] == 0
                assert len(pc["parameters"]) == 2  # [rho, nu]
                assert "gas_params" in pc
                assert "omega" in pc["gas_params"]
                assert "A" in pc["gas_params"]
                assert "B_effective" in pc["gas_params"]
            else:
                # Static edges keep BIC-selected family, no gas_params
                assert "gas_params" not in pc
