"""
Tests for vine model card export.

Verifies that model cards contain correct structure information and
pair-copula counts match theoretical expectations.
"""

import sys
import pathlib
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from vine_risk.benchmarks.static_vine import StaticDVineModel
from vine_risk.benchmarks.gas_vine import GASDVineModel


class TestModelCardStructure:
    """Test model card contains required fields."""

    @pytest.fixture
    def fitted_static_model(self):
        np.random.seed(42)
        U = pd.DataFrame(np.random.uniform(size=(200, 5)),
                         columns=['A', 'B', 'C', 'D', 'E'])
        model = StaticDVineModel(U)
        model.fit(U[:100])
        return model

    @pytest.fixture
    def fitted_gas_model(self):
        np.random.seed(42)
        U = pd.DataFrame(np.random.uniform(size=(200, 5)),
                         columns=['A', 'B', 'C', 'D', 'E'])
        model = GASDVineModel(U)
        model.fit(U[:100], fit_gas=True)
        return model

    def test_static_model_card_has_required_fields(self, fitted_static_model):
        """Model card must contain all required fields."""
        card = fitted_static_model.get_model_card()

        required_fields = [
            'dim', 'vine_type', 'variable_order', 'trunc_lvl',
            'is_full_vine', 'total_pair_copulas', 'family_set', 'pair_copulas'
        ]
        for field in required_fields:
            assert field in card, f"Missing required field: {field}"

        assert card['vine_type'] == 'D-vine'
        assert isinstance(card['pair_copulas'], list)

    def test_gas_model_card_has_gas_fields(self, fitted_gas_model):
        """GAS model card must include dynamics info (Option A' scope)."""
        card = fitted_gas_model.get_model_card()

        # Option A': GAS scope is elliptical-only
        assert 'gas_scope' in card
        assert card['gas_scope'] == 'tree1_elliptical_only'
        assert 'higher_tree_dynamics' in card
        assert card['higher_tree_dynamics'] == 'static'
        assert 'dynamic_edge_count' in card
        assert 'static_tree1_edge_count' in card

        # Tree-1 edges should have is_dynamic field
        tree1_edges = [pc for pc in card['pair_copulas'] if pc['tree'] == 1]
        for pc in tree1_edges:
            assert 'is_dynamic' in pc
            if pc['is_dynamic']:
                # Dynamic edges have GAS parameters
                assert 'gas_params' in pc, f"Dynamic edge missing gas_params"
                assert 'omega' in pc['gas_params']
                assert 'A' in pc['gas_params']
                assert 'B_effective' in pc['gas_params']
            else:
                # Static edges do NOT have gas_params
                assert 'gas_params' not in pc

    def test_pair_copula_fields(self, fitted_static_model):
        """Each pair copula must have required fields."""
        card = fitted_static_model.get_model_card()

        required_pc_fields = [
            'tree', 'edge_index', 'conditioned', 'conditioning',
            'family', 'rotation', 'parameters', 'kendall_tau'
        ]
        for pc in card['pair_copulas']:
            for field in required_pc_fields:
                assert field in pc, f"Pair copula missing field: {field}"

            # Conditioned should have 2 variables
            assert len(pc['conditioned']) == 2
            # Conditioning length should match tree - 1
            assert len(pc['conditioning']) == pc['tree'] - 1


class TestPairCopulaCount:
    """Test that pair copula counts match theoretical expectations."""

    def test_full_vine_pair_count(self):
        """Full D-vine: number_of_pairs == d*(d-1)/2."""
        for d in [3, 4, 5, 6]:
            np.random.seed(42)
            U = pd.DataFrame(np.random.uniform(size=(200, d)),
                             columns=[f'V{i}' for i in range(d)])
            model = StaticDVineModel(U)
            model.fit(U[:100])
            card = model.get_model_card()

            expected = d * (d - 1) // 2
            actual = card['total_pair_copulas']

            assert actual == expected, \
                f"d={d}: expected {expected} pairs, got {actual}"
            assert card['is_full_vine'] is True
            assert card['trunc_lvl'] == d - 1

    def test_truncated_vine_pair_count(self):
        """Truncated vine: number_of_pairs == sum_{i=1..k} (d-i)."""
        import pyvinecopulib as pvc

        d = 5
        k = 2  # Truncate at tree 2

        np.random.seed(42)
        U = pd.DataFrame(np.random.uniform(size=(200, d)),
                         columns=[f'V{i}' for i in range(d)])

        # Create truncated vine directly with pyvinecopulib
        order = [i + 1 for i in range(d)]
        struct = pvc.DVineStructure.from_order(order)

        controls = pvc.FitControlsVinecop(
            trunc_lvl=k,
            family_set=[pvc.BicopFamily.gaussian],
        )
        vine = pvc.Vinecop.from_data(U.values[:100], structure=struct,
                                     controls=controls)

        # Expected: sum_{tree=1..k} (d - tree)
        expected = sum(d - tree for tree in range(1, k + 1))
        actual_pairs = 0
        for tree in range(k):
            for edge in range(d - 1 - tree):
                actual_pairs += 1

        assert actual_pairs == expected, \
            f"Truncated at k={k}: expected {expected} pairs, got {actual_pairs}"

    def test_pair_copula_list_matches_count(self):
        """pair_copulas list length must match total_pair_copulas."""
        for d in [3, 5, 7]:
            np.random.seed(42)
            U = pd.DataFrame(np.random.uniform(size=(200, d)),
                             columns=[f'V{i}' for i in range(d)])
            model = StaticDVineModel(U)
            model.fit(U[:100])
            card = model.get_model_card()

            assert len(card['pair_copulas']) == card['total_pair_copulas'], \
                f"d={d}: pair_copulas list length doesn't match total_pair_copulas"


class TestModelCardSerialization:
    """Test that model cards can be serialized to JSON."""

    def test_json_serializable(self):
        """Model card must be JSON serializable."""
        np.random.seed(42)
        U = pd.DataFrame(np.random.uniform(size=(200, 4)),
                         columns=['A', 'B', 'C', 'D'])

        static_model = StaticDVineModel(U)
        static_model.fit(U[:100])
        static_card = static_model.get_model_card()

        # Should not raise
        json_str = json.dumps(static_card)
        assert len(json_str) > 0

        # Round-trip should preserve structure
        reloaded = json.loads(json_str)
        assert reloaded['dim'] == static_card['dim']
        assert len(reloaded['pair_copulas']) == len(static_card['pair_copulas'])

    def test_gas_model_json_serializable(self):
        """GAS model card must be JSON serializable."""
        np.random.seed(42)
        U = pd.DataFrame(np.random.uniform(size=(200, 4)),
                         columns=['A', 'B', 'C', 'D'])

        gas_model = GASDVineModel(U)
        gas_model.fit(U[:100], fit_gas=True)
        gas_card = gas_model.get_model_card()

        # Should not raise
        json_str = json.dumps(gas_card)
        assert len(json_str) > 0

        # Round-trip should preserve structure
        reloaded = json.loads(json_str)
        assert reloaded['gas_scope'] == 'tree1_elliptical_only'
        assert 'dynamic_edge_count' in reloaded

        # Verify is_dynamic field survives round-trip for all Tree-1 edges
        tree1_pcs = [pc for pc in reloaded['pair_copulas'] if pc['tree'] == 1]
        for pc in tree1_pcs:
            assert 'is_dynamic' in pc
            # Dynamic edges should have gas_params
            if pc['is_dynamic']:
                assert 'gas_params' in pc
