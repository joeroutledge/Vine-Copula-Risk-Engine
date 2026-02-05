"""
Tests for reproducibility contract.

Verifies that:
1. NumPy seeding produces deterministic results
2. pyvinecopulib seeding produces deterministic simulations
3. Determinism mode is recorded in model cards
4. make_pvc_seeds produces stable, distinct seeds
"""

import numpy as np
import pandas as pd
import pytest
import pyvinecopulib as pvc

from vine_risk.utils.random import (
    set_seed,
    make_pvc_seeds,
    get_determinism_info,
    get_determinism_mode,
    set_determinism_mode,
)
from vine_risk.benchmarks.static_vine import StaticDVineModel
from vine_risk.benchmarks.gas_vine import GASDVineModel


def generate_test_uniforms(n: int, d: int, seed: int = 42) -> pd.DataFrame:
    """Generate test uniform data."""
    np.random.seed(seed)
    from scipy.stats import norm

    corr = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            corr[i, j] = corr[j, i] = 0.4

    L = np.linalg.cholesky(corr)
    z = np.random.randn(n, d)
    x = z @ L.T
    U = norm.cdf(x)

    return pd.DataFrame(U, columns=[f"Asset_{i}" for i in range(d)])


class TestPVCSeeding:
    """Test pyvinecopulib deterministic seeding."""

    def test_same_seeds_same_output(self):
        """Same seeds produce identical simulations."""
        # Create a simple vine
        v = pvc.Vinecop(3)

        seeds = [123, 456, 789]

        samples1 = v.simulate(n=100, seeds=seeds)
        samples2 = v.simulate(n=100, seeds=seeds)

        np.testing.assert_array_equal(
            samples1, samples2,
            err_msg="pyvinecopulib simulate with same seeds should be identical"
        )

    def test_different_seeds_different_output(self):
        """Different seeds produce different simulations."""
        v = pvc.Vinecop(3)

        samples1 = v.simulate(n=100, seeds=[123, 456, 789])
        samples2 = v.simulate(n=100, seeds=[999, 888, 777])

        assert not np.allclose(samples1, samples2), (
            "Different seeds should produce different output"
        )


class TestMakePVCSeeds:
    """Test the make_pvc_seeds utility."""

    def test_deterministic_given_inputs(self):
        """Same inputs produce same seeds."""
        seeds1 = make_pvc_seeds(42, t_idx=0)
        seeds2 = make_pvc_seeds(42, t_idx=0)

        assert seeds1 == seeds2

    def test_different_t_idx_different_seeds(self):
        """Different t_idx produces different seeds."""
        seeds1 = make_pvc_seeds(42, t_idx=0)
        seeds2 = make_pvc_seeds(42, t_idx=1)

        assert seeds1 != seeds2

    def test_different_base_seed_different_seeds(self):
        """Different base_seed produces different seeds."""
        seeds1 = make_pvc_seeds(42, t_idx=0)
        seeds2 = make_pvc_seeds(99, t_idx=0)

        assert seeds1 != seeds2

    def test_seeds_are_valid_integers(self):
        """Seeds should be positive integers suitable for pyvinecopulib."""
        seeds = make_pvc_seeds(42, t_idx=5, n_seeds=5)

        assert len(seeds) == 5
        for s in seeds:
            assert isinstance(s, int)
            assert 0 < s < 2**31


class TestDeterminismInfo:
    """Test determinism info utilities."""

    def test_determinism_info_structure(self):
        """get_determinism_info returns expected structure."""
        info = get_determinism_info()

        assert "determinism_mode" in info
        assert "numpy_seeded" in info
        assert "pyvinecopulib_seeded" in info
        assert "note" in info

        assert info["numpy_seeded"] is True
        assert info["pyvinecopulib_seeded"] is True

    def test_determinism_mode_default(self):
        """Default determinism mode is strict."""
        assert get_determinism_mode() == "strict"

    def test_set_determinism_mode(self):
        """Determinism mode can be changed."""
        original = get_determinism_mode()
        try:
            set_determinism_mode("relaxed")
            assert get_determinism_mode() == "relaxed"

            set_determinism_mode("strict")
            assert get_determinism_mode() == "strict"
        finally:
            set_determinism_mode(original)

    def test_invalid_determinism_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            set_determinism_mode("invalid")


class TestStaticModelDeterminism:
    """Test StaticDVineModel simulation determinism."""

    def test_simulate_deterministic_with_seeds(self):
        """Simulation with seeds is deterministic."""
        np.random.seed(42)
        U = generate_test_uniforms(200, 4, seed=42)
        U_train = U.iloc[:150]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = StaticDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train)

        seeds = make_pvc_seeds(42, t_idx=0)

        sim1 = model.simulate(n_sim=100, t_idx=0, seeds=seeds)
        sim2 = model.simulate(n_sim=100, t_idx=0, seeds=seeds)

        np.testing.assert_array_equal(
            sim1, sim2,
            err_msg="StaticDVineModel.simulate with seeds should be deterministic"
        )

    def test_simulate_different_t_idx_different_output(self):
        """Different t_idx with deterministic seeds gives different output."""
        np.random.seed(42)
        U = generate_test_uniforms(200, 4, seed=42)
        U_train = U.iloc[:150]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = StaticDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train)

        seeds1 = make_pvc_seeds(42, t_idx=0)
        seeds2 = make_pvc_seeds(42, t_idx=1)

        sim1 = model.simulate(n_sim=100, t_idx=0, seeds=seeds1)
        sim2 = model.simulate(n_sim=100, t_idx=0, seeds=seeds2)

        # Note: t_idx doesn't affect static model, but seeds do
        assert not np.allclose(sim1, sim2)


class TestGASModelDeterminism:
    """Test GASDVineModel simulation determinism."""

    def test_simulate_deterministic_with_seeds(self):
        """Simulation with seeds is deterministic."""
        np.random.seed(42)
        U = generate_test_uniforms(200, 4, seed=42)
        U_train = U.iloc[:150]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = GASDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train, fit_gas=True)

        seeds = make_pvc_seeds(42, t_idx=10)

        sim1 = model.simulate(n_sim=100, t_idx=10, seeds=seeds)
        sim2 = model.simulate(n_sim=100, t_idx=10, seeds=seeds)

        np.testing.assert_array_equal(
            sim1, sim2,
            err_msg="GASDVineModel.simulate with seeds should be deterministic"
        )

    def test_model_card_includes_determinism(self):
        """GAS model card includes determinism info."""
        np.random.seed(42)
        U = generate_test_uniforms(200, 4, seed=42)
        U_train = U.iloc[:150]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = GASDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train, fit_gas=True)

        card = model.get_model_card()

        assert "determinism" in card
        assert card["determinism"]["numpy_seeded"] is True
        assert card["determinism"]["pyvinecopulib_seeded"] is True


class TestEndToEndDeterminism:
    """End-to-end determinism tests."""

    def test_full_workflow_reproducible(self):
        """
        Full workflow: data -> fit -> simulate is reproducible.
        """
        def run_workflow(seed):
            np.random.seed(seed)
            U = generate_test_uniforms(200, 4, seed=seed)
            U_train = U.iloc[:150]
            fixed_order = StaticDVineModel.compute_order_from_data(U_train)

            model = GASDVineModel(U, fixed_order=fixed_order)
            model.fit(U_train, fit_gas=True)

            pvc_seeds = make_pvc_seeds(seed, t_idx=5)
            sim = model.simulate(n_sim=50, t_idx=5, seeds=pvc_seeds)
            return sim

        sim1 = run_workflow(42)
        sim2 = run_workflow(42)

        np.testing.assert_array_equal(
            sim1, sim2,
            err_msg="Full workflow should be reproducible with same seed"
        )

    def test_different_seeds_produce_different_results(self):
        """Different seeds produce different simulations."""
        def run_workflow(seed):
            np.random.seed(seed)
            U = generate_test_uniforms(200, 4, seed=seed)
            U_train = U.iloc[:150]
            fixed_order = StaticDVineModel.compute_order_from_data(U_train)

            model = StaticDVineModel(U, fixed_order=fixed_order)
            model.fit(U_train)

            pvc_seeds = make_pvc_seeds(seed, t_idx=0)
            sim = model.simulate(n_sim=50, t_idx=0, seeds=pvc_seeds)
            return sim

        sim1 = run_workflow(42)
        sim2 = run_workflow(999)

        assert not np.allclose(sim1, sim2), (
            "Different seeds should produce different simulations"
        )
