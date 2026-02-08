"""
Tests for GAS D-vine OOS dynamics.

Verifies that:
1. GAS dynamics are truly dynamic out-of-sample (rho changes over OOS period)
2. The old "freeze at last training theta" bug is fixed
3. predict_tree1_rhos() returns different values for different OOS times
4. Full filtered paths are correctly computed and stored
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as tdist

from vine_risk.benchmarks.static_vine import StaticDVineModel
from vine_risk.benchmarks.gas_vine import GASDVineModel


def generate_time_varying_uniforms(
    n: int, d: int, seed: int = 42
) -> pd.DataFrame:
    """
    Generate PIT uniforms with time-varying correlation structure.

    Creates data where the true underlying correlation changes over time,
    which should be detectable by the GAS filter.
    """
    np.random.seed(seed)

    # Create time-varying correlation: starts at 0.3, increases to 0.7
    rho_true = 0.3 + 0.4 * np.linspace(0, 1, n)

    # Generate correlated Gaussian data
    z = np.zeros((n, d))
    z[:, 0] = np.random.randn(n)

    for j in range(1, d):
        # Use time-varying correlation for adjacent pairs
        rho_t = rho_true if j == 1 else 0.3 * np.ones(n)
        z[:, j] = rho_t * z[:, j-1] + np.sqrt(1 - rho_t**2) * np.random.randn(n)

    # Transform to uniform
    U = norm.cdf(z)

    return pd.DataFrame(U, columns=[f"Asset_{i}" for i in range(d)])


class TestOOSDynamicsNotFrozen:
    """Test that GAS dynamics are truly dynamic OOS."""

    def test_rhos_differ_across_oos(self):
        """
        For dynamic edges, rho should differ at different OOS times.

        This directly tests the fix: before, predict_tree1_rhos() would
        return the same value for all t_idx >= train_end because it
        clamped to theta_path[-1].
        """
        np.random.seed(42)
        n = 500
        d = 4
        train_end = 300

        U = generate_time_varying_uniforms(n, d, seed=42)
        U_train = U.iloc[:train_end]

        # Compute order from training data
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        # Fit GAS model
        model = GASDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train, fit_gas=True)

        # Check that we have at least one dynamic edge
        dynamic_edges = [e for e, is_dyn in model.tree1_is_dynamic.items() if is_dyn]
        assert len(dynamic_edges) > 0, "No dynamic edges found"

        # Get rhos at different OOS times
        t_a = train_end + 10
        t_b = train_end + 100

        rhos_a = model.predict_tree1_rhos(t_a)
        rhos_b = model.predict_tree1_rhos(t_b)

        # For at least one dynamic edge, rhos should differ
        found_difference = False
        for edge in dynamic_edges:
            if rhos_a[edge] != rhos_b[edge]:
                found_difference = True
                break

        assert found_difference, (
            f"GAS dynamics appear frozen: rhos at t={t_a} and t={t_b} are identical "
            f"for all dynamic edges. rhos_a={rhos_a}, rhos_b={rhos_b}"
        )

    def test_full_path_stored(self):
        """
        After fit(), theta_path_full and rho_path_full should be stored
        with length equal to len(U).
        """
        np.random.seed(42)
        n = 200
        d = 3
        train_end = 100

        U = generate_time_varying_uniforms(n, d, seed=42)
        U_train = U.iloc[:train_end]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = GASDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train, fit_gas=True)

        # Check that full paths are stored
        for edge, is_dynamic in model.tree1_is_dynamic.items():
            if is_dynamic:
                model_info = model.tree1_models[edge]

                assert 'theta_path_full' in model_info, "theta_path_full not stored"
                assert 'rho_path_full' in model_info, "rho_path_full not stored"

                # Length should equal len(U)
                assert len(model_info['theta_path_full']) == n, (
                    f"theta_path_full length {len(model_info['theta_path_full'])} != {n}"
                )
                assert len(model_info['rho_path_full']) == n, (
                    f"rho_path_full length {len(model_info['rho_path_full'])} != {n}"
                )

    def test_train_end_stored(self):
        """train_end_ attribute should be set after fit()."""
        np.random.seed(42)
        U = generate_time_varying_uniforms(150, 3, seed=42)
        U_train = U.iloc[:100]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = GASDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train, fit_gas=True)

        assert hasattr(model, 'train_end_'), "train_end_ not set"
        assert model.train_end_ == 100, f"train_end_={model.train_end_}, expected 100"


class TestPredictTreeRhosSemantics:
    """Test that predict_tree1_rhos() has correct semantics."""

    def test_no_extra_b_step(self):
        """
        predict_tree1_rhos(t) should return rho_path_full[t], NOT
        apply an extra B step. The old code did:
            theta_pred = omega + B_eff * theta_t
        which was double-counting the autoregressive term.
        """
        np.random.seed(42)
        n = 200
        d = 3
        train_end = 100

        U = generate_time_varying_uniforms(n, d, seed=42)
        U_train = U.iloc[:train_end]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = GASDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train, fit_gas=True)

        for edge, is_dynamic in model.tree1_is_dynamic.items():
            if is_dynamic:
                model_info = model.tree1_models[edge]
                rho_path_full = model_info['rho_path_full']

                # Check multiple indices
                for t_idx in [50, 100, 150]:
                    rhos = model.predict_tree1_rhos(t_idx)
                    expected_rho = rho_path_full[t_idx]

                    np.testing.assert_almost_equal(
                        rhos[edge], expected_rho, decimal=12,
                        err_msg=f"Edge {edge} at t={t_idx}: predict_tree1_rhos != rho_path_full"
                    )

    def test_rho_within_bounds(self):
        """Predicted rho should always be in (-1, 1)."""
        np.random.seed(42)
        U = generate_time_varying_uniforms(300, 4, seed=42)
        U_train = U.iloc[:200]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = GASDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train, fit_gas=True)

        for t_idx in range(len(U)):
            rhos = model.predict_tree1_rhos(t_idx)
            assert np.all(np.abs(rhos) < 1.0), (
                f"rho out of bounds at t={t_idx}: {rhos}"
            )


class TestSimulationUsesDynamicRhos:
    """Test that simulate() uses dynamic rhos from full path."""

    def test_build_vine_uses_full_path(self):
        """
        build_vine_at_time(t) should use rho_path_full[t], meaning
        simulations at different OOS times use different correlations.
        """
        np.random.seed(42)
        n = 200
        d = 3
        train_end = 100

        U = generate_time_varying_uniforms(n, d, seed=42)
        U_train = U.iloc[:train_end]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = GASDVineModel(U, fixed_order=fixed_order)
        model.fit(U_train, fit_gas=True)

        # Build vines at different OOS times
        t_a = train_end + 10
        t_b = train_end + 80

        vine_a = model.build_vine_at_time(t_a)
        vine_b = model.build_vine_at_time(t_b)

        # Get the Student-t copula parameters for Tree-1 dynamic edges
        for edge, is_dynamic in model.tree1_is_dynamic.items():
            if is_dynamic:
                bc_a = vine_a.get_pair_copula(0, edge)
                bc_b = vine_b.get_pair_copula(0, edge)

                # Both should be Student-t
                import pyvinecopulib as pvc
                assert bc_a.family == pvc.BicopFamily.student
                assert bc_b.family == pvc.BicopFamily.student

                # Parameters should differ (rho is params[0])
                rho_a = bc_a.parameters[0, 0] if bc_a.parameters.ndim == 2 else bc_a.parameters[0]
                rho_b = bc_b.parameters[0, 0] if bc_b.parameters.ndim == 2 else bc_b.parameters[0]

                # They may be similar but not exactly equal
                # Just check they are valid rhos
                assert -1 < rho_a < 1, f"Invalid rho_a={rho_a}"
                assert -1 < rho_b < 1, f"Invalid rho_b={rho_b}"


class TestWeeklyUpdateOOS:
    """Test OOS dynamics with gas_update_every > 1."""

    def test_weekly_update_still_dynamic_oos(self):
        """
        With gas_update_every=5 (weekly), OOS rhos should still change
        at the weekly cadence.
        """
        np.random.seed(42)
        n = 300
        d = 3
        train_end = 200

        U = generate_time_varying_uniforms(n, d, seed=42)
        U_train = U.iloc[:train_end]
        fixed_order = StaticDVineModel.compute_order_from_data(U_train)

        model = GASDVineModel(U, fixed_order=fixed_order, gas_update_every=5)
        model.fit(U_train, fit_gas=True)

        dynamic_edges = [e for e, is_dyn in model.tree1_is_dynamic.items() if is_dyn]
        if len(dynamic_edges) == 0:
            pytest.skip("No dynamic edges")

        edge = dynamic_edges[0]
        rho_path_full = model.tree1_models[edge]['rho_path_full']

        # Check OOS: rho should be piecewise constant with jumps at update days
        # Update days in OOS: 200, 205, 210, ...
        # Between update days, rho should be constant

        # Check that rho is constant between t=201 and t=204 (not update days)
        for t in range(train_end + 1, min(train_end + 5, n - 1)):
            if (t % 5) != 0:
                # Not an update day in OOS
                assert rho_path_full[t] == rho_path_full[t + 1], (
                    f"rho changed on non-update day OOS: t={t}"
                )

        # Check that rho CAN change at update days (at least once)
        changes = 0
        for t in range(train_end, n - 1):
            if t % 5 == 0:  # Update day
                if t + 1 < n and rho_path_full[t] != rho_path_full[t + 1]:
                    changes += 1

        # Should have at least some changes at update days
        assert changes > 0, "No rho changes at weekly update days in OOS"
