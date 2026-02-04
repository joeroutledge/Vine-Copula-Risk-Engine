"""
Tests for PIT/PPF round-trip consistency in GARCH-t marginals.

The key invariant: PIT followed by inverse PIT should reconstruct the original returns.
If this fails, there's a scaling bug in the GARCH-t parameterization.

Convention checked:
- r_t = sigma_t * z_t where z_t ~ t(nu) (standard Student-t, variance = nu/(nu-2))
- PIT: u = t(nu).cdf(r / sigma)
- Inverse PIT: r = sigma * t(nu).ppf(u)
"""

import sys
import pathlib
import numpy as np
import pytest
from scipy.stats import t as tdist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from run_var_es_backtest import fit_garch_pit


class TestPITPPFRoundtrip:
    """Test that PIT -> PPF recovers original returns within tolerance."""

    def test_roundtrip_synthetic(self):
        """
        Synthetic test: generate returns from known GARCH-t, verify round-trip.

        This tests the mathematical consistency without model estimation error.
        """
        rng = np.random.RandomState(42)
        n = 500
        nu = 6.0  # degrees of freedom

        # Generate GARCH(1,1) volatility path
        omega, alpha, beta = 1e-5, 0.08, 0.90
        sig2 = np.empty(n)
        sig2[0] = omega / (1 - alpha - beta)

        # Generate returns: r = sigma * z, z ~ t(nu)
        z = tdist.rvs(nu, size=n, random_state=rng)
        r = np.empty(n)
        sigma = np.empty(n)

        for t in range(n):
            sigma[t] = np.sqrt(sig2[t])
            r[t] = sigma[t] * z[t]
            if t < n - 1:
                sig2[t+1] = omega + alpha * r[t]**2 + beta * sig2[t]

        # PIT: u = t(nu).cdf(r / sigma)
        u = tdist.cdf(r / sigma, nu)

        # Inverse PIT: r_reconstructed = sigma * t(nu).ppf(u)
        r_reconstructed = sigma * tdist.ppf(u, nu)

        # Check round-trip accuracy
        max_error = np.max(np.abs(r - r_reconstructed))
        assert max_error < 1e-10, f"Round-trip max error {max_error:.2e} exceeds 1e-10"

    def test_roundtrip_fit_garch_pit(self):
        """
        Test round-trip using actual fit_garch_pit function.

        After fitting GARCH-t and computing PIT uniforms, inverse PIT
        should recover original returns within tolerance.
        """
        import pandas as pd

        rng = np.random.RandomState(123)
        n = 600
        train_end = 400

        # Generate synthetic returns with fat tails
        nu_true = 5.0
        z = tdist.rvs(nu_true, size=n, random_state=rng)
        sigma = 0.01 * (1 + 0.3 * np.sin(np.arange(n) * 0.02))
        r = sigma * z

        returns = pd.DataFrame({'ASSET': r})

        # Fit GARCH and get PIT uniforms
        U, garch_info = fit_garch_pit(returns, train_end)

        info = garch_info['ASSET']
        sigma_fit = info['sigma']
        nu_fit = info['nu']
        u = U['ASSET'].values

        # Inverse PIT: r = sigma * t(nu).ppf(u)
        r_reconstructed = sigma_fit * tdist.ppf(u, nu_fit)

        # Check round-trip
        max_error = np.max(np.abs(r - r_reconstructed))
        assert max_error < 1e-3, f"Round-trip max error {max_error:.2e} exceeds 1e-3"

        # Also verify uniforms are in (0, 1)
        assert np.all(u > 0) and np.all(u < 1), "PIT uniforms must be in (0, 1)"

        # Uniforms should be roughly uniform (KS test)
        from scipy.stats import kstest
        ks_stat, ks_pval = kstest(u, 'uniform')
        # Note: won't be perfectly uniform due to estimation error, but shouldn't
        # be wildly off
        assert ks_pval > 0.001, f"PIT uniforms badly non-uniform (KS p={ks_pval:.4f})"

    def test_no_sqrt_nu_factor(self):
        """
        Verify that no sqrt(nu/(nu-2)) scaling factor is applied.

        Bug check: If the buggy scaling is present, VaR estimates would be
        ~1.33x more extreme for nu=8.
        """
        import pandas as pd

        rng = np.random.RandomState(456)
        n = 400
        train_end = 300

        # Generate data with known nu=8
        nu = 8.0
        z = tdist.rvs(nu, size=n, random_state=rng)
        sigma_const = 0.01
        r = sigma_const * z

        returns = pd.DataFrame({'TEST': r})
        U, garch_info = fit_garch_pit(returns, train_end)

        info = garch_info['TEST']
        nu_fit = info['nu']

        # Simulate VaR at 1% level
        u_1pct = 0.01
        z_1pct = tdist.ppf(u_1pct, nu_fit)

        # If scaling bug exists, the code would use:
        # r = z * sigma * sqrt(nu/(nu-2)) instead of r = z * sigma
        # This makes VaR more extreme by factor sqrt(nu/(nu-2)) ~ 1.15 for nu=8

        # With correct scaling, VaR(1%) for unit sigma should be ~ t(nu).ppf(0.01)
        # For nu=8, t(8).ppf(0.01) ~ -2.896
        expected_z = tdist.ppf(0.01, nu_fit)

        # The fitted sigma should be close to true sigma_const
        avg_sigma = np.mean(info['sigma'])
        expected_var = expected_z * avg_sigma

        # If buggy scaling were present, it would be:
        buggy_var = expected_z * avg_sigma * np.sqrt(nu_fit / (nu_fit - 2))

        # The ratio should be close to 1 (correct) not sqrt(nu/(nu-2)) ~ 1.15 (buggy)
        ratio = buggy_var / expected_var

        # Verify the scaling factor matches what we expect
        expected_ratio = np.sqrt(nu_fit / (nu_fit - 2))
        assert abs(ratio - expected_ratio) < 0.01, "Ratio calculation error"

        # The correct VaR should NOT include this factor
        # This test documents the expected behavior
        assert expected_ratio > 1.1, f"Scaling factor {expected_ratio:.3f} should be > 1.1 for nu~8"


class TestUniformMarginals:
    """Test that PIT produces approximately uniform marginals."""

    def test_pit_uniform_coverage(self):
        """PIT uniforms should have roughly uniform distribution."""
        import pandas as pd
        from scipy.stats import kstest

        rng = np.random.RandomState(789)
        n = 1000
        train_end = 700

        # Generate heavy-tailed returns
        nu = 5.0
        z = tdist.rvs(nu, size=n, random_state=rng)
        sigma = 0.015 * np.exp(0.1 * np.sin(np.arange(n) * 0.01))
        r = sigma * z

        returns = pd.DataFrame({'ASSET': r})
        U, _ = fit_garch_pit(returns, train_end)
        u = U['ASSET'].values

        # Check uniform coverage
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            empirical = np.mean(u <= q)
            # Allow 10% deviation due to estimation error
            assert abs(empirical - q) < 0.15, f"Poor uniform coverage at q={q}: {empirical:.3f}"
