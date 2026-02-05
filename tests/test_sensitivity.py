"""
Tests for sensitivity analysis script.

Runs in tiny mode (SENSITIVITY_TINY=1) to minimize CI runtime.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


# Expected columns in sensitivity_summary.csv
EXPECTED_COLUMNS = [
    "model",
    "config_id",
    "n_sim",
    "nu_fixed",
    "alpha",
    "hit_rate",
    "kupiec_p",
    "christoffersen_p",
    "es_shortfall_ratio",
    "pinball_loss",
    "n_oos",
]


@pytest.fixture(scope="module")
def sensitivity_output():
    """Run sensitivity script in tiny mode and return output path."""
    out_dir = Path("outputs/sensitivity_quick")
    csv_path = out_dir / "sensitivity_summary.csv"

    # Run in tiny mode
    env = os.environ.copy()
    env["SENSITIVITY_TINY"] = "1"

    result = subprocess.run(
        [sys.executable, "scripts/run_sensitivity.py"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        pytest.fail(f"Sensitivity script failed with code {result.returncode}")

    return csv_path


class TestSensitivityOutput:
    """Test that sensitivity analysis produces valid output."""

    def test_csv_created(self, sensitivity_output):
        """CSV file should be created."""
        assert sensitivity_output.exists(), (
            f"Expected {sensitivity_output} to exist"
        )

    def test_required_columns(self, sensitivity_output):
        """CSV should have all required columns."""
        df = pd.read_csv(sensitivity_output)
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_deterministic_row_count(self, sensitivity_output):
        """Row count should be deterministic for tiny grid."""
        df = pd.read_csv(sensitivity_output)
        # Tiny grid: 1 model * 1 n_sim * 1 nu_fixed * 2 alphas = 2 rows
        expected_rows = 2
        assert len(df) == expected_rows, (
            f"Expected {expected_rows} rows in tiny mode, got {len(df)}"
        )

    def test_manifest_created(self, sensitivity_output):
        """Manifest JSON should be created."""
        manifest_path = sensitivity_output.parent / "manifest.json"
        assert manifest_path.exists(), (
            f"Expected {manifest_path} to exist"
        )

    def test_values_reasonable(self, sensitivity_output):
        """Check that output values are within reasonable ranges."""
        df = pd.read_csv(sensitivity_output)

        # hit_rate should be in [0, 1]
        assert (df["hit_rate"] >= 0).all() and (df["hit_rate"] <= 1).all(), (
            "hit_rate should be in [0, 1]"
        )

        # kupiec_p should be in [0, 1] or NaN
        kp = df["kupiec_p"].dropna()
        assert (kp >= 0).all() and (kp <= 1).all(), (
            "kupiec_p should be in [0, 1]"
        )

        # n_oos should be positive
        assert (df["n_oos"] > 0).all(), "n_oos should be positive"

        # pinball_loss should be non-negative
        pl = df["pinball_loss"].dropna()
        assert (pl >= 0).all(), "pinball_loss should be non-negative"


class TestSensitivityStructure:
    """Test that sensitivity analysis structure is consistent."""

    def test_consistent_structure(self):
        """Running twice should produce same structure (columns, row count, config)."""
        out_dir = Path("outputs/sensitivity_quick")
        csv_path = out_dir / "sensitivity_summary.csv"

        env = os.environ.copy()
        env["SENSITIVITY_TINY"] = "1"

        # Run first time
        subprocess.run(
            [sys.executable, "scripts/run_sensitivity.py"],
            env=env,
            capture_output=True,
            timeout=300,
        )
        df1 = pd.read_csv(csv_path)

        # Run second time
        subprocess.run(
            [sys.executable, "scripts/run_sensitivity.py"],
            env=env,
            capture_output=True,
            timeout=300,
        )
        df2 = pd.read_csv(csv_path)

        # Structure should be identical
        assert list(df1.columns) == list(df2.columns), "Columns should match"
        assert len(df1) == len(df2), "Row count should match"

        # Config columns should be exactly identical
        for col in ["model", "config_id", "n_sim", "nu_fixed", "alpha", "n_oos"]:
            pd.testing.assert_series_equal(
                df1[col], df2[col], check_names=False,
                obj=f"Column {col}"
            )

        # Note: Numeric columns (hit_rate, kupiec_p, etc.) may vary slightly
        # between runs due to pyvinecopulib internal randomness. This is
        # acceptable for a sensitivity analysis which tests robustness.
