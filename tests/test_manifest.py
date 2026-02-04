"""
Tests for manifest creation and validation.

Tests that:
1. Manifest excludes junk files (.DS_Store, .zip, etc.)
2. Manifest validation passes for valid files
3. Manifest validation fails for missing/modified files
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_manifest import sha256_file, validate_manifest


class TestManifestExclusion:
    """Test that junk files are excluded from manifest."""

    def test_ds_store_excluded(self, tmp_path):
        """Manifest should NOT include .DS_Store."""
        # Import build_manifest from backtest script
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_var_es_backtest import build_manifest

        # Create test files
        (tmp_path / "data.csv").write_text("col1,col2\n1,2\n")
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")

        manifest = build_manifest(tmp_path)

        assert "data.csv" in manifest["files"]
        assert ".DS_Store" not in manifest["files"]

    def test_zip_excluded(self, tmp_path):
        """Manifest should NOT include .zip files."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_var_es_backtest import build_manifest

        (tmp_path / "report.json").write_text('{"key": "value"}')
        (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04fake-zip")

        manifest = build_manifest(tmp_path)

        assert "report.json" in manifest["files"]
        assert "archive.zip" not in manifest["files"]

    def test_tar_gz_excluded(self, tmp_path):
        """Manifest should NOT include .tar.gz files."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_var_es_backtest import build_manifest

        (tmp_path / "results.csv").write_text("a,b\n1,2\n")
        (tmp_path / "backup.tar.gz").write_bytes(b"\x1f\x8bfake-gzip")

        manifest = build_manifest(tmp_path)

        assert "results.csv" in manifest["files"]
        assert "backup.tar.gz" not in manifest["files"]

    def test_gitkeep_excluded(self, tmp_path):
        """Manifest should NOT include .gitkeep."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_var_es_backtest import build_manifest

        (tmp_path / "metrics.json").write_text('{}')
        (tmp_path / ".gitkeep").write_text("")

        manifest = build_manifest(tmp_path)

        assert "metrics.json" in manifest["files"]
        assert ".gitkeep" not in manifest["files"]


class TestManifestValidation:
    """Test manifest validation logic."""

    def test_valid_manifest_passes(self, tmp_path):
        """Validation should pass when all files exist with correct hashes."""
        # Create test files
        file1 = tmp_path / "data.csv"
        file2 = tmp_path / "config.json"
        file1.write_text("col1,col2\n1,2\n3,4\n")
        file2.write_text('{"setting": true}')

        # Build manifest
        manifest = {
            "produced_utc": "2026-01-01T00:00:00Z",
            "files": {
                "data.csv": sha256_file(file1),
                "config.json": sha256_file(file2),
            }
        }

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Validate
        success, errors = validate_manifest(manifest_path, verbose=False)

        assert success is True
        assert len(errors) == 0

    def test_missing_file_fails(self, tmp_path):
        """Validation should fail when a manifest file is missing."""
        file1 = tmp_path / "existing.csv"
        file1.write_text("data")

        # Manifest references a non-existent file
        manifest = {
            "produced_utc": "2026-01-01T00:00:00Z",
            "files": {
                "existing.csv": sha256_file(file1),
                "missing.csv": "0" * 64,  # fake hash
            }
        }

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        success, errors = validate_manifest(manifest_path, verbose=False)

        assert success is False
        assert len(errors) == 1
        assert "MISSING" in errors[0]
        assert "missing.csv" in errors[0]

    def test_modified_file_fails(self, tmp_path):
        """Validation should fail when file content changes after manifest."""
        file1 = tmp_path / "data.csv"
        file1.write_text("original content")

        # Build manifest with original hash
        original_hash = sha256_file(file1)
        manifest = {
            "produced_utc": "2026-01-01T00:00:00Z",
            "files": {
                "data.csv": original_hash,
            }
        }

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Modify file AFTER manifest was written
        file1.write_text("modified content!")

        success, errors = validate_manifest(manifest_path, verbose=False)

        assert success is False
        assert len(errors) == 1
        assert "MISMATCH" in errors[0]
        assert "data.csv" in errors[0]

    def test_manifest_not_found(self, tmp_path):
        """Validation should fail gracefully for missing manifest."""
        fake_path = tmp_path / "nonexistent" / "manifest.json"

        success, errors = validate_manifest(fake_path, verbose=False)

        assert success is False
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_invalid_json(self, tmp_path):
        """Validation should fail gracefully for malformed JSON."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{invalid json")

        success, errors = validate_manifest(manifest_path, verbose=False)

        assert success is False
        assert len(errors) == 1
        assert "JSON" in errors[0]

    def test_empty_manifest_fails(self, tmp_path):
        """Validation should fail for manifest with no files."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text('{"produced_utc": "2026-01-01", "files": {}}')

        success, errors = validate_manifest(manifest_path, verbose=False)

        assert success is False
        assert "no files" in errors[0].lower()


class TestEndToEnd:
    """End-to-end test: build manifest then validate it."""

    def test_build_then_validate_roundtrip(self, tmp_path):
        """Built manifest should pass validation."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_var_es_backtest import build_manifest

        # Create legitimate output files
        (tmp_path / "metrics.json").write_text('{"sharpe": 1.5}')
        (tmp_path / "var_es.csv").write_text("date,var,es\n2020-01-01,0.02,0.03\n")
        (tmp_path / "model_card.json").write_text('{"dim": 5}')

        # Build manifest
        manifest = build_manifest(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Validate
        success, errors = validate_manifest(manifest_path, verbose=False)

        assert success is True
        assert len(errors) == 0
        assert len(manifest["files"]) == 3
