#!/usr/bin/env python3
"""
Manifest validation script.

Validates that all files in a manifest.json:
1. Exist in the directory
2. Have matching SHA256 hashes

Usage:
    python scripts/validate_manifest.py outputs/demo/manifest.json
    python scripts/validate_manifest.py outputs/demo/  # infers manifest.json

Exit codes:
    0 - All files valid
    1 - Validation failed (missing file or hash mismatch)
    2 - Manifest not found or invalid JSON
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Tuple, List, Dict


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(manifest_path: Path, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate a manifest file.

    Parameters
    ----------
    manifest_path : Path
        Path to manifest.json
    verbose : bool
        Print status messages

    Returns
    -------
    Tuple[bool, List[str]]
        (success, list of error messages)
    """
    errors = []
    base_dir = manifest_path.parent

    # Load manifest
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except FileNotFoundError:
        return False, [f"Manifest not found: {manifest_path}"]
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in manifest: {e}"]

    if "files" not in manifest:
        return False, ["Manifest missing 'files' key"]

    files = manifest["files"]
    if not files:
        return False, ["Manifest has no files listed"]

    # Validate each file
    for filename, expected_hash in files.items():
        file_path = base_dir / filename

        if not file_path.exists():
            errors.append(f"MISSING: {filename}")
            if verbose:
                print(f"  {filename}: MISSING")
            continue

        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            errors.append(f"MISMATCH: {filename} (expected {expected_hash[:12]}..., got {actual_hash[:12]}...)")
            if verbose:
                print(f"  {filename}: MISMATCH")
        else:
            if verbose:
                print(f"  {filename}: OK")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate manifest.json integrity")
    parser.add_argument("path", type=str, help="Path to manifest.json or directory containing it")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output except errors")
    args = parser.parse_args()

    path = Path(args.path)

    # If directory, look for manifest.json
    if path.is_dir():
        manifest_path = path / "manifest.json"
    else:
        manifest_path = path

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(2)

    if not args.quiet:
        print(f"Validating: {manifest_path}")

    success, errors = validate_manifest(manifest_path, verbose=not args.quiet)

    if success:
        if not args.quiet:
            print("OK: All files valid")
        sys.exit(0)
    else:
        print(f"FAILED: {len(errors)} error(s)", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
