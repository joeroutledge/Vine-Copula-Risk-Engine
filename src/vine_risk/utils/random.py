"""
Centralized random number generation utilities.

This module provides a single source of truth for RNG seeding across
the vine copula risk engine. It handles both NumPy and pyvinecopulib
seeding to ensure reproducible simulations.

Determinism Contract
--------------------
Given a fixed seed and identical inputs, simulations WILL produce
identical outputs when:
1. NumPy seed is set via set_seed() before simulations
2. pyvinecopulib simulate() is called with the seeds parameter

The seeds parameter in pyvinecopulib.Vinecop.simulate() controls the
internal RNG for copula simulation. Pass a list of integer seeds.

Usage
-----
>>> from vine_risk.utils.random import set_seed, make_pvc_seeds
>>> set_seed(42)
>>> pvc_seeds = make_pvc_seeds(42, t_idx=0)
>>> U_sim = vine.simulate(n=1000, seeds=pvc_seeds)
"""

import numpy as np
from typing import List, Optional


# Determinism mode tracks whether reproducibility is enforced
_determinism_mode = "strict"  # Options: "strict", "relaxed"


def set_seed(seed: int) -> None:
    """
    Set the global NumPy random seed.

    This should be called at the start of any reproducible workflow.

    Parameters
    ----------
    seed : int
        Random seed (must be non-negative integer)
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    np.random.seed(seed)


def make_pvc_seeds(base_seed: int, t_idx: int = 0, n_seeds: int = 3) -> List[int]:
    """
    Generate deterministic seeds for pyvinecopulib.simulate().

    The seeds are derived from a base seed and time index to ensure:
    - Same base_seed + t_idx produces same output
    - Different t_idx produces different (but deterministic) output

    Parameters
    ----------
    base_seed : int
        Base random seed
    t_idx : int
        Time index (used to create distinct seeds for different time steps)
    n_seeds : int
        Number of seeds to generate (pyvinecopulib uses multiple)

    Returns
    -------
    List[int]
        List of integer seeds for pyvinecopulib.simulate(seeds=...)
    """
    # Use a local RNG to avoid affecting global state
    rng = np.random.RandomState(base_seed + t_idx * 10000)
    # Generate positive 32-bit integers for pyvinecopulib
    seeds = [int(rng.randint(1, 2**31 - 1)) for _ in range(n_seeds)]
    return seeds


def get_determinism_mode() -> str:
    """
    Get the current determinism mode.

    Returns
    -------
    str
        "strict" if full determinism is expected
        "relaxed" if some variation is tolerated
    """
    return _determinism_mode


def set_determinism_mode(mode: str) -> None:
    """
    Set the determinism mode.

    Parameters
    ----------
    mode : str
        "strict" for full reproducibility
        "relaxed" if some variation is acceptable
    """
    global _determinism_mode
    if mode not in ("strict", "relaxed"):
        raise ValueError(f"mode must be 'strict' or 'relaxed', got {mode}")
    _determinism_mode = mode


def get_determinism_info() -> dict:
    """
    Get information about determinism for model cards.

    Returns
    -------
    dict
        Determinism mode and implementation details
    """
    return {
        "determinism_mode": _determinism_mode,
        "numpy_seeded": True,
        "pyvinecopulib_seeded": True,
        "note": (
            "Full determinism achieved via NumPy seeding and pyvinecopulib "
            "seeds parameter. Same seed produces identical simulations."
        ),
    }
