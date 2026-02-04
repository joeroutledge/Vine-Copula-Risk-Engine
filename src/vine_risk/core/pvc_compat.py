"""
pyvinecopulib API compatibility wrappers.

Ensures code works across pyvinecopulib versions (tested: 0.6.7+, 0.7.x).
Uses feature detection instead of version parsing for robustness.
"""

import numpy as np
import pyvinecopulib as pvc
from typing import List, Optional


def make_dvine_structure(dim: int, order: Optional[List[int]] = None) -> "pvc.RVineStructure":
    """
    Create a D-vine structure with the given dimension and order.

    Uses DVineStructure if available (0.7+), falls back to RVineStructure.

    Parameters
    ----------
    dim : int
        Number of variables
    order : List[int], optional
        Variable order (1-indexed). If None, uses [1, 2, ..., dim].

    Returns
    -------
    pvc.RVineStructure or pvc.DVineStructure
        The D-vine structure object
    """
    if order is None:
        order = list(range(1, dim + 1))

    if hasattr(pvc, "DVineStructure"):
        return pvc.DVineStructure.from_order(order)
    else:
        # Fallback for older pyvinecopulib
        return pvc.RVineStructure.from_order(order)


def fit_vine_from_data(
    data: np.ndarray,
    structure: "pvc.RVineStructure",
    controls: "pvc.FitControlsVinecop",
) -> "pvc.Vinecop":
    """
    Fit a vine copula from data.

    Uses Vinecop.from_data if available, otherwise Vinecop constructor + fit.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_obs x dim), values in (0, 1)
    structure : pvc.RVineStructure
        Vine structure
    controls : pvc.FitControlsVinecop
        Fit controls

    Returns
    -------
    pvc.Vinecop
        Fitted vine copula
    """
    data = np.ascontiguousarray(data, dtype=np.float64)

    if hasattr(pvc.Vinecop, "from_data"):
        return pvc.Vinecop.from_data(data, structure=structure, controls=controls)
    else:
        # Fallback: create empty vine and select
        vine = pvc.Vinecop(structure)
        vine.select(data, controls)
        return vine


def build_vine_from_pair_copulas(
    structure: "pvc.RVineStructure",
    pair_copulas: List[List["pvc.Bicop"]],
) -> "pvc.Vinecop":
    """
    Build a vine copula from pair copulas.

    Uses Vinecop.from_structure if available, otherwise Vinecop constructor.

    Parameters
    ----------
    structure : pvc.RVineStructure
        Vine structure
    pair_copulas : List[List[pvc.Bicop]]
        Pair copulas for each tree

    Returns
    -------
    pvc.Vinecop
        Vine copula with specified pair copulas
    """
    if hasattr(pvc.Vinecop, "from_structure"):
        return pvc.Vinecop.from_structure(structure=structure, pair_copulas=pair_copulas)
    else:
        # Fallback: create vine and set pair copulas manually
        vine = pvc.Vinecop(structure)
        for tree_idx, row in enumerate(pair_copulas):
            for edge_idx, bc in enumerate(row):
                vine.set_pair_copula(tree_idx, edge_idx, bc)
        return vine


def make_bicop_student(rho: float, nu: float) -> "pvc.Bicop":
    """
    Create a Student-t bivariate copula.

    Handles different parameter array formats across versions.

    Parameters
    ----------
    rho : float
        Correlation parameter, clipped to (-0.99, 0.99)
    nu : float
        Degrees of freedom, clipped to (2.1, 50)

    Returns
    -------
    pvc.Bicop
        Student-t bivariate copula
    """
    rho = float(np.clip(rho, -0.99, 0.99))
    nu = float(np.clip(nu, 2.10, 50.0))

    try:
        # Try 1D array format (newer versions)
        return pvc.Bicop(pvc.BicopFamily.student, 0, np.array([rho, nu], dtype=float))
    except Exception:
        # Fallback to 2D column format
        return pvc.Bicop(pvc.BicopFamily.student, 0,
                         np.asfortranarray([[rho], [nu]], dtype=float))


def get_pvc_version_info() -> dict:
    """
    Get pyvinecopulib version and available features.

    Returns
    -------
    dict
        Version info and feature flags
    """
    return {
        "version": getattr(pvc, "__version__", "unknown"),
        "has_DVineStructure": hasattr(pvc, "DVineStructure"),
        "has_from_data": hasattr(pvc.Vinecop, "from_data"),
        "has_from_structure": hasattr(pvc.Vinecop, "from_structure"),
    }
