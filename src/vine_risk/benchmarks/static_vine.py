"""
Static D-vine copula benchmark.

This module implements a static (time-invariant) D-vine copula model
for comparison with dynamic GAS specifications.

Pair-copula family selection uses pyvinecopulib with a restricted set:
{Gaussian, Student-t, Clayton, Gumbel, Frank} + rotations, selected by BIC.
"""

import numpy as np
import pandas as pd
import pyvinecopulib as pvc
from typing import Dict, List, Tuple, Optional
from scipy.stats import t as tdist, kendalltau
from scipy.optimize import minimize

from ..core.copulas import log_student_t_copula_density, clip01
from ..core.tail_dependence import lower_tail_dependence_t
from ..core.pvc_compat import (
    make_dvine_structure, fit_vine_from_data, build_vine_from_pair_copulas, make_bicop_student
)

# Defensible family set: parametric, well-studied, covers symmetric +
# upper/lower tail dependence patterns.
FAMILY_SET = [
    pvc.BicopFamily.gaussian,
    pvc.BicopFamily.student,
    pvc.BicopFamily.clayton,
    pvc.BicopFamily.gumbel,
    pvc.BicopFamily.frank,
]


class StaticDVineModel:
    """
    Static D-vine copula model with time-invariant parameters.

    Uses pyvinecopulib's native BIC-based family selection from a
    restricted family set for each pair-copula edge.

    Parameters
    ----------
    U : pd.DataFrame
        Uniform marginals from PIT
    nu_fixed : float, optional
        Degrees of freedom for copula edges. If None, estimated per-edge.
        Deprecated in favour of automatic family selection.
    fixed_order : List[int], optional
        Fixed variable ordering
    """

    def __init__(
        self,
        U: pd.DataFrame,
        nu_fixed: Optional[float] = None,
        fixed_order: Optional[List[int]] = None,
    ):
        self.U = U
        self.d = U.shape[1]
        self.assets = list(U.columns)
        self.nu_fixed = nu_fixed
        self.estimate_nu = nu_fixed is None

        # IMPORTANT: Track order source for OOS leakage prevention.
        # If fixed_order is provided, the order was computed externally
        # (ideally from training data only). If not, order is inferred
        # from the full U, which may leak OOS information.
        if fixed_order is not None:
            self.order = list(fixed_order)
            self._order_source = "train_only_fixed"
        else:
            self.order = self._determine_order(U)
            self._order_source = "inferred_from_full_u"

        self.U_ordered = U.iloc[:, self.order]
        self.static_params: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._vine: Optional[pvc.Vinecop] = None

    def _determine_order(self, U: pd.DataFrame) -> List[int]:
        """Determine D-vine order using greedy Kendall's tau algorithm."""
        return self.compute_order_from_data(U)

    @staticmethod
    def compute_order_from_data(U: pd.DataFrame) -> List[int]:
        """
        Compute D-vine order using greedy Kendall's tau algorithm.

        This is exposed as a static method to allow external code to compute
        ordering from training data only, preventing OOS leakage.

        Parameters
        ----------
        U : pd.DataFrame
            Uniform marginals (PIT)

        Returns
        -------
        List[int]
            Column indices specifying D-vine variable order
        """
        d = U.shape[1]
        tau_matrix = np.zeros((d, d))

        for i in range(d):
            for j in range(i + 1, d):
                tau, _ = kendalltau(U.iloc[:, i], U.iloc[:, j])
                tau_matrix[i, j] = tau_matrix[j, i] = abs(tau) if np.isfinite(tau) else 0.0

        start = int(np.argmax(tau_matrix.mean(axis=1)))
        path = [start]
        remaining = set(range(d)) - {start}

        while remaining:
            last = path[-1]
            next_node = max(remaining, key=lambda j: tau_matrix[last, j])
            path.append(next_node)
            remaining.remove(next_node)

        return path

    def fit(self, U_train: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Fit static vine copula using pyvinecopulib's BIC-based family
        selection from a restricted family set.

        Parameters
        ----------
        U_train : pd.DataFrame
            Training uniform marginals

        Returns
        -------
        Dict
            Fitted (rho, nu) for each (tree, edge) — for backward compat.
            For non-Student-t families, rho is Kendall's tau and nu=0.
        """
        U_ord = clip01(U_train.iloc[:, self.order].values)

        # Build D-vine structure using compat wrapper
        order_1 = [i + 1 for i in range(self.d)]
        struct = make_dvine_structure(self.d, order_1)

        controls = pvc.FitControlsVinecop(
            family_set=FAMILY_SET,
            selection_criterion="bic",
            allow_rotations=True,
        )

        # Fit using compat wrapper
        vine = fit_vine_from_data(U_ord, struct, controls)
        self._vine = vine

        # Extract parameters for backward compat with GAS model
        for tree in range(1, self.d):
            for edge in range(self.d - tree):
                bc = vine.get_pair_copula(tree - 1, edge)
                params = bc.parameters
                if bc.family == pvc.BicopFamily.student:
                    rho = float(params[0, 0]) if params.ndim == 2 else float(params[0])
                    nu = float(params[1, 0]) if params.ndim == 2 else float(params[1])
                elif bc.family == pvc.BicopFamily.gaussian:
                    rho = float(params[0, 0]) if params.ndim == 2 else float(params[0])
                    nu = 999.0  # Gaussian = t with nu->inf
                else:
                    # For Archimedean families, store Kendall's tau as rho proxy
                    tau = bc.tau
                    rho = float(tau) if np.isfinite(tau) else 0.0
                    nu = 0.0
                self.static_params[(tree, edge)] = (rho, nu)

        return self.static_params

    def refit_params(self, U_window: pd.DataFrame) -> None:
        """
        Refit pair-copula parameters on new data, keeping structure fixed.

        This allows rolling refits without re-selecting the D-vine order
        or pair-copula families. Used for rolling marginal/copula updates.

        Parameters
        ----------
        U_window : pd.DataFrame
            Rolling window of uniform marginals for parameter estimation
        """
        U_ord = clip01(U_window.iloc[:, self.order].values)

        # Keep same structure using compat wrapper
        order_1 = [i + 1 for i in range(self.d)]
        struct = make_dvine_structure(self.d, order_1)

        # Refit with same family set (BIC may select different families)
        controls = pvc.FitControlsVinecop(
            family_set=FAMILY_SET,
            selection_criterion="bic",
            allow_rotations=True,
        )

        # Fit using compat wrapper
        vine = fit_vine_from_data(U_ord, struct, controls)
        self._vine = vine

        # Update static_params for backward compat
        for tree in range(1, self.d):
            for edge in range(self.d - tree):
                bc = vine.get_pair_copula(tree - 1, edge)
                params = bc.parameters
                if bc.family == pvc.BicopFamily.student:
                    rho = float(params[0, 0]) if params.ndim == 2 else float(params[0])
                    nu = float(params[1, 0]) if params.ndim == 2 else float(params[1])
                elif bc.family == pvc.BicopFamily.gaussian:
                    rho = float(params[0, 0]) if params.ndim == 2 else float(params[0])
                    nu = 999.0
                else:
                    tau = bc.tau
                    rho = float(tau) if np.isfinite(tau) else 0.0
                    nu = 0.0
                self.static_params[(tree, edge)] = (rho, nu)

    def compute_log_likelihood(
        self,
        U_test: pd.DataFrame,
    ) -> np.ndarray:
        """Compute log-likelihood for test data."""
        vine = self._get_vine()
        U_ord = clip01(U_test.iloc[:, self.order].values)
        n = len(U_ord)

        ll = np.zeros(n)
        for t in range(n):
            u_row = U_ord[t:t + 1, :]
            ll[t] = float(vine.loglik(u_row))

        return ll

    def _get_vine(self) -> pvc.Vinecop:
        """Return the fitted vine copula object."""
        if self._vine is not None:
            return self._vine
        # Fallback: build from static_params (backward compat)
        return self._build_vine_from_params()

    def _build_vine_from_params(self) -> pvc.Vinecop:
        """Build pyvinecopulib Vinecop from stored static_params."""
        order_1 = [i + 1 for i in range(self.d)]
        struct = make_dvine_structure(self.d, order_1)

        pcs = []
        for tree in range(1, self.d):
            row = []
            for edge in range(self.d - tree):
                rho, nu = self.static_params[(tree, edge)]
                bicop = make_bicop_student(rho, nu)
                row.append(bicop)
            pcs.append(row)

        return build_vine_from_pair_copulas(struct, pcs)

    def simulate(
        self,
        n_sim: int,
        t_idx: int = 0,
        use_antithetic: bool = False,
        seeds: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Simulate from the static vine copula.

        Parameters
        ----------
        n_sim : int
            Number of simulations
        t_idx : int, optional
            Time index (ignored for static model, kept for interface compat)
        use_antithetic : bool
            Use antithetic variates for variance reduction
        seeds : List[int], optional
            Seeds for pyvinecopulib RNG. If provided, simulation is
            deterministic. Use make_pvc_seeds() to generate these.

        Returns
        -------
        np.ndarray
            Simulated uniforms (n_sim x d)
        """
        vine = self._get_vine()
        if seeds is not None:
            return vine.simulate(n_sim, seeds=seeds)
        else:
            return vine.simulate(n_sim)

    def predict_tree1_rhos(self, t_idx: int = 0) -> np.ndarray:
        """Get Tree-1 correlations (static, ignores t_idx)."""
        rhos = np.zeros(self.d - 1)
        for edge in range(self.d - 1):
            rho, _ = self.static_params[(1, edge)]
            rhos[edge] = rho
        return rhos

    def get_pair_copula_summary(self) -> pd.DataFrame:
        """Summary of selected pair-copula families and parameters."""
        vine = self._get_vine()
        rows = []
        for tree in range(1, self.d):
            for edge in range(self.d - tree):
                bc = vine.get_pair_copula(tree - 1, edge)
                rows.append({
                    'tree': tree, 'edge': edge,
                    'family': str(bc.family).split('.')[-1],
                    'rotation': bc.rotation,
                    'tau': float(bc.tau) if np.isfinite(bc.tau) else 0.0,
                })
        return pd.DataFrame(rows)

    def get_model_card(self) -> dict:
        """
        Export full model card with all pair-copula details.

        Returns a dictionary suitable for JSON serialization containing:
        - dim, vine_type, variable_order
        - trunc_lvl, is_full_vine
        - total_pair_copulas
        - pair_copulas: list of {tree, edge_index, conditioned, conditioning,
                                 family, rotation, parameters, kendall_tau}
        """
        vine = self._get_vine()
        d = vine.dim
        trunc_lvl = vine.trunc_lvl

        # Variable order: map vine order (1-indexed) to asset names
        vine_order = vine.order  # 1-indexed
        ordered_assets = [self.assets[self.order[i]] for i in range(d)]

        # For a full D-vine, trunc_lvl == d-1
        is_full = (trunc_lvl == d - 1)

        # Count pair copulas: sum_{tree=1..trunc_lvl} (d - tree)
        total_pairs = sum(d - tree for tree in range(1, trunc_lvl + 1))

        # Extract pair copula details
        # For D-vine with order [v_0, v_1, ..., v_{d-1}]:
        # Tree t (1-indexed): edge e connects v_e and v_{e+t}, conditioned on v_{e+1}..v_{e+t-1}
        pair_copulas = []
        for tree in range(trunc_lvl):  # 0-indexed in pvc
            num_edges = d - 1 - tree
            for edge in range(num_edges):
                bc = vine.get_pair_copula(tree, edge)

                # For D-vine: conditioned variables are at positions edge and edge+tree+1
                # Conditioning variables are positions edge+1, ..., edge+tree
                cond_left = ordered_assets[edge]
                cond_right = ordered_assets[edge + tree + 1]
                conditioned = [cond_left, cond_right]

                conditioning = []
                for k in range(1, tree + 1):
                    conditioning.append(ordered_assets[edge + k])

                # Extract family and parameters
                family = str(bc.family).split('.')[-1]
                rotation = int(bc.rotation)
                params = bc.parameters
                if params.size > 0:
                    param_list = params.flatten().tolist()
                else:
                    param_list = []
                tau = float(bc.tau) if np.isfinite(bc.tau) else 0.0

                pair_copulas.append({
                    "tree": tree + 1,  # 1-indexed for display
                    "edge_index": edge,
                    "conditioned": conditioned,
                    "conditioning": conditioning,
                    "family": family,
                    "rotation": rotation,
                    "parameters": param_list,
                    "kendall_tau": round(tau, 6),
                })

        return {
            "dim": d,
            "vine_type": "D-vine",
            "variable_order": ordered_assets,
            "order_indices": self.order,
            "order_source": self._order_source,
            "trunc_lvl": trunc_lvl,
            "is_full_vine": is_full,
            "total_pair_copulas": total_pairs,
            "family_set": [str(f).split('.')[-1] for f in FAMILY_SET],
            "pair_copulas": pair_copulas,
        }

    def get_tail_dependence(self) -> pd.DataFrame:
        """Get tail dependence for Tree-1 edges (Student-t edges only)."""
        rows = []
        for edge in range(self.d - 1):
            i = self.order[edge]
            j = self.order[edge + 1]
            rho, nu = self.static_params[(1, edge)]
            if nu > 0 and nu < 100:
                lambda_L = lower_tail_dependence_t(rho, nu)
            else:
                lambda_L = 0.0  # Gaussian or Archimedean

            rows.append({
                'edge': edge,
                'asset_i': self.assets[i],
                'asset_j': self.assets[j],
                'rho': rho,
                'nu': nu,
                'lambda_L': lambda_L,
            })

        return pd.DataFrame(rows)
