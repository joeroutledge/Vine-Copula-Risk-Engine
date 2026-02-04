"""
GAS D-vine copula model.

This module implements a standard GAS (Generalized Autoregressive Score)
D-vine copula model. The model adds score-driven time-varying correlations
to the static D-vine baseline.

Model hierarchy:
1. Static: Fixed rho, nu — family selection via BIC
2. GAS (this module): omega + A * s_tilde_t + B * theta_{t-1}

GAS dynamics apply only to Tree-1 edges with ELLIPTICAL families
(gaussian or student-t), where the Fisher-z transformation is well-defined.
Non-elliptical Tree-1 edges (Clayton, Gumbel, Frank, etc.) remain static
with their BIC-selected family. Higher trees always use the static vine's
family selection.

Design rationale:
- Fisher-z score requires correlation parameter (elliptical copulas only)
- Forcing Student-t on Archimedean edges would misspecify tail dependence
- This approach respects BIC family selection while adding dynamics where
  methodologically justified

References:
    Creal, D., Koopman, S.J., & Lucas, A. (2013).
    "Generalized Autoregressive Score Models with Applications"
"""

import numpy as np
import pandas as pd
import pyvinecopulib as pvc
from typing import Dict, List, Tuple, Optional
from scipy.stats import t as tdist
from scipy.optimize import minimize

from ..core.copulas import log_student_t_copula_density, clip01
from ..core.gas import GASParameters, GASFilterState, gas_score_theta_t, scale_opg, gas_filter
from ..core.tail_dependence import lower_tail_dependence_t
from ..core.pvc_compat import make_dvine_structure, build_vine_from_pair_copulas, make_bicop_student
from .static_vine import StaticDVineModel


class GASDVineModel(StaticDVineModel):
    """
    GAS D-vine copula model with score-driven Tree-1 dynamics.

    GAS dynamics apply ONLY to Tree-1 edges with elliptical families
    (gaussian or student-t). Non-elliptical edges (Clayton, Gumbel, Frank)
    remain static with their BIC-selected family.

    Higher trees always use the static vine's BIC-selected families.

    Parameters
    ----------
    U : pd.DataFrame
        Uniform marginals from PIT
    nu_fixed : float, optional
        Degrees of freedom for copula edges (if None, estimated)
    fixed_order : List[int], optional
        Fixed variable ordering

    Attributes
    ----------
    tree1_models : Dict[int, Dict]
        GAS model info for dynamic edges only
    tree1_is_dynamic : Dict[int, bool]
        Whether each Tree-1 edge has GAS dynamics (True) or is static (False)
    """

    # Elliptical families that support GAS dynamics via Fisher-z transformation
    ELLIPTICAL_FAMILIES = {pvc.BicopFamily.gaussian, pvc.BicopFamily.student}

    def __init__(
        self,
        U: pd.DataFrame,
        nu_fixed: Optional[float] = None,
        fixed_order: Optional[List[int]] = None,
    ):
        super().__init__(U, nu_fixed=nu_fixed, fixed_order=fixed_order)
        self.tree1_models: Dict[int, Dict] = {}
        self.tree1_is_dynamic: Dict[int, bool] = {}  # Track dynamic vs static edges

    def fit(
        self,
        U_train: pd.DataFrame,
        fit_gas: bool = True,
    ) -> Dict[int, Dict]:
        """
        Fit model parameters.

        Step 1: Fit static vine (family selection via BIC).
        Step 2: For Tree-1 edges with ELLIPTICAL families, add GAS dynamics.
                Non-elliptical edges remain static.

        Returns
        -------
        Dict[int, Dict]
            GAS model info for dynamic edges only
        """
        super().fit(U_train)

        if not fit_gas:
            return {}

        U_ord = clip01(U_train.iloc[:, self.order].values)
        static_vine = self._get_vine()

        for edge in range(self.d - 1):
            # Check the BIC-selected family for this Tree-1 edge
            bc = static_vine.get_pair_copula(0, edge)  # Tree-1 is index 0
            selected_family = bc.family

            # Only apply GAS to elliptical families
            if selected_family not in self.ELLIPTICAL_FAMILIES:
                # Non-elliptical: keep static, no GAS dynamics
                self.tree1_is_dynamic[edge] = False
                continue

            # Elliptical family: apply GAS dynamics
            self.tree1_is_dynamic[edge] = True

            u1 = U_ord[:, edge]
            u2 = U_ord[:, edge + 1]

            # For GAS dynamics, use Student-t parameterization.
            # If selected family is gaussian, convert to student with large nu.
            if selected_family == pvc.BicopFamily.gaussian:
                # Gaussian is Student-t with nu -> infinity
                # Use nu=100 as practical approximation, document in model card
                rho_static = self._extract_elliptical_rho(bc)
                nu = 100.0  # Large nu approximates Gaussian
            else:
                # Already Student-t: re-fit to get (rho, nu)
                rho_static, nu = self._fit_student_t_edge(u1, u2)

            # Override static_params for Tree-1 with t-copula params
            self.static_params[(1, edge)] = (rho_static, nu)

            theta_init = np.arctanh(np.clip(rho_static, -0.99, 0.99))

            gas_params, theta_path, ll_path, final_state = self._fit_gas_edge(
                u1, u2, nu, theta_init
            )

            rho_path = np.tanh(theta_path)
            lambda_path = np.array([
                lower_tail_dependence_t(rho, nu) for rho in rho_path
            ])

            self.tree1_models[edge] = {
                'type': 'GAS',
                'params': gas_params,
                'nu_fixed': nu,
                'rho_static': rho_static,
                'theta_init': theta_init,
                'theta_path': theta_path,
                'll_path': ll_path,
                'lambda_path': lambda_path,
                'final_state': final_state,
                'original_family': str(selected_family).split('.')[-1],
            }

        return self.tree1_models

    def _extract_elliptical_rho(self, bc: "pvc.Bicop") -> float:
        """Extract correlation parameter from elliptical copula."""
        params = bc.parameters
        rho = float(params[0, 0]) if params.ndim == 2 else float(params[0])
        return float(np.clip(rho, -0.995, 0.995))

    def refit_params(self, U_window: pd.DataFrame) -> None:
        """
        Refit copula parameters on new data, keeping structure fixed.

        For GAS model:
        1. Refit higher-tree pair-copulas (via parent class)
        2. For dynamic Tree-1 edges: refit Student-t parameters (rho, nu)
        3. For static Tree-1 edges: handled by parent class refit
        4. Reset GAS filter state for dynamic edges

        Note: GAS dynamics (omega, A, B) are NOT re-estimated — only the
        static correlation parameters are updated. This prevents overfitting
        on short rolling windows.
        """
        # Refit all trees via parent class (handles static edges too)
        super().refit_params(U_window)

        # Refit Tree-1 dynamic (elliptical) edges only
        U_ord = clip01(U_window.iloc[:, self.order].values)

        for edge in range(self.d - 1):
            # Skip non-dynamic edges (already handled by parent)
            if not self.tree1_is_dynamic.get(edge, False):
                continue
            if edge not in self.tree1_models:
                continue

            u1 = U_ord[:, edge]
            u2 = U_ord[:, edge + 1]

            model_info = self.tree1_models[edge]

            # Check if original family was gaussian (use large nu) or student
            if model_info.get('original_family') == 'gaussian':
                # Refit gaussian, extract rho, keep nu=100
                data = np.column_stack([clip01(u1), clip01(u2)])
                bc = pvc.Bicop(pvc.BicopFamily.gaussian)
                controls = pvc.FitControlsBicop(family_set=[pvc.BicopFamily.gaussian])
                bc.select(data, controls)
                rho_new = self._extract_elliptical_rho(bc)
                nu_new = 100.0
            else:
                # Re-estimate Student-t params via MLE
                rho_new, nu_new = self._fit_student_t_edge(u1, u2)

            self.static_params[(1, edge)] = (rho_new, nu_new)

            # Update the tree1_model info but keep GAS dynamics unchanged
            model_info['rho_static'] = rho_new
            model_info['nu_fixed'] = nu_new
            model_info['theta_init'] = np.arctanh(np.clip(rho_new, -0.99, 0.99))

            # Reset filter state to match new parameters
            model_info['final_state'] = None  # Will reinit from theta_init

    def _fit_student_t_edge(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
    ) -> Tuple[float, float]:
        """Fit Student-t copula to a pair via pyvinecopulib MLE."""
        data = np.column_stack([clip01(u1), clip01(u2)])
        bc = pvc.Bicop(pvc.BicopFamily.student)
        controls = pvc.FitControlsBicop(
            family_set=[pvc.BicopFamily.student],
        )
        bc.select(data, controls)
        params = bc.parameters
        rho = float(params[0, 0]) if params.ndim == 2 else float(params[0])
        nu = float(params[1, 0]) if params.ndim == 2 else float(params[1])
        return float(np.clip(rho, -0.995, 0.995)), float(np.clip(nu, 2.5, 50.0))

    def _fit_gas_edge(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        nu: float,
        theta_init: float,
    ) -> Tuple[GASParameters, np.ndarray, np.ndarray, GASFilterState]:
        """Fit GAS model for a single edge."""
        z1 = tdist.ppf(u1, nu)
        z2 = tdist.ppf(u2, nu)
        n = len(u1)

        def neg_log_lik(params):
            omega, A, B = params
            B_eff = np.tanh(B)

            theta = theta_init
            ll_sum = 0.0
            opg = 1.0
            lam = 0.95

            for t in range(n):
                rho = np.tanh(theta)
                ll = log_student_t_copula_density(z1[t], z2[t], rho, nu)
                ll_sum += ll

                score = gas_score_theta_t(z1[t], z2[t], theta, nu)
                opg = lam * opg + (1 - lam) * score**2
                scaled_score = scale_opg(score, opg)

                theta = omega + A * scaled_score + B_eff * theta
                theta = np.clip(theta, -3.8, 3.8)

            return -ll_sum

        x0 = [theta_init * 0.01, 0.1, 2.0]
        bounds = [(-1.0, 1.0), (0.0, 1.0), (0.0, 5.0)]

        result = minimize(neg_log_lik, x0, method='L-BFGS-B', bounds=bounds)

        if result.success:
            omega, A, B = result.x
        else:
            omega, A, B = x0

        gas_params = GASParameters(omega=omega, A=A, B=B)

        # Compute paths using the unified gas_filter (single source of truth)
        theta_path, ll_path, _, final_state = gas_filter(
            z1, z2, gas_params, nu,
            theta_init=theta_init,
            return_final_state=True,
        )

        return gas_params, theta_path, ll_path, final_state

    def compute_log_likelihood(
        self,
        U_test: pd.DataFrame,
        use_dynamic: bool = True,
    ) -> np.ndarray:
        """
        Compute log-likelihood for test data with TRUE RECURSIVE OOS filtering.

        For dynamic edges: continues the GAS recursion from training final state.
        For static edges: uses the fitted static pair copula.
        """
        if not use_dynamic or not self.tree1_models:
            return super().compute_log_likelihood(U_test)

        U_ord = U_test.iloc[:, self.order].values
        n = len(U_ord)

        tree1_ll = np.zeros(n)

        for edge in range(self.d - 1):
            # Skip non-dynamic edges (their contribution is in static vine ll)
            if not self.tree1_is_dynamic.get(edge, False):
                continue
            if edge not in self.tree1_models:
                continue

            u1 = clip01(U_ord[:, edge])
            u2 = clip01(U_ord[:, edge + 1])

            model_info = self.tree1_models[edge]
            nu = model_info['nu_fixed']
            params = model_info['params']
            final_state = model_info.get('final_state')

            if final_state is not None:
                theta_init = final_state.theta
                opg_init = final_state.opg
            else:
                theta_init = model_info['theta_path'][-1]
                opg_init = 1.0

            z1 = tdist.ppf(u1, nu)
            z2 = tdist.ppf(u2, nu)

            _, ll_path, _, _ = gas_filter(
                z1, z2, params, nu,
                theta_init=theta_init,
                opg_init=opg_init,
                return_final_state=False,
            )

            tree1_ll += ll_path

        return tree1_ll

    def predict_tree1_rhos(self, t_idx: int) -> np.ndarray:
        """
        Predict Tree-1 correlations at time t+1.

        For dynamic edges: uses GAS-predicted rho.
        For static edges: returns the static correlation parameter.

        Note: For non-elliptical static edges, the returned value is
        Kendall's tau (not Pearson correlation), as stored in static_params.
        """
        rhos = np.zeros(self.d - 1)

        for edge in range(self.d - 1):
            # Check if this edge is dynamic
            if self.tree1_is_dynamic.get(edge, False) and edge in self.tree1_models:
                model_info = self.tree1_models[edge]
                params = model_info['params']
                theta_path = model_info['theta_path']

                if t_idx < len(theta_path):
                    theta_t = theta_path[t_idx]
                else:
                    theta_t = theta_path[-1]

                B_eff = np.tanh(params.B)
                theta_pred = params.omega + B_eff * theta_t
                rhos[edge] = np.tanh(np.clip(theta_pred, -5.0, 5.0))
            else:
                # Static edge: use static params (may be tau for Archimedean)
                rhos[edge], _ = self.static_params[(1, edge)]

        return rhos

    def _make_bicop_t(self, rho: float, nu: float) -> "pvc.Bicop":
        """Create a Student-t bivariate copula using compat wrapper."""
        return make_bicop_student(rho, nu)

    def build_vine_at_time(self, t_idx: int) -> "pvc.Vinecop":
        """
        Build a vine copula using parameters at time t_idx.

        For dynamic Tree-1 edges: uses GAS-predicted rhos (Student-t).
        For static Tree-1 edges: uses the BIC-selected pair copula.
        Higher trees: always use the static vine's fitted pair-copulas.
        """
        rhos_tree1 = self.predict_tree1_rhos(t_idx)

        order_1 = [i + 1 for i in range(self.d)]
        struct = make_dvine_structure(self.d, order_1)

        static_vine = self._get_vine()
        pcs = []

        # Tree-1: mix of dynamic (GAS) and static pair copulas
        row1 = []
        for edge in range(self.d - 1):
            if self.tree1_is_dynamic.get(edge, False) and edge in self.tree1_models:
                # Dynamic edge: use GAS-predicted rho with Student-t
                nu = self.tree1_models[edge]['nu_fixed']
                row1.append(self._make_bicop_t(rhos_tree1[edge], nu))
            else:
                # Static edge: use the BIC-selected pair copula from static vine
                if static_vine is not None:
                    try:
                        bc = static_vine.get_pair_copula(0, edge)
                        row1.append(bc)
                        continue
                    except Exception:
                        pass
                # Fallback: use Student-t from static_params
                rho, nu = self.static_params[(1, edge)]
                row1.append(self._make_bicop_t(rho, nu if nu > 0 else 8.0))
        pcs.append(row1)

        # Higher trees: use fitted vine if available
        for tree in range(2, self.d):
            row = []
            for edge in range(self.d - tree):
                if static_vine is not None:
                    try:
                        bc = static_vine.get_pair_copula(tree - 1, edge)
                        row.append(bc)
                        continue
                    except Exception:
                        pass
                # Fallback to Student-t from static_params
                rho, nu = self.static_params[(tree, edge)]
                row.append(self._make_bicop_t(rho, nu if nu > 0 else 8.0))
            pcs.append(row)

        return build_vine_from_pair_copulas(struct, pcs)

    def simulate(
        self,
        n_sim: int,
        t_idx: int = 0,
        use_antithetic: bool = False,
    ) -> np.ndarray:
        """Simulate from the GAS D-vine copula at time t."""
        vine = self.build_vine_at_time(t_idx)
        return vine.simulate(n_sim)

    def get_model_card(self) -> dict:
        """
        Export full model card with GAS dynamics for elliptical Tree-1 edges.

        Extends parent class model card with:
        - gas_scope: "tree1_elliptical_only"
        - dynamic_edge_count / static_tree1_edge_count: summary stats
        - For each Tree-1 edge:
          - is_dynamic: true/false
          - If dynamic: gas_params, family="student", rho_static, nu_fixed
          - If static: keeps BIC-selected family and parameters
        """
        # Get base model card from parent
        card = super().get_model_card()

        # Count dynamic vs static Tree-1 edges
        dynamic_count = sum(1 for v in self.tree1_is_dynamic.values() if v)
        static_count = (self.d - 1) - dynamic_count

        # Add GAS scope summary
        card["gas_scope"] = "tree1_elliptical_only"
        card["dynamic_edge_count"] = dynamic_count
        card["static_tree1_edge_count"] = static_count
        card["higher_tree_dynamics"] = "static"

        # Update Tree-1 pair copulas based on dynamic/static status
        for pc in card["pair_copulas"]:
            if pc["tree"] == 1:
                edge = pc["edge_index"]
                is_dynamic = self.tree1_is_dynamic.get(edge, False)

                pc["is_dynamic"] = is_dynamic

                if is_dynamic and edge in self.tree1_models:
                    gas_info = self.tree1_models[edge]
                    params = gas_info["params"]

                    # Dynamic edge: override to student family with GAS params
                    original_family = gas_info.get("original_family", "student")
                    pc["family"] = "student"
                    pc["rotation"] = 0
                    pc["parameters"] = [
                        round(float(gas_info["rho_static"]), 6),
                        round(float(gas_info["nu_fixed"]), 4),
                    ]

                    # Add GAS dynamics parameters
                    pc["gas_params"] = {
                        "omega": round(float(params.omega), 6),
                        "A": round(float(params.A), 6),
                        "B": round(float(params.B), 6),
                        "B_effective": round(float(params.B_effective), 6),
                    }
                    pc["rho_static"] = round(float(gas_info["rho_static"]), 6)
                    pc["nu_fixed"] = round(float(gas_info["nu_fixed"]), 4)

                    # Document if gaussian was converted to student
                    if original_family == "gaussian":
                        pc["note"] = "Gaussian converted to Student-t(nu=100) for GAS"
                else:
                    # Static edge: keep BIC-selected family (no gas_params)
                    pass  # Family, rotation, parameters already set by parent

        return card
