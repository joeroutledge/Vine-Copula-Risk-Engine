"""
DCC-GARCH benchmark model.

This module implements a DCC-GARCH (Dynamic Conditional Correlation GARCH)
model as an alternative benchmark to the D-vine copula approach.

DCC-GARCH is a popular model for time-varying correlations in finance:
- Step 1: Fit univariate GARCH to each series
- Step 2: Fit DCC dynamics to standardized residuals

References:
    Engle, R. (2002). "Dynamic Conditional Correlation: A Simple Class
    of Multivariate Generalized Autoregressive Conditional Heteroskedasticity
    Models." Journal of Business & Economic Statistics, 20(3), 339-350.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.stats import norm, t as tdist
from dataclasses import dataclass


# ============================================================================
# DCC PARAMETERS
# ============================================================================

@dataclass
class DCCParameters:
    """
    DCC model parameters.

    Attributes
    ----------
    alpha : float
        Short-run persistence (α)
    beta : float
        Long-run persistence (β)
    Q_bar : np.ndarray
        Unconditional correlation matrix
    """
    alpha: float
    beta: float
    Q_bar: np.ndarray

    @property
    def persistence(self) -> float:
        """DCC persistence: α + β."""
        return self.alpha + self.beta

    @property
    def is_stationary(self) -> bool:
        """Check if DCC is covariance stationary."""
        return self.persistence < 1.0


# ============================================================================
# DCC-GARCH MODEL
# ============================================================================

class DCCGARCHModel:
    """
    DCC-GARCH model for multivariate volatility.

    This model combines:
    - Univariate GARCH(1,1) for each series
    - DCC dynamics for the correlation matrix

    Parameters
    ----------
    returns : pd.DataFrame
        Return series (decimals)
    dist : str
        Innovation distribution: 'normal' or 't'

    Attributes
    ----------
    d : int
        Number of series
    assets : List[str]
        Asset names
    garch_params : Dict
        GARCH parameters for each asset
    dcc_params : DCCParameters
        DCC parameters
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        dist: str = 'normal',
    ):
        self.returns = returns
        self.d = returns.shape[1]
        self.assets = list(returns.columns)
        self.dist = dist

        self.garch_params: Dict[str, Dict] = {}
        self.dcc_params: Optional[DCCParameters] = None
        self._sigma: Optional[pd.DataFrame] = None
        self._z: Optional[pd.DataFrame] = None
        self._R_path: Optional[np.ndarray] = None

    def fit(
        self,
        returns_train: pd.DataFrame,
    ) -> Tuple[Dict, DCCParameters]:
        """
        Fit DCC-GARCH model.

        Parameters
        ----------
        returns_train : pd.DataFrame
            Training returns

        Returns
        -------
        garch_params : Dict
            GARCH parameters for each asset
        dcc_params : DCCParameters
            DCC parameters
        """
        # Step 1: Fit univariate GARCH
        sigma = pd.DataFrame(index=returns_train.index, columns=returns_train.columns)
        z = pd.DataFrame(index=returns_train.index, columns=returns_train.columns)

        for col in returns_train.columns:
            params, sigma_series = self._fit_garch(returns_train[col])
            self.garch_params[col] = params
            sigma[col] = sigma_series
            z[col] = returns_train[col] / sigma_series.clip(lower=1e-10)

        self._sigma = sigma
        self._z = z

        # Step 2: Fit DCC
        z_arr = z.values
        self.dcc_params = self._fit_dcc(z_arr)

        return self.garch_params, self.dcc_params

    def _fit_garch(
        self,
        returns: pd.Series,
    ) -> Tuple[Dict, pd.Series]:
        """Fit GARCH(1,1) to a single series."""
        r = returns.values
        n = len(r)

        # Simple GARCH(1,1) estimation
        # σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

        # Initialize with sample variance
        var0 = np.var(r[:min(50, n)])

        # Starting values
        omega = var0 * 0.05
        alpha = 0.08
        beta = 0.90

        # Simple optimization (could use arch package for production)
        from scipy.optimize import minimize

        def neg_ll(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            sig2 = np.zeros(n)
            sig2[0] = var0

            for t in range(1, n):
                sig2[t] = omega + alpha * r[t - 1]**2 + beta * sig2[t - 1]
                sig2[t] = max(sig2[t], 1e-10)

            # Gaussian log-likelihood
            ll = -0.5 * np.sum(np.log(2 * np.pi * sig2) + r**2 / sig2)
            return -ll

        result = minimize(
            neg_ll,
            [omega, alpha, beta],
            method='L-BFGS-B',
            bounds=[(1e-10, None), (0, 0.5), (0, 0.999)],
        )

        if result.success:
            omega, alpha, beta = result.x
        else:
            omega, alpha, beta = var0 * 0.05, 0.08, 0.90

        # Compute variance path
        sig2 = np.zeros(n)
        sig2[0] = var0

        for t in range(1, n):
            sig2[t] = omega + alpha * r[t - 1]**2 + beta * sig2[t - 1]
            sig2[t] = max(sig2[t], 1e-10)

        sigma = pd.Series(np.sqrt(sig2), index=returns.index, name=returns.name)

        return {'omega': omega, 'alpha': alpha, 'beta': beta}, sigma

    def _fit_dcc(self, z: np.ndarray) -> DCCParameters:
        """Fit DCC dynamics to standardized residuals."""
        n, d = z.shape

        # Unconditional correlation
        Q_bar = np.corrcoef(z.T)

        # DCC estimation
        # Q_t = (1-α-β)Q̄ + α·ε_{t-1}ε'_{t-1} + β·Q_{t-1}
        # R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}

        from scipy.optimize import minimize

        def neg_ll(params):
            alpha, beta = params
            if alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            Q = Q_bar.copy()
            ll = 0.0

            for t in range(n):
                zt = z[t].reshape(-1, 1)

                # Correlation matrix from Q
                diag_Q = np.diag(Q)
                diag_Q = np.maximum(diag_Q, 1e-10)
                D_inv = np.diag(1.0 / np.sqrt(diag_Q))
                R = D_inv @ Q @ D_inv

                # Ensure valid correlation matrix
                np.fill_diagonal(R, 1.0)
                R = np.clip(R, -0.999, 0.999)

                # Log-likelihood contribution
                try:
                    sign, logdet = np.linalg.slogdet(R)
                    if sign <= 0:
                        return 1e10
                    R_inv = np.linalg.inv(R)
                    ll += -0.5 * (logdet + float(zt.T @ R_inv @ zt) - float(zt.T @ zt))
                except Exception:
                    return 1e10

                # DCC update
                Q = (1 - alpha - beta) * Q_bar + alpha * (zt @ zt.T) + beta * Q

            return -ll

        result = minimize(
            neg_ll,
            [0.02, 0.95],
            method='L-BFGS-B',
            bounds=[(0.001, 0.2), (0.7, 0.999)],
        )

        if result.success:
            alpha, beta = result.x
        else:
            alpha, beta = 0.02, 0.95

        return DCCParameters(alpha=alpha, beta=beta, Q_bar=Q_bar)

    def filter_correlations(
        self,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Filter time-varying correlations.

        Parameters
        ----------
        z : np.ndarray
            Standardized residuals (n x d)

        Returns
        -------
        np.ndarray
            Correlation matrices (n x d x d)
        """
        if self.dcc_params is None:
            raise RuntimeError("Model must be fitted first")

        n, d = z.shape
        alpha = self.dcc_params.alpha
        beta = self.dcc_params.beta
        Q_bar = self.dcc_params.Q_bar

        R_path = np.zeros((n, d, d))
        Q = Q_bar.copy()

        for t in range(n):
            zt = z[t].reshape(-1, 1)

            # Correlation from Q
            diag_Q = np.maximum(np.diag(Q), 1e-10)
            D_inv = np.diag(1.0 / np.sqrt(diag_Q))
            R = D_inv @ Q @ D_inv
            np.fill_diagonal(R, 1.0)
            R = np.clip(R, -0.999, 0.999)
            R_path[t] = R

            # Update Q
            Q = (1 - alpha - beta) * Q_bar + alpha * (zt @ zt.T) + beta * Q

        self._R_path = R_path
        return R_path

    def get_correlations(self) -> np.ndarray:
        """Get filtered correlation path."""
        if self._R_path is None:
            if self._z is not None:
                self.filter_correlations(self._z.values)
        return self._R_path

    def simulate(
        self,
        n_sim: int,
        h: int = 1,
    ) -> np.ndarray:
        """
        Simulate future returns.

        Parameters
        ----------
        n_sim : int
            Number of simulations
        h : int
            Forecast horizon

        Returns
        -------
        np.ndarray
            Simulated returns (n_sim x d)
        """
        if self.dcc_params is None or self._R_path is None:
            raise RuntimeError("Model must be fitted and filtered first")

        # Use last correlation matrix
        R = self._R_path[-1]

        # Simulate standard normal
        if self.dist == 't':
            # Use Student-t with estimated df
            nu = 8.0  # Could estimate this
            Z = tdist.rvs(nu, size=(n_sim, self.d))
        else:
            Z = np.random.standard_normal((n_sim, self.d))

        # Apply correlation via Cholesky
        try:
            L = np.linalg.cholesky(R)
            Z_corr = Z @ L.T
        except Exception:
            Z_corr = Z

        # Scale by volatility (use last sigma)
        sigma_last = self._sigma.iloc[-1].values

        return Z_corr * sigma_last

    def compute_var_es(
        self,
        weights: np.ndarray,
        n_sim: int = 10000,
        alphas: Tuple[float, ...] = (0.01, 0.025, 0.05),
    ) -> Dict:
        """
        Compute VaR/ES using Monte Carlo.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        n_sim : int
            Number of simulations
        alphas : Tuple[float, ...]
            Confidence levels

        Returns
        -------
        Dict
            VaR and ES for each alpha
        """
        r_sim = self.simulate(n_sim)
        r_portfolio = r_sim @ weights

        results = {'var': {}, 'es': {}}

        for alpha in alphas:
            var_alpha = np.quantile(r_portfolio, alpha)
            tail = r_portfolio[r_portfolio <= var_alpha]
            es_alpha = float(np.mean(tail)) if len(tail) > 0 else var_alpha

            results['var'][alpha] = float(var_alpha)
            results['es'][alpha] = float(es_alpha)

        return results

    def compare_to_vine(
        self,
        vine_model,
        U_test: pd.DataFrame,
    ) -> Dict:
        """
        Compare DCC to vine copula model.

        Parameters
        ----------
        vine_model : StaticDVineModel or similar
            Vine copula model
        U_test : pd.DataFrame
            Test uniforms

        Returns
        -------
        Dict
            Comparison statistics
        """
        # DCC log-likelihood on test
        if self._z is None:
            raise RuntimeError("Model must be fitted first")

        # Filter on test data
        returns_test = self.returns.loc[U_test.index]
        z_test = pd.DataFrame(index=returns_test.index, columns=returns_test.columns)

        for col in returns_test.columns:
            sigma = self._sigma[col].reindex(returns_test.index).fillna(method='ffill')
            z_test[col] = returns_test[col] / sigma.clip(lower=1e-10)

        R_test = self.filter_correlations(z_test.values)

        # Simple Gaussian log-likelihood
        n = len(z_test)
        ll_dcc = 0.0

        for t in range(n):
            zt = z_test.iloc[t].values.reshape(-1, 1)
            R = R_test[t]
            try:
                sign, logdet = np.linalg.slogdet(R)
                if sign > 0:
                    R_inv = np.linalg.inv(R)
                    ll_dcc += -0.5 * (logdet + float(zt.T @ R_inv @ zt) - float(zt.T @ zt))
            except Exception:
                pass

        # Vine log-likelihood
        ll_vine = vine_model.compute_log_likelihood(U_test)

        return {
            'll_dcc': float(ll_dcc),
            'll_vine': float(np.sum(ll_vine)),
            'll_diff': float(np.sum(ll_vine) - ll_dcc),
            'n_obs': n,
        }
