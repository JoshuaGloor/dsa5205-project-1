import numpy as np
from src.models.base_model import BaseModel


class Ridge(BaseModel):
    """SVD based Ridge

    We use SVD based Ridge for stability and to avoid having to recompute the inverse.
    Justification is included in the report.

    For documentation purposes, we assume S has dimension (m, n)
    """

    def _fit_impl(self, S: np.ndarray, R_next: np.ndarray):
        """Precompute SVD for faster computation of beta, given lambda"""

        U, self.D, Vt = np.linalg.svd(S, full_matrices=False)
        self.V = Vt.T
        self.Uty = U.T @ R_next  # U^\prime y

    def _fit_beta_hat(self, z: float, T_tr: int) -> np.ndarray:
        """Compute beta_hat with SVD trick

        Returns
        -------
        beta_hat : (n,) array
        """

        lam = z * T_tr
        diag = self.D / (self.D**2 + lam)
        return self.V @ np.diag(diag) @ self.Uty

    def _predict_impl(self, S_te: np.ndarray, z: float, T_tr: int) -> tuple[np.ndarray, np.ndarray]:
        """Estimate forecast

        Returns
        -------
        R_pred : (m,) array
        beta_hat : (n,) array
        """

        beta_hat = self._fit_beta_hat(z, T_tr)
        return S_te @ beta_hat, beta_hat
