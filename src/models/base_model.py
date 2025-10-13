from abc import ABC, abstractmethod
from typing import Self
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self):
        self._fitted = False

    def _ensure_fitted(self) -> None:
        """Ensures `fit` is called before `predict`"""

        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call .fit(...) first.")

    @abstractmethod
    def _fit_impl(self, S: np.ndarray, R_next: np.ndarray) -> None:
        """Do precomputations that hold for every shrinkage z, if possible"""
        ...

    def fit(self, S: np.ndarray, R_next: np.ndarray) -> Self:
        """Public training entrypoint"""

        self._fit_impl(S, R_next)
        self._fitted = True
        return self

    @abstractmethod
    def _fit_beta_hat(self, z: float, T_tr: int) -> np.ndarray:
        """Fit the model given shrinkage z and T_tr

        Method is internal because predict should return the beta_hat.
        """
        ...

    def predict(self, S_te: np.ndarray, z: float, T_tr: int) -> tuple[np.ndarray, np.ndarray]:
        """Public predict entrypoint"""

        self._ensure_fitted()
        return self._predict_impl(S_te, z, T_tr)

    @abstractmethod
    def _predict_impl(self, S_te: np.ndarray, z: float, T_tr: int) -> tuple[np.ndarray, np.ndarray]:
        """Predict R_next for the test set

        Return prediction and beta_hat.
        """
        ...
