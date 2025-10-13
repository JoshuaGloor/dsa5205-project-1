from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPRegressor as SkMLP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel


class MLP(BaseModel):
    """Simple MLP regressor where z and T_tr controls weight decay (alpha)"""

    def _fit_impl(self, S: np.ndarray, R_next: np.ndarray) -> None:

        self._S_tr = S
        self._R_tr = R_next
        self.hidden_layer_sizes = 16
        self.activation = "relu"
        self.solver = "adam"
        self.learning_rate_init = 0.01  # Non-default to speed up training a bit
        self.max_iter = 500  # Non-default to increase chance of convergence
        self.shuffle = False  # We don't want to mess with order of training set
        self.random_state = 1337

    def _transform_weight_matrix_to_vector(self) -> np.ndarray:
        """Turn weight matrices into one large vector

        We only use weight matrices for the parameter size and no bias because our Ridge also doesn't use intercept.
        """

        weight_matrices = self.pipe.named_steps["model"].coefs_
        return np.concatenate([W.ravel() for W in weight_matrices])

    def _fit_beta_hat(self, z: float, T_tr: int) -> np.ndarray:
        """Fit pipeline for (z, T_tr) and return flattened weights"""

        self.pipe = self._build_pipeline(alpha=z * T_tr)
        self.pipe.fit(self._S_tr, self._R_tr)
        return self._transform_weight_matrix_to_vector()

    def _predict_impl(self, S_te: np.ndarray, z: float, T_tr: int):
        """Predict using pipeline fitted for (z, T_tr)."""

        beta_hat = self._fit_beta_hat(z, T_tr)
        return self.pipe.predict(S_te), beta_hat

    def _build_pipeline(self, alpha: float):
        """Builds a simple standardized MLP pipeline

        We use weight decay alpha as analog to Ridge shrinkage
        """

        mlp = SkMLP(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            learning_rate_init=self.learning_rate_init,
            alpha=alpha,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        return Pipeline([("scalar", StandardScaler()), ("model", mlp)])
