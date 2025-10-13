import numpy as np


def r2_paper(R_te: np.ndarray, R_pred: np.ndarray) -> float:
    """Compute R-squared as in KMZ III.A."""

    SSR = ((R_te - R_pred) ** 2).mean()
    SST = (R_te**2).mean()
    return 1 - SSR / SST


def expected_timing_return(R_te: np.ndarray, R_pred: np.ndarray) -> float:
    """Compute E[R_{t + 1}^\pi]"""

    R_pi = R_pred * R_te
    return R_pi.mean()


def unconditional_sharpe(R_te: np.ndarray, R_pred: np.ndarray) -> float:
    """Compute unconditional sharpe, KMZ eq. (5)"""

    R_pi = R_pred * R_te
    exp_ret = R_pi.mean()
    std = np.sqrt((R_pi**2).mean())
    return exp_ret / std


def beta_norm_squared(beta_hat: np.ndarray) -> float:
    """Compute ||beta_hat||^2."""

    return np.linalg.norm(beta_hat) ** 2
