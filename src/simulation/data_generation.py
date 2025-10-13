import numpy as np
from numpy.random import Generator


def generate_signals(T: int, P: int, rng: Generator) -> np.ndarray:
    """Generate signals S with rows ~ N(0, I_P)

    Note, S_t = S[t, :]

    Returns
    -------
    S : (T, P) array
    """

    return rng.normal(size=(T, P))


def generate_beta_star(P: int, b_star: float, rng: Generator) -> np.ndarray:
    """Generate dense Gaussian beta and normalize to ||beta||^2 = b_star

    Returns
    -------
    beta_star : (P,) array
    """

    v = rng.normal(size=P)
    norm_2 = np.linalg.norm(v)
    return (np.sqrt(b_star) / norm_2) * v


def generate_returns(S: np.ndarray, beta_star: np.ndarray, T: int, rng: Generator) -> np.ndarray:
    """Generate returns via forward operator R_next = S_t' beta_star + eps_next

    Returns
    -------
    R_next : (T,) array
    """

    eps_next = rng.normal(size=T)
    return S @ beta_star + eps_next


def generate_run(T_tr: int, T_te: int, P: int, b_star: float, rng: Generator) -> tuple:
    """Generate one simulation run.

    Returns
    -------
    S_tr : (T_tr, P) array
    R_tr : (T_tr,) array
    S_te : (T_te, P) array
    R_te : (T_te,) array
    perm : (P,) permutation array
    """

    T = T_tr + T_te
    S = generate_signals(T, P, rng)
    beta_star = generate_beta_star(P, b_star, rng)
    R_next = generate_returns(S, beta_star, T, rng)
    perm = rng.permutation(P)

    # Split train/test
    S_tr, R_tr = S[:T_tr], R_next[:T_tr]
    S_te, R_te = S[T_tr:], R_next[T_tr:]

    return S_tr, R_tr, S_te, R_te, perm


def observed_block(S: np.ndarray, perm: np.ndarray, P_1: int) -> np.ndarray:
    """Keep first P_1 permuted columns of S.

    Returns
    -------
    S_{1, t} : (T, P1) array
    """

    if P_1 > S.shape[1]:
        raise ValueError("P_1 must be less than or equal to P.")

    observed_cols = perm[:P_1]
    return S[:, observed_cols]
