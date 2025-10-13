import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from src.models.factory import create_model
from src.metrics import unconditional_sharpe


class InvalidFeatureOrder(Exception):
    """Raised when the train features do not align with test features"""

    pass


def select_z_timecv(
    S: np.ndarray,
    R_next: np.ndarray,
    zs: np.ndarray,
    n_splits: int,
    model_name: str = "ridge",
) -> float:
    """Choose z by validation sharpe ratio using time-series-aware CV"""

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # We mainly/only care about sharpe
    sharpe = np.zeros((n_splits, len(zs)))

    for i, (tr_idx, va_idx) in enumerate(tscv.split(S)):
        S_tr, S_va = S[tr_idx], S[va_idx]
        R_tr, R_va = R_next[tr_idx], R_next[va_idx]

        model = create_model(model_name)
        model.fit(S_tr, R_tr)

        # Sweep z
        for j, z in enumerate(zs):
            R_pred, _ = model.predict(S_va, z, len(S_tr))
            sharpe[i, j] = unconditional_sharpe(R_va, R_pred)
            # corr = np.corrcoef(R_pred, R_va)[0, 1]
            # sr = unconditional_sharpe(R_va, R_pred)
            # print(
            #    f"(split={i}, z_j={j}, z={z:.2e}, corr={corr:.3f}, SR={sr:.4f}"
            # )

    w = np.arange(1, n_splits + 1)  # Weights 1, 2, ..., n
    # Mean sharpe weighting longer training sets as more important
    mean_sharpe = (sharpe.T @ w) / w.sum()
    # mean_sharpe = np.mean(sharpe, axis=0)  # Calculate mean by columns, i.e. for each z
    best_idx = np.argmax(mean_sharpe)
    return zs[best_idx], sharpe


def run_ridge_for_dataset(
    train_path: Path,
    test_path: Path,
    out_path: Path,
    zs: np.ndarray,
    n_splits: int,
) -> None:
    """Helper to read, train, predict, and write data"""

    # Read in data
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    tr_feat_cols = tr.columns[1:-1]  # The last column in tr is `return`
    te_feat_cols = te.columns[1:]

    if tr_feat_cols.tolist() != te_feat_cols.tolist():
        raise InvalidFeatureOrder(f"Train features are not the same or in the same order as the test features.")

    S_tr = tr[tr_feat_cols].to_numpy(dtype=float)
    R_next = tr["return"].to_numpy(dtype=float)
    S_te = te[te_feat_cols].to_numpy(dtype=float)

    # Perform model selection
    best_z, _ = select_z_timecv(S_tr, R_next, zs, n_splits)
    print(f"Selected z for Ridge: {best_z}")

    # Retrain on all training rows and predict test
    model = create_model("ridge")
    T_tr = S_tr.shape[0]

    model.fit(S_tr, R_next)
    R_pred, _ = model.predict(S_te, best_z, T_tr)

    # Write predictions to file
    out = pd.DataFrame({"t": te["t"].to_numpy(), "yhat": R_pred})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions of {out_path.stem} to {out_path.parent}.")


def run_lasso_for_dataset(
    train_path: Path,
    test_path: Path,
    out_path: Path,
    alphas: np.ndarray,
    n_splits: int,
) -> None:
    """Helper to read, train, predict, and write data

    WARNING: This is an ugly mess and lots of duplication, I ran out of time...
    """

    # Read in data
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    tr_feat_cols = tr.columns[1:-1]  # The last column in tr is `return`
    te_feat_cols = te.columns[1:]

    if tr_feat_cols.tolist() != te_feat_cols.tolist():
        raise InvalidFeatureOrder(f"Train features are not the same or in the same order as the test features.")

    S_tr = tr[tr_feat_cols].to_numpy(dtype=float)
    R_next = tr["return"].to_numpy(dtype=float)
    S_te = te[te_feat_cols].to_numpy(dtype=float)

    # Perform model selection
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = LassoCV(alphas=alphas, cv=tscv, random_state=42)
    model.fit(S_tr, R_next)
    print("Selected alpha for LASSO:", model.alpha_)

    # predict
    R_pred = model.predict(S_te)

    # Write predictions to file
    out = pd.DataFrame({"t": te["t"].to_numpy(), "yhat": R_pred})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions of {out_path.stem} to {out_path.parent}.")


if __name__ == "__main__":
    in_path = Path(input("Enter path to the directory of the input CSV files: ").strip())
    out_path = Path(input("Enter path to the directory of the output CSV files: ").strip())
    student_id = input("Enter your student id: ").strip()

    # Ensure input exists
    if not in_path.exists():
        raise FileNotFoundError("Looks like the path you specified does not exist on your machine, please try again.")

    files_that_must_exist = [
        "pairA_test_features.csv",
        "pairA_train.csv",
        "pairB_test_features.csv",
        "pairB_train.csv",
        "pairC_test_features.csv",
        "pairC_train.csv",
    ]

    # Ensure all required files exist in input path
    for f_name in files_that_must_exist:
        file_path = in_path / f_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"File {f_name} does not exist in your input directory, ensure {files_that_must_exist} exist."
            )

    zs = np.logspace(-4, 1, 100)
    alphas = np.logspace(-6, 0, 100)
    n_splits = 4

    for data_set in ["A", "B", "C"]:
        run_ridge_for_dataset(
            train_path=in_path / f"pair{data_set}_train.csv",
            test_path=in_path / f"pair{data_set}_test_features.csv",
            out_path=out_path / f"{student_id}_predictions_{data_set}.csv",
            zs=zs,
            n_splits=n_splits,
        )
