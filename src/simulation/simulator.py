import numpy as np
from numpy.random import Generator
from dataclasses import dataclass
from tqdm import tqdm

from src.simulation import data_generation as dg
from src.models.base_model import BaseModel
from src.metrics import r2_paper, expected_timing_return, unconditional_sharpe, beta_norm_squared


@dataclass(frozen=True)
class SimConfig:
    """Safety class to avoid accidental modification of parameters"""

    n_runs: int
    T_tr: int
    T_te: int
    P: int
    b_star: float
    cqs: tuple[float, ...]
    zs: tuple[float, ...]

    def __post_init__(self):
        # Validate number of runs
        if self.n_runs < 1:
            raise ValueError("n_runs must be at least 1")

        # Allow np.ndarray inputs for convenience
        if isinstance(self.cqs, np.ndarray):
            object.__setattr__(self, "cqs", tuple(self.cqs))
        if isinstance(self.zs, np.ndarray):
            object.__setattr__(self, "zs", tuple(self.zs))


class Simulator:
    """Simulation logic"""

    def __init__(self, cfg: SimConfig, model: BaseModel, rng: Generator):
        self.cfg = cfg
        self.n_runs = cfg.n_runs
        self.T_tr = cfg.T_tr
        self.T_te = cfg.T_te
        self.P = cfg.P
        self.b_star = cfg.b_star
        self.cqs = cfg.cqs
        self.zs = cfg.zs
        self._model = model
        self._rng = rng

    def run_simulation(self) -> dict:
        """Main method that executes multiple simulation run and combines them"""

        run_metrics = []

        for _ in tqdm(range(self.n_runs), desc="Running simulation"):
            metrics = self._simulate_one_run()
            run_metrics.append(metrics)

        avg_metric = self._compute_avg_metrics(run_metrics)
        return self._convert_from_gridtometric_to_metrictoztocq(avg_metric)

    def _convert_from_gridtometric_to_metrictoztocq(self, avg_metric):
        """Helper method to convert representation of metrics

        It converts from
        grid[i_cq][j_z] -> {metric1: avg1, metric2: avg2, ...}
        which makes more sense during collection to
        {metric1: {z1: [avg_cq1, avg_cq2, ...], z2: [avg_cq1, avg_cq2, ...], ...}, ...}
        which makes it easier to plot
        """

        metric_to_z_to_cqs = {metric: {z: [] for z in self.zs} for metric, _ in avg_metric[0][0].items()}

        for i in range(len(self.cqs)):
            for j, z in enumerate(self.zs):
                for metric, avg in avg_metric[i][j].items():
                    metric_to_z_to_cqs[metric][z].append(avg)
        return metric_to_z_to_cqs

    def _compute_avg_metrics(self, run_metrics) -> dict:
        """Compute average over multiple simulation runs"""

        if self.n_runs != len(run_metrics):
            raise ValueError("Number of run metrics recorded must equal n_runs")

        rows = len(run_metrics[0])
        cols = len(run_metrics[0][0])
        avg_metrics = [[{} for _ in range(cols)] for _ in range(rows)]

        for r, run in enumerate(run_metrics):
            for i in range(rows):
                for j in range(cols):
                    for k, v in run[i][j].items():
                        if r == 0:  # Base case - Init
                            avg_metrics[i][j][k] = v
                        else:  # Step case - accumulate
                            avg_metrics[i][j][k] += v
        for i in range(rows):
            for j in range(cols):
                for k, v in avg_metrics[i][j].items():
                    avg_metrics[i][j][k] = v / self.n_runs
        return avg_metrics

    def _simulate_one_run(self) -> dict:
        """Run one simulation

        Returns nested dict of metrics:
            results[q][z] = {
                "r2": float,
                "sr": float,
                "etr": float,
                "bns": float,
            }
        """

        metrics = [[{} for _ in range(len(self.zs))] for _ in range(len(self.cqs))]

        S_tr, R_tr, S_te, R_te, perm = dg.generate_run(self.T_tr, self.T_te, self.P, self.b_star, self._rng)

        for i, cq in enumerate(self.cqs):
            P_1 = int(cq * self.T_tr)
            S1_tr = dg.observed_block(S_tr, perm, P_1)
            S1_te = dg.observed_block(S_te, perm, P_1)

            self._model.fit(S1_tr, R_tr)

            # Sweet over zs
            for j, z in enumerate(self.zs):
                R_pred, beta_hat = self._model.predict(S1_te, z, self.T_tr)
                metrics[i][j] = {
                    "r2": r2_paper(R_te, R_pred),
                    "sr": unconditional_sharpe(R_te, R_pred),
                    "etr": expected_timing_return(R_te, R_pred),
                    "bns": beta_norm_squared(beta_hat),
                }
        return metrics
