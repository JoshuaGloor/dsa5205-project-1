import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def save_plot(metric: str) -> None:
    """Save plot for backup reasons"""

    current_dir = Path(__file__).resolve().parent

    # Use `plots` directory to save plots
    plots_dir = current_dir.parent / "plots"

    timestamp = datetime.now().strftime("%H%M")
    out_path = plots_dir / f"{timestamp}-{metric}.png"
    plt.savefig(out_path, dpi=500)


def plot_curves(
    metrics,
    metric: str,
    cqs,
    ylim,
    title: str = "",
    xlabel: str = "cq",
    figsize=(6, 4.5),
    dpi: int = 500,
) -> None:
    if metric not in metrics:
        raise ValueError(
            f"Specified metric code {metric} not part of available metrics: {', '.join(metric.keys())}."
        )

    # TODO make sure metric exists in metrics and that dim of cqs and z match with metrics
    _, ax = plt.subplots(dpi=dpi)
    z_to_avgs = metrics[metric]  # z -> avgs dictionary

    # Fix ylim because our plots are usually only interesting in a very small interval
    plt.ylim(ylim)

    # Plot individual lines
    for z, avgs in z_to_avgs.items():
        if z == 0:  # Treat ridgeless separately
            ax.plot(cqs, avgs, label="Ridgeless", lw=1, color="black")
        else:  # all others
            ax.plot(cqs, avgs, label=rf"$z = {z}$", lw=0.75)

    ax.axvline(1.0, color="gray", ls="--", lw=0.5, label=r"$c = 1$")

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # Make legend a bit smaller because we have many z
    plt.legend(
        fontsize=6,
        loc="upper right",
    )

    save_plot(metric)
    plt.show()
