from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_metric(ax: plt.Axes, x: np.ndarray, experiments: List[Dict[str, Any]], metric_base: str, title: str) -> None:
    for exp in experiments:
        mean = exp[f"{metric_base}_mean"]
        std = exp[f"{metric_base}_std"]
        ax.plot(x, mean, label=exp["label"])
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)


def plot_population_results(experiments: List[Dict[str, Any]], steps: int, filename: str = "output/figure_population.png", show: bool = True) -> None:
    x = np.arange(steps)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    plot_metric(axes[0, 0], x, experiments, "population_size", "Population size")
    plot_metric(axes[0, 1], x, experiments, "num_divisions", "Divisions per step")
    plot_metric(axes[1, 0], x, experiments, "num_apoptosis", "Apoptoses per step")
    plot_metric(axes[1, 1], x, experiments, "mean_kappa", "Mean residual division capacity")
    plot_metric(axes[2, 0], x, experiments, "mean_trace", "Mean trace norm")
    plot_metric(axes[2, 1], x, experiments, "state_var", "State variance")

    plt.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved population plot to {filename}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_final_state_and_lineages(experiments: List[Dict[str, Any]], filename: str = "output/figure_lineages.png", show: bool = True) -> None:
    fig, axes = plt.subplots(1, len(experiments), figsize=(4.5 * len(experiments), 4), sharex=False, sharey=False)
    if len(experiments) == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        snap = exp["final_snapshot_example"]
        H = snap["H"]
        lineage = snap["lineage_uid"]

        if H.size == 0:
            ax.set_title(exp["label"])
            ax.text(0.5, 0.5, "No surviving nodes", transform=ax.transAxes, ha="center", va="center")
            continue

        uniq = np.unique(lineage)
        cmap_vals = {u: i for i, u in enumerate(uniq)}
        colors = np.array([cmap_vals[u] for u in lineage])

        ax.scatter(H[:, 0], H[:, 1], c=colors, cmap="tab20", s=45, alpha=0.9)
        ax.set_title(exp["label"])
        ax.set_xlabel(r"$H^{(1)}$")
        ax.set_ylabel(r"$H^{(2)}$")
        ax.axhline(0, linewidth=0.5, color="gray")
        ax.axvline(0, linewidth=0.5, color="gray")
        ax.text(0.02, 0.98, f"N={H.shape[0]}\nL={len(uniq)}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved lineage plot to {filename}")
    if show:
        plt.show()
    else:
        plt.close(fig)
