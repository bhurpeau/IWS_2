import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def plot_final_state_projections(experiments: List[Dict[str, Any]], filename: str = "./output/figure2.png"):
    fig, axes = plt.subplots(1, len(experiments), figsize=(15, 4), sharex=True, sharey=True)
    if len(experiments) == 1:
        axes = [axes]
        
    all_H = np.vstack([exp["final_H_example"] for exp in experiments])
    xmin, xmax = all_H[:, 0].min(), all_H[:, 0].max()
    ymin, ymax = all_H[:, 1].min(), all_H[:, 1].max()
    
    for ax, exp in zip(axes, experiments):
        H = exp["final_H_example"]
        colors = np.linalg.norm(H, axis=1)
        ax.scatter(H[:, 0], H[:, 1], c=colors, s=50)
        ax.set_title(exp["label"])
        ax.set_xlabel(r"$H^{(1)}$")
        ax.set_ylabel(r"$H^{(2)}$")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axhline(0, linewidth=0.5, color='gray')
        ax.axvline(0, linewidth=0.5, color='gray')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved projection plot to {filename}")

def plot_metric(ax: plt.Axes, x: np.ndarray, experiments: List[Dict[str, Any]], metric_base: str, title: str):
    for exp in experiments:
        mean = exp[f"{metric_base}_mean"]
        std = exp[f"{metric_base}_std"]
        ax.plot(x, mean, label=exp["label"])
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.legend()

def plot_all_results(experiments: List[Dict[str, Any]], steps: int, filename: str = "./output/figure1.png"):
    x = np.arange(steps)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    plot_metric(axes[0, 0], x, experiments, "mean_trace", "Mean trace norm")
    plot_metric(axes[0, 1], x, experiments, "mean_pressure", "Mean internal pressure")
    plot_metric(axes[1, 0], x, experiments, "kairos_events", "Number of Kairos events")
    plot_metric(axes[1, 1], x, experiments, "graph_edges", "Number of graph edges")
    plot_metric(axes[2, 0], x, experiments, "state_norm", "Mean state norm")
    plot_metric(axes[2, 1], x, experiments, "state_var", "State variance")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved timeseries plot to {filename}")