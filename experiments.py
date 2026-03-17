import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from model import IWSSimulationPaper2, IWS2Parameters


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def make_seeds(n_runs: int, base_seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 1_000_000_000, size=n_runs, dtype=np.int64)


def stack_metric(results: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    """
    Stack metric arrays from multiple runs.

    Assumes all runs have the same number of steps.
    Output shape: (n_runs, n_steps)
    """
    return np.stack([r[key] for r in results], axis=0)


def aggregate_metric(results: list[dict[str, np.ndarray]], key: str) -> dict[str, np.ndarray]:
    data = stack_metric(results, key)
    return {
        "mean": data.mean(axis=0),
        "std": data.std(axis=0),
    }


def save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------

def run_single(seed: int, params: IWS2Parameters, steps: int = 3000) -> dict[str, np.ndarray]:
    """
    Run one simulation and return only lightweight arrays needed for analysis.
    """
    sim = IWSSimulationPaper2(params=params, seed=int(seed))

    for _ in range(steps):
        sim.step()

    # Optional final snapshot
    final_H = sim.H.copy()

    return {
        "population_size": np.asarray(sim.history["population_size"], dtype=float),
        "num_divisions": np.asarray(sim.history["num_divisions"], dtype=float),
        "num_apoptosis": np.asarray(sim.history["num_apoptosis"], dtype=float),
        "mean_kappa": np.asarray(sim.history["mean_kappa"], dtype=float),
        "mean_trace": np.asarray(sim.history["mean_trace"], dtype=float),
        "state_var": np.asarray(sim.history["state_var"], dtype=float),
        "final_H": final_H,
    }


# ---------------------------------------------------------------------
# Parallel experiment
# ---------------------------------------------------------------------

def run_experiment(
    label: str,
    params: IWS2Parameters,
    n_runs: int = 50,
    steps: int = 3000,
    n_jobs: int | None = None,
    base_seed: int = 42,
    save_raw: bool = True,
) -> dict[str, Any]:
    """
    Run many independent seeds in parallel for one regime.
    """
    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count - 1)

    seeds = make_seeds(n_runs=n_runs, base_seed=base_seed)

    print(f"\n=== Running {label} ===")
    print(f"steps={steps}, n_runs={n_runs}, n_jobs={n_jobs}")

    # tqdm doesn't integrate perfectly with joblib by default,
    # but it still gives a useful rough progression over the seeds iterator.
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_single)(seed, params, steps)
        for seed in tqdm(seeds, desc=label)
    )

    out = {
        "label": label,
        "params": params,
        "n_runs": n_runs,
        "steps": steps,
        "seeds": seeds,
        "results": results,
    }

    if save_raw:
        save_pickle(out, RESULTS_DIR / f"{label.replace(' ', '_').lower()}_raw.pkl")

    return out


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def summarize_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    """
    Build mean/std summaries for plotting.
    """
    results = exp["results"]

    summary = {
        "label": exp["label"],
        "n_runs": exp["n_runs"],
        "steps": exp["steps"],
        "population_size": aggregate_metric(results, "population_size"),
        "num_divisions": aggregate_metric(results, "num_divisions"),
        "num_apoptosis": aggregate_metric(results, "num_apoptosis"),
        "mean_kappa": aggregate_metric(results, "mean_kappa"),
        "mean_trace": aggregate_metric(results, "mean_trace"),
        "state_var": aggregate_metric(results, "state_var"),
        # keep first final_H as representative snapshot
        "final_H_example": results[0]["final_H"],
    }
    return summary


def save_summary(summary: dict[str, Any], name: str) -> None:
    save_pickle(summary, RESULTS_DIR / f"{name}_summary.pkl")


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_with_band(ax, x, y_mean, y_std, label):
    ax.plot(x, y_mean, label=label)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15)


def plot_population_dynamics(summaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    steps = summaries[0]["steps"]
    x = np.arange(steps)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Population size
    for s in summaries:
        plot_with_band(
            axes[0, 0], x,
            s["population_size"]["mean"],
            s["population_size"]["std"],
            s["label"]
        )
    axes[0, 0].set_title("Population size")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].legend()

    # Divisions
    for s in summaries:
        plot_with_band(
            axes[0, 1], x,
            s["num_divisions"]["mean"],
            s["num_divisions"]["std"],
            s["label"]
        )
    axes[0, 1].set_title("Divisions per step")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].legend()

    # Apoptosis
    for s in summaries:
        plot_with_band(
            axes[1, 0], x,
            s["num_apoptosis"]["mean"],
            s["num_apoptosis"]["std"],
            s["label"]
        )
    axes[1, 0].set_title("Apoptoses per step")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].legend()

    # kappa
    for s in summaries:
        plot_with_band(
            axes[1, 1], x,
            s["mean_kappa"]["mean"],
            s["mean_kappa"]["std"],
            s["label"]
        )
    axes[1, 1].set_title("Mean residual division capacity")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].legend()

    # trace
    for s in summaries:
        plot_with_band(
            axes[2, 0], x,
            s["mean_trace"]["mean"],
            s["mean_trace"]["std"],
            s["label"]
        )
    axes[2, 0].set_title("Mean trace norm")
    axes[2, 0].set_xlabel("Step")
    axes[2, 0].legend()

    # state variance
    for s in summaries:
        plot_with_band(
            axes[2, 1], x,
            s["state_var"]["mean"],
            s["state_var"]["std"],
            s["label"]
        )
    axes[2, 1].set_title("State variance")
    axes[2, 1].set_xlabel("Step")
    axes[2, 1].legend()

    plt.tight_layout()
    plt.show()


def plot_final_state_projections(summaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    all_H = np.vstack([s["final_H_example"] for s in summaries])
    xmin, xmax = all_H[:, 0].min(), all_H[:, 0].max()
    ymin, ymax = all_H[:, 1].min(), all_H[:, 1].max()

    fig, axes = plt.subplots(1, len(summaries), figsize=(5 * len(summaries), 4), sharex=True, sharey=True)

    if len(summaries) == 1:
        axes = [axes]

    for ax, s in zip(axes, summaries):
        H = s["final_H_example"]
        colors = np.arange(H.shape[0])
        ax.scatter(H[:, 0], H[:, 1], c=colors, s=50)
        ax.axhline(0.0, linewidth=0.5)
        ax.axvline(0.0, linewidth=0.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(s["label"])
        ax.set_xlabel(r"$H^{(1)}$")
        ax.set_ylabel(r"$H^{(2)}$")
        ax.text(
            0.02, 0.95,
            f"N={H.shape[0]}\nL={s['n_runs']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Default regimes for Paper II
# ---------------------------------------------------------------------

def build_regimes() -> list[tuple[str, IWS2Parameters]]:
    """
    Four regimes:
      E1: no structural plasticity
      E2: division only
      E3: division + crisis apoptosis
      E4: full plasticity
    """
    common = dict(
        n_nodes=10,
        dim=2,
        dt=0.05,
        #n_steps=3000,
        # tune these as needed
        theta_div=3.0,
        theta_safe=0.8,
        theta_death=1.2,
        deg_min=2,
        kappa0=10.0,
        delta_div=0.3,
        theta_kappa=0.5,
        rho_min=0.4,
        rho_max=0.6,
        p_inherit=0.5,
        max_nodes=1500,
    )

    e1 = IWS2Parameters(
        **common,
        use_division=False,
        use_apoptosis=False,
        use_senescence=False,
    )

    e2 = IWS2Parameters(
        **common,
        use_division=True,
        use_apoptosis=False,
        use_senescence=False,
    )

    e3 = IWS2Parameters(
        **common,
        use_division=True,
        use_apoptosis=True,
        use_senescence=False,
    )

    e4 = IWS2Parameters(
        **common,
        use_division=True,
        use_apoptosis=True,
        use_senescence=True,
    )

    return [
        ("E1: No structural plasticity", e1),
        ("E2: Division only", e2),
        ("E3: Division + crisis apoptosis", e3),
        ("E4: Full plasticity", e4),
    ]


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main():
    n_runs = 50
    steps = 3000

    cpu_count = os.cpu_count() or 2
    n_jobs = max(1, cpu_count - 1)

    experiments = []
    summaries = []

    for idx, (label, params) in enumerate(build_regimes()):
        exp = run_experiment(
            label=label,
            params=params,
            n_runs=n_runs,
            steps=steps,
            n_jobs=n_jobs,
            base_seed=42 + idx,
            save_raw=True,
        )
        summary = summarize_experiment(exp)

        name = label.replace(":", "").replace("+", "plus").replace(" ", "_").lower()
        save_summary(summary, name)

        experiments.append(exp)
        summaries.append(summary)

    plot_population_dynamics(summaries)
    plot_final_state_projections(summaries)


if __name__ == "__main__":
    main()