import os
import pickle
from pathlib import Path
from typing import Any
import argparse

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from model import IWSSimulationPaper2, IWS2Parameters
from plotting import plot_population_results, plot_final_state_and_lineages


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_effective_cpu_count() -> int:
    # cgroups v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().strip().split()
            if quota != "max":
                return max(1, int(int(quota) / int(period)))
    except Exception:
        pass

    # cgroups v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0:
            return max(1, int(quota / period))
    except Exception:
        pass

    return os.cpu_count() or 2


def make_seeds(n_runs: int, base_seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 1_000_000_000, size=n_runs, dtype=np.int64)


def stack_metric(results: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
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


def run_single(seed: int, params: IWS2Parameters, steps: int = 3000) -> dict[str, Any]:
    sim = IWSSimulationPaper2(params=params, seed=int(seed))

    for _ in range(steps):
        sim.step()

    final_H = sim.H.copy()

    if hasattr(sim, "lineage_uid"):
        lineage_uid = np.asarray(sim.lineage_uid)
    else:
        lineage_uid = np.arange(final_H.shape[0])

    return {
        "population_size": np.asarray(sim.history["population_size"], dtype=float),
        "num_divisions": np.asarray(sim.history["num_divisions"], dtype=float),
        "num_apoptosis": np.asarray(sim.history["num_apoptosis"], dtype=float),
        "mean_kappa": np.asarray(sim.history["mean_kappa"], dtype=float),
        "mean_trace": np.asarray(sim.history["mean_trace"], dtype=float),
        "state_var": np.asarray(sim.history["state_var"], dtype=float),
        "final_snapshot": {
            "H": final_H,
            "lineage_uid": lineage_uid,
        },
    }


def run_experiment(
    label: str,
    params: IWS2Parameters,
    n_runs: int = 50,
    steps: int = 3000,
    n_jobs: int | None = None,
    base_seed: int = 42,
    save_raw: bool = True,
    run_tag: str = "default",
) -> dict[str, Any]:
    if n_jobs is None:
        cpu_count = get_effective_cpu_count()
        n_jobs = max(1, cpu_count - 1)

    seeds = make_seeds(n_runs=n_runs, base_seed=base_seed)

    print(f"\n=== Running {label} ===")
    print(f"steps={steps}, n_runs={n_runs}, n_jobs={n_jobs}, run_tag={run_tag}")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
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
        safe_label = label.replace(" ", "_").replace(":", "").lower()
        fname = f"{safe_label}_{run_tag}_raw.pkl"
        save_pickle(out, RESULTS_DIR / fname)

    return out


def summarize_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    results = exp["results"]

    summary = {
        "label": exp["label"],
        "n_runs": exp["n_runs"],
        "steps": exp["steps"],
        "population_size_mean": aggregate_metric(results, "population_size")["mean"],
        "population_size_std": aggregate_metric(results, "population_size")["std"],
        "num_divisions_mean": aggregate_metric(results, "num_divisions")["mean"],
        "num_divisions_std": aggregate_metric(results, "num_divisions")["std"],
        "num_apoptosis_mean": aggregate_metric(results, "num_apoptosis")["mean"],
        "num_apoptosis_std": aggregate_metric(results, "num_apoptosis")["std"],
        "mean_kappa_mean": aggregate_metric(results, "mean_kappa")["mean"],
        "mean_kappa_std": aggregate_metric(results, "mean_kappa")["std"],
        "mean_trace_mean": aggregate_metric(results, "mean_trace")["mean"],
        "mean_trace_std": aggregate_metric(results, "mean_trace")["std"],
        "state_var_mean": aggregate_metric(results, "state_var")["mean"],
        "state_var_std": aggregate_metric(results, "state_var")["std"],
        "final_snapshot_example": results[0]["final_snapshot"],
    }
    return summary


def save_summary(summary: dict[str, Any], name: str) -> None:
    save_pickle(summary, RESULTS_DIR / f"{name}_summary.pkl")


def build_regimes(max_nodes: int) -> list[tuple[str, IWS2Parameters]]:
    common = dict(
        n_nodes=10,
        dim=2,
        dt=0.02,
        theta_div=3.0,
        theta_safe=0.8,
        theta_death=1.2,
        deg_min=2,
        kappa0=10.0,
        delta_div=0.3,
        theta_kappa=0.5,
        p_inherit=0.5,
        max_nodes=max_nodes,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max_nodes", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    run_tag = f"R{args.n_runs}_S{args.steps}_M{args.max_nodes}"
    output_dir = Path("output") / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_count = get_effective_cpu_count()
    n_jobs = max(1, cpu_count - 1)

    print(
        f"""
=== Experiment config ===
n_runs    : {args.n_runs}
steps     : {args.steps}
max_nodes : {args.max_nodes}
run_tag   : {run_tag}
n_jobs    : {n_jobs}
========================
"""
    )

    summaries = []

    for idx, (label, params) in enumerate(build_regimes(max_nodes=args.max_nodes)):
        exp = run_experiment(
            label=label,
            params=params,
            n_runs=args.n_runs,
            steps=args.steps,
            n_jobs=n_jobs,
            base_seed=42 + idx,
            save_raw=True,
            run_tag=run_tag,
        )
        summary = summarize_experiment(exp)

        name = label.replace(":", "").replace("+", "plus").replace(" ", "_").lower()
        save_summary(summary, f"{name}_{run_tag}")

        summaries.append(summary)

    plot_population_results(
        summaries,
        steps=args.steps,
        filename=output_dir / "figure_population.png",
        show=True,
    )

    plot_final_state_and_lineages(
        summaries,
        filename=output_dir / "figure_lineages.png",
        show=True,
    )


if __name__ == "__main__":
    main()