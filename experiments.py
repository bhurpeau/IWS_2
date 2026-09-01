import argparse
import numpy as np
from typing import List, Dict, Any
from model import IWSParameters, IWSSimulation
from plotting import plot_all_results, plot_final_state_projections

def run_experiment(label: str, seeds: List[int], params: IWSParameters, steps: int) -> Dict[str, Any]:
    print(f"Running {label} across {len(seeds)} seeds...")
    metrics = {
        "mean_trace": [], "mean_pressure": [], "kairos_events": [],
        "graph_edges": [], "state_norm": [], "state_var": [],
    }
    final_H_example = None

    for idx, seed in enumerate(seeds):
        sim = IWSSimulation(params=params, seed=seed)
        sim.run(steps=steps)
        if idx == 0:
            final_H_example = sim.H.copy()
        for key, hist_key in zip(metrics.keys(), ["mean_trace", "mean_pressure", "num_kairos", "num_edges", "h_norm", "h_var"]):
            metrics[key].append(sim.history[hist_key])

    out = {"label": label, "final_H_example": final_H_example}
    for key, series in metrics.items():
        arr = np.asarray(series, dtype=float)
        out[key + "_mean"] = arr.mean(axis=0)
        out[key + "_std"] = arr.std(axis=0)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inner World Systems (IWS) Simulations")
    parser.add_argument("--steps", type=int, default=5000, help="Number of simulation steps")
    parser.add_argument("--nodes", type=int, default=20, help="Number of nodes in the system")
    parser.add_argument("--seeds", type=int, nargs='+', default=[7, 11, 19, 23, 31], help="Random seeds to average over")
    parser.add_argument("--lambda_metric", type=float, default=2.0, help="Strength of trace-induced geometric deformation")
    
    args = parser.parse_args()

    # Base parameters with overrides from CLI
    base_params = IWSParameters(n_nodes=args.nodes, lambda_metric=args.lambda_metric)

    # E1: No-geometry regime
    params_e1 = IWSParameters(**{**base_params.__dict__, "use_memory_geometry": False, "use_kairos": True})
    exp1 = run_experiment("E1: No-geometry regime", args.seeds, params_e1, args.steps)

    # E2: No-Kairos regime
    params_e2 = IWSParameters(**{**base_params.__dict__, "use_memory_geometry": True, "use_kairos": False})
    exp2 = run_experiment("E2: No-Kairos regime", args.seeds, params_e2, args.steps)

    # E3: Full minimal IWS regime
    params_e3 = IWSParameters(**{**base_params.__dict__, "use_memory_geometry": True, "use_kairos": True})
    exp3 = run_experiment("E3: Full minimal IWS regime", args.seeds, params_e3, args.steps)

    plot_all_results([exp1, exp2, exp3], args.steps)
    plot_final_state_projections([exp1, exp2, exp3])