"""X0 -- Recalibration de base (Core Specification 0.1.1, Partie VI + revue).

Deux questions, rien d'autre :
  (1) le code reproduit-il la ligne de base historique ?  (-> run_checks C1)
  (2) la saturation SEULE modifie-t-elle qualitativement les trois regimes ?

Matrice :
  X0-P1   trace lineaire  | recablage legacy (2/2 + plafond) | u = eps*W
  X0-S    trace saturee   | identique a X0-P1 pour tout le reste
  X0-Core trace saturee   | conservatif m/m                  | u = 0
          (+ pression core, integrateur explicite, RNG separes, sans coup
           de bruit : bras d'effet CUMULE, ne sert PAS a attribuer un effet)

Chaque sortie contient : configuration complete, graine, version du code
(commit git + version du patch), temps de Kairos, violations d'invariants.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from iws_core.config import IWSConfig, PATCH_VERSION
from iws_core.engine import IWSEngine
from iws_core.validation import integrator_check

SEEDS = [7, 11, 19, 23, 31, 37, 41, 43, 53, 61]
STEPS = 5000
OUT = Path("output_x0")
KEYS = ["mean_trace", "mean_pressure", "num_kairos", "num_edges",
        "h_norm", "h_var", "homophily"]

ARMS = {
    "X0-P1": IWSConfig.legacy_p1(steps=STEPS),
    "X0-S": IWSConfig.legacy_p1(steps=STEPS, trace_mode="saturated"),
    "X0-Core": IWSConfig.core01(steps=STEPS),
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_arm(label: str, cfg: IWSConfig) -> dict:
    t0 = time.time()
    per_seed = {k: [] for k in KEYS}
    kairos_counts, violations, finals = [], [], []
    for seed in SEEDS:
        eng = IWSEngine(cfg, seed=seed).run()
        r = eng.result()
        for k in KEYS:
            per_seed[k].append(r["history"][k])
        kairos_counts.append(len(r["kairos_times"]))
        violations.append(len(r["violations"]))
        finals.append(r["final_H"])
    out = {"label": label, "seeds": SEEDS, "elapsed_s": time.time() - t0,
           "config": cfg.to_dict(), "git_commit": git_commit(),
           "patch_version": PATCH_VERSION,
           "total_kairos_per_seed": kairos_counts,
           "invariant_violations_per_seed": violations,
           "final_H_seed0": finals[0]}
    for k in KEYS:
        arr = np.asarray(per_seed[k], dtype=float)
        out[k + "_mean"] = arr.mean(axis=0)
        out[k + "_std"] = arr.std(axis=0)
        out[k + "_final_mean"] = float(arr[:, -1].mean())
        out[k + "_final_std"] = float(arr[:, -1].std())
    return out


def main():
    OUT.mkdir(exist_ok=True)
    summary = {"patch_version": PATCH_VERSION, "git_commit": git_commit(),
               "steps": STEPS, "seeds": SEEDS, "arms": {}}
    results = {}
    for label, cfg in ARMS.items():
        print(f"== {label}")
        res = run_arm(label, cfg)
        results[label] = res
        np.savez_compressed(
            OUT / f"{label}.npz",
            **{k + s: res[k + s] for k in KEYS for s in ("_mean", "_std")},
            final_H_seed0=res["final_H_seed0"])
        summary["arms"][label] = {
            "config": res["config"],
            "total_kairos_per_seed": res["total_kairos_per_seed"],
            "invariant_violations": res["invariant_violations_per_seed"],
            "finals": {k: [res[k + "_final_mean"], res[k + "_final_std"]]
                       for k in KEYS},
            "elapsed_s": round(res["elapsed_s"], 1),
        }
        print(f"   Kairos/graine: {res['total_kairos_per_seed']}")
        print(f"   finals: " + ", ".join(
            f"{k}={res[k + '_final_mean']:.3f}±{res[k + '_final_std']:.3f}"
            for k in ("mean_trace", "mean_pressure", "h_var", "num_edges")))

    # validation d'integrateur dt vs dt/2 (appariable : kick=0)
    print("== validation integrateur (dt vs dt/2)")
    ic = {}
    ic["X0-P1(kick=0)"] = integrator_check(
        IWSConfig.legacy_p1(steps=STEPS, kairos_kick_sigma=0.0), seed=SEEDS[0])
    ic["X0-S(kick=0)"] = integrator_check(
        IWSConfig.legacy_p1(steps=STEPS, trace_mode="saturated",
                            kairos_kick_sigma=0.0), seed=SEEDS[0])
    ic["X0-Core"] = integrator_check(ARMS["X0-Core"], seed=SEEDS[0])
    for k, v in ic.items():
        print(f"   {k}: max_delta={v.get('max_delta', float('nan')):.4f} "
              f"converged={v.get('converged')} "
              f"events={v.get('event_count')} "
              f"event_time_gap={v.get('event_time_max_gap', 0):.3f}")
    summary["integrator_checks"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "deltas"} | v.get("deltas", {})
        for k, v in ic.items()}

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    plot(results)
    print(f"\nSorties dans {OUT}/")


def plot(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"mean_trace": "O1 norme moyenne de trace",
              "mean_pressure": "O2 pression moyenne",
              "num_kairos": "O3 Kairos par pas",
              "num_edges": "O4 nombre d'aretes",
              "h_norm": "O5 norme moyenne d'etat",
              "h_var": "O6 variance d'etat",
              "homophily": "O7 homophilie de trace"}
    colors = {"X0-P1": "tab:blue", "X0-S": "tab:orange", "X0-Core": "tab:green"}
    fig, axes = plt.subplots(4, 2, figsize=(11, 13))
    axes = axes.ravel()
    t = np.arange(STEPS)
    for ax, k in zip(axes, KEYS):
        for label, res in results.items():
            m, s = res[k + "_mean"], res[k + "_std"]
            if k == "num_kairos":
                m = np.convolve(m, np.ones(100) / 100, mode="same")
                s = np.convolve(s, np.ones(100) / 100, mode="same")
            ax.plot(t, m, color=colors[label], label=label, lw=1.2)
            ax.fill_between(t, m - s, m + s, color=colors[label], alpha=0.15)
        ax.set_title(titles[k], fontsize=10)
        ax.set_xlabel("pas")
    axes[0].legend(fontsize=9)
    # etats finaux (graine 0)
    ax = axes[-1]
    for label, res in results.items():
        Hf = res["final_H_seed0"]
        ax.scatter(Hf[:, 0], Hf[:, 1], s=18, color=colors[label],
                   label=label, alpha=0.8)
    ax.set_title("Etats finaux H (graine 7)", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"X0 -- recalibration ({len(SEEDS)} graines, {STEPS} pas, "
                 f"{PATCH_VERSION})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "x0_observables.png", dpi=130)
    print(f"   figure: {OUT}/x0_observables.png")


if __name__ == "__main__":
    main()
