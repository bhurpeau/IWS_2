"""X4b -- Origine de la brisure de symetrie (erratum E-06' integre).

Distinction imposee par la revue :
  - symetrie du flot : preservee tant qu'aucun Kairos simultane n'est resolu ;
  - brisure CONVENTIONNELLE : le departage min(index) (D-08) rompt la symetrie
    lors d'un evenement simultane -- consequence de la specification, pas
    artefact du simulateur. Elle est journalisee, jamais comptee en violation.

Bras (base core01, u=0 sauf mention) :
  1a  circulant, etats identiques, Kairos OFF   (controle absolu du flot)
  1b  circulant, etats identiques, Kairos ON    (brisure par departage)
  2   idem 1b + perturbation eps0 sur le noeud 0, eps0 en balayage
  3   graphe aleatoire (ER), etats identiques
  4   circulant, etats aleatoires
  5   circulant, etats identiques, bruit blanc isotrope sigma=1e-6

Question : IWS cree-t-il de la differenciation, ou amplifie-t-il et
stabilise-t-il une heterogeneite initiale ?  (C-4 reformulee)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from iws_core.config import IWSConfig, PATCH_VERSION
from iws_core.engine import IWSEngine

STEPS = 5000
SAMPLE = 10  # echantillonnage du spread
OUT = Path("output_x4b")
EPS_SWEEP = [1e-8, 1e-6, 1e-4, 1e-2]
SEEDS_RANDOM = [7, 11, 19, 23, 31]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_one(cfg: IWSConfig, seed: int) -> dict:
    eng = IWSEngine(cfg, seed=seed)
    spread_t, spread_v = [], []

    def cb(e):
        if e.step_index % SAMPLE == 0:
            spread_t.append(e.step_index)
            spread_v.append(e.symmetry_spread())

    eng.run(callback=cb)
    r = eng.result()
    o6 = r["history"]["h_var"]
    first_kairos = r["kairos_log"][0]["step"] if r["kairos_log"] else None
    first_simult = (r["symmetry_log"][0]["step"] if r["symmetry_log"] else None)
    n_simult = (len(r["symmetry_log"][0]["simultaneous_nodes"])
                if r["symmetry_log"] else 0)
    return {
        "spread_t": np.array(spread_t), "spread": np.array(spread_v),
        "o6": o6, "o6_final": float(o6[-1]),
        "o2_final": float(r["history"]["mean_pressure"][-1]),
        "n_kairos": len(r["kairos_log"]),
        "first_kairos_step": first_kairos,
        "first_simultaneous": {"step": first_simult, "n_nodes": n_simult},
        "violations": len(r["violations"]),
        "config": r["config"], "seed": seed,
    }


def growth_rate(o6: np.ndarray, dt: float, floor: float = 1e-28) -> float:
    """Taux log-lineaire de O6 sur le segment central de la phase pre-plancher.

    ERRATUM E-X4b-G : la version anterieure ajustait, en cas de decroissance
    immediate, sur les 10 PREMIERS PAS (phase balistique de rotation
    position->vitesse de la perturbation), d'ou le G ~ -0.46 publie dans
    RAPPORT_X0_X4b.md -- artefact de fenetre, pas un taux dynamique.
    Reference corrigee (note T-1) : taux asymptotique au point actif
    ~ -0.093 (spectral, flot) / -0.082 (mesure, evenements inclus).
    """
    v = np.maximum(np.asarray(o6), floor)
    above = np.where(v > 10 * floor)[0]
    if len(above) < 100:
        return float("nan")
    i0, i_end = int(above[0]), int(above[-1])
    span = i_end - i0
    a, b = i0 + int(0.2 * span), i0 + int(0.8 * span)
    if b - a < 50:
        return float("nan")
    coef = np.polyfit(np.arange(a, b) * dt, np.log(v[a:b]), 1)
    return float(coef[0])


def main():
    OUT.mkdir(exist_ok=True)
    dt = 0.05
    report = {"patch_version": PATCH_VERSION, "git_commit": git_commit(),
              "steps": STEPS, "arms": {}}
    curves = {}

    base = dict(steps=STEPS, input_mode="zero")

    # --- 1a : Kairos OFF, controle absolu du flot symetrique
    cfg = IWSConfig.core01(graph_mode="circulant",
                           identical_initial_states=True, use_kairos=False,
                           **base)
    r = run_one(cfg, seed=3)
    report["arms"]["1a_flow_control"] = {
        "max_spread": float(r["spread"].max()), "o6_final": r["o6_final"],
        "interpretation": "symetrie du flot : PASS si ~precision machine"}
    curves["1a"] = r

    # --- 1b : Kairos ON, brisure conventionnelle par departage
    cfg = IWSConfig.core01(graph_mode="circulant",
                           identical_initial_states=True, **base)
    r = run_one(cfg, seed=3)
    report["arms"]["1b_tiebreak"] = {
        "first_kairos_step": r["first_kairos_step"],
        "first_simultaneous": r["first_simultaneous"],
        "spread_before_first_kairos": float(
            r["spread"][r["spread_t"] < (r["first_kairos_step"] or STEPS)].max()
            if r["first_kairos_step"] else r["spread"].max()),
        "o6_final": r["o6_final"], "n_kairos": r["n_kairos"],
        "G": growth_rate(r["o6"], dt),
        "interpretation": ("brisure CONVENTIONNELLE (D-08) au premier "
                           "evenement simultane, journalisee")}
    curves["1b"] = r

    # --- 2 : perturbation eps0 sur le noeud 0 (balayage)
    report["arms"]["2_perturbation"] = []
    for eps in EPS_SWEEP:
        cfg = IWSConfig.core01(graph_mode="circulant",
                               identical_initial_states=True,
                               perturb_node=0, perturb_eps=eps, **base)
        r = run_one(cfg, seed=3)
        amp = r["o6_final"] / (eps ** 2) if eps > 0 else float("nan")
        report["arms"]["2_perturbation"].append({
            "eps0": eps, "o6_final": r["o6_final"], "G": growth_rate(r["o6"], dt),
            "amplification_o6_over_eps0sq": amp,
            "first_kairos_step": r["first_kairos_step"],
            "n_kairos": r["n_kairos"]})
        curves[f"2_eps{eps:.0e}"] = r

    # --- 3 : graphe aleatoire, etats identiques
    arm3 = []
    for seed in SEEDS_RANDOM:
        cfg = IWSConfig.core01(graph_mode="erdos_renyi",
                               identical_initial_states=True, **base)
        r = run_one(cfg, seed=seed)
        arm3.append({"seed": seed, "o6_final": r["o6_final"],
                     "G": growth_rate(r["o6"], dt), "n_kairos": r["n_kairos"]})
        if seed == SEEDS_RANDOM[0]:
            curves["3"] = r
    report["arms"]["3_random_graph"] = arm3

    # --- 4 : graphe regulier, etats aleatoires
    arm4 = []
    for seed in SEEDS_RANDOM:
        cfg = IWSConfig.core01(graph_mode="circulant",
                               identical_initial_states=False, **base)
        r = run_one(cfg, seed=seed)
        arm4.append({"seed": seed, "o6_final": r["o6_final"],
                     "G": growth_rate(r["o6"], dt), "n_kairos": r["n_kairos"]})
        if seed == SEEDS_RANDOM[0]:
            curves["4"] = r
    report["arms"]["4_random_states"] = arm4

    # --- 5 : bruit isotrope infinitesimal
    arm5 = []
    for seed in SEEDS_RANDOM:
        cfg = IWSConfig.core01(graph_mode="circulant",
                               identical_initial_states=True,
                               input_mode="white_noise", input_sigma=1e-6,
                               steps=STEPS)
        r = run_one(cfg, seed=seed)
        arm5.append({"seed": seed, "o6_final": r["o6_final"],
                     "G": growth_rate(r["o6"], dt), "n_kairos": r["n_kairos"]})
        if seed == SEEDS_RANDOM[0]:
            curves["5"] = r
    report["arms"]["5_infinitesimal_noise"] = arm5

    with open(OUT / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    plot(curves)
    print(json.dumps(report["arms"], indent=2, default=str)[:4000])
    print(f"\nSorties dans {OUT}/")


def plot(curves: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) spread au cours du temps
    ax = axes[0]
    for key in ("1a", "1b"):
        r = curves[key]
        ax.semilogy(r["spread_t"], np.maximum(r["spread"], 1e-18),
                    label=f"bras {key}")
    ax.axhline(1e-12, color="grey", ls=":", lw=0.8, label="tolerance symetrie")
    ax.set_title("Ecart inter-noeuds (symetrie)")
    ax.set_xlabel("pas"); ax.legend(fontsize=8)
    # (b) O6 pour le balayage eps0
    ax = axes[1]
    for eps in EPS_SWEEP:
        r = curves[f"2_eps{eps:.0e}"]
        ax.semilogy(np.maximum(np.asarray(r["o6"]), 1e-30),
                    label=f"eps0={eps:.0e}", lw=1.0)
    ax.set_title("O6(t), bras 2 (perturbation)")
    ax.set_xlabel("pas"); ax.legend(fontsize=8)
    # (c) O6 bras 1b / 3 / 4 / 5
    ax = axes[2]
    for key, lab in (("1b", "1b regulier+departage"), ("3", "3 graphe aleatoire"),
                     ("4", "4 etats aleatoires"), ("5", "5 bruit 1e-6")):
        r = curves[key]
        ax.semilogy(np.maximum(np.asarray(r["o6"]), 1e-30), label=lab, lw=1.0)
    ax.set_title("O6(t), sources d'heterogeneite")
    ax.set_xlabel("pas"); ax.legend(fontsize=8)
    fig.suptitle("X4b -- origine de la brisure de symetrie", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "x4b.png", dpi=130)
    print(f"   figure: {OUT}/x4b.png")


if __name__ == "__main__":
    main()
