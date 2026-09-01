"""Criteres de reussite du patch (revue 0.1.1) -- verification automatique.

C1  la configuration legacy reproduit BIT-A-BIT model.IWSSimulation ;
C2  la borne de trace saturee ||tau_i|| <= max(||tau_i(0)||, alpha*sqrt(d)/beta)
    est respectee (tolerance integrateur) ;
C3  |E| strictement invariant en mode conservatif ;
C4  u = 0 produit exactement une entree nulle ;
C5  flot symetrique (graphe circulant, etats identiques, u=0, Kairos OFF) :
    symetrie preservee a la precision machine ;
C6  Kairos ON sur configuration symetrique : le premier evenement est
    simultane, la brisure est CONVENTIONNELLE (departage D-08) et journalisee
    -- ce n'est pas une erreur du simulateur (erratum E-06') ;
C7  generateurs separes : les bras zero/white_noise sont apparies (memes
    graphe et etats initiaux a input_mode different).
"""
from __future__ import annotations

import numpy as np

from model import IWSParameters, IWSSimulation
from iws_core.config import IWSConfig
from iws_core.engine import IWSEngine

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


# ------------------------------------------------------------------ C1
def c1_legacy_bit_exact(steps=800, seeds=(7, 11, 42)):
    keys = ["mean_trace", "mean_pressure", "num_kairos", "num_edges", "h_norm", "h_var"]
    for seed in seeds:
        ref = IWSSimulation(IWSParameters(), seed=seed)
        ref.run(steps)
        eng = IWSEngine(IWSConfig.legacy_p1(steps=steps), seed=seed).run()
        for k in keys:
            a = np.asarray(ref.history[k], dtype=float)
            b = np.asarray(eng.history[k], dtype=float)
            if not np.array_equal(a, b):
                i = int(np.argmax(a != b))
                check(f"C1 legacy bit-exact (seed {seed})", False,
                      f"divergence sur '{k}' au pas {i}: {a[i]} vs {b[i]}")
                return
        if not (np.array_equal(ref.H, eng.H) and np.array_equal(ref.A, eng.A)):
            check(f"C1 legacy bit-exact (seed {seed})", False, "etats finaux differents")
            return
    check("C1 legacy bit-exact (3 graines, 800 pas, 6 observables + etats finaux)", True)


# ------------------------------------------------------------------ C2
def c2_trace_bound(steps=5000, seed=7):
    cfg = IWSConfig.core01(steps=steps, input_mode="fixed_heterogeneous")
    eng = IWSEngine(cfg, seed=seed).run()
    bound = cfg.trace_bound()
    max_tau = max(eng.history["mean_trace"])  # moyenne <= max, mais on verifie le vrai max
    hard = float(np.max(np.linalg.norm(eng.tau, axis=1)))
    viol = [v for v in eng.violations if v["invariant"] == "trace_bound"]
    check("C2 borne de trace saturee",
          len(viol) == 0 and hard <= bound + 1e-6,
          f"max final ||tau||={hard:.4f}, borne={bound:.4f}, violations={len(viol)}")


# ------------------------------------------------------------------ C3
def c3_edge_invariant(steps=5000, seeds=(7, 11, 19)):
    for seed in seeds:
        eng = IWSEngine(IWSConfig.core01(steps=steps,
                                         input_mode="fixed_heterogeneous"),
                        seed=seed).run()
        edges = np.asarray(eng.history["num_edges"])
        viol = [v for v in eng.violations if v["invariant"] == "edge_count"]
        n_kairos = len(eng.kairos_log)
        if not (np.all(edges == edges[0]) and not viol):
            check(f"C3 invariant |E| (seed {seed})", False,
                  f"|E| in [{edges.min()},{edges.max()}], {n_kairos} Kairos")
            return
    check("C3 invariant |E| en mode conservatif (3 graines, Kairos actifs)", True,
          f"|E|={int(edges[0])} constant, {n_kairos} Kairos (derniere graine)")


# ------------------------------------------------------------------ C4
def c4_zero_input():
    eng = IWSEngine(IWSConfig.core01(steps=10), seed=1)
    ok = all(not np.any(eng.input.sample(s)) for s in range(10))
    check("C4 u=0 exactement nul", ok)


# ------------------------------------------------------------------ C5 / C6
def c5_c6_symmetry(steps=3000, seed=3):
    base = IWSConfig.core01(
        graph_mode="circulant", identical_initial_states=True,
        input_mode="zero", steps=steps)
    # C5 : Kairos OFF -> flot exactement symetrique
    cfg_a = IWSConfig.core01(graph_mode="circulant",
                             identical_initial_states=True,
                             input_mode="zero", steps=steps,
                             use_kairos=False)
    eng_a = IWSEngine(cfg_a, seed=seed)
    spreads = []
    eng_a.run(callback=lambda e: spreads.append(e.symmetry_spread()))
    check("C5 symetrie du flot (X4b-1a, Kairos OFF)",
          max(spreads) <= 1e-12,
          f"ecart max inter-noeuds = {max(spreads):.2e}")

    # C6 : Kairos ON -> premier evenement simultane, brisure conventionnelle
    eng_b = IWSEngine(base, seed=seed)
    pre_spread, first_event_step = [], None

    def cb(e):
        nonlocal first_event_step
        if first_event_step is None:
            if e.kairos_log:
                first_event_step = e.kairos_log[0]["step"]
            else:
                pre_spread.append(e.symmetry_spread())

    eng_b.run(callback=cb)
    sym_before = (max(pre_spread) <= 1e-12) if pre_spread else True
    simultaneous = (len(eng_b.symmetry_log) > 0 and
                    len(eng_b.symmetry_log[0]["simultaneous_nodes"]) == eng_b.n)
    check("C6 brisure conventionnelle journalisee (X4b-1b)",
          first_event_step is not None and sym_before and simultaneous,
          f"1er Kairos au pas {first_event_step}, symetrie avant = "
          f"{max(pre_spread):.2e}" if pre_spread else "")


# ------------------------------------------------------------------ C7
def c7_paired_streams(steps=50, seed=5):
    e0 = IWSEngine(IWSConfig.core01(steps=steps, input_mode="zero"), seed=seed)
    e1 = IWSEngine(IWSConfig.core01(steps=steps, input_mode="white_noise",
                                    input_sigma=0.1), seed=seed)
    ok = (np.array_equal(e0.A, e1.A) and np.array_equal(e0.H, e1.H)
          and np.array_equal(e0.V, e1.V))
    check("C7 flux RNG separes : bras apparies (graphe/etats identiques "
          "entre input_mode)", ok)


if __name__ == "__main__":
    c1_legacy_bit_exact()
    c2_trace_bound()
    c3_edge_invariant()
    c4_zero_input()
    c5_c6_symmetry()
    c7_paired_streams()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} verifications passees.")
    raise SystemExit(1 if n_fail else 0)
