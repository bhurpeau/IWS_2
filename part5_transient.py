"""PART 5 -- resolution de l'ecart entre G_asymptotique (~-0.09) et G_X4b (-0.4577).

Hypothese : le -0.4577 de X4b est une MOYENNE SUR TOUTE LA TRAJECTOIRE
(transitoire d'allumage inclus), pas le taux asymptotique au point actif.

(5a) replique exacte du bras 2 de X4b ; G calcule sur la fenetre entiere
     (definition X4b) puis sur une fenetre tardive (asymptotique) ;
(5b) integration variationnelle transverse (mode rho fixe, graphe gele,
     relaxations d'evenements delta_v <- r_V delta_v incluses, saltation
     ignoree) le long de la trajectoire sur Delta -- prediction du taux moyen.
"""
from __future__ import annotations

import json
import numpy as np

from iws_core.config import IWSConfig
from iws_core.engine import IWSEngine, circulant_graph, row_normalize_with_self_loops

ZETA, LAM, ALPHA, BETA = 0.8, 2.0, 0.45, 0.12
G0, G1 = 0.25, 1.25
K1, K2, K3, THETA_K = 0.6, 0.4, 0.5, 1.25
RP, RV = 0.35, 0.5
DT, STEPS, N = 0.05, 5000, 20

out = {}

# ---------------------------------------------------------------- (5a)
def x4b_replica(seed=7, eps0=1e-6):
    cfg = IWSConfig.core01(steps=STEPS, graph_mode="circulant",
                           identical_initial_states=True,
                           perturb_node=0, perturb_eps=eps0)
    eng = IWSEngine(cfg, seed=seed).run()
    o6 = np.asarray(eng.history["h_var"])
    t = np.arange(STEPS) * DT
    floor = o6 > 1e-28
    # definition X4b : pente moyenne sur toute la trajectoire
    i0 = 10
    i1 = int(np.max(np.where(floor)[0])) if floor.any() else STEPS - 1
    G_whole = float((np.log(o6[i1]) - np.log(o6[i0])) / (t[i1] - t[i0]))
    # fenetre tardive (asymptotique), si au-dessus du plancher
    a, b = int(0.6 * i1), i1
    G_late = float(np.polyfit(t[a:b], np.log(o6[a:b]), 1)[0]) if b - a > 50 else float("nan")
    hf = float(np.mean(np.linalg.norm(eng.H, axis=1)))
    return G_whole, G_late, len(eng.kairos_log), hf, i1

print("(5a) replique X4b bras 2 (etats identiques + eps0, Kairos ON)")
for seed in (7, 11, 19):
    Gw, Gl, nk, hf, i1 = x4b_replica(seed)
    print(f"  seed {seed}: G(fenetre entiere)={Gw:+.4f}  G(tardif)={Gl:+.4f}  "
          f"Kairos={nk}  ||h||_final={hf:.3f}  (plancher au pas {i1})")
    out[f"replica_seed{seed}"] = {"G_whole": Gw, "G_late": Gl,
                                  "kairos": nk, "h_final": hf}

# ---------------------------------------------------------------- (5b)
print("(5b) variationnel transverse le long du transitoire (rho fixe)")

def sech2(x): return 1.0 / np.cosh(x) ** 2

def Mmat(th):
    n2 = float(th @ th)
    return np.linalg.inv(np.eye(2) + LAM * np.outer(th, th) / (1 + n2))

def DM_at(th, h, eps=1e-7):
    D = np.zeros((2, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = eps
        D[:, k] = (Mmat(th + e) @ h - Mmat(th - e) @ h) / (2 * eps)
    return D

def variational_run(rho, seed=7, steps=STEPS):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[1])
    h = rng.normal(0, 0.6, 2)           # meme tirage que identical_initial_states
    v = rng.normal(0, 0.1, 2)
    th = np.zeros(2); p = 0.0
    dh = np.array([1e-8, 0.0]); dv = np.zeros(2); dth = np.zeros(2)
    logs, norm0 = [], np.linalg.norm(dh)
    for step in range(steps):
        # champ sur Delta (ordre engine, Euler explicite)
        gam = G0 + G1 * np.linalg.norm(th)
        M = Mmat(th)
        dv_dt = -gam * v - M @ h + ZETA * np.tanh(h)
        # variationnel (avant mise a jour, memes points d'evaluation)
        nth = np.linalg.norm(th)
        gam_th = (G1 * th / nth) if nth > 1e-12 else np.zeros(2)
        S = np.diag(sech2(h))
        ddv = (-gam * dv - np.outer(v, gam_th) @ dth
               - M @ dh + ZETA * rho * S @ dh - DM_at(th, h) @ dth)
        v_old, dv_old = v.copy(), dv.copy()
        v = v + DT * dv_dt;      dv = dv + DT * ddv
        h = h + DT * v_old;      dh = dh + DT * dv_old
        th = th + DT * (ALPHA * np.tanh(h) - BETA * th)
        dth = dth + DT * (ALPHA * rho * S @ dh - BETA * dth)
        p = p + DT * (K1 * np.linalg.norm(h) + K2 * np.linalg.norm(v) - K3 * p)
        if p > THETA_K:                  # cascade sur Delta : tous les noeuds
            p *= RP
            v *= RV
            dv *= RV                     # relaxation transverse de v
        logs.append(np.log(np.linalg.norm(np.concatenate([dh, dv, dth])) / norm0))
    logs = np.asarray(logs)
    t = np.arange(steps) * DT
    G_whole = 2 * logs[-1] / t[-1]       # O6 ~ ||delta||^2
    a = int(0.6 * steps)
    G_late = 2 * float(np.polyfit(t[a:], logs[a:], 1)[0])
    return G_whole, G_late, float(np.linalg.norm(h))

A = circulant_graph(N, (1, 2))
At = row_normalize_with_self_loops(A)
rhos = np.sort(np.linalg.eigvalsh((At + At.T) / 2))
for rho in (rhos[-2], rhos[0]):          # mode le plus lent et le plus rapide
    Gw, Gl, hf = variational_run(rho)
    print(f"  rho={rho:+.4f}: G_pred(fenetre entiere)={Gw:+.4f}  "
          f"G_pred(tardif)={Gl:+.4f}  ||h||_final={hf:.3f}")
    out[f"variational_rho_{rho:+.3f}"] = {"G_whole": Gw, "G_late": Gl}

with open("output_theory/part5_transient.json", "w") as f:
    json.dump(out, f, indent=2)
print("-> output_theory/part5_transient.json")
