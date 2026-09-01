"""Note T-3 (partie B) -- Kairos comme operateur de transition entre paysages.

Objet : l'equilibre bipartitionne (bloc 1 au puits +h*, bloc 2 en -h*) sur un
graphe deux-blocs regulier (d_in, d_out) se reduit, par symetrie antipodale,
a UN bloc couple a lui-meme via r = rho_part = (1+d_in-d_out)/(1+d_in+d_out) :

    x = zeta * tanh(r x) / m(sqrt(2) (a/b) tanh(r x))      (puits diagonal, d=2)

En coordonnee canonique u = tanh(r x) :  Psi_2(u) = r * zeta.
=> la branche bipartitionnee existe ssi  r > r_c := zeta_min,2 / zeta.
Le seuil de transition est un INVARIANT de Psi.

B1  calcul de r_c ; verification sur le modele reduit deux-blocs (fusion/persistance) ;
B2  reseau N=20 (Kairos OFF) : bipartition sur graphes r<r_c et r>r_c ;
B3  reseau, Kairos ON depuis r0<r_c : les recablages homophiles transportent
    rho_part(t) a travers r_c ; la bipartition survit ssi la traversee precede
    la fusion.
"""
from __future__ import annotations

import json
import numpy as np

from iws_core.config import IWSConfig
from iws_core.engine import IWSEngine

ALPHA, BETA, ZETA, G0, G1, LAM = 0.45, 0.12, 0.8, 0.25, 1.25, 2.0
AB = ALPHA / BETA
N, NB = 20, 10
OUT = {}

m_rat = lambda t: (1 + t**2) / (1 + 3 * t**2)

# ------------------------------------------------------------------ B1
print("B1 -- seuil r_c = zeta_min,2 / zeta et modele reduit deux-blocs")
u = np.linspace(1e-6, 1 - 1e-9, 300000)
Psi2 = np.arctanh(u) * m_rat(np.sqrt(2) * AB * u) / u
zmin2 = float(np.min(Psi2))
r_c = zmin2 / ZETA
print(f"  zeta_min (q=2) = {zmin2:.4f}  ->  r_c = {r_c:.4f}")
OUT["zeta_min_q2"], OUT["r_c"] = zmin2, r_c

def block_reduced(r, x0=2.2959, steps=120000, dt=0.01):
    """Bloc unique (d=2 diagonal, par composante) couple par r (antipodal)."""
    x, v, th = x0, 0.0, AB * np.tanh(r * x0) * 0 + AB * np.tanh(x0)
    for _ in range(steps):
        nth = np.sqrt(2) * abs(th)
        gam = G0 + G1 * nth
        mm = m_rat(nth)
        dv = -gam * v - mm * x + ZETA * np.tanh(r * x)
        x, v, th = x + dt * v, v + dt * dv, th + dt * (ALPHA * np.tanh(r * x) - BETA * th)
    return x

for r in (r_c - 0.06, r_c - 0.01, r_c + 0.01, r_c + 0.06):
    xf = block_reduced(r)
    print(f"  r = {r:.4f} ({'<' if r < r_c else '>'} r_c) : x_bloc(inf) = {xf:8.4f}"
          f"  -> {'FUSION (0)' if abs(xf) < 0.2 else 'BIPARTITION PERSISTE'}")
    OUT.setdefault("B1_reduced", {})[round(r, 4)] = float(xf)

# ------------------------------------------------------------------ B2
print("B2 -- reseau N=20, Kairos OFF : bipartition sur graphes deux-blocs")

def two_block_graph(k_in, d_out):
    A = np.zeros((N, N), dtype=int)
    for b in (0, 1):
        for i in range(NB):
            for j in range(1, k_in + 1):
                a, c = b * NB + i, b * NB + (i + j) % NB
                A[a, c] = A[c, a] = 1
    for i in range(NB):
        for j in range(d_out):
            a, c = i, NB + (i + j) % NB
            A[a, c] = A[c, a] = 1
    return A

def rho_part(A):
    phi = np.concatenate([np.ones(NB), -np.ones(NB)])
    deg = A.sum(1) + 1
    At = (A + np.eye(N)) / deg[:, None]
    return float(phi @ (At @ phi) / (phi @ phi))

X_WELL = 2.2959

def run_bipartition(A, use_kairos, steps=4000, seed=5, track=False):
    cfg = IWSConfig.core01(steps=steps, use_kairos=use_kairos)
    eng = IWSEngine(cfg, seed=seed)
    eng.A = A.copy()
    h = np.array([X_WELL, X_WELL])
    eng.H = np.vstack([np.tile(h, (NB, 1)), np.tile(-h, (NB, 1))])
    eng.H += np.random.default_rng(seed).normal(0, 0.05, (N, 2))
    eng.V = np.zeros((N, 2))
    eng.tau = np.vstack([np.tile(AB * np.tanh(h), (NB, 1)),
                         np.tile(-AB * np.tanh(h), (NB, 1))])
    eng.P = np.zeros(N)
    hist = {"sep": [], "r": []}
    def cb(e):
        if track and e.step_index % 10 == 0:
            hist["sep"].append(float(np.linalg.norm(
                e.H[:NB].mean(0) - e.H[NB:].mean(0))))
            hist["r"].append(rho_part(e.A))
    eng.run(callback=cb if track else None)
    sep = float(np.linalg.norm(eng.H[:NB].mean(0) - eng.H[NB:].mean(0)))
    return sep, rho_part(eng.A), len(eng.kairos_log), hist

for k_in, d_out in [(2, 3), (3, 1), (4, 1)]:
    A = two_block_graph(k_in, d_out)
    r0 = rho_part(A)
    sep, _, _, _ = run_bipartition(A, use_kairos=False)
    verdict = "PERSISTE" if sep > 1.0 else "FUSION"
    pred = "persiste" if r0 > r_c else "fusion"
    print(f"  d_in={2*k_in}, d_out={d_out} : r0={r0:.4f} "
          f"({'>' if r0 > r_c else '<'} r_c={r_c:.3f})  sep_finale={sep:6.3f} "
          f"-> {verdict}  [predit : {pred}]")
    OUT.setdefault("B2", []).append({"d_in": 2 * k_in, "d_out": d_out,
                                     "r0": r0, "sep": sep, "verdict": verdict})

# ------------------------------------------------------------------ B3
print("B3 -- Kairos ON depuis r0 < r_c : traversee de la surface d'existence")
for k_in, d_out, label in [(2, 3, "r0=0.20"), (3, 2, "r0=0.45")]:
    A = two_block_graph(k_in, d_out)
    r0 = rho_part(A)
    sep, rf, nk, hist = run_bipartition(A, use_kairos=True, track=True, steps=5000)
    r_arr = np.asarray(hist["r"]); s_arr = np.asarray(hist["sep"])
    crossed = r_arr > r_c
    t_cross = int(np.argmax(crossed)) * 10 if crossed.any() else None
    print(f"  {label} : r0={r0:.3f} -> r_final={rf:.3f}  Kairos={nk}  "
          f"sep_finale={sep:.3f}  traversee r_c au pas {t_cross}  "
          f"-> {'BIPARTITION SURVIT' if sep > 1.0 else 'FUSION'}")
    OUT.setdefault("B3", []).append({
        "r0": r0, "r_final": rf, "n_kairos": nk, "sep_final": sep,
        "t_cross": t_cross,
        "r_t": r_arr[::5].tolist(), "sep_t": s_arr[::5].tolist()})

with open("output_theory/transitions_report.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("-> output_theory/transitions_report.json")
