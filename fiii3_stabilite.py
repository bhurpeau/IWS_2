"""F-III.3 -- Stabilite de la structure d'observabilite sous accroissement
du budget d'histoires.

Cadre. Filtration declaree B1 c B2 c B3 :
  B1 : sondes simples (u1, G2, T2), T2 = 40..140 pas 10 (et controle -u1) ;
  B2 : B1 + (amorce u1, T_am in {10,20,30,40}) . (oubli 150) . sonde B1 ;
  B3 : B2 + deux amorces (T1,T2) in {20,40}^2, oubli 150 apres chacune, . sonde.

Population. La droite des contenus S(s) = repos + s.u1, s in [0.002, 0.0135].
Si l'issue de chaque histoire est monotone en s (verifie en Y1), chaque
histoire H definit au plus un seuil b_H ; la partition induite par B_k sur la
droite des contenus est decoupee par Theta_k = {b_H : H in B_k}, et le
quotient limite est la droite quotientee par l'adherence de Theta_infini.
La stabilisation de la structure equivaut alors a la stationnarite de Theta.

Experiences :
  Y0  validation de l'integrateur (reference F-III.0 P4, ecart exige < 1e-9).
  Y1  monotonicite des issues en s (6 histoires-temoins x 8 points).
  Y2  seuils b_H par bisection pour toutes les histoires de B1, B2, B3.
  Y3  raffinement : Theta_k cumulatifs, n_classes(k), seuils nouveaux par
      niveau, ecart minimal (accumulation ?), verdict de stationnarite
      observee -- qui n'est jamais une preuve de stabilite (Prop. III-4b).
  Y4  paire inter-especes (G_lin, G_sat) depuis s = 0 : niveau minimal de
      separation DANS la classe declaree (l est relatif a B).
  Y5  figure : partition en couches de la droite des contenus par niveau.
"""
from __future__ import annotations

import json
import math

ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
W = math.tanh(2.2959)
LCHI, KQ, RCHI, KCHI, K3, THK, RP = 0.2, 0.6, 0.8, 2.5, 0.5, 1.25, 0.35
EPS_S, CS, G2 = 0.005, 8.0, 0.18
SQ2 = math.sqrt(2.0)
U1 = (1.0 / SQ2, 1.0 / SQ2)
MU1 = (-U1[0], -U1[1])
S0 = 0.05
B_LIN = lambda d, nc: 1.0 + CS * d / (nc + 1e-12)
B_SAT = lambda d, nc: 1.0 + CS * S0 * math.tanh((d / (nc + 1e-12)) / S0)
S_LO, S_HI = 0.002, 0.0135
GRID_T = list(range(40, 141, 10))
OUT = {}


def run(st, u, g, T, gb=B_LIN, dt=0.01):
    x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc = st
    u1, u2 = u
    gw1, gw2 = g * W * u1, g * W * u2
    for _ in range(int(T / dt)):
        nt = math.sqrt(t1 * t1 + t2 * t2)
        cc = 2.0 / (1.0 + 3.0 * nt * nt)
        thx = t1 * x1 + t2 * x2
        m1 = x1 - cc * thx * t1
        m2 = x2 - cc * thx * t2
        gam = G0 + G1 * nt
        dv1 = -gam * v1 - m1 + ZETA * math.tanh(x1)
        dv2 = -gam * v2 - m2 + ZETA * math.tanh(x2)
        x1, v1 = x1 + dt * v1, v1 + dt * dv1
        x2, v2 = x2 + dt * v2, v2 + dt * dv2
        t1 = t1 + dt * (ALPHA * math.tanh(x1) - BETA * t1)
        t2 = t2 + dt * (ALPHA * math.tanh(x2) - BETA * t2)
        c1 = c1 + dt * (gw1 - LCHI * c1)
        c2 = c2 + dt * (gw2 - LCHI * c2)
        s1 = s1 + dt * EPS_S * (c1 - s1)
        s2 = s2 + dt * EPS_S * (c2 - s2)
        nc = math.sqrt(c1 * c1 + c2 * c2)
        ssc = ssc + dt * EPS_S * (nc - ssc)
        nx = math.sqrt(x1 * x1 + x2 * x2)
        nv = math.sqrt(v1 * v1 + v2 * v2)
        p = p + dt * ((0.6 * nx + 0.4 * nv) / SQ2 + KCHI * nc - K3 * p)
        if p > THK:
            d = s1 * c1 + s2 * c2
            boost = gb(d, nc)
            if boost < 0.0:
                boost = 0.0
            t1 += boost * KQ * c1
            t2 += boost * KQ * c2
            c1 *= (1.0 - RCHI)
            c2 *= (1.0 - RCHI)
            p *= RP
    return (x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc)


def fresh(s=0.0):
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, s * U1[0], s * U1[1], 0.0)


def outcome(s, ams, T2, u_s=U1, gb=B_LIN):
    """Issue de l'histoire (amorces `ams` . oubli 150 chacune) . sonde (u_s, T2)."""
    st = fresh(s)
    for Ta in ams:
        st = run(st, U1, G2, float(Ta), gb)
        st = run(st, U1, 0.0, 150.0, gb)
    st = (1e-3 * u_s[0], 1e-3 * u_s[1], 0.0, 0.0) + st[4:]
    st = run(st, u_s, G2, float(T2), gb)
    st = run(st, u_s, 0.0, 300.0, gb)
    return math.hypot(st[0], st[1]) > 1.0


# ------------------------------------------------------------------- Y0
print("Y0 -- validation integrateur (reference F-III.0 P4)")
pr = run(run(fresh(), U1, G2, 41.9), U1, 0.0, 200.0)
st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + pr[4:]
st = run(run(st, U1, G2, 100.0), U1, 0.0, 200.0)
pc = math.hypot(st[0], st[1])
assert abs(pc - 3.246573235196175) < 1e-9, "integrateur non conforme"
print(f"  ||x_PC|| = {pc:.12f} : conforme", flush=True)
OUT["Y0"] = {"PC": pc}

# ------------------------------------------------------------------- Y1
print("Y1 -- monotonicite des issues en s (temoins)")
witnesses = [((), 60), ((), 100), ((20.0,), 80), ((40.0,), 60),
             ((20.0, 20.0), 60), ((40.0, 40.0), 50)]
gridS = [S_LO + i * (S_HI - S_LO) / 7 for i in range(8)]
mono_all = True
for ams, T2 in witnesses:
    patt = "".join("A" if outcome(s, ams, T2) else "." for s in gridS)
    mono = "." not in patt.split("A", 1)[-1] if "A" in patt else True
    mono_all &= mono
    print(f"  ams={ams} T2={T2} : [{patt}] monotone : {mono}", flush=True)
OUT["Y1"] = {"monotone_partout": mono_all}

# ------------------------------------------------------------------- Y2
print("Y2 -- seuils b_H par bisection (B1, B2, B3)", flush=True)
LEVELS = {
    1: [((), T2) for T2 in GRID_T],
    2: [((Ta,), T2) for Ta in (10.0, 20.0, 30.0, 40.0) for T2 in GRID_T],
    3: [((Ta, Tb), T2) for Ta in (20.0, 40.0) for Tb in (20.0, 40.0)
        for T2 in GRID_T],
}
thresholds = {k: [] for k in LEVELS}
for k, hists in LEVELS.items():
    for ams, T2 in hists:
        o_lo, o_hi = outcome(S_LO, ams, T2), outcome(S_HI, ams, T2)
        if o_lo == o_hi:
            continue
        lo, hi = S_LO, S_HI
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if outcome(mid, ams, T2) == o_lo:
                lo = mid
            else:
                hi = mid
        b = 0.5 * (lo + hi)
        thresholds[k].append({"ams": list(ams), "T2": T2, "b": b})
    print(f"  niveau {k} : {len(LEVELS[k])} histoires, "
          f"{len(thresholds[k])} seuils dans la plage", flush=True)
# controle -u1 au niveau 1 (extremites seulement)
flip_mu1 = [T2 for T2 in GRID_T
            if outcome(S_LO, (), T2, MU1) != outcome(S_HI, (), T2, MU1)]
print(f"  controle -u1 (niveau 1) : seuils dans la plage pour T2 = "
      f"{flip_mu1 if flip_mu1 else 'aucun'}", flush=True)
OUT["Y2"] = {"seuils": thresholds, "controle_mu1": flip_mu1}

# ------------------------------------------------------------------- Y3
print("Y3 -- raffinement et stationnarite observee")
TOL = 1e-4
cum, sched = [], {}
for k in (1, 2, 3):
    new = []
    for t in sorted(x["b"] for x in thresholds[k]):
        if all(abs(t - c) > TOL for c in cum + new):
            new.append(t)
    sched[k] = new
    cum = sorted(cum + new)
    gaps = [round(b - a, 5) for a, b in zip(cum, cum[1:])]
    print(f"  niveau {k} : +{len(new)} seuils nouveaux "
          f"{[round(t, 5) for t in new]} -> n_classes = {len(cum) + 1} ; "
          f"ecart min = {min(gaps) if gaps else '-'}", flush=True)
stationnaire = len(sched[3]) == 0
print(f"  stationnarite observee au niveau 3 : {stationnaire}")
OUT["Y3"] = {"nouveaux_par_niveau": {k: sched[k] for k in sched},
             "seuils_cumules": cum, "n_classes": len(cum) + 1,
             "stationnaire_niveau_3": stationnaire}

# ------------------------------------------------------------------- Y4
print("Y4 -- paire inter-especes (G_lin, G_sat) depuis s = 0")
lvl, wit = None, None
for k, hists in LEVELS.items():
    for ams, T2 in hists:
        ol = outcome(0.0, ams, T2, U1, B_LIN)
        os_ = outcome(0.0, ams, T2, U1, B_SAT)
        if ol != os_:
            lvl, wit = k, {"ams": list(ams), "T2": T2,
                           "lin": "A" if ol else ".",
                           "sat": "A" if os_ else "."}
            break
    if lvl:
        break
print(f"  separation au niveau {lvl} : {wit}", flush=True)
OUT["Y4"] = {"niveau": lvl, "temoin": wit}

# ------------------------------------------------------------------- Y5
print("Y5 -- figure : partition en couches de la droite des contenus")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 3.6))
colors = ["#4878a8", "#c44e52", "#55a868"]
for row, k in enumerate((1, 2, 3)):
    cuts = sorted(t for kk in range(1, k + 1) for t in sched[kk])
    edges = [S_LO] + cuts + [S_HI]
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        ax.plot([a, b], [3 - row] * 2, lw=9, solid_capstyle="butt",
                color=colors[i % 3], alpha=0.85)
    for t in cuts:
        ax.plot([t, t], [3 - row - 0.28, 3 - row + 0.28], color="k", lw=1)
    ax.text(S_LO - 0.0006, 3 - row,
            f"$\\mathfrak{{B}}_{k}$ ({len(cuts) + 1} cl.)",
            ha="right", va="center", fontsize=10)
ax.set_yticks([])
ax.set_xlim(S_LO - 0.0022, S_HI + 0.0002)
ax.set_ylim(0.4, 3.6)
ax.set_xlabel("contenu mnesique  s  (le long de $u_1$)")
ax.set_title("F-III.3 -- partition de la droite des contenus par budget "
             "d'histoires (classes = intervalles ; traits = seuils $b_H$)")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("output_theory/fiii3_stabilite.png", dpi=150)
print("  -> output_theory/fiii3_stabilite.png")

with open("output_theory/fiii3_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii3_report.json")
