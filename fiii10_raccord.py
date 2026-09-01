"""F-III.10 -- Le raccord des nappes : Sigma_P stratifiee par les calendriers
de la sonde (QO-79, programme F-III.9.1-R8).

Methode (calendrier par calendrier, pas de cartographie aveugle) :
  1. signature evenementielle n(theta, s) = nombre de Kairos charges de la
     sonde 80, sur la region de reorganisation, a p = chi = 0 ;
  2. frontieres de changement de calendrier = bords des cellules de n ;
  3. Sigma_P dans le plan = transitions d'issue ; chaque transition est
     classee K (n change au meme endroit) ou C (n constant de part et
     d'autre) ;
  4. terminaisons de nappes : la branche superieure s_c^+ et la bascule
     vers tout-approprie coincident-elles avec des bords de cellules ?
     Colonne du temoin (p, chi) != 0 en controle.

Verdict vise : Sigma_P = union des Sigma_{P,C} sur l'exemple -- ou son
refus, chiffre.

Protocole declare : sonde fil-rouge (u1, G2, T2 = 80, regraine canonique,
relaxation 300), rho = bassin binaire ; calendrier = Kairos charges
(||chi|| > 0.01) de la phase active ; grille theta in [0.05, 0.15] pas
0.005, s in [0.001, 0.021] pas 0.001 ; p = chi = 0 sauf X3b.
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
NC_MIN = 0.01
OUT = {}


def run(st, u, g, T, rec=None, dt=0.01):
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
            boost = 1.0 + CS * d / (nc + 1e-12)
            if boost < 0.0:
                boost = 0.0
            if rec is not None and nc > NC_MIN:
                rec.append(round(_ * dt, 2))
            t1 += boost * KQ * c1
            t2 += boost * KQ * c2
            c1 *= (1.0 - RCHI)
            c2 *= (1.0 - RCHI)
            p *= RP
    return (x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc)


def lecture(pfix, cfix, theta, s):
    """(issue, n) de la sonde 80 sur l'etat lent injecte."""
    st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0,
          theta * U1[0], theta * U1[1], cfix * U1[0], cfix * U1[1],
          pfix, s * U1[0], s * U1[1], 0.0)
    rec = []
    st = run(st, U1, G2, 80.0, rec)
    st = run(st, U1, 0.0, 300.0)
    return (math.hypot(st[0], st[1]) > 1.0), len(rec), tuple(rec)


# ------------------------------------------------------------------- X0
print("X0 -- validation integrateur")
fr = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0)
pr = run(run(fr, U1, G2, 41.9), U1, 0.0, 200.0)
st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + pr[4:]
st = run(run(st, U1, G2, 100.0), U1, 0.0, 200.0)
assert abs(math.hypot(st[0], st[1]) - 3.246573235196175) < 1e-9
print("  conforme", flush=True)

# ------------------------------------------------------------------- X1
print("X1 -- carte (issue, n) sur la region de reorganisation, p = chi = 0")
TH = [round(0.05 + 0.005 * i, 3) for i in range(21)]
SS = [round(0.001 + 0.001 * j, 3) for j in range(21)]
ISS, NN, DT = {}, {}, {}
for i, th in enumerate(TH):
    row_i, row_n = "", []
    for s in SS:
        o, n, dates = lecture(0.0, 0.0, th, s)
        ISS[(th, s)] = o
        NN[(th, s)] = n
        DT[(th, s)] = dates
        row_i += "A" if o else "."
        row_n.append(n)
    print(f"  theta={th:5.3f} issue=[{row_i}] n={row_n}", flush=True)
OUT["X1"] = {"TH": TH, "SS": SS,
             "issue": {f"{th}": "".join("A" if ISS[(th, s)] else "."
                                        for s in SS) for th in TH},
             "n": {f"{th}": [NN[(th, s)] for s in SS] for th in TH}}

# ------------------------------------------------------------------- X2
print("X2 -- classification des transitions d'issue : K ou C")
T_JUMP = 5.0


def type_trans(a, b):
    if NN[a] != NN[b]:
        return "K"
    if any(abs(x - y) > T_JUMP for x, y in zip(DT[a], DT[b])):
        return "K"
    return "C"


trans = {"K": 0, "C": 0, "K_compte": 0, "detail_C": [], "toutes": []}
paires = [((TH[i], s), (TH[i + 1], s)) for i in range(len(TH) - 1)
          for s in SS] + \
         [((th, SS[j]), (th, SS[j + 1])) for th in TH
          for j in range(len(SS) - 1)]
for a, b in paires:
    if ISS[a] != ISS[b]:
        t = type_trans(a, b)
        trans[t] += 1
        trans["toutes"].append({"a": list(a), "b": list(b), "type": t})
        if t == "K" and NN[a] != NN[b]:
            trans["K_compte"] += 1
        if t == "C":
            trans["detail_C"].append({"a": list(a), "b": list(b),
                                      "n": NN[a]})
n_tot = trans["K"] + trans["C"]
print(f"  transitions d'issue : {n_tot} ; K (compte ou saut de date "
      f"> {T_JUMP} u) : {trans['K']} ({100*trans['K']/n_tot:.0f}%), dont "
      f"par compte seul : {trans['K_compte']} ; C stricts : {trans['C']}",
      flush=True)
for d in trans["detail_C"][:8]:
    print(f"    C : {d['a']} -> {d['b']} (n = {d['n']})", flush=True)
OUT["X2"] = {"K": trans["K"], "C": trans["C"], "K_compte": trans["K_compte"], "toutes": trans["toutes"],
             "detail_C": trans["detail_C"][:20]}

# ------------------------------------------------------------------- X3
print("X3 -- terminaisons de nappes")
# (a) la branche superieure a theta = 0.10 : n sous / sur s_c^+ = 0.01873
o1, n1, d1 = lecture(0.0, 0.0, 0.10, 0.0180)
o2, n2, d2 = lecture(0.0, 0.0, 0.10, 0.0192)
saut1 = max((abs(x - y) for x, y in zip(d1, d2)), default=0.0)
print(f"  s_c^+ (theta=0.10) : sous (s=0.0180) issue={'A' if o1 else '.'} "
      f"n={n1} ; sur (s=0.0192) issue={'A' if o2 else '.'} n={n2} ; "
      f"saut de date max = {saut1:.2f} u -> "
      f"{'K' if (n1 != n2 or saut1 > 5.0) else 'C'}", flush=True)
# (b) la bascule tout-approprie : colonne s = 0.002, theta croissant fin
print("  bascule tout-A a bas contenu (s = 0.002), theta = 0.110..0.130 :")
basc = []
for th in [round(0.110 + 0.002 * i, 3) for i in range(11)]:
    o, n, _d = lecture(0.0, 0.0, th, 0.002)
    basc.append({"theta": th, "issue": "A" if o else ".", "n": n})
print("   " + " ".join(f"{b['theta']}:{b['issue']}{b['n']}" for b in basc),
      flush=True)
# (c) colonne du temoin (p, chi) != 0 : la frontiere elevee est-elle K ?
print("  colonne du temoin (p=0.053, chi=0.0064, theta=0.1044) :")
col = []
for s in [round(0.014 + 0.001 * i, 3) for i in range(7)]:
    o, n, _d = lecture(0.053, 0.0064, 0.1044, s)
    col.append({"s": s, "issue": "A" if o else ".", "n": n})
print("   " + " ".join(f"{c['s']}:{c['issue']}{c['n']}" for c in col),
      flush=True)
OUT["X3"] = {"s_plus": {"sous": [o1, n1], "sur": [o2, n2]},
             "bascule": basc, "colonne_temoin": col}

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9.5, 6.0))
nvals = sorted({NN[k] for k in NN})
cmap = plt.cm.viridis
for (th, s), n in NN.items():
    col_ = cmap((nvals.index(n)) / max(1, len(nvals) - 1))
    ax.scatter(th, s, c=[col_], s=46, marker="s",
               edgecolors="k" if ISS[(th, s)] else "none",
               linewidths=1.1 if ISS[(th, s)] else 0.0)
handles = [mpatches.Patch(color=cmap(i / max(1, len(nvals) - 1)),
                          label=f"n = {n}")
           for i, n in enumerate(nvals)]
handles.append(mpatches.Patch(facecolor="white", edgecolor="k",
                              label="bord noir = approprie"))
ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=2)
for tr in trans["toutes"]:
    (ta, sa), (tb, sb) = tr["a"], tr["b"]
    col_t = "#d62728" if tr["type"] == "K" else "#222222"
    lw_t = 2.4 if tr["type"] == "K" else 1.0
    ax.plot([ta, tb], [sa, sb], "-", color=col_t, lw=lw_t, alpha=0.9,
            zorder=3)
ax.plot([], [], "-", color="#d62728", lw=2.4, label="transition K")
ax.plot([], [], "-", color="#222222", lw=1.0, label="transition C")
ax.annotate("plafond $s_c^+$ (pli C)", (0.100, 0.0187),
            xytext=(0.062, 0.0205), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("fermeture de bande\n(candidat fronce, C)", (0.1025, 0.013),
            xytext=(0.118, 0.0165), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("ligne morte (C)", (0.105, 0.004),
            xytext=(0.088, 0.0015), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("raccords K\n(flanc droit)", (0.122, 0.004),
            xytext=(0.135, 0.007), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("$\\theta$")
ax.set_ylabel("$s$")
ax.set_title("F-III.10 -- la region de reorganisation : calendrier n "
             "(couleur) et issue (bord)\nles nappes de $\\Sigma_P$ se "
             "raccordent-elles aux bords de cellules ?")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("output_theory/fiii10_raccord.png", dpi=150)
print("-> output_theory/fiii10_raccord.png")

with open("output_theory/fiii10_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii10_report.json")
