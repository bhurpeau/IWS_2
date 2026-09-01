"""F-III.11 -- Le test frontal de la dualite protocole/etat (QO-81).

C-III.10 (F-III.10) : les discontinuites de la geometrie de seuils le long
des directions de protocole sont K ; le long des directions d'etat, la
frontiere evolue par plis C. Anticipation enregistree (F-III.10.1-R10) :
affaiblissement en propension attendu. Deux attaques :

  Y1  chercher K le long d'un axe d'ETAT : l'axe p aux grandes valeurs
      (candidat designe : le premier evenement de la sonde peut apparaitre
      ou disparaitre). Colonnes p in [0, 1.3] a s fixe pres de la frontiere,
      lecture (issue, n, dates), classification de chaque transition
      d'issue au critere de F-III.4 (compte ou saut de date > 5 u).
  Y2  chercher C le long d'un axe de PROTOCOLE : l'axe T2 (duree de sonde)
      a etat fixe. n(T2) est en escalier croissant ; une transition d'issue
      tombant strictement a l'interieur d'une marche (n constant de part et
      d'autre, dates communes continues) est un pli C protocolaire.
  Y3  bilan des propensions par axe : T_am (F-III.5), (theta, s)
      (F-III.10), p (Y1), T2 (Y2) -- et le verdict sur C-III.10.

Protocole declare : sonde fil-rouge 80 (Y1) et sonde de duree variable
(u1, G2, T2, regraine canonique, relaxation 300) (Y2) ; rho = bassin
binaire ; calendrier = Kairos charges (||chi|| > 0.01) de la phase active ;
etat lent aligne, p = chi = 0 sauf axe explore ; classification K :
compte different OU saut > 5 u sur les dates communes.
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
T_JUMP = 5.0
OUT = {}


def run(st, u, g, T, rec=None, dt=0.01):
    x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc = st
    u1, u2 = u
    gw1, gw2 = g * W * u1, g * W * u2
    for k in range(int(T / dt)):
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
                rec.append(round(k * dt, 2))
            t1 += boost * KQ * c1
            t2 += boost * KQ * c2
            c1 *= (1.0 - RCHI)
            c2 *= (1.0 - RCHI)
            p *= RP
    return (x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc)


def lecture(p0, s, T2=80.0):
    st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          p0, s * U1[0], s * U1[1], 0.0)
    rec = []
    st = run(st, U1, G2, T2, rec)
    st = run(st, U1, 0.0, 300.0)
    return (math.hypot(st[0], st[1]) > 1.0), len(rec), tuple(rec)


def type_trans(la, lb):
    """K si compte different ou saut > T_JUMP sur les dates communes."""
    if la[1] != lb[1]:
        return "K"
    if any(abs(x - y) > T_JUMP for x, y in zip(la[2], lb[2])):
        return "K"
    return "C"


# ------------------------------------------------------------------- Y0
print("Y0 -- validation integrateur")
fr = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0)
pr = run(run(fr, U1, G2, 41.9), U1, 0.0, 200.0)
st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + pr[4:]
st = run(run(st, U1, G2, 100.0), U1, 0.0, 200.0)
assert abs(math.hypot(st[0], st[1]) - 3.246573235196175) < 1e-9
print("  conforme", flush=True)

# ------------------------------------------------------------------- Y1
print("Y1 -- axe d'ETAT p in [0, 1.3] (chercher K)")
P_GRID = [round(0.05 * i, 2) for i in range(27)]
res_p = {"trans": []}
for s in (0.0150, 0.0155, 0.0160):
    col = [lecture(p0, s) for p0 in P_GRID]
    motif = "".join("A" if c[0] else "." for c in col)
    ns = sorted({c[1] for c in col})
    print(f"  s={s} issue=[{motif}] n in {ns}", flush=True)
    for i in range(len(P_GRID) - 1):
        if col[i][0] != col[i + 1][0]:
            t = type_trans(col[i], col[i + 1])
            saut = max((abs(x - y) for x, y in
                        zip(col[i][2], col[i + 1][2])), default=0.0)
            res_p["trans"].append(
                {"s": s, "p": [P_GRID[i], P_GRID[i + 1]],
                 "n": [col[i][1], col[i + 1][1]], "saut": round(saut, 2),
                 "type": t})
            print(f"    transition p={P_GRID[i]}->{P_GRID[i+1]} "
                  f"n={col[i][1]}->{col[i+1][1]} saut={saut:.2f} : {t}",
                  flush=True)
K_p = sum(1 for t in res_p["trans"] if t["type"] == "K")
print(f"  bilan axe p : {len(res_p['trans'])} transitions, {K_p} K",
      flush=True)
OUT["Y1"] = {"transitions": res_p["trans"], "K": K_p}

# ------------------------------------------------------------------- Y2
print("Y2 -- axe de PROTOCOLE T2 in [78, 92] (chercher C)")
T2_GRID = [round(78.0 + 0.25 * i, 2) for i in range(57)]
res_t = {"trans": []}
for s in (0.0157, 0.0159, 0.0161):
    col = [lecture(0.0, s, T2) for T2 in T2_GRID]
    motif = "".join("A" if c[0] else "." for c in col)
    print(f"  s={s} issue=[{motif}]", flush=True)
    for i in range(len(T2_GRID) - 1):
        if col[i][0] != col[i + 1][0]:
            t = type_trans(col[i], col[i + 1])
            res_t["trans"].append(
                {"s": s, "T2": [T2_GRID[i], T2_GRID[i + 1]],
                 "n": [col[i][1], col[i + 1][1]], "type": t})
            print(f"    transition T2={T2_GRID[i]}->{T2_GRID[i+1]} "
                  f"n={col[i][1]}->{col[i+1][1]} : {t}", flush=True)
C_t = sum(1 for t in res_t["trans"] if t["type"] == "C")
print(f"  bilan axe T2 : {len(res_t['trans'])} transitions, {C_t} C",
      flush=True)
OUT["Y2"] = {"transitions": res_t["trans"], "C": C_t}

# ------------------------------------------------------------------- Y3
print("Y3 -- propensions par axe et verdict sur C-III.10")
nK_p, nT_p = OUT["Y1"]["K"], len(OUT["Y1"]["transitions"])
nC_t, nT_t = OUT["Y2"]["C"], len(OUT["Y2"]["transitions"])
tableau = [
    ("T_am (protocole, F-III.5)", "2/2 K (100 %)"),
    ("(theta, s) (etat, F-III.10)", "14/54 K (26 %)"),
    (f"p (etat, Y1)", f"{nK_p}/{nT_p} K"),
    (f"T2 (protocole, Y2)", f"{nT_t - nC_t}/{nT_t} K"),
]
for axe, bilan in tableau:
    print(f"  {axe:32s} : {bilan}", flush=True)
dicho_cassee_etat = nK_p > 0
dicho_cassee_proto = nC_t > 0
print(f"  K trouve sur un axe d'etat : {dicho_cassee_etat} ; "
      f"C trouve sur un axe de protocole : {dicho_cassee_proto}", flush=True)
OUT["Y3"] = {"tableau": tableau, "K_sur_etat": dicho_cassee_etat,
             "C_sur_protocole": dicho_cassee_proto}

with open("output_theory/fiii11_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii11_report.json")
