"""F-III.7 -- Axiomatique des histoires admissibles : instance numerique de
l'axe rho (la double filtration en acte).

Question. La paire E1 (s = 0.006 vs 0.008) est inseparable au budget B1
(sondes simples) en resolution binaire rho_bin (bassin), et separable au
niveau 2 seulement : l_{B2, rho_bin} = 2 (F-III.2/F-III.3). L'axe rho de la
double filtration predit qu'un raffinement de LECTURE peut se substituer a
un enrichissement d'ACTION. Test : a budget B1 fixe, la resolution
rho_cal = (bassin, nombre de Kairos charges pendant la sonde) separe-t-elle
la paire au niveau 1 ?

Protocole declare. B1 = sondes simples (u in {+u1, -u1}, T2 = 40..140 pas
10), graine appariee, relaxation 300. Lecture rho_cal : issue binaire
(||x|| > 1 apres relaxation) + compte des Kairos charges (||chi|| > 0.01)
pendant la phase active de la sonde (relaxation exclue). rho_cal raffine
rho_bin par construction (projection = oubli du compte).

  U0  validation de l'integrateur (reference F-III.0 P4).
  U1  cartes de la paire E1 aux deux resolutions, budget B1.
  U2  consequence sur l : l_{B, rho_cal} vs l_{B, rho_bin} ; symetrie des
      deux axes (budget vs resolution) rapportee.
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
                rec.append(1)
            t1 += boost * KQ * c1
            t2 += boost * KQ * c2
            c1 *= (1.0 - RCHI)
            c2 *= (1.0 - RCHI)
            p *= RP
    return (x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc)


def fresh(s=0.0):
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, s * U1[0], s * U1[1], 0.0)


def sonde_rho(st0, u_p, T2):
    """Lecture rho_cal : (issue binaire, nombre de Kairos charges actifs)."""
    st = (1e-3 * u_p[0], 1e-3 * u_p[1], 0.0, 0.0) + st0[4:]
    rec = []
    st = run(st, u_p, G2, float(T2), rec)
    st = run(st, u_p, 0.0, 300.0, None)
    return ("A" if math.hypot(st[0], st[1]) > 1.0 else "."), len(rec)


# ------------------------------------------------------------------- U0
print("U0 -- validation integrateur")
pr = run(run(fresh(), U1, G2, 41.9), U1, 0.0, 200.0)
st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + pr[4:]
st = run(run(st, U1, G2, 100.0), U1, 0.0, 200.0)
assert abs(math.hypot(st[0], st[1]) - 3.246573235196175) < 1e-9
print("  conforme", flush=True)

# ------------------------------------------------------------------- U1
print("U1 -- la paire E1 au budget B1, resolutions rho_bin et rho_cal")
GRID = list(range(40, 141, 10))
cartes = {}
for s in (0.006, 0.008):
    binp, calp = [], []
    for u in (U1, MU1):
        for T2 in GRID:
            o, n = sonde_rho(fresh(s), u, T2)
            binp.append(o)
            calp.append(f"{o}{n}")
    cartes[s] = {"bin": "".join(binp), "cal": " ".join(calp)}
    print(f"  S({s}) rho_bin : [{cartes[s]['bin']}]", flush=True)
    print(f"  S({s}) rho_cal : [{cartes[s]['cal']}]", flush=True)
sep_bin = cartes[0.006]["bin"] != cartes[0.008]["bin"]
sep_cal = cartes[0.006]["cal"] != cartes[0.008]["cal"]
temoins = []
c6, c8 = cartes[0.006]["cal"].split(), cartes[0.008]["cal"].split()
for i, (a, b) in enumerate(zip(c6, c8)):
    if a != b:
        u = "+u1" if i < len(GRID) else "-u1"
        temoins.append({"u": u, "T2": GRID[i % len(GRID)],
                        "S(0.006)": a, "S(0.008)": b})
print(f"  separation rho_bin : {sep_bin} ; separation rho_cal : {sep_cal} ; "
      f"temoins : {len(temoins)}", flush=True)
for t in temoins[:4]:
    print(f"    {t}", flush=True)
OUT["U1"] = {"cartes": {str(k): v for k, v in cartes.items()},
             "sep_bin": sep_bin, "sep_cal": sep_cal, "temoins": temoins}

# ------------------------------------------------------------------- U2
print("U2 -- consequence sur l : les deux axes de la double filtration")
l_bin = 2  # etabli en F-III.2 (X2) : temoin unique (amorce 20, oubli 150, sonde 80)
l_cal = 1 if sep_cal else None
print(f"  l(B, rho_bin) = {l_bin} (F-III.2) ; l(B, rho_cal) = {l_cal}")
print(f"  axe budget (F-III.3-Y4) : enrichir les sondes (T<=130 -> T<=140) "
      f"fait chuter l ; axe resolution (ici) : enrichir la lecture "
      f"(bassin -> bassin+compte) {'fait de meme' if l_cal == 1 else 'ne suffit pas'}",
      flush=True)
OUT["U2"] = {"l_bin": l_bin, "l_cal": l_cal}

with open("output_theory/fiii7_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii7_report.json")
