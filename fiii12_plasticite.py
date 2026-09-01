"""F-III.12 -- La plasticite de la loi d'action (QO-56) : detection et
indiscernabilite.

Extension plastique minimale (declaree). Le gain de la loi d'action devient
une variable gamma : boost = 1 + gamma * CS * pr, avec
  - fatigue evenementielle : gamma *= (1 - R_GA) a chaque Kairos charge ;
  - recuperation lente : dgamma/dt = EPS_GA * (1 - gamma), EPS_GA = 5e-4
    (tau = 2000 u : l'horloge la plus lente du systeme, sous s).
La classe d'axiomes canonique est retrouvee pour R_GA = EPS_GA = 0 (Z0 :
validation bit a bit contre la reference F-III.0).

Protocole (F-III.11, remarque 12, cinq etapes) :
  Z1  deux histoires d'usage inegal -- H1 lourde (3 amorces 41.9, oublis
      150) et H2 legere (1 amorce 41.9) -- puis oubli final : D2 = 160 fixe,
      D1 bisecte pour egaliser ||s||_regraine a 1e-6 ; p, chi, theta morts
      des deux cotes ; gamma_1 != gamma_2 (usages differents).
  Z2  detection : cartes de sondes des deux etats reels (memes Psi, meme z
      a 1e-6) -- different-elles ? Et la frontiere s*(gamma) par injection :
      Sigma^(1) vs Sigma^(2) a z egal.
  Z3  fermeture etendue : le resume (p,chi,theta,s) echoue ; le resume
      (p,chi,theta,s,gamma) reproduit cellule a cellule. P-III.6 gagne un
      etage (tau = 2000).
  Z4  indiscernabilite (Theoreme III-11) : la presentation 'loi plastique'
      et la presentation 'loi figee lisant une variable d'etat w' sont le
      meme systeme hybride -- executions identiques au bit pres.

(B, rho) declares : sondes (u1, G2, T2 = 40..140, regraine canonique,
relaxation 300), bassin binaire ; famille alignee.
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
R_GA, EPS_GA = 0.005, 5e-4
OUT = {}


def run(st, u, g, T, r_ga=R_GA, e_ga=EPS_GA, rec=None, dt=0.01):
    """Integrateur plastique ; r_ga = e_ga = 0 redonne le canonique
    (gamma reste a 1 et 1.0 * x = x exactement en IEEE)."""
    x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc, ga = st
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
        ga = ga + dt * e_ga * (1.0 - ga)
        if p > THK:
            d = s1 * c1 + s2 * c2
            boost = 1.0 + ga * CS * d / (nc + 1e-12)
            if boost < 0.0:
                boost = 0.0
            t1 += boost * KQ * c1
            t2 += boost * KQ * c2
            c1 *= (1.0 - RCHI)
            c2 *= (1.0 - RCHI)
            p *= RP
            if nc > NC_MIN:
                ga *= (1.0 - r_ga)
                if rec is not None:
                    rec.append(1)
    return (x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc, ga)


def fresh(s=0.0, ga=1.0):
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, s * U1[0], s * U1[1], 0.0, ga)


def reseed(st):
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + st[4:]


def carte(st, grid=range(40, 141, 10)):
    out = ""
    for T2 in grid:
        s = run(reseed(st), U1, G2, float(T2))
        s = run(s, U1, 0.0, 300.0)
        out += "A" if math.hypot(s[0], s[1]) > 1.0 else "."
    return out


# ------------------------------------------------------------------- Z0
print("Z0 -- validation : r_ga = e_ga = 0 redonne le canonique bit a bit")
pr = run(run(fresh(), U1, G2, 41.9, 0.0, 0.0), U1, 0.0, 200.0, 0.0, 0.0)
st = reseed(pr)
st = run(run(st, U1, G2, 100.0, 0.0, 0.0), U1, 0.0, 200.0, 0.0, 0.0)
pc = math.hypot(st[0], st[1])
assert abs(pc - 3.246573235196175) < 1e-12 and st[12] == 1.0
print(f"  ||x_PC|| = {pc:.12f}, gamma = {st[12]} : conforme", flush=True)

# ------------------------------------------------------------------- Z1
print("Z1 -- deux histoires d'usage inegal, egalisees sur la cible s = 0.030")
S_CIBLE = 0.030


def H_legere(D2, rec=None):
    st = run(fresh(), U1, G2, 41.9, rec=rec)
    return run(st, U1, 0.0, float(D2), rec=rec)


def H_lourde(D1, rec=None):
    st = fresh()
    for _ in range(3):
        st = run(st, U1, G2, 41.9, rec=rec)
        st = run(st, U1, 0.0, 150.0, rec=rec)
    return run(st, U1, 0.0, float(D1), rec=rec)


def bisect_D(H, lo, hi):
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        if math.hypot(H(mid)[9], H(mid)[10]) > S_CIBLE:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


D1 = bisect_D(H_lourde, 5.0, 600.0)
D2 = bisect_D(H_legere, 5.0, 600.0)
r1, r2 = [], []
st1, st2 = H_lourde(D1, r1), H_legere(D2, r2)
s1n, s2n = math.hypot(st1[9], st1[10]), math.hypot(st2[9], st2[10])
print(f"  H1 (lourde, D1={D1:.2f}) : ||s||={s1n:.6f} gamma={st1[12]:.4f} "
      f"Kairos charges = {len(r1)}")
print(f"  H2 (legere, D2={D2:.2f}) : ||s||={s2n:.6f} gamma={st2[12]:.4f} "
      f"Kairos charges = {len(r2)}")
print(f"  ecart a la cible : ({abs(s1n - S_CIBLE):.1e}, "
      f"{abs(s2n - S_CIBLE):.1e}) ; morts : "
      f"theta=({math.hypot(st1[4], st1[5]):.1e},{math.hypot(st2[4], st2[5]):.1e}) "
      f"p=({st1[8]:.1e},{st2[8]:.1e})", flush=True)
OUT["Z1"] = {"D1": round(D1, 2), "D2": round(D2, 2), "s_cible": S_CIBLE,
             "gamma_lourde": round(st1[12], 4),
             "gamma_legere": round(st2[12], 4),
             "kairos": [len(r1), len(r2)]}

# ------------------------------------------------------------------- Z2
print("Z2 -- detection : cartes a z egal, et la frontiere s*(gamma)")
c1, c2 = carte(st1), carte(st2)
print(f"  carte H1 (gamma={st1[12]:.3f}) : [{c1}]")
print(f"  carte H2 (gamma={st2[12]:.3f}) : [{c2}]")
print(f"  cartes differentes a z egal (1e-6) : {c1 != c2}", flush=True)


def s_star(ga, fige=False):
    lo, hi = 0.005, 0.06
    def om(s):
        args = (0.0, 0.0) if fige else (R_GA, EPS_GA)
        st = run(fresh(s, ga), U1, G2, 80.0, *args)
        st = run(st, U1, 0.0, 300.0, *args)
        return math.hypot(st[0], st[1]) > 1.0
    if om(lo) == om(hi):
        return None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if om(mid) == om(lo):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


courbe_p = {round(g, 2): (lambda v: round(v, 5) if v else None)(s_star(g))
            for g in (0.6, 0.7, 0.8, 0.9, 1.0)}
courbe_f = {round(g, 2): (lambda v: round(v, 5) if v else None)(s_star(g, True))
            for g in (0.6, 0.7, 0.8, 0.9, 1.0)}
print(f"  s*(gamma), sonde plastique : {courbe_p}")
print(f"  s*(gamma), sonde a gamma fige : {courbe_f}", flush=True)
OUT["Z2"] = {"carte_H1": c1, "carte_H2": c2, "detection": c1 != c2,
             "s_star_plastique": courbe_p, "s_star_fige": courbe_f}

# ------------------------------------------------------------------- Z3
print("Z3 -- fermeture : le resume a 4 horloges echoue, a 5 il ferme")
sy_5 = fresh(s1n, st1[12])          # (0,0,0,s, gamma_1)
sy_4 = fresh(s1n, 1.0)              # gamma omis (:= 1)
c5, c4 = carte(sy_5), carte(sy_4)
print(f"  reel H1        : [{c1}]")
print(f"  synth (z,gamma): [{c5}] accord : {c5 == c1}")
print(f"  synth (z seul) : [{c4}] accord : {c4 == c1}", flush=True)
OUT["Z3"] = {"accord_5": c5 == c1, "accord_4": c4 == c1,
             "carte_synth5": c5, "carte_synth4": c4}

# ------------------------------------------------------------------- Z4
print("Z4 -- indiscernabilite (Theoreme III-11) : les deux presentations")


def run_B(st, u, g, T, dt=0.01):
    """Presentation B : loi FIGEE G(w, pr) = 1 + w*CS*pr lisant la
    coordonnee d'etat w (= 13e composante) a dynamique propre. Meme
    arithmetique que run : le rebapteme est la preuve."""
    return run(st, u, g, T, R_GA, EPS_GA, None, dt)


stA = run(run(fresh(0.006), U1, G2, 41.9), U1, 0.0, 150.0)
stA = run(run(reseed(stA), U1, G2, 80.0), U1, 0.0, 300.0)
stB = run_B(run_B(fresh(0.006), U1, G2, 41.9), U1, 0.0, 150.0)
stB = run_B(run_B(reseed(stB), U1, G2, 80.0), U1, 0.0, 300.0)
identiques = all(a == b for a, b in zip(stA, stB))
print(f"  executions identiques au bit pres : {identiques}", flush=True)
OUT["Z4"] = {"identiques": identiques}

with open("output_theory/fiii12_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii12_report.json")
