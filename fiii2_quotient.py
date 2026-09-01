"""F-III.2 -- Le quotient de Nerode d'un IWS.

Objets implementes :
  ~ (Nerode)   X ~ Y  ssi  R_X = R_Y sur la classe d'histoires declaree B.
  l(X,Y)       separabilite : plus petit nombre d'episodes d'une histoire
               de B qui separe X et Y (inf = pas de separation dans B).
  d = e^{-l}   candidat pseudo-ultrametrique (Prop. III-3 : inegalite forte
               d(X,Z) <= max(d(X,Y), d(Y,Z)) pour TOUTE mesure de complexite).

Experiences :
  X0  validation de l'integrateur scalaire contre la reference F-III.0
      (P4 : T_C=100, ||x_PC||=3.246573, ||x_CP||~0).
  X1  cartes de sondes (longueur 1) pour S(s), s in {0.006, 0.008, 0.012} :
      la paire E1 (0.006, 0.008) a des cartes identiques ; 0.012 s'en separe
      -> l = 1 pour les paires eloignees.
  X2  separabilite de la paire E1 : recherche de la plus courte histoire
      separatrice (amorce + oubli + episode) -> l = 2.
  X3  verification numerique de l'inegalite ultrametrique sur le triplet
      (configuration isocele attendue : deux grands cotes egaux).
  X4  separabilite de la paire E2 (G_lin vs G_sat, meme (Psi, s=0)) :
      niveau minimal d'histoire cumulative qui les separe.
  X5  resume suffisant de la classe conditionnelle : la signature de l'etat
      post-histoire PC est-elle reproduite par l'injection (A, s_vec) ?
      Ablations : s_vec seul (bassin omis), A seul (contenu omis).

Protocole declare : sondes = episodes (u_p, G2, T2) a graine appariee,
relaxation 300 ; avant toute sonde de X5, chi et p sont ramenes a 0
(variables rapides et pression au repos -- meme hypothese que III-1).
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
S0 = 0.05  # echelle de saturation de G_sat (F-III.1, pente identique en 0)

# Lois d'action G : boost(d, nc) avec d = s.chi, nc = ||chi||  (pr = d/(nc+eps))
B_LIN = lambda d, nc: 1.0 + CS * d / (nc + 1e-12)
B_SAT = lambda d, nc: 1.0 + CS * S0 * math.tanh((d / (nc + 1e-12)) / S0)

OUT = {}


def run(st, u, g, T, gb=B_LIN, dt=0.01):
    """Integrateur scalaire, arithmetique identique a fiii0_fin.run."""
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
    """Repos, graine 1e-3 le long de u1, contenu mnesique s.U1."""
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, s * U1[0], s * U1[1], 0.0)


def reseed(st, u_p):
    """Graine appariee : x = 1e-3 u_p, v = 0, variables lentes conservees."""
    return (1e-3 * u_p[0], 1e-3 * u_p[1], 0.0, 0.0) + st[4:]


def sonde(st, u_p, T2, gb=B_LIN):
    """Issue ('A' / '.') de l'episode-test (u_p, G2, T2) + relaxation 300."""
    s = reseed(st, u_p)
    s = run(s, u_p, G2, float(T2), gb)
    s = run(s, u_p, 0.0, 300.0, gb)
    return math.hypot(s[0], s[1]) > 1.0


MU1 = (-U1[0], -U1[1])


def carte(st, grid, gb=B_LIN, dirs=(U1, MU1)):
    return "".join("A" if sonde(st, u, T, gb) else "."
                   for u in dirs for T in grid)


# ------------------------------------------------------------------- X0
print("X0 -- validation de l'integrateur scalaire (reference F-III.0 P4)")
prime = run(run(fresh(), U1, G2, 41.9), U1, 0.0, 200.0)
stPC = run(run(reseed(prime, U1), U1, G2, 100.0), U1, 0.0, 200.0)
stC = run(run(fresh(), U1, G2, 100.0), U1, 0.0, 200.0)
stCP = run(run(stC, U1, G2, 41.9), U1, 0.0, 200.0)
pc, cp = math.hypot(stPC[0], stPC[1]), math.hypot(stCP[0], stCP[1])
REF_PC = 3.246573235196175
print(f"  ||x_PC|| = {pc:.12f} (ref {REF_PC:.12f}, ecart {abs(pc-REF_PC):.2e})")
print(f"  ||x_CP|| = {cp:.3e} (ref ~0)")
assert abs(pc - REF_PC) < 1e-6 and cp < 1e-6, "integrateur non conforme"
OUT["X0"] = {"PC": pc, "CP": cp, "ecart_ref": abs(pc - REF_PC)}

# ------------------------------------------------------------------- X1
print("X1 -- cartes de longueur 1 (classe B1 : u in {+u1,-u1}, T2=40..140)")
GRID = list(range(40, 141, 10))
SYS = {"a": 0.006, "b": 0.008, "c": 0.012}
cards = {k: carte(fresh(s), GRID) for k, s in SYS.items()}
for k in SYS:
    print(f"  S({SYS[k]}) : [{cards[k]}]")
OUT["X1"] = {"grid": "T2=40..140 pas 10, u in {+u1,-u1}",
             "cartes": {k: cards[k] for k in SYS}}


def sep1(k1, k2):
    return cards[k1] != cards[k2]


# ------------------------------------------------------------------- X2
print("X2 -- separabilite de la paire E1 (a,b) : histoires de longueur 2")
# prefixes : amorce (u1, G2, T_am) + oubli 150 ; sondes B1 ensuite
PREF_T = (10.0, 20.0, 30.0, 40.0)
pref = {k: {} for k in SYS}
for k, s in SYS.items():
    for T_am in PREF_T:
        pref[k][T_am] = run(run(fresh(s), U1, G2, T_am), U1, 0.0, 150.0)
sep2_ab = []
for T_am in PREF_T:
    for u in (U1, MU1):
        for T2 in GRID:
            oa = sonde(pref["a"][T_am], u, T2)
            ob = sonde(pref["b"][T_am], u, T2)
            if oa != ob:
                sep2_ab.append((T_am, "+u1" if u == U1 else "-u1", T2,
                                "A" if oa else ".", "A" if ob else "."))
first = sep2_ab[0] if sep2_ab else None
print(f"  separations longueur 2 (a,b) : {len(sep2_ab)} ; premiere : {first}")
L = {("a", "b"): (1 if sep1("a", "b") else (2 if sep2_ab else None)),
     ("a", "c"): 1 if sep1("a", "c") else None,
     ("b", "c"): 1 if sep1("b", "c") else None}
print(f"  l(a,b)={L[('a','b')]}  l(a,c)={L[('a','c')]}  l(b,c)={L[('b','c')]}")
OUT["X2"] = {"separations_ab": sep2_ab[:6], "n_sep_ab": len(sep2_ab),
             "l": {f"{p[0]}{p[1]}": L[p] for p in L}}

# ------------------------------------------------------------------- X3
print("X3 -- inegalite ultrametrique d(X,Z) <= max(d(X,Y),d(Y,Z)), d=e^{-l}")
d = {p: math.exp(-L[p]) for p in L if L[p] is not None}
dd = {("a", "b"): d[("a", "b")], ("b", "a"): d[("a", "b")],
      ("a", "c"): d[("a", "c")], ("c", "a"): d[("a", "c")],
      ("b", "c"): d[("b", "c")], ("c", "b"): d[("b", "c")]}
ok_all = True
for X, Y, Z in (("a", "b", "c"), ("a", "c", "b"), ("b", "a", "c")):
    lhs, rhs = dd[(X, Z)], max(dd[(X, Y)], dd[(Y, Z)])
    ok = lhs <= rhs + 1e-15
    ok_all &= ok
    print(f"  d({X},{Z})={lhs:.4f} <= max(d({X},{Y}),d({Y},{Z}))={rhs:.4f} : "
          f"{'OK' if ok else 'VIOLATION'}")
iso = (abs(dd[("a", "c")] - dd[("b", "c")]) < 1e-15
       and dd[("a", "b")] < dd[("a", "c")])
print(f"  configuration isocele (deux grands cotes egaux) : {iso}")
OUT["X3"] = {"d": {f"{p[0]}{p[1]}": v for p, v in d.items()},
             "ultrametrique_verifiee": ok_all, "isocele": iso}

# ------------------------------------------------------------------- X4
print("X4 -- separabilite de la paire E2 (G_lin vs G_sat, meme (Psi, s=0))")
GRID_E2 = list(range(40, 131, 10))  # classe bornee declaree (F-III.1)
c_lin = carte(fresh(), GRID_E2, B_LIN)
c_sat = carte(fresh(), GRID_E2, B_SAT)
print(f"  longueur 1 : G_lin=[{c_lin}] G_sat=[{c_sat}] "
      f"identiques : {c_lin == c_sat}")
lvl_found, wit = None, None
if c_lin != c_sat:
    lvl_found = 1
else:
    st_lin, st_sat = fresh(), fresh()
    for n_am in (1, 2, 3, 4):
        st_lin = run(run(st_lin, U1, G2, 41.9, B_LIN), U1, 0.0, 150.0, B_LIN)
        st_sat = run(run(st_sat, U1, G2, 41.9, B_SAT), U1, 0.0, 150.0, B_SAT)
        sl = math.hypot(st_lin[9], st_lin[10])
        ss = math.hypot(st_sat[9], st_sat[10])
        for T2 in GRID_E2:
            ol = sonde(st_lin, U1, T2, B_LIN)
            os_ = sonde(st_sat, U1, T2, B_SAT)
            if ol != os_:
                lvl_found, wit = n_am + 1, (n_am, T2, "A" if ol else ".",
                                            "A" if os_ else ".")
                break
        print(f"  niveau {n_am + 1} ({n_am} amorces 41.9u + oubli 150 + "
              f"episode) : ||s||_lin={sl:.4f} ||s||_sat={ss:.4f} "
              f"-> {'SEPARATION ' + str(wit) if wit else 'rien'}")
        if lvl_found:
            break
print(f"  l(G_lin, G_sat) = {lvl_found}")
OUT["X4"] = {"cartes_l1_identiques": c_lin == c_sat,
             "l_E2": lvl_found, "temoin": wit}

# ------------------------------------------------------------------- X5
print("X5 -- resume suffisant de la classe conditionnelle : (A, s_vec)")
GRID5 = list(range(40, 150, 20))  # grille de signature F-III.0


def zero_fast(st):
    """chi := 0, p := 0 (protocole declare avant sonde)."""
    return st[:6] + (0.0, 0.0, 0.0) + st[9:]


st_ref = zero_fast(stPC)                      # etat vrai post-histoire PC
s_ref = (st_ref[9], st_ref[10])
# etat canonique du bassin approprie : graine forte + relaxation autonome
stA = (3.0 * U1[0], 3.0 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0)
stA = zero_fast(run(stA, U1, 0.0, 400.0))
print(f"  bassin : ||x_ref||={math.hypot(st_ref[0], st_ref[1]):.4f} "
      f"||x_A||={math.hypot(stA[0], stA[1]):.4f} ; "
      f"s_ref=({s_ref[0]:.5f},{s_ref[1]:.5f})")
st_sum = stA[:9] + (s_ref[0], s_ref[1], stA[11])       # (A, s_vec)
st_s_only = zero_fast(fresh())[:9] + (s_ref[0], s_ref[1], 0.0)  # s sans A
st_A_only = stA                                         # A sans s
sig = {name: carte(st, GRID5)
       for name, st in (("ref_PC", st_ref), ("A+s", st_sum),
                        ("s_seul", st_s_only), ("A_seul", st_A_only))}
for name, patt in sig.items():
    print(f"  R({name:7s}) = [{patt}]")
print(f"  suffisance (A,s) : {sig['ref_PC'] == sig['A+s']} ; "
      f"bassin necessaire : {sig['ref_PC'] != sig['s_seul']} ; "
      f"contenu necessaire : {sig['ref_PC'] != sig['A_seul']}")
OUT["X5"] = {"signatures": sig, "s_ref": list(s_ref),
             "suffisance_A_s": sig["ref_PC"] == sig["A+s"],
             "bassin_necessaire": sig["ref_PC"] != sig["s_seul"],
             "contenu_necessaire": sig["ref_PC"] != sig["A_seul"]}

with open("output_theory/fiii2_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii2_report.json")
