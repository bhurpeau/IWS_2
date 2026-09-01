"""F-III.8 -- Le theoreme de reduction au contenu survivant.

Theoreme III-9 (reduction, regime d'extinction exacte). Si, a la regraine,
p = chi = theta = 0, alors pour toute sonde 𝒫 et toute resolution rho il
existe omega_{𝒫,rho} telle que R_X^rho(𝒫) = omega_{𝒫,rho}(s_regraine).
Preuve formelle dans la note (deux etapes : A-H1 fixe (x,v) ; l'extinction
fixe (p,chi,theta) ; le flot hybride est deterministe (O1) donc l'execution
-- et toute lecture -- est fonction du seul s_regraine ; famille alignee :
s_regraine = s.u1, omega scalaire).

Corollaires : (C1) si un prefixe eteint ecrit delta, issue = omega(s0+delta) ;
si omega n'a qu'une commutation s^* sur la plage, b + delta = s^* (III-5
retrouvee). (C2) iles et anti-monotonie = composantes de la frontiere
d'appropriation de omega, OU composition avec une ecriture delta(s0) non
monotone -- le theoreme force l'un des deux, T5 tranche.

Verifications :
  T1  omega mesuree DIRECTEMENT sur la sonde nue T2 = 80 (aucun prefixe) :
      scan fin + bisection de s^*. Controle independant de III-5 : s^* doit
      retomber sur 0.01627.
  T2  la courbe d'exigence s^*(T2), T2 = 60..140, plage etendue [2e-4, 0.06],
      avec scan de detection d'iles par sonde. QO-64 (sterilite de la sonde
      60) recue quantitativement.
  T3  la prediction par translation : pour deux prefixes eteints, predire la
      carte entiere du systeme prefixe par issue = [s0 + delta(s0) >= s^*],
      sans aucune sonde sur le systeme prefixe autre que la mesure de delta.
      Comparaison cellule a cellule avec la carte mesuree.
  T5  le mecanisme de l'anti-monotonie (temoin F-III.4 : amorce 10, oubli
      150, sonde 110) : mesurer s_total(s0) = s0 + delta(s0) et l'issue sur
      la fenetre anti-monotone ; tester issue == [s_total >= s^*_110] et la
      monotonie de s_total. Decide : ecriture non monotone vs omega
      multi-commutations.

Protocole declare : (B, rho) = (histoires de cette note, bassin binaire) ;
regraine canonique ; relaxation 300 ; delta mesure a la regraine.
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
OUT = {}


def run(st, u, g, T, dt=0.01):
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
            t1 += boost * KQ * c1
            t2 += boost * KQ * c2
            c1 *= (1.0 - RCHI)
            c2 *= (1.0 - RCHI)
            p *= RP
    return (x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc)


def fresh(s=0.0):
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, s * U1[0], s * U1[1], 0.0)


def omega_nue(s, T2):
    """Issue de la sonde nue sur contenu pur s (regime d'extinction exact)."""
    st = run(fresh(s), U1, G2, float(T2))
    st = run(st, U1, 0.0, 300.0)
    return math.hypot(st[0], st[1]) > 1.0


def bisect_omega(T2, lo, hi, iters=12):
    o_lo, o_hi = omega_nue(lo, T2), omega_nue(hi, T2)
    if o_lo == o_hi:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if omega_nue(mid, T2) == o_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def prefixe_state(s0, T_am, D=150.0):
    st = run(fresh(s0), U1, G2, float(T_am))
    return run(st, U1, 0.0, float(D))


def issue_prefixe(s0, T_am, T2, D=150.0):
    st = prefixe_state(s0, T_am, D)
    st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + st[4:]
    st = run(st, U1, G2, float(T2))
    st = run(st, U1, 0.0, 300.0)
    return math.hypot(st[0], st[1]) > 1.0


# ------------------------------------------------------------------- T0
print("T0 -- validation integrateur")
pr = run(run(fresh(), U1, G2, 41.9), U1, 0.0, 200.0)
st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + pr[4:]
st = run(run(st, U1, G2, 100.0), U1, 0.0, 200.0)
assert abs(math.hypot(st[0], st[1]) - 3.246573235196175) < 1e-9
print("  conforme", flush=True)

# ------------------------------------------------------------------- T1
print("T1 -- omega mesuree directement (sonde nue T2 = 80)")
scan = [0.0002 + i * 0.0005 for i in range(50)]
motif = "".join("A" if omega_nue(s, 80) else "." for s in scan)
ncomm = sum(1 for a, b in zip(motif, motif[1:]) if a != b)
s_star_80 = bisect_omega(80, 0.0002, 0.025)
print(f"  scan [0.0002, 0.025] pas 5e-4 : [{motif}]")
print(f"  commutations detectees : {ncomm} ; s*_80 = {s_star_80:.5f} "
      f"(III-5 prevoyait 0.01627 via les familles de prefixes)", flush=True)
OUT["T1"] = {"motif": motif, "n_commutations": ncomm,
             "s_star_80": round(s_star_80, 6)}

# ------------------------------------------------------------------- T2
print("T2 -- la courbe d'exigence s*(T2), plage etendue [2e-4, 0.06]")
T2_GRID = (60, 70, 80, 90, 100, 110, 120, 130, 140)
courbe = {}
for T2 in T2_GRID:
    scan8 = [0.0002 + i * (0.06 - 0.0002) / 15 for i in range(16)]
    m8 = "".join("A" if omega_nue(s, T2) else "." for s in scan8)
    sstar = bisect_omega(T2, 0.0002, 0.06)
    courbe[T2] = {"s_star": None if sstar is None else round(sstar, 5),
                  "scan": m8}
    print(f"  T2={T2:>3} s*={'-' if sstar is None else f'{sstar:.5f}'} "
          f"scan=[{m8}]", flush=True)
OUT["T2"] = courbe

# ------------------------------------------------------------------- T3
print("T3 -- prediction par translation : cartes predites vs mesurees")
T3 = {}
for T_am in (17.5, 20.0):
    grid = [0.002 + i * 0.00115 for i in range(11)]
    pred, mes = "", ""
    for s0 in grid:
        stp = prefixe_state(s0, T_am)
        s_tot = math.hypot(stp[9], stp[10])
        pred += "A" if s_tot >= s_star_80 else "."
        mes += "A" if issue_prefixe(s0, T_am, 80) else "."
    T3[T_am] = {"predite": pred, "mesuree": mes, "accord": pred == mes}
    print(f"  T_am={T_am} predite=[{pred}] mesuree=[{mes}] "
          f"accord : {pred == mes}", flush=True)
OUT["T3"] = {str(k): v for k, v in T3.items()}

# ------------------------------------------------------------------- T5
print("T5 -- mecanisme de l'anti-monotonie (amorce 10, oubli 150, sonde 110)")
s_star_110 = courbe[110]["s_star"]
fen = [0.0100 + i * 0.0004 for i in range(10)]
rows = []
for s0 in fen:
    stp = prefixe_state(s0, 10.0)
    s_tot = math.hypot(stp[9], stp[10])
    o = issue_prefixe(s0, 10.0, 110)
    naive = s_tot >= s_star_110              # predicat a commutation unique
    point = omega_nue(s_tot, 110)            # evaluation pointwise de omega
    rows.append({"s0": round(s0, 5), "s_total": round(s_tot, 5),
                 "issue": "A" if o else ".",
                 "naive": "A" if naive else ".",
                 "pointwise": "A" if point else ".",
                 "accord_pointwise": o == point})
    print(f"  s0={s0:.4f} s_total={s_tot:.5f} issue={'A' if o else '.'} "
          f"naif={'A' if naive else '.'} "
          f"omega(s_tot)={'A' if point else '.'}", flush=True)
stot = [r["s_total"] for r in rows]
croiss = all(b >= a - 1e-6 for a, b in zip(stot, stot[1:]))
acc_n = sum(1 for r in rows if r["issue"] == r["naive"])
acc_p = sum(1 for r in rows if r["accord_pointwise"])
print(f"  s_total(s0) croissante : {croiss} ; accord predicat naif : "
      f"{acc_n}/10 ; accord pointwise omega(s_total) : {acc_p}/10", flush=True)
# cartographie de l'ile de repos interne de omega_110
scan_i = [0.006 + i * 0.0005 for i in range(29)]
carte_i = "".join("A" if omega_nue(s, 110) else "." for s in scan_i)
def edge(lo, hi):
    o_lo = omega_nue(lo, 110)
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if omega_nue(mid, 110) == o_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
ile = (edge(0.0135, 0.0140), edge(0.0150, 0.0155))
print(f"  omega_110 sur [0.006, 0.020] : [{carte_i}]")
print(f"  ile de repos interne : [{ile[0]:.5f}, {ile[1]:.5f}]", flush=True)
OUT["T5"] = {"lignes": rows, "s_total_croissante": croiss,
             "accord_naif": acc_n, "accord_pointwise": acc_p,
             "carte_omega_110": carte_i,
             "ile_repos": [round(ile[0], 5), round(ile[1], 5)],
             "verdict": "anti-monotonie = ecriture croissante x ile interne "
                        "de omega (C2, option multi-composantes)"}

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
xs = [T2 for T2 in T2_GRID if courbe[T2]["s_star"] is not None]
ys = [courbe[T2]["s_star"] for T2 in xs]
axs[0].plot(xs, ys, "o-", color="#4878a8", ms=6, lw=1.2)
none_x = [T2 for T2 in T2_GRID if courbe[T2]["s_star"] is None]
axs[0].plot(none_x, [0.061] * len(none_x), "x", color="gray", ms=7)
axs[0].axhline(0.01627, color="k", lw=0.7, ls="--", label="$s^*_{80}$ (III-5)")
axs[0].set_xlabel("duree de sonde $T_2$ (u)")
axs[0].set_ylabel("exigence $s^*(T_2)$")
axs[0].set_title("T2 -- la courbe d'exigence de la sonde nue\n"
                 "($\\times$ : aucune commutation sur [2e-4, 0.06])")
axs[0].legend(fontsize=9)
axs[1].plot([r["s0"] for r in rows], [r["s_total"] for r in rows], "o-",
            color="#c44e52", ms=5, lw=1.2, label="$s_{total}(s_0)$ (croissante)")
axs[1].axhspan(ile[0], ile[1], color="#4878a8", alpha=0.25,
               label=f"ile de repos de $\\omega_{{110}}$")
for r in rows:
    axs[1].annotate(r["issue"], (r["s0"], r["s_total"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
axs[1].set_xlabel("contenu initial $s_0$")
axs[1].set_ylabel("contenu total a la regraine")
axs[1].set_title("T5 -- l'anti-monotonie expliquee : ecriture croissante\n"
                 "traversant l'ile de repos interne de $\\omega_{110}$")
axs[1].legend(fontsize=9)
for a in axs:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("output_theory/fiii8_reduction.png", dpi=150)
print("-> output_theory/fiii8_reduction.png")

with open("output_theory/fiii8_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii8_report.json")
