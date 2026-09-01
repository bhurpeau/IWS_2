"""F-III.6 -- Sortie du regime de separation : ou vit le regime theta-vivant ?
(QO-66)

La prediction naive de F-III.5 (l'equation scalaire "doit tomber" a oubli
court) est corrigee par l'experience en deux temps :

  1. Pour les prefixes FECONDS (T_am dans la fenetre de F-III.5), le regime
     theta-vivant est INACCESSIBLE : la separation des echelles (tau_theta =
     8.3 u contre tau_s = 200 u) fait que, lorsque le contenu total est
     redescendu dans la plage des seuils, theta est mort depuis longtemps.
     A D court, la saturation observee n'est pas un nouveau regime : c'est
     III-5 qui la PREDIT (delta seul depasse s*). Test V2 : la bascule sur
     l'axe D doit se produire exactement quand ||s||_regraine croise s* --
     une prediction inter-axes de l'invariant.

  2. Le regime theta-vivant existe, mais pour les prefixes FAIBLES
     (T_am ~ 6 u, ecriture insuffisante pour saturer) a oubli court
     (D ~ 12-25 u). Test V3 : la deviation s* - (b + delta) y est-elle une
     fonction croissante de theta_regraine, commune a toutes les
     combinaisons (courbe de facilitation) ? Test V4 : ablations (complet /
     (theta, s) / s seul) au voisinage du seuil.

Protocole declare. Sonde fixe T2 = 80, relaxation 300 ; plage de contenus
s in [0.0002, 0.0135] ; regraine canonique (x = 1e-3 u1, v = 0, variables
lentes conservees) ; delta = ||s||_regraine - s0 ; theta, chi, p mesures a
la regraine ; bisection en s (11 iterations) et en D (10 iterations).
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
S_LO, S_HI = 0.0002, 0.0135
T2_FIX = 80
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


def reseed(st):
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + st[4:]


def prefixe(s0, T_am, D):
    st = run(fresh(s0), U1, G2, float(T_am))
    return run(st, U1, 0.0, float(D))


def sonde_issue(st):
    st = run(reseed(st), U1, G2, float(T2_FIX))
    st = run(st, U1, 0.0, 300.0)
    return math.hypot(st[0], st[1]) > 1.0


def issue(s0, T_am, D):
    return sonde_issue(prefixe(s0, T_am, D))


def bisect_b(T_am, D, lo=S_LO, hi=S_HI, iters=11):
    o_lo, o_hi = issue(lo, T_am, D), issue(hi, T_am, D)
    if o_lo == o_hi:
        return None, ("A" if o_lo else ".") * 2
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if issue(mid, T_am, D) == o_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), None


# ------------------------------------------------------------------- V0
print("V0 -- validation integrateur")
pr = run(run(fresh(), U1, G2, 41.9), U1, 0.0, 200.0)
st = run(run(reseed(pr), U1, G2, 100.0), U1, 0.0, 200.0)
assert abs(math.hypot(st[0], st[1]) - 3.246573235196175) < 1e-9
print("  conforme", flush=True)

# ------------------------------------------------------------------- V1
print("V1 -- diagnostic : etat de regraine le long de D (T_am=17.5, s0=0.0002)")
V1 = []
for D in (10, 20, 30, 50, 60, 70, 80, 90, 100, 110, 120, 150):
    st = prefixe(0.0002, 17.5, float(D))
    e = {"D": D, "theta": round(math.hypot(st[4], st[5]), 5),
         "chi": round(math.hypot(st[6], st[7]), 5),
         "p": round(st[8], 4),
         "s": round(math.hypot(st[9], st[10]), 5),
         "issue": "A" if sonde_issue(st) else "."}
    V1.append(e)
    print(f"  D={D:>4} theta={e['theta']:.5f} chi={e['chi']:.5f} "
          f"||s||={e['s']:.5f} issue={e['issue']}", flush=True)
OUT["V1"] = V1

# ------------------------------------------------------------------- V2
print("V2 -- prediction inter-axes : la bascule en D se produit a ||s|| = s*")
S_STAR = 0.01627
V2 = []
for T_am in (15.5, 17.5, 20.0):
    lo, hi = 50.0, 150.0
    assert issue(0.0002, T_am, lo) and not issue(0.0002, T_am, hi)
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if issue(0.0002, T_am, mid):
            lo = mid
        else:
            hi = mid
    Dstar = 0.5 * (lo + hi)
    st = prefixe(0.0002, T_am, Dstar)
    s_at = math.hypot(st[9], st[10])
    th_at = math.hypot(st[4], st[5])
    V2.append({"T_am": T_am, "D_star": round(Dstar, 2),
               "s_a_la_bascule": round(s_at, 5), "theta": round(th_at, 6),
               "ecart_a_s_star": round(s_at - S_STAR, 6)})
    print(f"  T_am={T_am:5.1f} D*={Dstar:6.2f} ||s||(D*)={s_at:.5f} "
          f"(s*={S_STAR}, ecart={s_at - S_STAR:+.5f}, theta={th_at:.6f})",
          flush=True)
OUT["V2"] = V2

# ------------------------------------------------------------------- V3
print("V3 -- le regime theta-vivant (prefixes faibles) : courbe de facilitation")
V3 = []
for T_am in (5.0, 5.5, 6.0, 6.5, 7.0):
    for D in (12.0, 15.0, 18.0, 21.0, 25.0, 150.0):
        b, edge = bisect_b(T_am, D)
        e = {"T_am": T_am, "D": D, "b": None if b is None else round(b, 5),
             "bord": edge}
        if b is not None:
            stp = prefixe(b, T_am, D)
            e.update({"theta": round(math.hypot(stp[4], stp[5]), 4),
                      "chi": round(math.hypot(stp[6], stp[7]), 5),
                      "p": round(stp[8], 4),
                      "delta": round(math.hypot(stp[9], stp[10]) - b, 5)})
            e["b_plus_delta"] = round(b + e["delta"], 5)
            e["facilitation"] = round(S_STAR - e["b_plus_delta"], 5)
            print(f"  T_am={T_am:4.1f} D={D:5.1f} b={b:.5f} "
                  f"b+delta={e['b_plus_delta']:.5f} "
                  f"facil={e['facilitation']:+.5f} theta={e['theta']:.4f} "
                  f"chi={e['chi']:.5f}", flush=True)
        else:
            print(f"  T_am={T_am:4.1f} D={D:5.1f} b=- ({edge})", flush=True)
        V3.append(e)
OUT["V3"] = V3
okv3 = [e for e in V3 if e["b"] is not None]
def strate(sel, nom):
    pts = sorted(sel, key=lambda e: e["theta"])
    paires = [(a, c) for i, a in enumerate(pts) for c in pts[i + 1:]
              if abs(c["theta"] - a["theta"]) < 0.004
              and (a["T_am"], a["D"]) != (c["T_am"], c["D"])]
    disp = max((abs(c["facilitation"] - a["facilitation"]) for a, c in paires),
               default=None)
    mono = all(c["facilitation"] >= a["facilitation"] - 5e-4
               for a, c in zip(pts, pts[1:]))
    print(f"  strate {nom} : monotone en theta : {mono} ; dispersion "
          f"transverse max : {disp} ({len(paires)} paires)", flush=True)
    return {"monotone": mono, "dispersion": disp, "n_paires": len(paires)}
syn = {"toutes_D": strate(okv3, "toutes D"),
       "D_ge_15": strate([e for e in okv3 if e["D"] >= 15.0], "D >= 15"),
       "D_12": strate([e for e in okv3 if e["D"] < 15.0], "D = 12")}
OUT["V3_synthese"] = syn

# ------------------------------------------------------------------- V4
print("V4 -- l'echelle des resumes : ablations pres du seuil (T_am=6)")
V4 = {}
for D in (150.0, 21.0, 12.0):
    b, _ = bisect_b(6.0, D)
    if b is None:
        V4[str(D)] = {"b": None}
        print(f"  D={D:5.1f} : pas de seuil en plage", flush=True)
        continue
    ss = [b + k * 4e-4 for k in (-3, -2, -1, 0, 1, 2, 3)]
    patt = {"complet": "", "chi_theta_s": "", "theta_s": "", "s_seul": ""}
    for s0 in ss:
        stp = prefixe(s0, 6.0, D)
        variantes = {"complet": stp,
                     "chi_theta_s": stp[:8] + (0.0,) + stp[9:],
                     "theta_s": stp[:6] + (0.0, 0.0, 0.0) + stp[9:],
                     "s_seul": stp[:4] + (0.0, 0.0, 0.0, 0.0, 0.0) + stp[9:]}
        for k, s in variantes.items():
            patt[k] += "A" if sonde_issue(s) else "."
    V4[str(D)] = {"b": round(b, 5), "motifs": patt,
                  "chi_theta_s_suffit": patt["complet"] == patt["chi_theta_s"],
                  "theta_s_suffit": patt["complet"] == patt["theta_s"],
                  "s_seul_suffit": patt["complet"] == patt["s_seul"]}
    print(f"  D={D:5.1f} b={b:.5f} complet=[{patt['complet']}] "
          f"(chi,theta,s)=[{patt['chi_theta_s']}] "
          f"(theta,s)=[{patt['theta_s']}] s_seul=[{patt['s_seul']}]", flush=True)
OUT["V4"] = V4

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
axs[0].plot([e["D"] for e in V1], [e["s"] for e in V1], "o-",
            color="#4878a8", ms=5, lw=1.2, label="$\\|s\\|$ a la regraine")
axs[0].axhline(S_STAR, color="k", lw=0.9, ls="--", label="$s^*$ (III-5)")
Ds = OUT["V2"][1]["D_star"]
axs[0].axvline(Ds, color="#c44e52", lw=0.9, ls=":",
               label=f"bascule mesuree $D^*$={Ds}")
for e in V1:
    axs[0].annotate(e["issue"], (e["D"], e["s"]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
axs[0].set_xlabel("oubli $D$ (u)")
axs[0].set_ylabel("contenu total a la regraine")
axs[0].set_title("V1/V2 -- l'invariant $s^*$ predit la bascule sur l'axe $D$\n"
                 "($T_{am}$=17.5 ; $\\theta$ mort a la bascule)")
axs[0].legend(fontsize=8)
mk = {5.0: "o", 5.5: "s", 6.0: "^", 6.5: "D", 7.0: "v"}
cmap = plt.cm.viridis
for e in okv3:
    col = cmap(min(e["D"], 30.0) / 30.0)
    axs[1].plot(e["theta"], e["facilitation"], mk[e["T_am"]], color=col, ms=6)
axs[1].axhline(0.0, color="k", lw=0.8, ls="--")
axs[1].set_xlabel("$\\|\\theta\\|$ a la regraine")
axs[1].set_ylabel("facilitation  $s^* - (b + \\delta)$")
axs[1].set_title("V3 -- le regime $\\theta$-vivant : courbe de facilitation\n"
                 "(marqueurs : $T_{am}$ ; couleur : $D$, jaune = long)")
for a in axs:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("output_theory/fiii6_regimes.png", dpi=150)
print("-> output_theory/fiii6_regimes.png")

with open("output_theory/fiii6_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii6_report.json")
