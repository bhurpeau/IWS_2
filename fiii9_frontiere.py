"""F-III.9 -- La frontiere d'appropriation dans l'etat lent complet.

Objet. Troisieme piece du chemin critique. Pour la sonde fil-rouge P =
(u1, G2, T2=80, regraine canonique, relaxation 300) et rho = bassin binaire,
on definit sur l'etat lent aligne z = (p, chi, theta, s) (composantes le
long de u1, hors-axe nul, ssc = 0) :

    A_P = { z : R_z(P) = A },     Sigma_P = frontiere de A_P.

Le theoreme III-9 decrit la section {p = chi = theta = 0} : A_P inter cet
axe = A_Omega. F-III.6-V4 en a sonde les sections theta et p par ablations.
Ici on cartographie les sections par INJECTION DIRECTE d'etats (aucun
prefixe), on teste la fermeture (l'etat lent aligne suffit-il a classer les
etats REELS produits par des histoires ?), et on classifie l'activite des
horloges aux trois niveaux (ponctuelle / locale / globale, F-III.8.1-R8).

Experiences :
  W0  validation de l'integrateur.
  W1  section (theta, s) a p = chi = 0 : fibres scannees + bisection des
      bords ; detection de fibres multicomposantes.
  W2  sections (chi, s) a p = theta = 0 et (p, s) a chi = theta = 0.
  W3  test de fermeture (Prop. III-10) : etats de regraine REELS produits
      par des histoires (T_am, D) varies, y compris l'etage profond D = 12 ;
      l'etat synthetique aligne portant les quatre coordonnees lentes
      reproduit-il l'issue ? Residus hors-axe mesures.
  W4  la facilitation de F-III.6 expliquee : les points (theta, facilitation)
      de la strate D >= 15 sont-ils sur la section (theta, s) de Sigma_P ?
      Residus rapportes contre chi.
  W5  activite des horloges aux trois niveaux, en quatre points de l'etat
      lent (pres de la frontiere, aux etages successifs).

Protocole declare : (B, rho) = (sondes de cette note, bassin binaire) ;
plage s in [0.0002, 0.020] ; grilles theta in [0, 0.30], chi in [0, 0.03],
p in [0, 1.2] ; bisections 10 iterations ; fibres scannees en 14 points.
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
S_LO, S_HI = 0.0002, 0.020
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


def zstate(p, chi, theta, s):
    """Etat lent aligne injecte sur la graine canonique."""
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0,
            theta * U1[0], theta * U1[1], chi * U1[0], chi * U1[1],
            p, s * U1[0], s * U1[1], 0.0)


def issue_z(z, T2=80):
    st = run(zstate(*z), U1, G2, float(T2))
    st = run(st, U1, 0.0, 300.0)
    return math.hypot(st[0], st[1]) > 1.0


def fibre(pfix, cfix, tfix, n=14, T2=80):
    """Scan de la fibre en s + bisection de tous les bords detectes."""
    ss = [S_LO + i * (S_HI - S_LO) / (n - 1) for i in range(n)]
    motif = "".join("A" if issue_z((pfix, cfix, tfix, s), T2) else "."
                    for s in ss)
    bords = []
    for i in range(n - 1):
        if motif[i] != motif[i + 1]:
            lo, hi = ss[i], ss[i + 1]
            o_lo = motif[i] == "A"
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if issue_z((pfix, cfix, tfix, mid), T2) == o_lo:
                    lo = mid
                else:
                    hi = mid
            bords.append(round(0.5 * (lo + hi), 5))
    return motif, bords


def fresh(s=0.0):
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, s * U1[0], s * U1[1], 0.0)


# ------------------------------------------------------------------- W0
print("W0 -- validation integrateur")
pr = run(run(fresh(), U1, G2, 41.9), U1, 0.0, 200.0)
st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + pr[4:]
st = run(run(st, U1, G2, 100.0), U1, 0.0, 200.0)
assert abs(math.hypot(st[0], st[1]) - 3.246573235196175) < 1e-9
print("  conforme", flush=True)

# ------------------------------------------------------------------- W1
print("W1 -- section (theta, s) a p = chi = 0")
TH_GRID = [0.30 * i / 12 for i in range(13)]
sec_th = []
for th in TH_GRID:
    motif, bords = fibre(0.0, 0.0, th)
    sec_th.append({"theta": round(th, 3), "motif": motif, "bords": bords})
    print(f"  theta={th:5.3f} [{motif}] bords={bords}", flush=True)
multi_th = [e["theta"] for e in sec_th if len(e["bords"]) > 1]
print(f"  fibres multicomposantes : {multi_th if multi_th else 'aucune'}",
      flush=True)
OUT["W1"] = sec_th

# ------------------------------------------------------------------- W2
print("W2 -- sections (chi, s) a p = theta = 0 et (p, s) a chi = theta = 0")
CH_GRID = [0.030 * i / 6 for i in range(7)]
sec_ch = []
for ch in CH_GRID:
    motif, bords = fibre(0.0, ch, 0.0)
    sec_ch.append({"chi": round(ch, 4), "motif": motif, "bords": bords})
    print(f"  chi={ch:6.4f} [{motif}] bords={bords}", flush=True)
P_GRID = [1.2 * i / 6 for i in range(7)]
sec_p = []
for pv in P_GRID:
    motif, bords = fibre(pv, 0.0, 0.0)
    sec_p.append({"p": round(pv, 3), "motif": motif, "bords": bords})
    print(f"  p={pv:5.3f} [{motif}] bords={bords}", flush=True)
OUT["W2"] = {"chi": sec_ch, "p": sec_p}

# ------------------------------------------------------------------- W3
print("W3 -- test de fermeture : etats reels vs etats synthetiques alignes")
combos = [(6.0, D, s0) for D in (12.0, 15.0, 21.0, 150.0)
          for s0 in (0.004, 0.006)] + \
         [(17.5, D, s0) for D in (50.0, 150.0) for s0 in (0.008, 0.010)]
n_ok, res_max, rows = 0, 0.0, []
for T_am, D, s0 in combos:
    stp = run(fresh(s0), U1, G2, T_am)
    stp = run(stp, U1, 0.0, D)
    # coordonnees lentes alignees et residus hors-axe
    thpar = (stp[4] + stp[5]) / SQ2
    chpar = (stp[6] + stp[7]) / SQ2
    spar = (stp[9] + stp[10]) / SQ2
    resid = max(abs(stp[4] - stp[5]), abs(stp[6] - stp[7]),
                abs(stp[9] - stp[10])) / SQ2
    res_max = max(res_max, resid)
    st_r = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + stp[4:]
    st_r = run(run(st_r, U1, G2, 80.0), U1, 0.0, 300.0)
    o_reel = math.hypot(st_r[0], st_r[1]) > 1.0
    o_syn = issue_z((stp[8], chpar, thpar, spar))
    ok = o_reel == o_syn
    n_ok += ok
    rows.append({"T_am": T_am, "D": D, "s0": s0,
                 "z": [round(stp[8], 4), round(chpar, 4),
                       round(thpar, 4), round(spar, 5)],
                 "reel": "A" if o_reel else ".",
                 "synthetique": "A" if o_syn else ".", "accord": ok})
    print(f"  T_am={T_am:4} D={D:5} s0={s0} z=({stp[8]:.3f},{chpar:.4f},"
          f"{thpar:.4f},{spar:.5f}) reel={'A' if o_reel else '.'} "
          f"syn={'A' if o_syn else '.'} accord={ok}", flush=True)
print(f"  fermeture : {n_ok}/{len(combos)} ; residu hors-axe max = "
      f"{res_max:.2e} ; ssc reel non nul, synthetique = 0", flush=True)
OUT["W3"] = {"lignes": rows, "accord": f"{n_ok}/{len(combos)}",
             "residu_hors_axe_max": res_max}

# ------------------------------------------------------------------- W4
print("W4 -- la facilitation de F-III.6 sur la section (theta, s)")
rep6 = json.load(open("output_theory/fiii6_report.json"))
S_STAR = 0.01627
pts6 = [e for e in rep6["V3"] if e.get("b") is not None and e["D"] >= 15.0]


def s_c(theta):
    lo, hi = S_LO, S_HI
    o_lo = issue_z((0.0, 0.0, theta, lo))
    if o_lo == issue_z((0.0, 0.0, theta, hi)):
        return None
    for _ in range(11):
        mid = 0.5 * (lo + hi)
        if issue_z((0.0, 0.0, theta, mid)) == o_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


W4 = []
for e in pts6:
    pred = S_STAR - s_c(e["theta"])
    W4.append({"T_am": e["T_am"], "D": e["D"], "theta": e["theta"],
               "chi": e["chi"], "fac_mesuree": e["facilitation"],
               "fac_section": round(pred, 5),
               "residu": round(e["facilitation"] - pred, 5)})
resid = [abs(r["residu"]) for r in W4]
print(f"  {len(W4)} points strate D>=15 : residu |mesure - section| "
      f"median ~ {sorted(resid)[len(resid)//2]:.5f}, max = {max(resid):.5f}",
      flush=True)
for r in sorted(W4, key=lambda r: -abs(r["residu"]))[:4]:
    print(f"    pire : T_am={r['T_am']} D={r['D']} theta={r['theta']} "
          f"chi={r['chi']} mesure={r['fac_mesuree']} section="
          f"{r['fac_section']} (residu {r['residu']:+.5f})", flush=True)
OUT["W4"] = W4

# ------------------------------------------------------------------- W5
print("W5 -- activite des horloges aux trois niveaux")
Z_PTS = [("tardif", (0.0, 0.0, 0.0, 0.0160)),
         ("theta-vivant", (0.0, 0.0, 0.12, 0.0110)),
         ("chi-vivant", (0.0, 0.015, 0.05, 0.0125)),
         ("profond", (0.6, 0.015, 0.10, 0.0100))]
W5 = {}
for nom, z in Z_PTS:
    base = issue_z(z)
    ligne = {"z": list(z), "issue": "A" if base else ".", "horloges": {}}
    for i, hn in enumerate(("p", "chi", "theta")):
        za = list(z)
        za[i] = 0.0
        ponct = issue_z(tuple(za)) != base
        # activite locale : trois voisins en s (+-6e-4)
        loc = any(issue_z((za[0], za[1], za[2], z[3] + d))
                  != issue_z((z[0], z[1], z[2], z[3] + d))
                  for d in (-6e-4, 0.0, 6e-4))
        ligne["horloges"][hn] = {"ponctuelle": ponct, "locale": loc}
    W5[nom] = ligne
    print(f"  {nom:12s} z={z} issue={ligne['issue']} " + " ".join(
        f"{h}:{'P' if v['ponctuelle'] else '-'}{'L' if v['locale'] else '-'}"
        for h, v in ligne["horloges"].items()), flush=True)
print("  (P = active ponctuellement, L = active localement ; "
      "l'activite globale sur B decoule des sections W1-W2)", flush=True)
OUT["W5"] = W5

# ------------------------------------------------------------------- W6
print("W6 -- le temoin anti-facilitant de F-III.6 sur la frontiere")
stp = run(fresh(0.0080), U1, G2, 5.5)
stp = run(stp, U1, 0.0, 12.0)
zt = (stp[8], (stp[6] + stp[7]) / SQ2, (stp[4] + stp[5]) / SQ2,
      (stp[9] + stp[10]) / SQ2)
motif_t, bords_t = fibre(zt[0], zt[1], zt[2])
fac_pred = 0.01627 - (bords_t[0] if bords_t else float("nan"))
print(f"  z_temoin = (p={zt[0]:.3f}, chi={zt[1]:.4f}, theta={zt[2]:.4f}, "
      f"s={zt[3]:.5f})")
print(f"  fibre a (p,chi,theta) du temoin : [{motif_t}] bords={bords_t}")
print(f"  facilitation predite par la frontiere : {fac_pred:+.5f} "
      f"(mesuree en F-III.6 : -0.0006)", flush=True)
OUT["W6"] = {"z_temoin": [round(v, 5) for v in zt], "motif": motif_t,
             "bords": bords_t, "fac_predite": round(fac_pred, 5),
             "fac_mesuree_fiii6": -0.0006}

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
ths = [e["theta"] for e in sec_th if e["bords"]]
scs = [e["bords"][0] for e in sec_th if e["bords"]]
axs[0].plot(ths, scs, "o-", color="#4878a8", ms=5, lw=1.2,
            label="branche inferieure $s_c^-$")
th2 = [e["theta"] for e in sec_th if len(e["bords"]) > 1]
sc2 = [e["bords"][1] for e in sec_th if len(e["bords"]) > 1]
axs[0].plot(th2, sc2, "^", color="#4878a8", ms=8, mfc="white",
            label="branche superieure $s_c^+$ (fibre bornee)")
axs[0].plot([r["theta"] for r in W4],
            [S_STAR - r["fac_mesuree"] for r in W4], "s",
            color="#c44e52", ms=5, alpha=0.8,
            label="histoires reelles (F-III.6, $D\\geq 15$)")
axs[0].set_xlabel("$\\theta$")
axs[0].set_title("section $(\\theta, s)$,  $p=\\chi=0$")
axs[0].legend(fontsize=8)
chs = [e["chi"] for e in sec_ch if e["bords"]]
sccs = [e["bords"][0] for e in sec_ch if e["bords"]]
axs[1].plot(chs, sccs, "o-", color="#55a868", ms=5, lw=1.2)
axs[1].set_xlabel("$\\chi$")
axs[1].set_title("section $(\\chi, s)$,  $p=\\theta=0$")
pvs = [e["p"] for e in sec_p if e["bords"]]
spcs = [e["bords"][0] for e in sec_p if e["bords"]]
axs[2].plot(pvs, spcs, "o-", color="#8172b2", ms=5, lw=1.2)
axs[2].set_xlabel("$p$")
axs[2].set_title("section $(p, s)$,  $\\chi=\\theta=0$")
axs[0].set_ylabel("frontiere en $s$")
for a in axs:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("F-III.9 -- trois sections de la frontiere d'appropriation "
             "$\\Sigma_P$ (sonde 80)", y=1.02)
fig.tight_layout()
fig.savefig("output_theory/fiii9_frontiere.png", dpi=150,
            bbox_inches="tight")
print("-> output_theory/fiii9_frontiere.png")

with open("output_theory/fiii9_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii9_report.json")
