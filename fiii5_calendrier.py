"""F-III.5 -- Le calendrier comme resume suffisant de la naissance des seuils.

Question (remarque 9 de la relecture de F-III.4) : si deux prefixes possedent
le meme calendrier charge, produisent-ils les memes morceaux de b(T_am) ?
Si oui, l'histoire est remplacable par son calendrier d'evenements informatifs
-- une compression plus forte encore que le resume mnesique.

Protocole declare. Le calendrier pertinent d'un seuil est celui du prefixe
PRIS AU SEUIL : apres bisection de b(T_am), le prefixe (amorce T_am, oubli
150) est rejoue instrumente avec s = b, et sa signature est extraite :
  n      : nombre de Kairos charges (||chi|| > 0.01) ;
  dates  : dates des evenements ;
  Q      : charge totale sum(nc * boost) ;
  theta  : ||theta|| a la regraine (etat lent transmis a la sonde).
Cellule de signature : n constant et dates sans saut > 3 u entre points
adjacents (dT_am = 0.25). Morceau de b : |db| < 1.5e-3 entre points adjacents.
C-III.4 predit : frontieres de morceaux = frontieres de cellules.
Suffisance (sens teste ici, declare) : meme n de prefixe sur des cellules
DISJOINTES => meme valeur de b (a la tolerance des morceaux). Le test porte
sur le compte n ; dates et charges sont rapportees en diagnostic.

Experiences :
  W0  validation de l'integrateur (reference F-III.0 P4).
  W1  grille fine T_am = 13.00 .. 24.00, pas 0.25 : b(T_am) + signature au
      seuil pour chaque point (sonde fixe T2 = 80).
  W2  decomposition en morceaux (continuite de b) et en cellules (signature)
      -- test frontal de C-III.4 / QO-63 : les frontieres coincident-elles ?
  W3  test de suffisance par recurrence : les cellules disjointes de meme n
      portent-elles le meme b ? (les quasi-plateaux 19/21/23 sont les
      candidats). Contre-test : cellules de meme n et b differents = refutation.
  W4  sonde croisee : prefixes composites (Ta, Tb) -- collision de signature
      avec la famille simple => comparaison des b.
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
S_LO, S_HI = 0.002, 0.0135
NC_MIN = 0.01
T2_FIX = 80
DB_PIECE, DT_JUMP = 1.5e-3, 3.0
OUT = {}


def run(st, u, g, T, rec=None, t0=0.0, dt=0.01):
    x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc = st
    u1, u2 = u
    gw1, gw2 = g * W * u1, g * W * u2
    t = t0
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
        t += dt
        if p > THK:
            d = s1 * c1 + s2 * c2
            boost = 1.0 + CS * d / (nc + 1e-12)
            if boost < 0.0:
                boost = 0.0
            if rec is not None and nc > NC_MIN:
                rec.append((t, nc, boost))
            t1 += boost * KQ * c1
            t2 += boost * KQ * c2
            c1 *= (1.0 - RCHI)
            c2 *= (1.0 - RCHI)
            p *= RP
    return (x1, x2, v1, v2, t1, t2, c1, c2, p, s1, s2, ssc), t


def fresh(s=0.0):
    return (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, s * U1[0], s * U1[1], 0.0)


def prefixe(s, ams, record=False):
    """Execute les amorces + oublis ; retourne (etat, calendrier)."""
    rec = [] if record else None
    st, t = fresh(s), 0.0
    for Ta in ams:
        st, t = run(st, U1, G2, float(Ta), rec, t)
        st, t = run(st, U1, 0.0, 150.0, rec, t)
    return st, (rec or [])


def issue(s, ams, T2=T2_FIX):
    st, _ = prefixe(s, ams)
    st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + st[4:]
    st, _ = run(st, U1, G2, float(T2), None, 0.0)
    st, _ = run(st, U1, 0.0, 300.0, None, 0.0)
    return math.hypot(st[0], st[1]) > 1.0


def bisect_b(ams, lo=S_LO, hi=S_HI, iters=10):
    o_lo, o_hi = issue(lo, ams), issue(hi, ams)
    if o_lo == o_hi:
        return None, ("A" if o_lo else ".") + ("A" if o_hi else ".")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if issue(mid, ams) == o_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), None


def signature(ams, b):
    """Calendrier du prefixe pris au seuil s = b."""
    st, cal = prefixe(b, ams, record=True)
    return {"n": len(cal), "dates": [round(e[0], 2) for e in cal],
            "Q": round(sum(e[1] * e[2] for e in cal), 4),
            "theta": round(math.hypot(st[4], st[5]), 4)}


# ------------------------------------------------------------------- W0
print("W0 -- validation integrateur")
pr, t = run(fresh(), U1, G2, 41.9)
pr, t = run(pr, U1, 0.0, 200.0, None, t)
st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + pr[4:]
st, t = run(st, U1, G2, 100.0, None, t)
st, _ = run(st, U1, 0.0, 200.0, None, t)
assert abs(math.hypot(st[0], st[1]) - 3.246573235196175) < 1e-9
print("  conforme", flush=True)

# ------------------------------------------------------------------- W1
print("W1 -- grille fine T_am = 13..24 pas 0.25, sonde T2 = 80")
W1 = []
Ta = 13.0
while Ta <= 24.0 + 1e-9:
    b, edge = bisect_b((Ta,))
    e = {"T_am": Ta, "b": None if b is None else round(b, 5), "bord": edge}
    if b is not None:
        e["sig"] = signature((Ta,), b)
    W1.append(e)
    if b is not None:
        print(f"  T_am={Ta:6.2f} b={b:.5f} n={e['sig']['n']:2d} "
              f"Q={e['sig']['Q']:.3f} theta={e['sig']['theta']:.3f}",
              flush=True)
    else:
        print(f"  T_am={Ta:6.2f} b=-      ({edge})", flush=True)
    Ta += 0.25
OUT["W1"] = W1

# ------------------------------------------------------------------- W2
print("W2 -- morceaux de b vs cellules de signature (test C-III.4 / QO-63)")
pts = [e for e in W1 if e["b"] is not None]
front_piece, front_cell = [], []
for a, c in zip(pts, pts[1:]):
    if abs(c["T_am"] - a["T_am"] - 0.25) > 1e-9:
        continue  # trou (point mort) : frontiere triviale des deux cotes
    mid = 0.5 * (a["T_am"] + c["T_am"])
    if abs(c["b"] - a["b"]) > DB_PIECE:
        front_piece.append(mid)
    jump = (a["sig"]["n"] != c["sig"]["n"]
            or any(abs(x - y) > DT_JUMP
                   for x, y in zip(a["sig"]["dates"], c["sig"]["dates"])))
    if jump:
        front_cell.append(mid)
print(f"  frontieres de morceaux  ({len(front_piece)}) : {front_piece}")
print(f"  frontieres de cellules  ({len(front_cell)}) : {front_cell}")
fp, fc = set(front_piece), set(front_cell)
print(f"  coincidence : morceaux c cellules : {fp <= fc} ; "
      f"cellules c morceaux : {fc <= fp}", flush=True)
OUT["W2"] = {"front_morceaux": front_piece, "front_cellules": front_cell,
             "morceaux_dans_cellules": sorted(fp - fc) == [],
             "cellules_dans_morceaux": sorted(fc - fp) == []}

# ------------------------------------------------------------------- W3
print("W3 -- suffisance par recurrence : cellules disjointes de meme n")
cells, cur = [], [pts[0]]
for a, c in zip(pts, pts[1:]):
    coupe = (abs(c["T_am"] - a["T_am"] - 0.25) > 1e-9
             or 0.5 * (a["T_am"] + c["T_am"]) in fc)
    if coupe:
        cells.append(cur)
        cur = []
    cur.append(c)
cells.append(cur)
res = [{"T_am": [c[0]["T_am"], c[-1]["T_am"]], "n": c[0]["sig"]["n"],
        "b": [min(e["b"] for e in c), max(e["b"] for e in c)],
        "Q": [min(e["sig"]["Q"] for e in c), max(e["sig"]["Q"] for e in c)]}
       for c in cells]
for r in res:
    print(f"  cellule T_am [{r['T_am'][0]:.2f},{r['T_am'][1]:.2f}] n={r['n']} "
          f"b in [{r['b'][0]:.5f},{r['b'][1]:.5f}] Q in "
          f"[{r['Q'][0]:.3f},{r['Q'][1]:.3f}]", flush=True)
suff_ok, suff_ko = [], []
for i in range(len(res)):
    for j in range(i + 1, len(res)):
        if res[i]["n"] == res[j]["n"]:
            bi, bj = res[i]["b"], res[j]["b"]
            proche = abs(0.5 * (bi[0] + bi[1]) - 0.5 * (bj[0] + bj[1])) < DB_PIECE
            (suff_ok if proche else suff_ko).append(
                {"n": res[i]["n"], "cellules": [res[i]["T_am"], res[j]["T_am"]],
                 "b": [bi, bj]})
print(f"  paires de cellules disjointes de meme n : "
      f"{len(suff_ok)} concordantes, {len(suff_ko)} DISCORDANTES", flush=True)
for e in suff_ko[:6]:
    print(f"    refutation candidate : n={e['n']} cellules {e['cellules']} "
          f"b {e['b']}", flush=True)
OUT["W3"] = {"cellules": res, "concordantes": suff_ok, "discordantes": suff_ko}

# ------------------------------------------------------------------- W4
print("W4 -- sonde croisee : prefixes composites")
W4 = []
for ams in ((8.0, 8.0), (8.0, 12.0), (12.0, 8.0), (10.0, 10.0), (6.0, 14.0)):
    b, edge = bisect_b(ams)
    e = {"ams": list(ams), "b": None if b is None else round(b, 5),
         "bord": edge}
    if b is not None:
        e["sig"] = signature(ams, b)
        match = [r for r in res if r["n"] == e["sig"]["n"]]
        e["collision_n"] = [{"T_am": m["T_am"], "b": m["b"]} for m in match]
        print(f"  ams={list(ams)} b={b:.5f} n={e['sig']['n']} "
              f"Q={e['sig']['Q']:.3f} ; cellules simples de meme n : "
              f"{[m['T_am'] for m in match]}", flush=True)
    else:
        print(f"  ams={list(ams)} b=- ({edge})", flush=True)
    W4.append(e)
OUT["W4"] = W4

# ------------------------------------------------------------------- W5
print("W5 -- l'equation de seuil : b + delta(T_am, b) = s*")
W5 = []
for e in pts[::2]:
    st, _ = prefixe(e["b"], (e["T_am"],))
    delta = math.hypot(st[9], st[10]) - e["b"]
    W5.append({"T_am": e["T_am"], "n": e["sig"]["n"], "b": e["b"],
               "delta": round(delta, 5),
               "b_plus_delta": round(e["b"] + delta, 5)})
    print(f"  T_am={e['T_am']:6.2f} n={e['sig']['n']} b={e['b']:.5f} "
          f"delta={delta:.5f} b+delta={e['b'] + delta:.5f}", flush=True)
vals = [r["b_plus_delta"] for r in W5]
print(f"  b+delta : etendue = {max(vals) - min(vals):.5f} "
      f"(vs etendue de b = {max(p['b'] for p in pts) - min(p['b'] for p in pts):.5f})")
OUT["W5"] = {"points": W5, "s_star": round(sum(vals) / len(vals), 5),
             "etendue_b_plus_delta": round(max(vals) - min(vals), 6)}

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axs[0]
ns = sorted({e["sig"]["n"] for e in pts})
cmap = plt.cm.tab10
for i, n in enumerate(ns):
    xs = [e["T_am"] for e in pts if e["sig"]["n"] == n]
    ys = [e["b"] for e in pts if e["sig"]["n"] == n]
    ax.plot(xs, ys, "o", ms=4.5, color=cmap(i % 10), label=f"n = {n}")
dead = [e["T_am"] for e in W1 if e["b"] is None and e["bord"] == ".."]
sat = [e["T_am"] for e in W1 if e["b"] is None and e["bord"] == "AA"]
ax.plot(dead, [S_HI + 4e-4] * len(dead), "x", color="gray", ms=6,
        label="point mort")
ax.plot(sat, [S_LO - 4e-4] * len(sat), "v", color="#55a868", ms=6,
        label="sature")
for f in front_cell:
    ax.axvline(f, color="k", lw=0.5, ls=":", alpha=0.5)
ax.set_xlabel("duree d'amorce $T_{am}$ (u)")
ax.set_ylabel("seuil $b(T_{am})$")
ax.set_title("$b(T_{am})$ colore par le compte $n$ du calendrier au seuil")
ax.legend(ncol=2, fontsize=8)
ax2 = axs[1]
sstar = OUT["W5"]["s_star"]
for i, n in enumerate(ns):
    xs = [r["delta"] for r in W5 if r["n"] == n]
    ys = [r["b"] for r in W5 if r["n"] == n]
    ax2.plot(xs, ys, "o", ms=5, color=cmap(i % 10))
dd = [0.003, 0.015]
ax2.plot(dd, [sstar - d for d in dd], "k-", lw=1,
         label=f"$b + \\delta = s^* = {sstar}$")
ax2.set_xlabel("contenu ecrit par le prefixe  $\\delta$")
ax2.set_ylabel("seuil $b$")
ax2.set_title("L'equation de seuil : toutes les cellules\nsur une meme ligne de niveau")
ax2.legend(fontsize=9)
for a in axs:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("output_theory/fiii5_calendrier.png", dpi=150)
print("-> output_theory/fiii5_calendrier.png")

with open("output_theory/fiii5_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii5_report.json")
