"""F-III.4 -- La naissance d'un seuil.

Question (remarque 9 de la relecture de F-III.3) : plutot que photographier
Theta, comprendre le MECANISME qui cree un seuil. L'issue d'une histoire est
decidee par son calendrier Kairos (nombre, dates, charges des evenements).
Le calendrier est a valeurs discretes ; la charge cumulee est continue en s.
D'ou deux types de naissance :

  Type K (combinatoire) : au seuil b, le calendrier lui-meme change --
    un Kairos charge apparait, disparait ou saute ; bifurcation de protocole.
  Type C (continu) : calendrier identique de part et d'autre ; la
    fonctionnelle continue des charges croise le niveau d'appropriation ;
    zero regulier.

Protocole declare. Calendrier = evenements p > THK avec ||chi|| > 0.01
(Kairos charges) survenant pendant les phases actives et les oublis de
l'histoire, hors relaxation finale (les Kairos vides du bassin approprie,
chi ~ 0, ne portent aucune information et pollueraient la comparaison).
Classification : K si le nombre d'evenements differe entre s = b-delta et
s = b+delta, ou si une date saute de plus de 5 u entre deux points de scan
adjacents (ds = 5e-5) ; C sinon. delta = 2e-4, 9 points de scan.

Experiences :
  Z0  validation de l'integrateur instrumente (reference F-III.0 P4).
  Z1  classification K/C des dix seuils de Theta_B3 (temoins de F-III.3).
  Z2  carte generatrice T_am -> b(T_am) a sonde fixe (T2 = 60 et 80) :
      branches, plateaux, discontinuites = le mecanisme rendu visible.
  Z3  zoom sur la paire serree {0.00463, 0.00504} : leurs calendriers, et
      une famille dense de prefixes bissectee dans l'intervalle etroit
      [0.0046, 0.0051] -- de nouveaux seuils s'inserent-ils dans le trou ?
  Z4  synthese (note) : mecanisme dominant, conjecture pour Theta_infini.
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
NC_MIN = 0.01     # charge minimale d'un Kairos "charge"
DELTA, NSCAN, T_JUMP = 2e-4, 9, 5.0
OUT = {}


def run(st, u, g, T, rec=None, t0=0.0, dt=0.01):
    """Integrateur scalaire canonique ; si rec est une liste, enregistre les
    Kairos charges (t_abs, ||chi||, boost). Retourne (etat, t_abs final)."""
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


def histoire(s, ams, T2, record=False):
    """Execute (amorces . oubli 150 chacune) . sonde(T2) . relax 300.
    Retourne (issue, calendrier) ; calendrier = liste (t, nc, boost) des
    Kairos charges hors relaxation finale."""
    rec = [] if record else None
    st, t = fresh(s), 0.0
    for Ta in ams:
        st, t = run(st, U1, G2, float(Ta), rec, t)
        st, t = run(st, U1, 0.0, 150.0, rec, t)
    st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + st[4:]
    st, t = run(st, U1, G2, float(T2), rec, t)
    st, _ = run(st, U1, 0.0, 300.0, None, t)
    return math.hypot(st[0], st[1]) > 1.0, (rec or [])


def bisect_b(ams, T2, lo, hi, iters=10):
    o_lo, _ = histoire(lo, ams, T2)
    o_hi, _ = histoire(hi, ams, T2)
    if o_lo == o_hi:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if histoire(mid, ams, T2)[0] == o_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def classify(ams, T2, b):
    """Scan s = b +/- DELTA (NSCAN points) : type K ou C + diagnostic."""
    ss = [b - DELTA + i * 2 * DELTA / (NSCAN - 1) for i in range(NSCAN)]
    cals, outs = [], []
    for s in ss:
        o, cal = histoire(s, ams, T2, record=True)
        outs.append(o)
        cals.append(cal)
    ns = [len(c) for c in cals]
    jump = 0.0
    for c0, c1 in zip(cals, cals[1:]):
        for e0, e1 in zip(c0, c1):
            jump = max(jump, abs(e1[0] - e0[0]))
    n_lo, n_hi = ns[0], ns[-1]
    typ = "K" if (n_lo != n_hi or jump > T_JUMP) else "C"
    charge = [sum(e[1] * e[2] for e in c) for c in cals]
    return {"type": typ, "n_evts": [n_lo, n_hi], "saut_date_max": round(jump, 2),
            "charge_lo_hi": [round(charge[0], 4), round(charge[-1], 4)],
            "issues": "".join("A" if o else "." for o in outs)}


# ------------------------------------------------------------------- Z0
print("Z0 -- validation integrateur instrumente")
pr, t = run(fresh(), U1, G2, 41.9)
pr, t = run(pr, U1, 0.0, 200.0, None, t)
st = (1e-3 * U1[0], 1e-3 * U1[1], 0.0, 0.0) + pr[4:]
st, t = run(st, U1, G2, 100.0, None, t)
st, _ = run(st, U1, 0.0, 200.0, None, t)
pc = math.hypot(st[0], st[1])
assert abs(pc - 3.246573235196175) < 1e-9, "integrateur non conforme"
print(f"  ||x_PC|| = {pc:.12f} : conforme", flush=True)

# ------------------------------------------------------------------- Z1
print("Z1 -- classification K/C des dix seuils de Theta_B3")
rep3 = json.load(open("output_theory/fiii3_report.json"))
seuils, wit = rep3["Y3"]["nouveaux_par_niveau"], []
for k in ("1", "2", "3"):
    for b in seuils[k]:
        for e in rep3["Y2"]["seuils"][k]:
            if abs(e["b"] - b) < 1e-4:
                wit.append((int(k), tuple(e["ams"]), e["T2"], b))
                break
Z1 = []
for k, ams, T2, b in wit:
    c = classify(ams, T2, b)
    c.update({"niveau": k, "ams": list(ams), "T2": T2, "b": round(b, 5)})
    Z1.append(c)
    print(f"  b={b:.5f} (niv {k}, ams={list(ams)}, T2={T2}) : type {c['type']} "
          f"n={c['n_evts']} saut={c['saut_date_max']}u issues=[{c['issues']}]",
          flush=True)
n_K = sum(1 for c in Z1 if c["type"] == "K")
print(f"  bilan : {n_K} K / {len(Z1) - n_K} C", flush=True)
OUT["Z1"] = Z1

# ------------------------------------------------------------------- Z2
print("Z2 -- carte generatrice T_am -> b(T_am), sondes T2 = 60 et 80")
Z2 = {}
for T2 in (60, 80):
    branch = []
    Ta = 5.0
    while Ta <= 60.0 + 1e-9:
        b = bisect_b((Ta,), T2, S_LO, S_HI)
        branch.append({"T_am": Ta, "b": None if b is None else round(b, 5)})
        Ta += 2.5
    Z2[T2] = branch
    txt = " ".join(f"{e['T_am']:.0f}:{'-' if e['b'] is None else f'{e['b']:.4f}'}"
                   for e in branch)
    print(f"  T2={T2} : {txt}", flush=True)
OUT["Z2"] = Z2

# ------------------------------------------------------------------- Z2b
print("Z2b -- densification de la branche (T2=80) et diagnostic des bords")
Z2b = []
for Ta in (13.0, 14.0, 16.0, 17.0, 19.0, 21.0, 23.0, 24.0):
    b = bisect_b((Ta,), 80, S_LO, S_HI)
    o_lo = histoire(S_LO, (Ta,), 80)[0]
    o_hi = histoire(S_HI, (Ta,), 80)[0]
    Z2b.append({"T_am": Ta, "b": None if b is None else round(b, 5),
                "o_S_LO": "A" if o_lo else ".", "o_S_HI": "A" if o_hi else "."})
    print(f"  T_am={Ta:>4} : b={'-' if b is None else f'{b:.5f}'} "
          f"(extremites {Z2b[-1]['o_S_LO']}|{Z2b[-1]['o_S_HI']})", flush=True)
OUT["Z2b"] = Z2b

# ------------------------------------------------------------------- Z2c
print("Z2c -- multi-croisements en s le long du zigzag (T2=80)")
Z2c = {}
for Ta in (17.0, 19.0):
    ss = [S_LO + i * (S_HI - S_LO) / 14 for i in range(15)]
    patt = "".join("A" if histoire(s, (Ta,), 80)[0] else "." for s in ss)
    ncross = sum(1 for a, b in zip(patt, patt[1:]) if a != b)
    Z2c[Ta] = {"motif": patt, "croisements": ncross}
    print(f"  T_am={Ta} : [{patt}] croisements >= {ncross}", flush=True)
OUT["Z2c"] = {str(k): v for k, v in Z2c.items()}

# ------------------------------------------------------------------- Z3
print("Z3 -- la paire serree {0.00463, 0.00504} et le trou entre elle")
pair = [c for c in Z1 if abs(c["b"] - 0.00463) < 2e-4
        or abs(c["b"] - 0.00504) < 2e-4]
for c in pair:
    print(f"  b={c['b']} : type {c['type']} n={c['n_evts']} "
          f"(niv {c['niveau']}, ams={c['ams']}, T2={c['T2']})", flush=True)
inserts = []
for Ta in (10.0, 15.0, 25.0, 30.0, 35.0):
    for Tb in (None, 10.0, 15.0, 25.0, 30.0, 35.0):
        ams = (Ta,) if Tb is None else (Ta, Tb)
        for T2 in (70, 80, 90):
            b = bisect_b(ams, T2, 0.0046, 0.0051, iters=8)
            if b is not None:
                inserts.append({"ams": list(ams), "T2": T2, "b": round(b, 6)})
uniq = []
for e in sorted(inserts, key=lambda e: e["b"]):
    if all(abs(e["b"] - u) > 2e-5 for u in uniq):
        uniq.append(e["b"])
print(f"  histoires denses testees : 90 ; seuils dans ]0.0046, 0.0051[ : "
      f"{len(inserts)} temoins, {len(uniq)} seuils distincts : {uniq}",
      flush=True)
OUT["Z3"] = {"paire": pair, "inserts": inserts, "seuils_distincts": uniq}

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
pts80 = ([(e["T_am"], e["b"]) for e in Z2[80] if e["b"] is not None]
         + [(e["T_am"], e["b"]) for e in Z2b if e["b"] is not None])
axs[0].plot([p[0] for p in pts80], [p[1] for p in pts80], "s",
            color="#c44e52", ms=5, label="sonde $T_2$=80 (un temoin par $T_{am}$)")
dead = [e["T_am"] for e in Z2b if e["b"] is None and e["o_S_HI"] == "."]
sat = [e["T_am"] for e in Z2b if e["b"] is None and e["o_S_LO"] == "A"]
axs[0].plot(dead, [S_HI + 4e-4] * len(dead), "x", color="gray", ms=6,
            label="point mort (tout-repos)")
axs[0].plot(sat, [S_LO - 4e-4] * len(sat), "v", color="#55a868", ms=6,
            label="sature (seuil sous la plage)")
axs[0].axhline(S_HI, color="gray", lw=0.6, ls=":")
axs[0].axhline(S_LO, color="gray", lw=0.6, ls=":")
axs[0].set_xlabel("duree d'amorce $T_{am}$ (u)")
axs[0].set_ylabel("seuil $b(T_{am})$")
axs[0].set_title("Z2 -- carte generatrice ($\\times$ = hors plage)")
axs[0].legend()
types = [c["type"] for c in Z1]
bs = [c["b"] for c in Z1]
for b, ty, c in zip(bs, types, Z1):
    axs[1].axvline(b, color="#c44e52" if ty == "K" else "#4878a8", lw=1.4)
    axs[1].text(b, 1.02 + 0.06 * (c["niveau"] - 1), ty, ha="center",
                fontsize=8, color="#c44e52" if ty == "K" else "#4878a8")
for u in uniq:
    axs[1].axvline(u, color="#55a868", lw=0.9, ls="--")
axs[1].set_xlim(S_LO, S_HI)
axs[1].set_ylim(0.95, 1.25)
axs[1].set_yticks([])
axs[1].set_xlabel("contenu $s$")
axs[1].set_title("Z1/Z3 -- seuils : K rouge, C bleu ; inserts Z3 en vert")
for a in axs:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("output_theory/fiii4_naissance.png", dpi=150)
print("-> output_theory/fiii4_naissance.png")

with open("output_theory/fiii4_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii4_report.json")
