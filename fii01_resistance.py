"""F-II.0.1 -- Theorie analytique de l'influence unidirectionnelle.

Reduction scalaire (diagonale, par composante) du systeme B force par A fixe :
  x' = v
  v' = -Gamma(sqrt2|th|) v - m(sqrt2|th|) x + zeta tanh(x)
  th' = alpha tanh(x) + g w - beta th ,   w = tanh(x_A) = 0.98006

PART 1  expansions locales en g_c (verifiees contre les mesures F-II.0) :
        - bifurcation TRANSCRITIQUE en g_c : la selle d'allumage traverse
          l'origine et ressort stabilisee en branche resistante ;
        - x_res(g) = -(w/alpha)(g - g_c) + O((g-g_c)^2) ;
        - m sur la branche : m = zeta - (zeta/3) x^2 + o(x^2)  (C = zeta/3).
PART 2  continuation complete des equilibres de F_g(x) sur x in [-4,4],
        g in [0, 0.14], stabilite par jacobien 3D -> nature de g_ign.
PART 3  seuil structurel (fin de branche) vs seuil dynamique (graine +1e-3)
        vs relapse depuis la branche.
PART 4  loi d'appropriation : predicteur lineaire non autonome (croissance
        integree le long de la charge, decouplage inclus) vs frontiere T*(g)
        bissectee dans la dyade complete.
"""
from __future__ import annotations

import json
import numpy as np

ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
X_A = 2.2959
W = float(np.tanh(X_A))
TAU_IGN = float(np.sqrt((1 - ZETA) / (3 * ZETA - 1)))
G_C = BETA * TAU_IGN / (np.sqrt(2) * W)
OUT = {"g_c": G_C, "tau_ign": TAU_IGN}

m_of = lambda n: (1 + n**2) / (1 + 3 * n**2)
mp_of = lambda n: -4 * n / (1 + 3 * n**2) ** 2

def theta_star(x, g):
    return (ALPHA * np.tanh(x) + g * W) / BETA

def F_g(x, g):
    n = np.sqrt(2) * np.abs(theta_star(x, g))
    return m_of(n) * x - ZETA * np.tanh(x)

def field(y, g):
    x, v, th = y
    n = np.sqrt(2) * abs(th)
    return np.array([v,
                     -(G0 + G1 * n) * v - m_of(n) * x + ZETA * np.tanh(x),
                     ALPHA * np.tanh(x) + g * W - BETA * th])

def jac3(y, g, h=1e-7):
    J = np.zeros((3, 3)); f0 = field(y, g)
    for k in range(3):
        yp = y.copy(); yp[k] += h
        J[:, k] = (field(yp, g) - f0) / h
    return J

# ---------------------------------------------------------------- PART 1
print("PART 1 -- expansions locales (verification contre F-II.0)")
print(f"  g_c = {G_C:.5f}  tau_ign = {TAU_IGN:.4f}")
for gmul, x_meas, m_meas in [(1.1, -0.0071, 0.79999), (1.5, -0.0356, 0.79966)]:
    d = (gmul - 1) * G_C
    x_pred = -(W / ALPHA) * d
    m_pred = ZETA - (ZETA / 3) * x_pred**2
    print(f"  g={gmul}g_c : x_res pred {x_pred:+.5f} vs mesure {x_meas:+.4f} | "
          f"m pred {m_pred:.5f} vs mesure {m_meas:.5f}")
OUT["expansion"] = {"slope": -W / ALPHA, "C": ZETA / 3}

# ---------------------------------------------------------------- PART 2
print("PART 2 -- continuation des equilibres (jacobien 3D)")
gs = np.linspace(0.0, 0.14, 281)
xgrid = np.linspace(-4, 4, 32001)
branches = []          # (g, x, max Re eig)
res_branch = {}        # g -> x de la branche resistante
for g in gs:
    vals = F_g(xgrid, g)
    idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    roots = []
    for i in idx:
        a, b = xgrid[i], xgrid[i + 1]
        for _ in range(60):
            mdl = 0.5 * (a + b)
            if F_g(a, g) * F_g(mdl, g) <= 0: b = mdl
            else: a = mdl
        roots.append(0.5 * (a + b))
    roots.append(0.0)  # x=0 toujours equilibre (peut echapper au scan)
    for r in sorted(set(np.round(roots, 8))):
        y = np.array([r, 0.0, theta_star(r, g)])
        ev = np.linalg.eigvals(jac3(y, g))
        mre = float(np.max(ev.real))
        branches.append((float(g), float(r), mre))
        if -0.35 < r < -1e-5 and mre < 0:
            res_branch[float(g)] = float(r)

g_res = sorted(res_branch)
g_end = g_res[-1] if g_res else None
print(f"  branche resistante stable : g in [{g_res[0]:.4f}, {g_end:.4f}]"
      f"  (g_c={G_C:.4f})")
print(f"  x_res a la fin de branche : {res_branch[g_end]:+.4f}")
OUT["branch_range"] = [g_res[0], g_end]

# que se passe-t-il a la fin ? inventaire des racines juste avant/apres
for gq in (g_end - 0.001, g_end + 0.001):
    rs = sorted({round(r, 4) for gg, r, _ in branches if abs(gg - gq) < 2.6e-4})
    print(f"  racines a g={gq:.4f} : {rs}")

# ---------------------------------------------------------------- PART 3
print("PART 3 -- structurel vs dynamique")
def dyn_outcome(g, x0, steps=300000, dt=0.01):
    y = np.array([x0, 0.0, 0.0])
    for _ in range(steps):
        y = y + dt * field(y, g)
    return y[0]

lo, hi = 1.2 * G_C, 2.2 * G_C
for _ in range(14):
    mid = 0.5 * (lo + hi)
    if dyn_outcome(mid, 1e-3) > 1: hi = mid
    else: lo = mid
g_ign_dyn = 0.5 * (lo + hi)
print(f"  seuil dynamique (graine +1e-3, th0=0) : g_ign_dyn = {g_ign_dyn:.4f} "
      f"= {g_ign_dyn/G_C:.2f} g_c")
# relapse depuis la branche elle-meme, juste au-dela de la fin structurelle
if g_end:
    xb = res_branch[g_end]
    xf = dyn_outcome(g_end + 0.002, xb)
    print(f"  depuis la branche (x0={xb:+.4f}) a g=g_end+0.002 : x(inf) = {xf:+.3f}")
OUT["g_ign_dyn"] = g_ign_dyn

# ---------------------------------------------------------------- PART 4
print("PART 4 -- loi d'appropriation : predicteur lineaire vs dyade complete")
X_SADDLE0 = 0.0717     # selle autonome (g=0), par composante

def T_star_pred(g, dt=0.005, Tmax=200.0):
    """Contact de duree T puis decouplage : plus petit T avec evasion lineaire."""
    def escapes(T):
        x, v, th, t = 1e-3, 0.0, 0.0, 0.0
        while t < T + 400.0:
            drive = g * W if t < T else 0.0
            n = np.sqrt(2) * abs(th)
            a = -(G0 + G1 * n) * v - m_of(n) * x + ZETA * x   # linearise en x
            x, v = x + dt * v, v + dt * a
            th = th + dt * (ALPHA * x + drive - BETA * th)     # ecriture linearisee
            t += dt
            if abs(x) > X_SADDLE0:
                return True
            if t > T and n < TAU_IGN and abs(x) < 1e-6:
                return False
        return abs(x) > X_SADDLE0
    lo_, hi_ = 1.0, Tmax
    if not escapes(Tmax):
        return None
    for _ in range(20):
        mid = 0.5 * (lo_ + hi_)
        if escapes(mid): hi_ = mid
        else: lo_ = mid
    return 0.5 * (lo_ + hi_)

def T_star_full(g, dt=0.01):
    def appro(T):
        y = np.array([1e-3, 0.0, 0.0])
        for _ in range(int(T / dt)):
            y = y + dt * field(y, g)
        for _ in range(int(300 / dt)):
            y = y + dt * field(y, 0.0)
        return abs(y[0]) > 1.0
    if not appro(200.0):
        return None
    lo_, hi_ = 1.0, 200.0
    for _ in range(16):
        mid = 0.5 * (lo_ + hi_)
        if appro(mid): hi_ = mid
        else: lo_ = mid
    return 0.5 * (lo_ + hi_)

law = {}
for g in (0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
    tp, tf = T_star_pred(g), T_star_full(g)
    law[g] = {"pred": tp, "full": tf}
    print(f"  g={g:4.2f} : T* predit = {str(round(tp,1)) if tp else 'jamais':>7} ; "
          f"T* mesure (dyade) = {str(round(tf,1)) if tf else 'jamais':>7}")
OUT["law"] = {str(k): v for k, v in law.items()}

with open("output_theory/fii01_report.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("-> output_theory/fii01_report.json")
