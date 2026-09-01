"""Etape A/B -- Theorie sur la variete homogene Delta (note T-1).

PART 0  verification P-11 : lambda=0, zeta<1 => extinction globale sur Delta ;
PART 1  equilibres sur Delta (d=2) : classification (composantes dans {0,+-x_q}),
        seuils d'allumage x_u, stabilite (jacobien 6x6 sur Delta) ;
PART 2  d=1 : Routh-Hurwitz explicite, diagramme de bifurcation x*(zeta),
        carte de regimes (lambda, zeta) ;
PART 3  bassins d'attraction sur Delta (bistabilite historique) ;
PART 4  spectre transverse par mode de A~ au point actif ; prediction de G
        (flot seul) vs mesure simulateur (Kairos OFF puis ON) vs -0.4577 (X4b).

Parametres Core 0.1.1 / [P1] : zeta=0.8, lambda=2, alpha=0.45, beta=0.12,
gamma0=0.25, gamma1=1.25, kappa=(0.6,0.4,0.5), Theta=1.25, r_P=0.35, r_V=0.5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from iws_core.config import IWSConfig
from iws_core.engine import IWSEngine, circulant_graph, row_normalize_with_self_loops

OUT = Path("output_theory")
OUT.mkdir(exist_ok=True)
REPORT: dict = {}

# parametres
ZETA, LAM = 0.8, 2.0
ALPHA, BETA = 0.45, 0.12
G0, G1 = 0.25, 1.25
D = 2

# ---------------------------------------------------------------- outils
def Mmat(theta: np.ndarray, lam: float = LAM) -> np.ndarray:
    n2 = float(theta @ theta)
    Q = np.eye(len(theta)) + lam * np.outer(theta, theta) / (1.0 + n2)
    return np.linalg.inv(Q)


def reduced_field(y: np.ndarray, lam: float = LAM, zeta: float = ZETA) -> np.ndarray:
    """Champ reduit sur Delta, d=2 : y = (h, v, theta) in R^6 (trace saturee, u=0)."""
    h, v, th = y[0:2], y[2:4], y[4:6]
    gam = G0 + G1 * np.linalg.norm(th)
    dv = -gam * v - Mmat(th, lam) @ h + zeta * np.tanh(h)
    dth = ALPHA * np.tanh(h) - BETA * th
    return np.concatenate([v, dv, dth])


def jac_num(f, y, eps=1e-7):
    n = len(y)
    J = np.zeros((n, n))
    f0 = f(y)
    for k in range(n):
        yp = y.copy(); yp[k] += eps
        J[:, k] = (f(yp) - f0) / eps
    return J


def rk4(f, y, dt, steps):
    for _ in range(steps):
        k1 = f(y); k2 = f(y + dt / 2 * k1)
        k3 = f(y + dt / 2 * k2); k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def lambda_q(x: float, q: int, lam=LAM, ab=ALPHA / BETA) -> float:
    r2 = q * (ab * np.tanh(x)) ** 2
    return 1.0 + lam * r2 / (1.0 + r2)


def scalar_roots(q: int, lam=LAM, zeta=ZETA, ab=ALPHA / BETA):
    """Racines positives de x = zeta * Lambda_q(x) * tanh(x)."""
    F = lambda x: zeta * lambda_q(x, q, lam, ab) * np.tanh(x) - x
    xs = np.linspace(1e-6, 12.0, 24000)
    vals = F(xs)
    roots = []
    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] <= 0 and vals[i] != vals[i + 1]:
            a, b = xs[i], xs[i + 1]
            for _ in range(80):
                m = 0.5 * (a + b)
                if F(a) * F(m) <= 0: b = m
                else: a = m
            roots.append(0.5 * (a + b))
    # dedoublonner
    out = []
    for r in roots:
        if not out or abs(r - out[-1]) > 1e-6:
            out.append(r)
    return out


# ================================================================ PART 0
print("=" * 72)
print("PART 0 -- P-11 : lambda=0, zeta<1 => extinction sur Delta")
y = np.array([1.7, -2.3, 0.4, 0.1, 0.0, 0.0])
yf = rk4(lambda z: reduced_field(z, lam=0.0), y, 0.01, 60000)
print(f"  ||h(inf)|| = {np.linalg.norm(yf[0:2]):.2e}  (attendu : 0)")
roots_l0 = scalar_roots(1, lam=0.0)
print(f"  racines positives (lam=0, q=1) : {roots_l0}  (attendu : aucune)")
REPORT["P11_extinction_lambda0"] = {"h_final": float(np.linalg.norm(yf[0:2])),
                                    "nonzero_roots": roots_l0}

# ================================================================ PART 1
print("=" * 72)
print("PART 1 -- equilibres sur Delta (d=2), classification et stabilite")
eq_info = {}
for q in (1, 2):
    roots = scalar_roots(q)
    print(f"  q={q} : racines positives x = {[f'{r:.4f}' for r in roots]}"
          f"  -> ||h*|| = {[f'{r*np.sqrt(q):.4f}' for r in roots]}")
    eq_info[q] = roots
REPORT["equilibria"] = {str(q): eq_info[q] for q in eq_info}

# seuil d'allumage (approx analytique petite amplitude)
ab = ALPHA / BETA
for q in (1, 2):
    x_u_approx = np.sqrt((1 / ZETA - 1) / (LAM * q * ab**2 - 1 / 3))
    print(f"  seuil d'allumage approx (q={q}) : x_u ~ {x_u_approx:.4f}"
          f"  (exact : {eq_info[q][0]:.4f})")

# verification h* // tanh(h*) : Newton multi-departs sur le champ reduit
rng = np.random.default_rng(0)
found = set()
for _ in range(400):
    y = rng.normal(0, 2.5, 6)
    for _ in range(200):
        J = jac_num(lambda z: reduced_field(z), y)
        try:
            step = np.linalg.solve(J, reduced_field(y))
        except np.linalg.LinAlgError:
            break
        y = y - step
        if np.linalg.norm(step) < 1e-12:
            break
    if np.linalg.norm(reduced_field(y)) < 1e-9:
        h = np.round(y[0:2], 4)
        found.add(tuple(np.sort(np.abs(h))))
print(f"  composantes |h*| trouvees (Newton, 400 departs) : {sorted(found)}")
REPORT["newton_equilibria_abs_components"] = sorted([list(t) for t in found])

# stabilite sur Delta : jacobien 6x6
def eq_point(q, x):
    h = np.zeros(2); h[:q] = x
    th = (ALPHA / BETA) * np.tanh(h)
    return np.concatenate([h, np.zeros(2), th])

stab = {}
for label, q, x in [("origine", 1, 0.0),
                    ("axe_instable", 1, eq_info[1][0]),
                    ("axe_stable", 1, eq_info[1][-1]),
                    ("diag_instable", 2, eq_info[2][0]),
                    ("diag_stable", 2, eq_info[2][-1])]:
    y = eq_point(q, x)
    ev = np.linalg.eigvals(jac_num(lambda z: reduced_field(z), y))
    stab[label] = {"x": x, "max_Re": float(np.max(ev.real)),
                   "eig_Re_sorted": sorted(np.round(ev.real, 4).tolist())}
    print(f"  {label:14s} x={x:8.4f}  max Re(eig on-Delta) = {np.max(ev.real):+.4f}")
REPORT["on_delta_stability"] = stab

# ================================================================ PART 2
print("=" * 72)
print("PART 2 -- d=1 : Routh-Hurwitz explicite + cartes")

def rh_d1(x, lam=LAM, zeta=ZETA):
    """Coefficients du polynome caracteristique au point (x,0,theta*) (d=1)."""
    t = np.tanh(x); s2 = 1 - t * t
    th = ab * t
    den = 1 + (1 + lam) * th * th
    m = (1 + th * th) / den
    b = 2 * lam * th * x / den**2          # +dv/dtheta
    gam = G0 + G1 * abs(th)
    c = zeta * s2
    a2 = gam + BETA
    a1 = gam * BETA + (m - c)
    a0 = BETA * (m - c) - ALPHA * b * s2
    return a2, a1, a0

for x in eq_info[1]:
    a2, a1, a0 = rh_d1(x)
    stable = (a2 > 0) and (a0 > 0) and (a2 * a1 > a0)
    print(f"  x={x:7.4f} : a2={a2:.3f} a1={a1:.3f} a0={a0:+.5f} "
          f"a2*a1-a0={a2*a1-a0:+.3f} -> {'STABLE' if stable else 'INSTABLE'}")

# carte (lambda, zeta)
lams = np.linspace(0.0, 6.0, 121)
zets = np.linspace(0.05, 1.6, 121)
phase = np.zeros((len(zets), len(lams)), dtype=int)
# codes : 0 extinction seule ; 1 bistable ; 2 actif seul (zeta>1) ;
#         3 bistable avec branche haute oscillante (RH viole) ; 4 actif oscillant
for i, z in enumerate(zets):
    for j, l in enumerate(lams):
        roots = scalar_roots(1, lam=l, zeta=z)
        has_active = len(roots) >= 1 and roots[-1] > 1e-3
        if z < 1.0:
            if not has_active or len(roots) < 2:
                phase[i, j] = 0
            else:
                a2, a1, a0 = rh_d1(roots[-1], lam=l, zeta=z)
                ok = (a0 > 0) and (a2 * a1 > a0)
                phase[i, j] = 1 if ok else 3
        else:
            if has_active:
                a2, a1, a0 = rh_d1(roots[-1], lam=l, zeta=z)
                ok = (a0 > 0) and (a2 * a1 > a0)
                phase[i, j] = 2 if ok else 4
            else:
                phase[i, j] = 2  # origine instable, pas de racine detectee (bord)
frac_osc = float(np.mean((phase == 3) | (phase == 4)))
print(f"  fraction de la carte avec RH viole (oscillatoire) : {frac_osc:.3f}")
REPORT["phase_map_osc_fraction"] = frac_osc

# frontiere d'existence a zeta<1 : zeta_min(lambda)
zmin = []
for l in lams:
    lo, hi = 0.01, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if len(scalar_roots(1, lam=l, zeta=mid)) >= 2:
            hi = mid
        else:
            lo = mid
    zmin.append(hi)
REPORT["zeta_min_curve"] = {"lams": lams.tolist(), "zeta_min": zmin}

# ================================================================ PART 3
print("=" * 72)
print("PART 3 -- bassins sur Delta (h0 in [-3,3]^2, v0=0, theta0=0)")

def batch_field(Y: np.ndarray) -> np.ndarray:
    """Champ reduit vectorise : Y (B,6). Sherman-Morrison pour M(theta)h."""
    h, v, th = Y[:, 0:2], Y[:, 2:4], Y[:, 4:6]
    n2 = np.sum(th * th, axis=1)
    coef = LAM / (1.0 + (1.0 + LAM) * n2)
    Mh = h - (coef * np.sum(th * h, axis=1))[:, None] * th
    gam = G0 + G1 * np.sqrt(n2)
    dv = -gam[:, None] * v - Mh + ZETA * np.tanh(h)
    dth = ALPHA * np.tanh(h) - BETA * th
    return np.hstack([v, dv, dth])

def batch_rk4(Y, dt, steps):
    for _ in range(steps):
        k1 = batch_field(Y); k2 = batch_field(Y + dt / 2 * k1)
        k3 = batch_field(Y + dt / 2 * k2); k4 = batch_field(Y + dt * k3)
        Y = Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return Y

ngrid = 61
xs = np.linspace(-3, 3, ngrid)
X1, X2 = np.meshgrid(xs, xs, indexing="ij")
Y0 = np.zeros((ngrid * ngrid, 6))
Y0[:, 0], Y0[:, 1] = X1.ravel(), X2.ravel()
Yf = batch_rk4(Y0, 0.05, 6000)
basin = (np.linalg.norm(Yf[:, 0:2], axis=1) >= 0.2).astype(int).reshape(ngrid, ngrid)
frac_active = float(basin.mean())
print(f"  fraction du carre atteignant l'etat actif : {frac_active:.3f}")
# fraction sous la loi initiale N(0, 0.6) de [P1]
mc = rng.normal(0, 0.6, (2000, 2))
Ymc = np.zeros((2000, 6)); Ymc[:, 0:2] = mc
Ymcf = batch_rk4(Ymc, 0.05, 6000)
p_act = float(np.mean(np.linalg.norm(Ymcf[:, 0:2], axis=1) >= 0.2))
print(f"  P(actif | h0 ~ N(0,0.6), v0=0, th0=0) ~ {p_act:.3f}  (2000 tirages)")
REPORT["basin"] = {"grid_fraction_active": frac_active,
                   "p_active_under_P1_init": p_act}

# ================================================================ PART 4
print("=" * 72)
print("PART 4 -- spectre transverse au point actif + validation de G")

N = 20
A = circulant_graph(N, (1, 2))
At = row_normalize_with_self_loops(A)
rho = np.sort(np.linalg.eigvalsh((At + At.T) / 2))  # symetrique ici
print(f"  modes de A~ (circulant) : rho_min={rho[0]:.4f}, "
      f"rho_2max={rho[-2]:.4f}, rho_1={rho[-1]:.4f}")

# point actif diagonal (celui observe : ||h*||=3.2468)
x_s = eq_info[2][-1]
h_star = np.array([x_s, x_s]); th_star = (ALPHA / BETA) * np.tanh(h_star)
S = np.diag(1.0 / np.cosh(h_star) ** 2)
M_star = Mmat(th_star)
Gam_star = G0 + G1 * np.linalg.norm(th_star)

# D_M = d/dtheta [ M(theta) h* ]  (2x2, differences finies)
def Mh(th): return Mmat(th) @ h_star
DM = np.zeros((2, 2))
for k in range(2):
    e = np.zeros(2); e[k] = 1e-7
    DM[:, k] = (Mh(th_star + e) - Mh(th_star - e)) / 2e-7
# d/dtheta [ Gamma(theta) v ] = 0 en v=0

def mode_jac(r):
    Z, I = np.zeros((2, 2)), np.eye(2)
    top = np.hstack([Z, I, Z])
    mid = np.hstack([-M_star + ZETA * r * S, -Gam_star * I, -DM])
    bot = np.hstack([ALPHA * r * S, Z, -BETA * I])
    return np.vstack([top, mid, bot])

max_re_T = -np.inf
per_mode = []
for r in rho[:-1]:                      # modes transverses (rho != 1)
    ev = np.linalg.eigvals(mode_jac(r))
    per_mode.append((float(r), float(np.max(ev.real))))
    max_re_T = max(max_re_T, np.max(ev.real))
ev_sync = np.linalg.eigvals(mode_jac(rho[-1]))
print(f"  mode synchro (rho=1)  : max Re = {np.max(ev_sync.real):+.4f} "
      f"(doit ~ stabilite sur Delta)")
print(f"  transverse (flot seul): max Re = {max_re_T:+.4f} "
      f"-> G_pred(flot) = {2*max_re_T:+.4f}")
REPORT["transverse"] = {"per_mode_maxRe": per_mode,
                        "G_pred_flow": 2 * max_re_T,
                        "sync_maxRe": float(np.max(ev_sync.real))}

# ---- mesure simulateur : Kairos OFF (flot pur) puis ON
def measure_G(use_kairos: bool, steps=3000, fit=(300, 1800), eps0=1e-4):
    cfg = IWSConfig.core01(steps=steps, graph_mode="circulant",
                           use_kairos=use_kairos)
    eng = IWSEngine(cfg, seed=3)
    eng.A = A.copy()
    eng.H = np.tile(h_star, (N, 1))
    eng.V = np.zeros((N, 2))
    eng.tau = np.tile(th_star, (N, 1))
    eng.P = np.zeros(N)
    eng.H[0, 0] += eps0
    eng.run()
    o6 = np.asarray(eng.history["h_var"])
    t = np.arange(steps) * cfg.dt
    a, b = fit
    mask = o6[a:b] > 1e-27   # au-dessus du plancher machine (h*~3.2, eps~1e-16)
    coef = np.polyfit(t[a:b][mask], np.log(o6[a:b][mask]), 1)
    return float(coef[0]), len(eng.kairos_log), o6

G_flow, nk0, o6_off = measure_G(False)
G_full, nk1, o6_on = measure_G(True)
print(f"  G mesure (Kairos OFF) = {G_flow:+.4f}   [{nk0} evenements]")
print(f"  G mesure (Kairos ON)  = {G_full:+.4f}   [{nk1} evenements]"
      f"   (X4b : -0.4577)")
print(f"  ecart prediction flot vs mesure OFF : "
      f"{abs(2*max_re_T - G_flow):.4f}")
REPORT["G"] = {"pred_flow": 2 * max_re_T, "meas_flow_kairos_off": G_flow,
               "meas_kairos_on": G_full, "x4b_reference": -0.4577,
               "events_off": nk0, "events_on": nk1}

# contribution des evenements : contraction de dv par r_V a chaque saut
if nk1 > 0:
    T_ev = (3000 * 0.05) / (nk1 / N)     # periode moyenne par noeud
    print(f"  periode Kairos par noeud ~ {T_ev:.3f} ; "
          f"contraction brute ln(r_V)/T_ev = {np.log(0.5)/T_ev:+.3f} "
          f"(sur la composante v seulement)")
    REPORT["G"]["event_period_per_node"] = T_ev

with open(OUT / "theory_report.json", "w") as f:
    json.dump(REPORT, f, indent=2, default=str)

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# (a) bifurcation x*(zeta) a lambda=2, q=1
ax = axes[0, 0]
zz = np.linspace(0.05, 1.5, 300)
for z in zz:
    rts = scalar_roots(1, zeta=z)
    for k, r in enumerate(rts):
        a2, a1, a0 = rh_d1(r, zeta=z)
        st = (a0 > 0) and (a2 * a1 > a0)
        ax.plot(z, r, ".", ms=2, color="tab:green" if st else "tab:red")
ax.axvline(1.0, color="k", lw=0.6, ls="--")
ax.plot([0.05, 1.0], [0, 0], color="tab:green", lw=2)
ax.plot([1.0, 1.5], [0, 0], color="tab:red", lw=2)
ax.set_xlabel("zeta"); ax.set_ylabel("x*")
ax.set_title("(a) bifurcation x*(zeta), lambda=2, d=1\nvert=stable, rouge=instable")

# (b) carte (lambda, zeta)
ax = axes[0, 1]
cmap = matplotlib.colors.ListedColormap(
    ["#d9d9d9", "#4daf4a", "#377eb8", "#ff7f00", "#e41a1c"])
ax.pcolormesh(lams, zets, phase, cmap=cmap, vmin=0, vmax=4)
ax.plot(lams, zmin, "k-", lw=1.2, label="zeta_min(lambda)")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.plot(LAM, ZETA, "k*", ms=14, label="[P1]")
ax.set_xlabel("lambda"); ax.set_ylabel("zeta")
ax.set_title("(b) carte d=1 : gris=extinction, vert=bistable,\n"
             "bleu=actif seul, orange/rouge=RH viole")
ax.legend(fontsize=8, loc="lower right")

# (c) bassins
ax = axes[1, 0]
ax.pcolormesh(xs, xs, basin.T, cmap="Greens", vmin=0, vmax=1.4)
for sgn in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
    ax.plot(sgn[0] * x_s, sgn[1] * x_s, "r*", ms=10)
    ax.plot(sgn[0] * eq_info[1][-1], 0, "b^", ms=7)
    ax.plot(0, sgn[1] * eq_info[1][-1], "b^", ms=7)
ax.plot(0, 0, "ko", ms=6)
ax.set_xlabel("h0[0]"); ax.set_ylabel("h0[1]")
ax.set_title(f"(c) bassins sur Delta (blanc=extinction, vert=actif)\n"
             f"P(actif|N(0,0.6)) ~ {p_act:.2f}")

# (d) spectre transverse + G
ax = axes[1, 1]
pm = np.array(per_mode)
ax.plot(pm[:, 0], 2 * pm[:, 1], "o-", ms=4, label="2*maxRe (flot, par mode)")
ax.axhline(G_flow, color="tab:green", ls="--",
           label=f"G mesure Kairos OFF = {G_flow:.3f}")
ax.axhline(G_full, color="tab:red", ls="--",
           label=f"G mesure Kairos ON = {G_full:.3f}")
ax.axhline(-0.4577, color="k", ls=":", label="X4b = -0.4577")
ax.set_xlabel("rho (mode de A~)"); ax.set_ylabel("taux de O6")
ax.set_title("(d) prediction spectrale de G au point actif")
ax.legend(fontsize=8)

fig.suptitle("Theorie sur Delta -- Etape A/B (note T-1)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "theory_delta.png", dpi=130)
print(f"\nFigure : {OUT}/theory_delta.png ; rapport : {OUT}/theory_report.json")
