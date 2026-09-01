"""Note T-2 -- Pourquoi M(tau) ? Verification numerique du theoreme de robustesse.

Classe K x Sigma :
  m : [0,inf) -> (0,1], continue, m(0)=1, non croissante, m_inf > 0  (affaiblissement sature)
  sigma : R+ -> R+, sigma(0)=0, sigma'(0)=1, strictement concave, bornee (u_max = sigma_inf)

Reduction universelle (d=1, sur Delta, trace saturee) :
  equilibres <=>  Psi(u) = sigma^{-1}(u) * m((alpha/beta) u) / u = zeta,  u in (0, u_max)
  Psi(0+) = 1, Psi(u_max-) = +inf  =>  trichotomie pilotee par zeta_min = min Psi.

PART 1  universalite : Psi, zeta_min, racines, stabilite (RH) pour 4 m x 4 sigma ;
        coherence avec la courbe zeta_min(lambda) de T-1 ;
PART 2  lemmes de destruction :
        (i) renforcement (m croissante / constante)  -> extinction seule ;
        (ii) sigma lineaire                          -> seuil-repulseur + fuite ;
        (iii) rappel evanescent + trace non bornee   -> derive sans equilibre (surcharge) ;
PART 3  verification reseau (N=20) : operateur exponentiel m_exp injecte dans le
        moteur -> puits diagonal predit, bistabilite, decroissance transverse.
"""
from __future__ import annotations

import json
import numpy as np

from iws_core.config import IWSConfig
from iws_core.engine import IWSEngine

OUTJ = "output_theory/robustness_report.json"
REPORT: dict = {}

ALPHA, BETA, G0, G1, ZETA = 0.45, 0.12, 0.25, 1.25, 0.8
AB = ALPHA / BETA

# ------------------------------------------------------------- familles
M_FAMILY = {
    "rationnel(l=2)": lambda t: (1 + t**2) / (1 + 3 * t**2),
    "exp(k=0.5,m8=1/3)": lambda t: 1/3 + (2/3) * np.exp(-0.5 * t),
    "alg(m8=1/3)": lambda t: 1/3 + (2/3) / (1 + t**2),
    "logis(k=0.6,m8=0.4)": lambda t: 0.4 + 0.6 * (1 - np.tanh(0.6 * t)),
}
from scipy.special import erf as _serf, erfinv as _serfinv

def _erf(x):
    return _serf(np.sqrt(np.pi) / 2 * x)          # sigma'(0)=1, u_max=1

def _erf_inv(u):
    return 2 / np.sqrt(np.pi) * _serfinv(u)

SIGMA_FAMILY = {
    "tanh": (np.tanh, np.arctanh, 1.0),
    "alg x/(1+x)": (lambda x: x / (1 + np.abs(x)), lambda u: u / (1 - u), 1.0),
    "erf": (_erf, _erf_inv, 1.0),
    "2tanh(x/2)": (lambda x: 2 * np.tanh(x / 2), lambda u: 2 * np.arctanh(u / 2), 2.0),
}


def psi(u, m, sig_inv):
    return sig_inv(u) * m(AB * u) / u


def zeta_min(m, sig_inv, umax):
    u = np.linspace(1e-6, umax * (1 - 1e-9), 200000)
    vals = psi(u, m, sig_inv)
    i = int(np.argmin(vals))
    return float(vals[i]), float(u[i])


def roots_and_stability(m, sig, sig_inv, umax, zeta=ZETA):
    """Racines de Psi(u)=zeta ; stabilite via RH generalise (numerique)."""
    u = np.linspace(1e-7, umax * (1 - 1e-9), 400000)
    f = psi(u, m, sig_inv) - zeta
    roots = []
    for i in range(len(u) - 1):
        if f[i] * f[i + 1] <= 0 and f[i] != f[i + 1]:
            roots.append(float(u[i] - f[i] * (u[i + 1] - u[i]) / (f[i + 1] - f[i])))
    out = []
    h = 1e-6
    for ur in roots:
        x = float(sig_inv(ur))
        th = AB * sig(x)
        mp = (m(th + h) - m(th - h)) / (2 * h)          # m'
        sp = (sig(x + h) - sig(x - h)) / (2 * h)        # sigma'
        mm, c, b = float(m(th)), zeta * sp, -mp * x
        gam = G0 + G1 * th
        a2, a1 = gam + BETA, gam * BETA + (mm - c)
        a0 = BETA * (mm - c) - ALPHA * b * sp
        stable = (a0 > 0) and (a2 * a1 > a0)
        out.append({"x": x, "u": ur, "a0": a0, "m-c": mm - c,
                    "a2a1-a0": a2 * a1 - a0, "stable": bool(stable)})
    return out


print("=" * 76)
print("PART 1 -- universalite de la trichotomie sur K x Sigma  (zeta = 0.8)")
tab = {}
for mn, m in M_FAMILY.items():
    for sn, (sig, sig_inv, umax) in SIGMA_FAMILY.items():
        zm, ustar = zeta_min(m, sig_inv, umax)
        rs = roots_and_stability(m, sig, sig_inv, umax)
        pat = "".join("S" if r["stable"] else "U" for r in rs)
        mc_ok = all(r["m-c"] > 0 for r in rs)
        rh_ok = all(r["a2a1-a0"] > 0 for r in rs)
        n = len(rs)
        ok = ((zm < ZETA and n == 2 and pat == "US") or (zm >= ZETA and n == 0))
        tab[f"{mn} | {sn}"] = {"zeta_min": zm, "n_roots": n, "pattern": pat,
                               "x_roots": [round(r["x"], 4) for r in rs],
                               "m-c>0": mc_ok, "a2a1>a0": rh_ok, "conforme": ok}
        print(f"  {mn:22s} x {sn:12s}: zmin={zm:.4f}  racines={n} [{pat}] "
              f"x={[round(r['x'],3) for r in rs]}  m-c>0:{mc_ok}  RH:{rh_ok}"
              f"  {'OK' if ok else '<<< ANOMALIE'}")
REPORT["part1"] = tab

# coherence avec la courbe zeta_min(lambda) de T-1
prev = json.load(open("output_theory/theory_report.json"))
lams = np.array(prev["zeta_min_curve"]["lams"])
zt1 = np.array(prev["zeta_min_curve"]["zeta_min"])
zpsi = []
for l in lams:
    m = lambda t, l=l: (1 + t**2) / (1 + (1 + l) * t**2)
    zpsi.append(zeta_min(m, np.arctanh, 1.0)[0])
gap = float(np.max(np.abs(np.array(zpsi) - zt1)))
print(f"  coherence zeta_min(lambda) [T-1 bissection vs min Psi] : ecart max = {gap:.4f}")
REPORT["consistency_T1"] = gap

# ------------------------------------------------------------- PART 2
print("=" * 76)
print("PART 2 -- lemmes de destruction (simulations reduites d=1)")

def simulate_reduced(m, sig, trace_saturated=True, x0=2.5, v0=0.0, th0=0.0,
                     steps=60000, dt=0.01, track=False):
    x, v, th = x0, v0, th0
    xs = []
    for _ in range(steps):
        gam = G0 + G1 * abs(th)
        dv = -gam * v - m(abs(th)) * x + ZETA * sig(x)
        w = sig(x) if trace_saturated else x
        dth = ALPHA * w - BETA * th
        x, v, th = x + dt * v, v + dt * dv, th + dt * dth
        if track:
            xs.append(x)
    return (x, v, th, np.asarray(xs)) if track else (x, v, th)

# (i) trace passive (m=1) et renforcement (m croissante >= 1, hors classe)
for name, m in [("m=1 (trace passive)", lambda t: 1.0),
                ("m croissante (renforcement)",
                 lambda t: (1 + 3 * np.asarray(t)**2) / (1 + np.asarray(t)**2))]:
    xf, _, _ = simulate_reduced(m, np.tanh, x0=3.0)
    print(f"  (i) {name:30s}: x(inf) depuis x0=3 -> {xf:+.2e} (attendu : 0)")
    REPORT.setdefault("part2", {})[name] = float(xf)

# (ii) sigma lineaire : seuil-repulseur (test sur la variete lente theta=(a/b)x)
m_rat = M_FAMILY["rationnel(l=2)"]
xs = np.linspace(1e-4, 30, 300000)
vals = m_rat(AB * xs) - ZETA                # equilibres : m(theta(x)) = zeta
sgn = np.where(np.diff(np.sign(vals)))[0]
xc = float(xs[sgn[0]]) if len(sgn) else None
x_below, _, _ = simulate_reduced(m_rat, lambda x: x, x0=xc*0.9, th0=AB*xc*0.9)
xa, _, _, tr = simulate_reduced(m_rat, lambda x: x, x0=xc*1.1, th0=AB*xc*1.1,
                                steps=40000, track=True)
print(f"  (ii) sigma=identite : seuil x_c={xc:.4f} ; "
      f"x0=0.9xc -> {x_below:.2e} (extinction) ; "
      f"x0=1.1xc -> x(T)={xa:.2f}, x monotone croissant: {bool(np.all(np.diff(tr[::100])>0))} (fuite)")
REPORT["part2"]["sigma_lineaire"] = {"x_c": xc, "below": float(x_below),
                                     "above_final": float(xa)}

# (iii) rappel evanescent + trace lineaire (non bornee) : derive ~ sqrt(t)
m_vanish = lambda t: 1.0 / (1.0 + np.asarray(t) ** 2)     # m_inf = 0, o(1/t)
_, _, _, traj = simulate_reduced(m_vanish, np.tanh, trace_saturated=False,
                                 x0=1.0, steps=200000, dt=0.01, track=True)
t_ax = np.arange(len(traj)) * 0.01
half = len(traj) // 2
slope = np.polyfit(np.log(t_ax[half:]), np.log(traj[half:]), 1)[0]
print(f"  (iii) m->0 + trace lineaire : x(T)={traj[-1]:.2f}, "
      f"exposant log-log tardif ~ {slope:.3f} (derive sans equilibre ; ~0.5 attendu)")
REPORT["part2"]["surcharge"] = {"x_final": float(traj[-1]), "exponent": float(slope)}

# ------------------------------------------------------------- PART 3
print("=" * 76)
print("PART 3 -- reseau : existence/stabilite du puits vs bassin dynamique (m_exp)")
m_exp = M_FAMILY["exp(k=0.5,m8=1/3)"]

def metric_action_exp(self, tau_i, grad_i):
    nrm = float(np.linalg.norm(tau_i))
    if nrm < 1e-12:
        return grad_i
    that = tau_i / nrm
    mval = float(m_exp(nrm))
    return grad_i - (1.0 - mval) * that * float(that @ grad_i)

# prediction du puits diagonal (q=2)
xs = np.linspace(1e-4, 8, 400000)
Fq2 = ZETA * np.tanh(xs) / m_exp(np.sqrt(2) * AB * np.tanh(xs)) - xs
sgn = np.where(np.diff(np.sign(Fq2)))[0]
roots = [float(xs[i]) for i in sgn]
x_pred = roots[-1]
print(f"  puits diagonal predit (m_exp, q=2) : x={x_pred:.4f} -> ||h*||={x_pred*np.sqrt(2):.4f}"
      f"  (selle : {roots[0]:.4f})")

# (3a) bassin dynamique sur Delta (d=1) sous N(0,0.6), theta0=0 : m_exp vs rationnel
def basin_frac(m, n=400, seed=1):
    rr = np.random.default_rng(seed)
    act = 0
    for x0 in rr.normal(0, 0.6, n):
        xf, _, _ = simulate_reduced(m, np.tanh, x0=float(x0), steps=30000)
        act += int(abs(xf) > 0.2)
    return act / n
pa_rat = basin_frac(M_FAMILY["rationnel(l=2)"])
pa_exp = basin_frac(m_exp)
print(f"  (3a) P(actif | N(0,0.6), d=1) : rationnel={pa_rat:.3f}  exp={pa_exp:.3f}"
      f"  -> meme trichotomie, bassins NON universels")
REPORT["part3a"] = {"p_active_rational": pa_rat, "p_active_exp": pa_exp}

# (3b) reseau initialise AU puits predit + bruit : doit y rester (stabilite reseau)
orig = IWSEngine._metric_action
IWSEngine._metric_action = metric_action_exp
res = []
for seed in (7, 11, 19):
    cfg = IWSConfig.core01(steps=2500, graph_mode="circulant")
    eng = IWSEngine(cfg, seed=seed)
    rngp = np.random.default_rng(seed)
    eng.H = np.tile([x_pred, x_pred], (20, 1)) + rngp.normal(0, 0.1, (20, 2))
    eng.V = np.zeros((20, 2))
    eng.tau = np.tile((ALPHA/BETA)*np.tanh([x_pred, x_pred]), (20, 1))
    eng.P = np.zeros(20)
    eng.run()
    hn = float(np.mean(np.linalg.norm(eng.H, axis=1)))
    res.append({"seed": seed, "h_norm": hn,
                "o6": float(eng.history["h_var"][-1]),
                "err_puits": abs(hn - x_pred*np.sqrt(2))})
    print(f"  (3b) seed {seed}: ||h||_final={hn:.4f} (predit {x_pred*np.sqrt(2):.4f}, "
          f"ecart {res[-1]['err_puits']:.4f})  O6={res[-1]['o6']:.2e}")
IWSEngine._metric_action = orig
REPORT["part3b"] = res

with open(OUTJ, "w") as f:
    json.dump(REPORT, f, indent=2, default=str)
print(f"\n-> {OUTJ}")
