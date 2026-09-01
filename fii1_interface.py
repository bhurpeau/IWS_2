"""F-II.1 -- la plus petite interface qui transforme une influence en histoire.

PART 0  corrections F-II.0.1 :
        (a) a3 = zeta/3 + (1/2) m''(tau_ign) (sqrt2 alpha/beta)^2 (terme dominant) ;
            verification : branche a l'ordre 2 vs continuation exacte ;
        (b) au-dela de g_H la branche persiste INSTABLE (rupture, pas mort) ;
        (c) g_ign depend du protocole : montee progressive vs exposition brutale.
PART 1  R1 (permeabilite dynamique)  q' = rho(1-q) - eta q g w :
        g_eff(g) = rho g/(rho + eta g w) sature ;
        g_c^{R1}, g_H^{R1} par transport de g_eff ; protection absolue si
        rho/(eta w) < g_c^0 ; T*(g) recalcule.
PART 2  R2 (validation evenementielle) : chi accumule a l'interface, transfert
        quantifie a tau aux Kairos (pression d'exposition) :
        (a) N_K*(g) : nombre minimal de Kairos pour l'appropriation (escalier) ;
        (b) sommation d'expositions sous-critiques (fenetre de memoire) ;
        (c) amorcage : une exposition oubliee abaisse T* de la suivante.
"""
from __future__ import annotations

import json
import numpy as np

ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
X_A = 2.2959
W = float(np.tanh(X_A))
TAU_IGN = float(np.sqrt((1 - ZETA) / (3 * ZETA - 1)))
G_C0 = BETA * TAU_IGN / (np.sqrt(2) * W)
G_H0 = 0.1285
X_SADDLE0 = 0.0717
OUT = {}

m_of = lambda n: (1 + n**2) / (1 + 3 * n**2)
mp = lambda n: -4 * n / (1 + 3 * n**2) ** 2
mpp = lambda n: -4 * (1 - 9 * n**2) / (1 + 3 * n**2) ** 3

def theta_star(x, g): return (ALPHA * np.tanh(x) + g * W) / BETA
def F_g(x, g):
    n = np.sqrt(2) * np.abs(theta_star(x, g))
    return m_of(n) * x - ZETA * np.tanh(x)
def res_root(g, lo=-0.5, hi=-1e-5):
    xs = np.linspace(lo, hi, 200001); v = F_g(xs, g)
    i = np.where(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    return float(xs[i[-1]]) if len(i) else None

# ---------------------------------------------------------------- PART 0
print("PART 0 -- corrections")
a1 = np.sqrt(2) * mp(TAU_IGN) * W / BETA
a2 = np.sqrt(2) * mp(TAU_IGN) * ALPHA / BETA
a3 = ZETA / 3 + 0.5 * mpp(TAU_IGN) * (np.sqrt(2) * ALPHA / BETA) ** 2
print(f"  (a) a1={a1:.3f} a2={a2:.3f} a3={a3:.3f} "
      f"(zeta/3={ZETA/3:.3f} ; terme m'' = {a3-ZETA/3:.3f} DOMINANT)")
for gmul in (2.0, 3.0):
    d = (gmul - 1) * G_C0
    x_lin = a1 * d / (-a2)
    disc = a2**2 - 4 * a3 * a1 * d
    x_quad = (-a2 - np.sqrt(disc)) / (2 * a3)
    x_ex = res_root(gmul * G_C0)
    print(f"      g={gmul}g_c : lineaire {x_lin:+.4f} | ordre 2 {x_quad:+.4f} "
          f"| exact {x_ex:+.4f}")
OUT["a3"] = a3

def field(y, g):
    x, v, th = y
    n = np.sqrt(2) * abs(th)
    return np.array([v, -(G0 + G1 * n) * v - m_of(n) * x + ZETA * np.tanh(x),
                     ALPHA * np.tanh(x) + g * W - BETA * th])

# (b) persistance instable au-dela de g_H
def jac3(y, g, h=1e-7):
    J = np.zeros((3, 3)); f0 = field(y, g)
    for k in range(3):
        yp = y.copy(); yp[k] += h
        J[:, k] = (field(yp, g) - f0) / h
    return J
r = res_root(0.145)
ev = np.linalg.eigvals(jac3(np.array([r, 0, theta_star(r, 0.145)]), 0.145))
print(f"  (b) g=0.145 > g_H : racine resistante persiste x={r:+.4f}, "
      f"max Re = {np.max(ev.real):+.4f} (INSTABLE, pas disparue)")

# (c) dependance au protocole : rampe vs brutal, cible g=0.10
def run_ramp(g_target, T_ramp, T_hold=800.0, dt=0.01):
    y = np.array([1e-3, 0.0, 0.0]); t = 0.0
    while t < T_ramp + T_hold:
        g = g_target * min(1.0, t / T_ramp) if T_ramp > 0 else g_target
        y = y + dt * field(y, g); t += dt
    return y[0]
for T_ramp in (0.0, 20.0, 400.0):
    xf = run_ramp(0.10, T_ramp)
    print(f"  (c) g->0.10, rampe T={T_ramp:5.0f} : x(inf)={xf:+.4f} "
          f"-> {'APPROPRIATION' if xf > 1 else 'RESISTANCE'}")
OUT["ramp"] = {str(T): float(run_ramp(0.10, T)) for T in (0.0, 20.0, 400.0)}

# ---------------------------------------------------------------- PART 1
print("PART 1 -- R1 : permeabilite dynamique  q' = rho(1-q) - eta q g w")
RHO, ETA = 0.05, 1.0
g_eff = lambda g: RHO * g / (RHO + ETA * g * W)
g_max_eff = RHO / (ETA * W)
print(f"  rho={RHO}, eta={ETA} : g_eff sature a {g_max_eff:.4f} "
      f"({'PROTECTION ABSOLUE (< g_c0)' if g_max_eff < G_C0 else '> g_c0'})")
# transport des seuils
def invert_geff(target):
    if target >= g_max_eff: return None
    return RHO * target / (RHO - ETA * W * target)
gc1, gh1 = invert_geff(G_C0), invert_geff(G_H0)
print(f"  g_c^R1 = {gc1 if gc1 else 'jamais'}   g_H^R1 = {gh1 if gh1 else 'jamais'}")
OUT["R1"] = {"rho": RHO, "eta": ETA, "g_max_eff": g_max_eff,
             "g_c_R1": gc1, "g_H_R1": gh1}

def field_q(y, g):
    x, v, th, q = y
    n = np.sqrt(2) * abs(th)
    return np.array([v, -(G0 + G1 * n) * v - m_of(n) * x + ZETA * np.tanh(x),
                     ALPHA * np.tanh(x) + q * g * W - BETA * th,
                     RHO * (1 - q) - ETA * q * g * W])

# verification dynamique de g_c^R1 (bisection sur l'allumage au contact permanent)
def ignites_R1(g, steps=500000, dt=0.01):
    y = np.array([1e-3, 0.0, 0.0, 1.0])
    for _ in range(steps):
        y = y + dt * field_q(y, g)
    return abs(y[0]) > 0.02
if gc1:
    lo, hi = 0.6 * gc1, 2.5 * gc1
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if ignites_R1(mid): hi = mid
        else: lo = mid
    print(f"  g_c^R1 dynamique = {0.5*(lo+hi):.4f} (formule : {gc1:.4f})")

# T*(g) recalcule sous R1
def T_star_R1(g, dt=0.01):
    def appro(T):
        y = np.array([1e-3, 0.0, 0.0, 1.0])
        for _ in range(int(T / dt)):
            y = y + dt * field_q(y, g)
        for _ in range(int(400 / dt)):
            y = y + dt * np.append(field(y[:3], 0.0), RHO * (1 - y[3]))
        return abs(y[0]) > 1.0
    if not appro(400.0): return None
    lo_, hi_ = 1.0, 400.0
    for _ in range(14):
        mid = 0.5 * (lo_ + hi_)
        if appro(mid): hi_ = mid
        else: lo_ = mid
    return 0.5 * (lo_ + hi_)
print("  T*(g) sous R1 (vs R0)  [R0 : 24.8 a g=0.10 ; 15.0 a 0.20 ; 12.3 a 0.30]")
for g in (0.10, 0.20, 0.30):
    t1 = T_star_R1(g)
    print(f"    g={g:.2f} : T*_R1 = {round(t1,1) if t1 else 'jamais'}")
    OUT.setdefault("R1_Tstar", {})[g] = t1

# ---------------------------------------------------------------- PART 2
print("PART 2 -- R2 : validation evenementielle (quanta d'integration)")
LCHI, KQ, RCHI = 0.2, 0.6, 0.8
KCHI, K3, THK, RP = 2.5, 0.5, 1.25, 0.35

def run_R2(g_sched, T_total, dt=0.01, seed_x=1e-3):
    """g_sched(t) -> g ; retourne (x_f, n_kairos, traj tau norme, temps kairos)."""
    x, v, th, chi, p = seed_x, 0.0, 0.0, 0.0, 0.0
    nk, taus, tks, t = 0, [], [], 0.0
    n_steps = int(T_total / dt)
    for k in range(n_steps):
        g = g_sched(t)
        n = np.sqrt(2) * abs(th)
        dv = -(G0 + G1 * n) * v - m_of(n) * x + ZETA * np.tanh(x)
        x, v = x + dt * v, v + dt * dv
        th = th + dt * (ALPHA * np.tanh(x) - BETA * th)          # PAS d'ecriture directe
        chi = chi + dt * (g * W - LCHI * chi)                    # accumulation interface
        p = p + dt * (0.6 * abs(x) + 0.4 * abs(v) + KCHI * np.sqrt(2) * abs(chi) - K3 * p)
        if p > THK:                                              # Kairos : validation
            th += KQ * chi
            chi *= (1 - RCHI)
            p *= RP
            nk += 1; tks.append(t)
        if k % 100 == 0: taus.append(np.sqrt(2) * abs(th))
        t += dt
    return x, nk, np.array(taus), tks

# (a) N_K* : contact permanent, compter les Kairos avant franchissement
print("  (a) quantification : Kairos avant appropriation (contact permanent)")
for g in (0.06, 0.10, 0.20):
    xf, nk, taus, tks = run_R2(lambda t: g, 600.0)
    # nombre d'evenements avant que x depasse la selle
    print(f"    g={g:.2f} : x_final={xf:+.3f}  Kairos={nk}  "
          f"tau max={taus.max():.3f} (tau_ign={TAU_IGN:.3f})  "
          f"-> {'APPROPRIE' if abs(xf) > 1 else 'non'}")
    OUT.setdefault("R2_NK", {})[g] = {"x_final": float(xf), "n_kairos": nk,
                                       "tau_max": float(taus.max())}

# (b) sommation sous-critique : deux expositions T_sub separees d'un delai D
print("  (b) sommation de deux expositions sous-critiques (g=0.10)")
def two_pulse(T_sub, D):
    sched = lambda t: 0.10 if (t < T_sub or T_sub + D <= t < 2 * T_sub + D) else 0.0
    xf, nk, _, _ = run_R2(sched, 2 * T_sub + D + 400.0)
    return abs(xf) > 1
# T_sub choisi sous-critique seul :
T_sub = 25.0
x1, nk1, _, _ = run_R2(lambda t: 0.10 if t < T_sub else 0.0, T_sub + 400.0)
print(f"    une exposition T={T_sub} seule : appropriee = {abs(x1) > 1}")
for D in (5.0, 30.0, 80.0, 200.0):
    print(f"    deux expositions T={T_sub}, delai D={D:5.1f} : "
          f"appropriee = {two_pulse(T_sub, D)}")

# (c) amorcage : T* de la seconde exposition apres une premiere oubliee
print("  (c) amorcage (une exposition sous-critique prealable, D=100)")
def Tstar_after_prime(prime_T, D, g=0.10):
    def appro(T2):
        sched = lambda t: g if (t < prime_T or prime_T + D <= t < prime_T + D + T2) else 0.0
        xf, _, _, _ = run_R2(sched, prime_T + D + T2 + 400.0)
        return abs(xf) > 1
    lo_, hi_ = 1.0, 200.0
    if not appro(200.0): return None
    for _ in range(12):
        mid = 0.5 * (lo_ + hi_)
        if appro(mid): hi_ = mid
        else: lo_ = mid
    return 0.5 * (lo_ + hi_)
t_noprime = Tstar_after_prime(0.0, 0.0)
t_prime = Tstar_after_prime(20.0, 100.0)
print(f"    T* sans amorce = {round(t_noprime,1) if t_noprime else 'jamais'} ; "
      f"T* apres amorce (20 u., oubliee 100 u.) = "
      f"{round(t_prime,1) if t_prime else 'jamais'}")
OUT["R2_priming"] = {"no_prime": t_noprime, "primed": t_prime}

with open("output_theory/fii1_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fii1_report.json")
