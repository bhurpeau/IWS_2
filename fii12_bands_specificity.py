"""F-II.1.2 -- Bandes de phase et specificite de la memoire d'interface.

PART 1  theorie des bandes : la seconde exposition = perturbation de la carte.
        Predicteur : x_lin(D + t_c) extrapole par la dynamique LIBRE linearisee
        (spirale avec theta(t) = theta_d e^{-beta t}), t_c = temps de charge de
        l'echelle jusqu'a theta_ign (mesure une fois), A_crit calibre sur UNE
        frontiere de bande ; prediction du motif complet + mecanisme d'asymetrie
        (capture par la resistance pulsee du cote negatif).
PART 2  trois controles de l'amorcage :
        (a) gain constant vs boucle dynamique ; (b) charge livree egale Q_net ;
        (c) signe c_s < 0 (habituation).
PART 3  specificite : modele vectoriel d=2 ; s scalaire -> amorcage GLOBAL
        (confirmation de la limite) ; s directionnel + regle de compatibilite
        kappa_Q(1 + c_s <s, chi_hat>) -> table meme/orthogonal/oppose.
"""
from __future__ import annotations

import json
import numpy as np

ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
W = float(np.tanh(2.2959))
TH_IGN = float(np.sqrt((1 - ZETA) / (3 * ZETA - 1))) / np.sqrt(2)
LCHI, KQ, RCHI, KCHI, K3, THK, RP = 0.2, 0.6, 0.8, 2.5, 0.5, 1.25, 0.35
m_of = lambda n: (1 + n**2) / (1 + 3 * n**2)
OUT = {}
G2, T_SUB = 0.18, 41.9

def run_scalar(g_sched, T, state=(1e-3, 0, 0, 0, 0), dt=0.01, ret_final=True):
    x, v, th, chi, p = state
    t = 0.0
    for _ in range(int(T / dt)):
        g = g_sched(t)
        n = np.sqrt(2) * abs(th)
        dv = -(G0 + G1 * n) * v - m_of(n) * x + ZETA * np.tanh(x)
        x, v = x + dt * v, v + dt * dv
        th = th + dt * (ALPHA * np.tanh(x) - BETA * th)
        chi = chi + dt * (g * W - LCHI * chi)
        p = p + dt * (0.6 * abs(x) + 0.4 * abs(v) + KCHI * np.sqrt(2) * abs(chi) - K3 * p)
        if p > THK:
            th += KQ * chi; chi *= (1 - RCHI); p *= RP
        t += dt
    return (x, v, th, chi, p)

# ================================================================ PART 1
print("PART 1 -- theorie des bandes")
# etat au decouplage de l'exposition 1
sd = run_scalar(lambda t: G2, T_SUB)
x_d, v_d, th_d = sd[0], sd[1], sd[2]
print(f"  decouplage : x_d={x_d:+.4f} v_d={v_d:+.4f} theta_d={th_d:.4f}")

# t_c : temps de charge de l'echelle jusqu'a theta_ign (depuis le repos, g=0.18)
chi, p, th, t, dt = 0.0, 0.0, 0.0, 0.0, 0.002
while th < TH_IGN:
    chi = chi + dt * (G2 * W - LCHI * chi)
    p = p + dt * (KCHI * np.sqrt(2) * abs(chi) - K3 * p)
    th = th + dt * (-BETA * th)
    if p > THK:
        th += KQ * chi; chi *= (1 - RCHI); p *= RP
    t += dt
t_c = t
print(f"  t_c (charge jusqu'a theta_ign) = {t_c:.2f}")

# extrapolation LIBRE linearisee depuis le decouplage : theta(t)=th_d e^{-bt}
def x_free(t_end, dt=0.002):
    x, v, th, t = x_d, v_d, th_d, 0.0
    traj = []
    while t < t_end:
        n = np.sqrt(2) * abs(th)
        a = -(G0 + G1 * n) * v - (m_of(n) - ZETA) * x
        x, v = x + dt * v, v + dt * a
        th = th + dt * (-BETA * th)
        traj.append((t, x, v))
        t += dt
    return np.array(traj)

traj = x_free(66 + t_c)
def x_at(t):
    i = int(t / 0.002)
    return traj[min(i, len(traj) - 1), 1]

# calibrage de A_crit sur UNE frontiere : D=20 (S) / D=22 (.)
A_lo, A_hi = x_at(22 + t_c), x_at(20 + t_c)
A_crit = 0.5 * (A_lo + A_hi)
print(f"  A_crit calibre sur la frontiere 20/22 : {A_crit:+.5f}")

# prediction du motif D=10..60 (hors regime de charge D<10)
meas = "S.....SSSS....SSS....SSS......"      # D=2..60 pas 2 (F-II.1.1)
pred, hits, tot = "", 0, 0
for i, D in enumerate(range(2, 62, 2)):
    if D < 10:
        pred += "?"
        continue
    yes = x_at(D + t_c) > A_crit
    pred += "S" if yes else "."
    tot += 1
    hits += int(("S" if yes else ".") == meas[i])
print(f"  mesure  : [{meas}]")
print(f"  predit  : [{pred}]   score (D>=10) : {hits}/{tot}")
OUT["bands"] = {"t_c": t_c, "A_crit": A_crit, "meas": meas, "pred": pred,
                "score": f"{hits}/{tot}"}

# mecanisme d'asymetrie : ou atterrissent les sommations ? et les echecs ?
def summed_final(D):
    sch = lambda t, D=D: G2 if (t < T_SUB or T_SUB + D <= t < 2 * T_SUB + D) else 0.0
    return run_scalar(sch, 2 * T_SUB + D + 400.0)[0]
signs = {D: summed_final(float(D)) for D in (14, 20, 32, 44, 26, 38)}
print("  atterrissages : " + "  ".join(f"D={D}:{x:+.2f}" for D, x in signs.items()))
OUT["landings"] = {str(k): float(v) for k, v in signs.items()}

# ================================================================ PART 2
print("PART 2 -- trois controles de l'amorcage")
EPS_S = 0.005
def run_s(g_sched, T, s0=0.0, cs=8.0, eps_s=EPS_S, dt=0.01, track_Q=False):
    x, v, th, chi, p, sl, t, Q = 1e-3, 0.0, 0.0, 0.0, 0.0, s0, 0.0, 0.0
    for _ in range(int(T / dt)):
        g = g_sched(t)
        n = np.sqrt(2) * abs(th)
        dv = -(G0 + G1 * n) * v - m_of(n) * x + ZETA * np.tanh(x)
        x, v = x + dt * v, v + dt * dv
        th = th + dt * (ALPHA * np.tanh(x) - BETA * th)
        chi = chi + dt * (g * W - LCHI * chi)
        sl = sl + dt * eps_s * (chi - sl)
        p = p + dt * (0.6 * abs(x) + 0.4 * abs(v) + KCHI * np.sqrt(2) * abs(chi) - K3 * p)
        if p > THK:
            q = KQ * (1 + cs * sl) * chi
            th += q; Q += q; chi *= (1 - RCHI); p *= RP
        t += dt
    return (x, sl, Q)

S_RES = 0.0095
def bands_count(s0, cs, eps_s=EPS_S):
    n = 0
    firsts = None
    for T2 in range(40, 160, 5):
        xf, _, _ = run_s(lambda t, T2=T2: G2 if t < T2 else 0.0, T2 + 400.0,
                         s0=s0, cs=cs, eps_s=eps_s)
        if abs(xf) > 1:
            n += 1
            if firsts is None: firsts = T2
    return n, firsts

n_ref, f_ref = bands_count(0.0, 8.0)                 # non amorce, boucle active
n_pri, f_pri = bands_count(S_RES, 8.0)               # amorce, boucle active
n_cst, f_cst = bands_count(S_RES, 8.0, eps_s=0.0)    # (a) gain constant fige
print(f"  (a) cellules/1ere duree : non-amorce {n_ref}/{f_ref} ; "
      f"amorce+boucle {n_pri}/{f_pri} ; amorce gain-constant {n_cst}/{f_cst}")
# (b) charge livree egale : Q_net a deux delais de meme dose, issues opposees
def two_pulse_Q(D):
    sch = lambda t, D=D: G2 if (t < T_SUB or T_SUB + D <= t < 2 * T_SUB + D) else 0.0
    xf, _, Q = run_s(sch, 2 * T_SUB + D + 400.0, s0=0.0, cs=0.0)
    return xf, Q
for D in (14.0, 26.0):
    xf, Q = two_pulse_Q(D)
    print(f"  (b) D={D}: Q_net={Q:.3f}  issue={'appropriee' if abs(xf)>1 else 'oubliee'}")
# (c) habituation : c_s < 0
n_hab, f_hab = bands_count(S_RES, -8.0)
print(f"  (c) c_s=-8 (habituation) : cellules {n_hab} (1ere duree {f_hab}) "
      f"vs non-amorce {n_ref}")
OUT["controls"] = {"ref": [n_ref, f_ref], "primed_loop": [n_pri, f_pri],
                   "primed_const": [n_cst, f_cst], "habituation": [n_hab, f_hab]}

# ================================================================ PART 3
print("PART 3 -- specificite (modele vectoriel d=2)")
U_DIAG = np.array([1.0, 1.0]) / np.sqrt(2)
U_ORTH = np.array([1.0, -1.0]) / np.sqrt(2)

def run_vec(dir1, T1, gap, dir2, T2, s_mode, cs=8.0, dt=0.01):
    """s_mode: 'scalar' (norme) ou 'vector' (compatibilite <s, chi_hat>)."""
    x = np.array([1e-3, 1e-3]) / np.sqrt(2)
    v = np.zeros(2); th = np.zeros(2); chi = np.zeros(2)
    s_sc, s_vec = 0.0, np.zeros(2)
    p, t = 0.0, 0.0
    Ttot = T1 + gap + T2 + 400.0
    for _ in range(int(Ttot / dt)):
        if t < T1: g, u = G2, dir1
        elif t < T1 + gap: g, u = 0.0, dir1
        elif t < T1 + gap + T2: g, u = G2, dir2
        else: g, u = 0.0, dir2
        nt = np.linalg.norm(th)
        c = 2.0 / (1 + 3 * nt**2)
        Mx = x - c * float(th @ x) * th
        dv = -(G0 + G1 * nt) * v - Mx + ZETA * np.tanh(x)
        x, v = x + dt * v, v + dt * dv
        th = th + dt * (ALPHA * np.tanh(x) - BETA * th)
        chi = chi + dt * (g * W * u - LCHI * chi)
        s_sc = s_sc + dt * EPS_S * (np.linalg.norm(chi) - s_sc)
        s_vec = s_vec + dt * EPS_S * (chi - s_vec)
        p = p + dt * (0.6 * np.linalg.norm(x) + 0.4 * np.linalg.norm(v)
                      + KCHI * np.linalg.norm(chi) * np.sqrt(2) - K3 * p)
        if p > THK:
            nc = np.linalg.norm(chi)
            if s_mode == "scalar":
                boost = 1 + cs * s_sc
            else:
                boost = 1 + cs * float(s_vec @ chi) / (nc + 1e-12)
            th = th + max(boost, 0.0) * KQ * chi
            chi = chi * (1 - RCHI); p *= RP
        t += dt
    return float(np.linalg.norm(x))

def cells_vec(dir1, dir2, s_mode, prime=True):
    n = 0
    T1 = T_SUB if prime else 0.0
    gap = 300.0 if prime else 0.0
    for T2 in range(40, 160, 15):
        if run_vec(dir1, T1, gap, dir2, float(T2), s_mode) > 1:
            n += 1
    return n

base = cells_vec(U_DIAG, U_DIAG, "scalar", prime=False)
print(f"  base (sans amorce) : {base}/8 cellules")
for mode in ("scalar", "vector"):
    same = cells_vec(U_DIAG, U_DIAG, mode)
    orth = cells_vec(U_DIAG, U_ORTH, mode)
    oppo = cells_vec(U_DIAG, -U_DIAG, mode)
    print(f"  s {mode:6s} : meme={same}/8  orthogonal={orth}/8  oppose={oppo}/8")
    OUT.setdefault("specificity", {})[mode] = {"same": same, "orth": orth,
                                               "oppo": oppo, "base": base}

with open("output_theory/fii12_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fii12_report.json")
