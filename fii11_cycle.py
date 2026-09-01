"""F-II.1.1 -- Theorie du cycle evenementiel de l'interface (R2).

PART A  cycle au repos (x=0 invariant) : l'application (chi,p) -> cycle periodique
        Delta(g) ; echelle de trace pulsee tau_n -> tau_inf = kQ chi*/(1-e^{-b Delta}) ;
        moyenne de barreau  theta_bar = kQ chi*/(beta Delta) ;
        seuils :  g_K (formule exacte)  et  g_F (quasi-statique + Floquet exact).
PART B  resistance pulsee : capture, pinning pulse m(sqrt2 theta_bar) ~ zeta ;
        g_H^{R2} par continuation dynamique (rampe lente depuis l'attracteur).
PART C  fenetres de sommation : scan fin en D (g=0.18) + diagnostics d'etat a
        l'arrivee de la seconde exposition (phase (x,v), p, chi, theta).
PART D  QO-34 : peau qui se souvient -- variable lente s, kQ(s) ;
        experience decisive T2* < T1* apres oubli complet de (tau,chi,p,x,v).
"""
from __future__ import annotations

import json
import numpy as np

ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
W = float(np.tanh(2.2959))
TAU_IGN = float(np.sqrt((1 - ZETA) / (3 * ZETA - 1)))     # seuil en NORME
TH_IGN = TAU_IGN / np.sqrt(2)                              # par composante
LCHI, KQ, RCHI, KCHI, K3, THK, RP = 0.2, 0.6, 0.8, 2.5, 0.5, 1.25, 0.35
m_of = lambda n: (1 + n**2) / (1 + 3 * n**2)
OUT = {}

def step_R2(state, g, dt=0.01, kq=KQ):
    x, v, th, chi, p = state
    n = np.sqrt(2) * abs(th)
    dv = -(G0 + G1 * n) * v - m_of(n) * x + ZETA * np.tanh(x)
    x, v = x + dt * v, v + dt * dv
    th = th + dt * (ALPHA * np.tanh(x) - BETA * th)
    chi = chi + dt * (g * W - LCHI * chi)
    p = p + dt * (0.6 * abs(x) + 0.4 * abs(v) + KCHI * np.sqrt(2) * abs(chi) - K3 * p)
    fired = p > THK
    if fired:
        th += kq * chi; chi *= (1 - RCHI); p *= RP
    return (x, v, th, chi, p), fired

# ================================================================ PART A
print("PART A -- cycle au repos et seuils g_K, g_F")
g_K = THK * K3 * LCHI / (KCHI * np.sqrt(2) * W)
print(f"  g_K (formule) = Theta k3 lchi/(kchi sqrt2 w) = {g_K:.5f}")

def rest_cycle(g, warm=400.0, dt=0.002):
    """Cycle (chi,p,theta) a x=0 exactement ; renvoie Delta, chi-, theta_bar."""
    chi, p, th = 0.0, 0.0, 0.0
    t, events = 0.0, []
    chi_minus = []
    ths = []
    while t < warm:
        chi = chi + dt * (g * W - LCHI * chi)
        p = p + dt * (KCHI * np.sqrt(2) * abs(chi) - K3 * p)
        th = th + dt * (-BETA * th)
        if p > THK:
            events.append(t); chi_minus.append(chi)
            th += KQ * chi; chi *= (1 - RCHI); p *= RP
        if t > warm * 0.6:
            ths.append(th)
        t += dt
    if len(events) < 6:
        return None
    Delta = float(np.mean(np.diff(events[-8:])))
    return Delta, float(np.mean(chi_minus[-8:])), float(np.mean(ths))

for g in (0.05, 0.08, 0.10, 0.16):
    rc = rest_cycle(g)
    if rc is None:
        print(f"  g={g}: pas de cycle (g < g_K)"); continue
    Delta, chim, thbar = rc
    th_ladder = KQ * chim / (BETA * Delta)                 # prediction moyenne
    print(f"  g={g:.2f}: Delta={Delta:6.3f}  chi-={chim:.4f}  "
          f"theta_bar mesure={thbar:.4f} / predit kQ chi-/(b Delta)={th_ladder:.4f}"
          f"  m(sqrt2 th)={m_of(np.sqrt(2)*thbar):.4f}")
    OUT.setdefault("cycles", {})[g] = {"Delta": Delta, "chi_minus": chim,
                                       "theta_bar": thbar, "ladder_pred": th_ladder}

# g_F quasi-statique : theta_bar(g) = TH_IGN
def theta_bar(g):
    rc = rest_cycle(g)
    return rc[2] if rc else 0.0
lo, hi = g_K * 1.02, 0.16
for _ in range(18):
    mid = 0.5 * (lo + hi)
    if theta_bar(mid) > TH_IGN: hi = mid
    else: lo = mid
g_F_qs = 0.5 * (lo + hi)
print(f"  g_F (quasi-statique, theta_bar = th_ign) = {g_F_qs:.4f}")

# g_F exact : multiplicateur de Floquet du repos sur le cycle
def floquet_rest(g, dt=0.002):
    """Monodromie (x,v) sur un cycle converge (theta(t) en dents de scie)."""
    chi, p, th = 0.0, 0.0, 0.0
    t, last_ev, evs = 0.0, None, 0
    # convergence
    while evs < 60:
        chi = chi + dt * (g * W - LCHI * chi)
        p = p + dt * (KCHI * np.sqrt(2) * abs(chi) - K3 * p)
        th = th + dt * (-BETA * th)
        if p > THK:
            th += KQ * chi; chi *= (1 - RCHI); p *= RP
            evs += 1; last_ev = t
        t += dt
    # une periode : integrer (theta(t)) et la monodromie
    M = np.eye(2)
    fired = False
    while not fired:
        n = np.sqrt(2) * abs(th)
        A = np.array([[0.0, 1.0], [ZETA - m_of(n), -(G0 + G1 * n)]])
        M = M + dt * (A @ M)
        chi = chi + dt * (g * W - LCHI * chi)
        p = p + dt * (KCHI * np.sqrt(2) * abs(chi) - K3 * p)
        th = th + dt * (-BETA * th)
        if p > THK:
            th += KQ * chi; chi *= (1 - RCHI); p *= RP
            fired = True
    return float(np.max(np.abs(np.linalg.eigvals(M))))

lo, hi = g_K * 1.02, 0.16
for _ in range(16):
    mid = 0.5 * (lo + hi)
    if floquet_rest(mid) > 1.0: hi = mid
    else: lo = mid
g_F = 0.5 * (lo + hi)
print(f"  g_F (Floquet exact) = {g_F:.4f}   "
      f"[hierarchie attendue : g_K={g_K:.4f} < g_F < g_app=0.1632]")
OUT["g_K"], OUT["g_F_qs"], OUT["g_F"] = g_K, g_F_qs, g_F

# ================================================================ PART B
print("PART B -- resistance pulsee et g_H^R2")
def run(g_sched, T, state=(1e-3, 0.0, 0.0, 0.0, 0.0), dt=0.01, kq=KQ):
    s, t = tuple(state), 0.0
    for _ in range(int(T / dt)):
        s, _ = step_R2(s, g_sched(t), dt, kq)
        t += dt
    return s

# capture entre g_F et g_app : caracteriser l'attracteur pulse
s_end = run(lambda t: 0.12, 800.0)
xs, ths = [], []
s = s_end
for k in range(40000):
    s, _ = step_R2(s, 0.12)
    if k % 20 == 0: xs.append(s[0]); ths.append(s[2])
xbar, thbar = float(np.mean(xs)), float(np.mean(ths))
print(f"  g=0.12 : resistance pulsee x_bar={xbar:+.4f}  "
      f"m(sqrt2 theta_bar)={m_of(np.sqrt(2)*thbar):.5f} (zeta={ZETA})"
      f"  -> pinning pulse")
OUT["pulsed_res"] = {"x_bar": xbar, "m_at_thbar": m_of(np.sqrt(2) * thbar)}

# g_H^R2 : rampe lente depuis l'attracteur pulse (continuation dynamique)
s = s_end; g_now, dt = 0.12, 0.01
while g_now < 0.30:
    s, _ = step_R2(s, g_now, dt)
    g_now += 2e-6 * dt / 0.01          # rampe tres lente
    if abs(s[0]) > 1.0:
        break
print(f"  g_H^R2 (rampe lente depuis la resistance pulsee) ~ {g_now:.4f}")
OUT["g_H_R2"] = g_now

# ================================================================ PART C
print("PART C -- fenetres de sommation : scan fin en D (g=0.18, T_sub=41.9)")
g2, T_sub = 0.18, 41.9
def arrival_state(D):
    return run(lambda t: g2 if t < T_sub else 0.0, T_sub + D)
def summed(D):
    sch = lambda t: g2 if (t < T_sub or T_sub + D <= t < 2 * T_sub + D) else 0.0
    s = run(sch, 2 * T_sub + D + 400.0)
    return abs(s[0]) > 1.0
pattern, diags = "", {}
for D in range(2, 62, 2):
    ok = summed(float(D))
    pattern += "S" if ok else "."
    if D % 6 == 2:
        x, v, th, chi, p = arrival_state(float(D))
        diags[D] = {"ok": ok, "x": x, "v": v, "phase": float(np.arctan2(v, x)),
                    "th": th, "chi": chi, "p": p}
print(f"  D=2..60 pas 2 : [{pattern}]")
for D, d in diags.items():
    print(f"    D={D:2d}: {'S' if d['ok'] else '.'}  x={d['x']:+.2e} v={d['v']:+.2e} "
          f"phase={d['phase']:+.2f}  p={d['p']:.3f}  chi={d['chi']:.3f}  th={d['th']:.3f}")
OUT["window_pattern"] = pattern

# ================================================================ PART D
print("PART D -- QO-34 : peau qui se souvient (variable lente s, kQ(s))")
EPS_S, CS = 0.005, 3.0
def run_s(g_sched, T, dt=0.01):
    x, v, th, chi, p, sl, t = 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(int(T / dt)):
        g = g_sched(t)
        n = np.sqrt(2) * abs(th)
        dv = -(G0 + G1 * n) * v - m_of(n) * x + ZETA * np.tanh(x)
        x, v = x + dt * v, v + dt * dv
        th = th + dt * (ALPHA * np.tanh(x) - BETA * th)
        chi = chi + dt * (g * W - LCHI * chi)
        sl = sl + dt * EPS_S * (chi - sl)
        p = p + dt * (0.6 * abs(x) + 0.4 * abs(v) + KCHI * np.sqrt(2) * abs(chi) - K3 * p)
        if p > THK:
            th += KQ * (1 + CS * sl) * chi; chi *= (1 - RCHI); p *= RP
        t += dt
    return x, v, th, chi, p, sl

def Tstar_s(prime_T, D, g=0.18):
    lo_, hi_ = 2.0, 300.0
    def ap(T2):
        sch = lambda t: g if (t < prime_T or prime_T + D <= t < prime_T + D + T2) else 0.0
        out = run_s(sch, prime_T + D + T2 + 400.0)
        return abs(out[0]) > 1.0
    if not ap(300.0): return None
    for _ in range(12):
        mid = 0.5 * (lo_ + hi_)
        if ap(mid): hi_ = mid
        else: lo_ = mid
    return 0.5 * (lo_ + hi_)

T1 = Tstar_s(0.0, 0.0)
# etat apres amorce + gap : tout est-il oublie sauf s ?
st = run_s(lambda t: 0.18 if t < 30.0 else 0.0, 30.0 + 300.0)
print(f"  apres amorce 30u + gap 300 : |x|={abs(st[0]):.1e} th={st[2]:.1e} "
      f"chi={st[3]:.1e} p={st[4]:.1e}  MAIS s={st[5]:.4f} (persiste)")
T2 = Tstar_s(30.0, 300.0)
print(f"  T1* (nu) = {T1:.1f}   T2* (apres amorce oubliee) = {T2:.1f}"
      f"   -> {'AMORCAGE (T2*<T1*)' if T2 and T1 and T2 < T1 else 'pas d amorcage'}")
# controle : sans memoire de peau (eps_s effectif nul via CS=0)
CS_save = CS; CS = 0.0
T2c = Tstar_s(30.0, 300.0)
CS = CS_save
print(f"  controle CS=0 : T2* = {T2c:.1f} (attendu : >= T1*, erosion de graine)")
OUT["priming"] = {"T1": T1, "T2_with_s": T2, "T2_control": T2c,
                  "s_residual": st[5]}

with open("output_theory/fii11_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fii11_report.json")
