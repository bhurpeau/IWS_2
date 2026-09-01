"""Tome II, note F-II.0 -- dyade IWS ouverte minimale (niveau Delta, d=2).

Couplage conforme aux axiomes d'interface :
  A-I1  B ne recoit qu'une projection saturee de l'etat de A : g*sigma(h_A)
        (jamais tau_A, jamais le graphe de A -- la memoire est privee) ;
  A-I4  l'interaction n'entre JAMAIS comme force sur H : elle s'ecrit dans tau.
        => tau_B' = alpha*sigma(h_B) + g*sigma(h_A) - beta*tau_B

Consequences calculees :
  II-1  invariance du repos : h_B=0 reste un equilibre quel que soit h_A ;
  II-2  destabilisation mnesique : le repos devient instable ssi
        m(||tau_B||) < zeta  <=>  ||tau_B|| > tau_ign = m^{-1}(zeta) ;
  II-3  contact permanent avec A actif : allumage ssi g > g_c = beta*tau_ign/||sigma(h_A)|| ;
  II-4  contact TRANSITOIRE de duree T : frontiere d'appropriation dans (g, T)
        (persistance apres decouplage : B rejoint SON puits autonome) ;
  II-5  competition (A,B actifs antipodaux, couplage mutuel) : reduction exacte
        alpha_eff = alpha - g  =>  seuil g_comp tel que zeta_min(alpha_eff/beta)=zeta ;
  II-6  alliance (memes puits) : alpha_eff = alpha + g => abaissement partage de zeta_min.
"""
from __future__ import annotations

import json
import numpy as np

ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
AB = ALPHA / BETA
X_WELL = 2.2959          # puits diagonal (T-1)
H_A = np.array([X_WELL, X_WELL])
OUT = {}

m_rat = lambda t: (1 + t**2) / (1 + 3 * t**2)

def Mdot(tau, h):
    n2 = float(tau @ tau)
    c = 2.0 / (1 + 3 * n2)          # lambda=2 : M = I - c tau tau^T
    return h - c * float(tau @ h) * tau

def dyad_field(y, gAB, gBA):
    """y = (hA, vA, tA, hB, vB, tB) in R^12 ; couplage trace sature."""
    hA, vA, tA = y[0:2], y[2:4], y[4:6]
    hB, vB, tB = y[6:8], y[8:10], y[10:12]
    dvA = -(G0 + G1 * np.linalg.norm(tA)) * vA - Mdot(tA, hA) + ZETA * np.tanh(hA)
    dvB = -(G0 + G1 * np.linalg.norm(tB)) * vB - Mdot(tB, hB) + ZETA * np.tanh(hB)
    dtA = ALPHA * np.tanh(hA) + gAB * np.tanh(hB) - BETA * tA
    dtB = ALPHA * np.tanh(hB) + gBA * np.tanh(hA) - BETA * tB
    return np.concatenate([vA, dvA, dtA, vB, dvB, dtB])

def integrate(y, gAB, gBA, steps, dt=0.01):
    for _ in range(steps):
        y = y + dt * dyad_field(y, gAB, gBA)   # Euler (champs doux ici)
    return y

def state(hA=None, tA=None, hB=None, tB=None):
    z = np.zeros(12)
    if hA is not None: z[0:2] = hA
    if tA is not None: z[4:6] = tA
    if hB is not None: z[6:8] = hB
    if tB is not None: z[10:12] = tB
    return z

# ---------------------------------------------------------------- II-1 / II-2
tau_ign = float(np.sqrt((1 - ZETA) / (3 * ZETA - 1)))   # m(t)=zeta pour m_rat
print(f"II-2  tau_ign = m^-1(zeta) = {tau_ign:.4f}")
y = state(hA=H_A, tA=AB * np.tanh(H_A), hB=[0, 0], tB=[0, 0])
yf = integrate(y, 0.0, 0.4, steps=40000)                # g fort, hB EXACTEMENT 0
print(f"II-1  invariance du repos : g=0.4, ||h_B(T)|| = {np.linalg.norm(yf[6:8]):.2e}"
      f"  (||tau_B|| = {np.linalg.norm(yf[10:12]):.3f} > tau_ign : paysage charge,"
      f" etat inerte -- l'exterieur ne pousse pas)")
OUT["tau_ign"] = tau_ign

# ---------------------------------------------------------------- II-3
g_c = BETA * tau_ign / float(np.linalg.norm(np.tanh(H_A)))
print(f"II-3  g_c = beta*tau_ign/||sigma(h_A)|| = {g_c:.5f}")
res = {}
for g in (0.5 * g_c, 0.9 * g_c, 1.1 * g_c, 2 * g_c, 5 * g_c):
    y = state(hA=H_A, tA=AB * np.tanh(H_A), hB=[1e-3, 1e-3], tB=[0, 0])
    yf = integrate(y, 0.0, g, steps=250000)
    hB = float(np.linalg.norm(yf[6:8]))
    res[round(g / g_c, 2)] = hB
    print(f"      g = {g/g_c:.2f} g_c : ||h_B(inf)|| = {hB:7.4f}"
          f"  -> {'ALLUME' if hB > 1 else 'eteint'}")
OUT["II3_gc"] = g_c
OUT["II3_scan"] = res

# ---------------------------------------------------------------- II-4
print("II-4  frontiere d'appropriation (contact transitoire de duree T, graine 1e-3)")
def appropriation(g, T, dt=0.01, after=300.0):
    y = state(hA=H_A, tA=AB * np.tanh(H_A), hB=[1e-3, 1e-3], tB=[0, 0])
    y = integrate(y, 0.0, g, steps=int(T / dt), dt=dt)      # contact
    y = integrate(y, 0.0, 0.0, steps=int(after / dt), dt=dt)  # decouplage
    return float(np.linalg.norm(y[6:8]))

grid = {}
for g in (0.05, 0.1, 0.2, 0.4):
    Ts, verdicts = (5, 10, 20, 40, 80, 160), []
    for T in Ts:
        hB = appropriation(g, T)
        verdicts.append("A" if hB > 1 else ".")
    # frontiere : premiere duree appropriee
    T_star = next((T for T, v in zip(Ts, verdicts) if v == "A"), None)
    grid[g] = {"pattern": "".join(verdicts), "T_star": T_star}
    print(f"      g = {g:4.2f} : T = {Ts} -> [{''.join(verdicts)}]  T* ~ {T_star}")
OUT["II4_boundary"] = {str(k): v for k, v in grid.items()}
# permanence : le puits atteint est le puits AUTONOME de B
hB_final = appropriation(0.2, 160)
print(f"      permanence : ||h_B|| apres decouplage long = {hB_final:.4f}"
      f"  (puits autonome = {np.linalg.norm([X_WELL, X_WELL]):.4f})")
OUT["II4_permanence"] = hB_final

# ---------------------------------------------------------------- II-5 / II-6
print("II-5  competition : reduction exacte alpha_eff = alpha - g")
u = np.linspace(1e-6, 1 - 1e-9, 200000)
def zmin_of_ab(ab):
    return float(np.min(np.arctanh(u) * m_rat(np.sqrt(2) * ab * u) / u))
lo, hi = 0.0, AB
for _ in range(60):                      # zeta_min((alpha-g)/beta) = zeta
    mid = 0.5 * (lo + hi)
    if zmin_of_ab(mid) > ZETA: lo = mid
    else: hi = mid
ab_c = hi
g_comp = ALPHA - BETA * ab_c
print(f"      ab_c = {ab_c:.3f}  ->  g_comp = alpha - beta*ab_c = {g_comp:.4f}")
for g in (0.5 * g_comp, 1.2 * g_comp):
    y = state(hA=H_A, tA=AB * np.tanh(H_A), hB=-H_A, tB=-AB * np.tanh(H_A))
    y[0:2] += 0.02; y[6:8] -= 0.013      # brisure legere de symetrie
    yf = integrate(y, g, g, steps=300000)
    sA, sB = np.linalg.norm(yf[0:2]), np.linalg.norm(yf[6:8])
    print(f"      g = {g:.4f} ({'<' if g < g_comp else '>'} g_comp) : "
          f"||h_A||={sA:.3f}, ||h_B||={sB:.3f} "
          f"-> {'COEXISTENCE' if min(sA, sB) > 1 else 'EXTINCTION MUTUELLE/CAPTURE'}")
    OUT.setdefault("II5", {})[round(g, 4)] = [float(sA), float(sB)]

print("II-6  alliance (memes puits) : alpha_eff = alpha + g")
for g in (0.0, 0.2):
    zm = zmin_of_ab((ALPHA + g) / BETA)
    print(f"      g = {g:.1f} : zeta_min effectif = {zm:.4f}"
          + ("" if g == 0 else "  (abaissement partage de la frontiere d'existence)"))
    OUT.setdefault("II6_zmin", {})[g] = zm

with open("output_theory/dyad_report.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("-> output_theory/dyad_report.json")
