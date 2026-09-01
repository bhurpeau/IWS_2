"""F-III.1 -- Ruptures de compressibilite et les trois strates de l'individu.

E1  Separation par l'histoire : deux individus (s1 != s2) INDISCERNABLES par
    tout episode-test de la grille fine, SEPARES par une histoire composee
    (amorce + episode conditionnel) -> l'individu est un operateur sur les
    histoires, pas une carte de stimuli.
E2  Geometrie des transitions (3e strate) : deux lois d'action G1 (lineaire)
    et G2 (saturante, meme pente a l'origine) -> memes paysages, memes
    contenus aux sondes faibles, cartes d'episodes identiques ; une histoire
    CUMULATIVE (repetition) les separe.
"""
import numpy as np, json

ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
W = float(np.tanh(2.2959))
LCHI, KQ, RCHI, KCHI, K3, THK, RP = 0.2, 0.6, 0.8, 2.5, 0.5, 1.25, 0.35
m_of = lambda n: (1+n**2)/(1+3*n**2)
EPS_S, CS, G2 = 0.005, 8.0, 0.18
U1 = np.array([1., 1.])/np.sqrt(2)
OUT = {}

def run(state, u, g, T, gfun, dt=0.01):
    x, v, th, chi, p, s = state
    x, v, th, chi, s = x.copy(), v.copy(), th.copy(), chi.copy(), s.copy()
    for _ in range(int(T/dt)):
        nt = np.linalg.norm(th)
        c = 2./(1+3*nt**2)
        Mx = x - c*float(th@x)*th
        dv = -(G0+G1*nt)*v - Mx + ZETA*np.tanh(x)
        x, v = x+dt*v, v+dt*dv
        th = th + dt*(ALPHA*np.tanh(x)-BETA*th)
        chi = chi + dt*(g*W*u-LCHI*chi)
        s = s + dt*EPS_S*(chi-s)
        p = p + dt*((0.6*np.linalg.norm(x)+0.4*np.linalg.norm(v))/np.sqrt(2)
                    + KCHI*np.linalg.norm(chi) - K3*p)
        if p > THK:
            proj = float(s@chi)/(np.linalg.norm(chi)+1e-12)
            th = th + max(1+gfun(proj), 0.)*KQ*chi
            chi = chi*(1-RCHI); p *= RP
    return (x, v, th, chi, p, s)

def st0(s_vec):
    return (np.array([0., 0.]), np.zeros(2), np.zeros(2), np.zeros(2), 0.0,
            np.array(s_vec, dtype=float))

G_lin = lambda pr: CS*pr

def episode_outcome(s_vec, T2, gfun=G_lin):
    st = st0(s_vec)
    st = (1e-3*U1, np.zeros(2)) + st[2:]
    st = run(st, U1, G2, float(T2), gfun)
    st = run(st, U1, 0.0, 300.0, gfun)
    return np.linalg.norm(st[0]) > 1

# ---------------------------------------------------------------- E1
print("E1 -- separation par l'histoire")
# seuils en s (composante le long de U1) pour quelques durees : bisection
def s_threshold(T2, gfun=G_lin):
    lo, hi = 0.0, 0.03
    if episode_outcome(lo*U1, T2, gfun): return 0.0
    if not episode_outcome(hi*U1, T2, gfun): return None
    for _ in range(10):
        mid = 0.5*(lo+hi)
        if episode_outcome(mid*U1, T2, gfun): hi = mid
        else: lo = mid
    return 0.5*(lo+hi)

ths = {}
for T2 in (60, 80, 100, 120, 140):
    b = s_threshold(T2)
    ths[T2] = b
    print(f"  seuil s pour T2={T2}: {b if b is None else round(b,5)}")
# choisir s1 < s2 dans un meme intervalle entre seuils
bs = sorted(v for v in ths.values() if v is not None)
s1, s2 = None, None
for i in range(len(bs)-1):
    if bs[i+1]-bs[i] > 0.004:
        s1, s2 = bs[i]+0.001, bs[i+1]-0.001
        break
if s1 is None:
    s1, s2 = bs[0]+0.001, bs[0]+0.004
print(f"  choix : s1={s1:.4f}, s2={s2:.4f}")
sig1 = "".join("A" if episode_outcome(s1*U1, T) else "." for T in range(40, 150, 10))
sig2 = "".join("A" if episode_outcome(s2*U1, T) else "." for T in range(40, 150, 10))
print(f"  cartes d'episodes (T=40..140 pas 10) : s1=[{sig1}]  s2=[{sig2}]"
      f"  identiques : {sig1 == sig2}")
# histoire composee : petite amorce (decalage delta) puis episode conditionnel
def composed(s_vec, T_am, D, T_C, gfun=G_lin):
    st = st0(s_vec)
    st = run(st, U1, G2, T_am, gfun)          # amorce courte
    st = run(st, U1, 0.0, D, gfun)            # oubli
    st = (1e-3*U1, np.zeros(2)) + st[2:]      # episode conditionnel
    st = run(st, U1, G2, T_C, gfun)
    st = run(st, U1, 0.0, 300.0, gfun)
    return np.linalg.norm(st[0]) > 1, st

found = None
for T_am in (8.0, 12.0, 16.0):
    for T_C in (60.0, 80.0, 100.0, 120.0):
        o1, _ = composed(s1*U1, T_am, 150.0, T_C)
        o2, _ = composed(s2*U1, T_am, 150.0, T_C)
        if o1 != o2:
            found = (T_am, T_C, o1, o2)
            break
    if found: break
if found:
    T_am, T_C, o1, o2 = found
    print(f"  histoire separatrice : amorce {T_am}u + oubli 150 + episode {T_C}u")
    print(f"    individu s1 -> {'APPROPRIE' if o1 else 'repos'} ; "
          f"individu s2 -> {'APPROPRIE' if o2 else 'repos'}   SEPARES")
    OUT["E1"] = {"s1": s1, "s2": s2, "cartes_identiques": sig1 == sig2,
                 "histoire": [T_am, 150.0, T_C], "issues": [o1, o2]}
else:
    print("  pas de separation trouvee sur la grille testee")
    OUT["E1"] = {"s1": s1, "s2": s2, "cartes_identiques": sig1 == sig2,
                 "histoire": None}

# ---------------------------------------------------------------- E2
print("E2 -- geometrie des transitions (3e strate)")
S0 = 0.05
G_sat = lambda pr: CS*S0*np.tanh(pr/S0)       # meme pente a l'origine
# cartes d'episodes depuis le vierge : identiques ?
c1 = "".join("A" if episode_outcome([0,0], T, G_lin) else "." for T in range(40, 150, 10))
c2 = "".join("A" if episode_outcome([0,0], T, G_sat) else "." for T in range(40, 150, 10))
print(f"  cartes vierges : G_lin=[{c1}]  G_sat=[{c2}]  identiques : {c1 == c2}")
# histoire cumulative : trois amorces puis episode conditionnel
def cumulative(gfun, n_am=3, T_am=41.9, D=120.0, T_C=90.0):
    st = st0([0., 0.])
    for _ in range(n_am):
        st = run(st, U1, G2, T_am, gfun)
        st = run(st, U1, 0.0, D, gfun)
    s_before = float(st[5] @ U1)
    st = (1e-3*U1, np.zeros(2)) + st[2:]
    st = run(st, U1, G2, T_C, gfun)
    st = run(st, U1, 0.0, 300.0, gfun)
    return np.linalg.norm(st[0]) > 1, s_before
sep = None
for T_C in (60.0, 70.0, 80.0, 90.0, 100.0):
    o1, sb1 = cumulative(G_lin, T_C=T_C)
    o2, sb2 = cumulative(G_sat, T_C=T_C)
    print(f"  T_C={T_C}: G_lin (s={sb1:.4f}) -> {'A' if o1 else '.'} ; "
          f"G_sat (s={sb2:.4f}) -> {'A' if o2 else '.'}")
    if o1 != o2 and sep is None:
        sep = T_C
if sep:
    print(f"  -> SEPARATION par l'histoire cumulative a T_C={sep} : "
          f"meme paysage, meme carte d'episodes, lois d'action differentes")
OUT["E2"] = {"cartes_identiques": c1 == c2, "T_C_separateur": sep}

json.dump(OUT, open("output_theory/fiii1_report.json", "w"), indent=2, default=str)
print("-> output_theory/fiii1_report.json")
