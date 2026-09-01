"""F-III.0 -- Histoires, signatures, et la naissance de l'effet d'ordre.

Definitions implementees :
  episode  E = (u, g, T)          (protocole P : contact brutal, graine appariee)
  histoire H = (E1, D1, E2, D2, ...)   ;   X_fin = Phi_H(X0)
  etat etendu X = (x, v, tau, chi, p, s_vec [, s_sc])
  signature R_X : pour u_probe in {u1, -u1}, motif d'appropriation sur
                  T2 in {40..140 pas 20} (graine appariee au debut du probe).

P1  superposition : s_vec apres AB / BA = combinaison lineaire a poids de
    recence exacts (lecture directe vs prediction par episodes solo).
P2  suffisance (Prop. III-1) : injecter s_vec dans un systeme vierge reproduit
    la signature de l'histoire complete -> l'individu minimal = s_vec ;
    l'effet d'ordre = arithmetique de recence.
P3  annulation : (u puis -u) ajuste pour s_vec(test)=0 : signature = vierge
    (limitation demontree) ; memoire duale (canal scalaire) : != vierge.
P4  composition propre : episode CONDITIONNEL C (T_C reussit si amorce,
    echoue sinon) : P puis C -> APPROPRIE ; C puis P -> repos amorce.
    Premier effet d'ordre irreductible a la recence (bassins finaux differents).
"""
from __future__ import annotations

import json
import numpy as np

ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
W = float(np.tanh(2.2959))
LCHI, KQ, RCHI, KCHI, K3, THK, RP = 0.2, 0.6, 0.8, 2.5, 0.5, 1.25, 0.35
m_of = lambda n: (1 + n**2) / (1 + 3 * n**2)
EPS_S, CS, G2 = 0.005, 8.0, 0.18
U1 = np.array([1.0, 1.0]) / np.sqrt(2)
OUT = {}

def phi(episodes, probe=None, s0=None, s_sc0=0.0, dual=False, c0=4.0, dt=0.01):
    """episodes : liste de (u, g, T) separes par des gaps (u, 0, D).
    probe : (u_p, T2) ajoute en fin avec graine appariee, +400 de relaxation.
    Renvoie l'etat etendu final (x, v, th, chi, p, s_vec, s_sc)."""
    x = np.array([1e-3, 1e-3]) / np.sqrt(2)
    v = np.zeros(2); th = np.zeros(2); chi = np.zeros(2)
    s_vec = np.zeros(2) if s0 is None else np.array(s0, dtype=float)
    s_sc = s_sc0; p = 0.0
    seq = list(episodes)
    if probe is not None:
        u_p, T2 = probe
        seq = seq + [("RESEED", u_p, 0.0), (u_p, G2, T2), (u_p, 0.0, 400.0)]
    for item in seq:
        if item[0] is not None and isinstance(item[0], str) and item[0] == "RESEED":
            x = 1e-3 * item[1]; v = np.zeros(2)
            continue
        u, g, T = item
        t = 0.0
        for _ in range(int(T / dt)):
            nt = np.linalg.norm(th)
            c = 2.0 / (1 + 3 * nt**2)
            Mx = x - c * float(th @ x) * th
            dv = -(G0 + G1 * nt) * v - Mx + ZETA * np.tanh(x)
            x, v = x + dt * v, v + dt * dv
            th = th + dt * (ALPHA * np.tanh(x) - BETA * th)
            chi = chi + dt * (g * W * u - LCHI * chi)
            s_vec = s_vec + dt * EPS_S * (chi - s_vec)
            s_sc = s_sc + dt * EPS_S * (np.linalg.norm(chi) - s_sc)
            p = p + dt * ((0.6 * np.linalg.norm(x) + 0.4 * np.linalg.norm(v)) / np.sqrt(2)
                          + KCHI * np.linalg.norm(chi) - K3 * p)
            if p > THK:
                nc = np.linalg.norm(chi)
                boost = 1 + CS * float(s_vec @ chi) / (nc + 1e-12)
                if dual:
                    boost *= (1 + c0 * s_sc)
                th = th + max(boost, 0.0) * KQ * chi
                chi = chi * (1 - RCHI); p *= RP
            t += dt
    return x, v, th, chi, p, s_vec, s_sc

def signature(episodes, s0=None, s_sc0=0.0, dual=False):
    """Motifs d'appropriation pour u_probe in {U1, -U1}."""
    sig = {}
    for name, u_p in (("+u1", U1), ("-u1", -U1)):
        patt = ""
        for T2 in range(40, 150, 20):
            xf = phi(episodes, probe=(u_p, float(T2)), s0=s0, s_sc0=s_sc0, dual=dual)[0]
            patt += "A" if np.linalg.norm(xf) > 1 else "."
        sig[name] = patt
    return sig

E_A = (U1, G2, 41.9)             # amorce le long de u1
GAP = (U1, 0.0, 200.0)
U2 = np.array([1.0, -1.0]) / np.sqrt(2)
E_B = (U2, G2, 41.9)             # amorce orthogonale
GAP2 = (U2, 0.0, 200.0)

# ---------------------------------------------------------------- P1
print("P1 -- superposition des residus s_vec")
# sigma(T) par episodes solo (lecture au temps de test relatif)
sA = phi([E_A, (U1, 0.0, 400.0)])[5]           # A puis 400
sB = phi([E_B, (U2, 0.0, 400.0)])[5]
sA_far = phi([E_A, (U1, 0.0, 200.0 + 41.9 + 200.0)])[5]
sB_far = phi([E_B, (U2, 0.0, 200.0 + 41.9 + 200.0)])[5]
s_AB = phi([E_A, GAP, E_B, GAP2])[5]
s_BA = phi([E_B, GAP2, E_A, GAP])[5]
pred_AB = sA_far + sB * 0 + phi([E_B, (U2, 0.0, 200.0)])[5]
pred_BA = sB_far + phi([E_A, (U1, 0.0, 200.0)])[5]
print(f"  s(AB) lu = {np.round(s_AB,5)}  predit (superposition) = {np.round(pred_AB,5)}")
print(f"  s(BA) lu = {np.round(s_BA,5)}  predit                = {np.round(pred_BA,5)}")
OUT["P1"] = {"s_AB": s_AB.tolist(), "pred_AB": pred_AB.tolist(),
             "s_BA": s_BA.tolist(), "pred_BA": pred_BA.tolist()}

# ---------------------------------------------------------------- P2
print("P2 -- suffisance de s_vec (Prop. III-1)")
sig_AB = signature([E_A, GAP, E_B, GAP2])
sig_BA = signature([E_B, GAP2, E_A, GAP])
sig_injAB = signature([], s0=s_AB)
sig_injBA = signature([], s0=s_BA)
print(f"  R(hist AB)    = {sig_AB}")
print(f"  R(inject sAB) = {sig_injAB}   identiques : {sig_AB == sig_injAB}")
print(f"  R(hist BA)    = {sig_BA}")
print(f"  R(inject sBA) = {sig_injBA}   identiques : {sig_BA == sig_injBA}")
print(f"  effet d'ordre AB vs BA (recence pure) : {'present' if sig_AB != sig_BA else 'absent'}")
OUT["P2"] = {"AB": sig_AB, "BA": sig_BA, "injAB": sig_injAB, "injBA": sig_injBA}

# ---------------------------------------------------------------- P3
print("P3 -- annulation (u puis -u) et memoire duale")
# ajuster T2e episode pour s_vec(test) ~ 0 : sigma_2 = sigma_1 * w_far/w_near
E_A2 = (-U1, G2, 41.9)
# recherche grossiere de la duree T2e annulant s au test
best_T, best_norm = None, 1e9
for T2e in (10, 15, 20, 25, 30, 41.9):
    s_end = phi([E_A, GAP, (-U1, G2, float(T2e)), (U1, 0.0, 200.0)])[5]
    if np.linalg.norm(s_end) < best_norm:
        best_norm, best_T = float(np.linalg.norm(s_end)), T2e
print(f"  meilleure annulation : T2e={best_T}, ||s(test)||={best_norm:.5f} "
      f"(vs solo {np.linalg.norm(sA):.4f})")
H_cancel = [E_A, GAP, (-U1, G2, float(best_T)), (U1, 0.0, 200.0)]
sig_cancel = signature(H_cancel)
sig_virgin = signature([])
print(f"  R(annulee) = {sig_cancel}")
print(f"  R(vierge)  = {sig_virgin}   identiques : {sig_cancel == sig_virgin}")
# memoire duale : canal scalaire c0=4
st = phi(H_cancel, dual=True)
sig_cancel_d = signature(H_cancel, dual=True)
sig_virgin_d = signature([], dual=True)
print(f"  duale : s_sc residuel = {st[6]:.4f} ; "
      f"R(annulee)={sig_cancel_d} vs R(vierge)={sig_virgin_d} "
      f"-> differentes : {sig_cancel_d != sig_virgin_d}")
OUT["P3"] = {"cancel": sig_cancel, "virgin": sig_virgin,
             "cancel_dual": sig_cancel_d, "virgin_dual": sig_virgin_d}

# ---------------------------------------------------------------- P4
print("P4 -- composition propre : episode conditionnel")
# fenetre conditionnelle : T_C reussit si amorce (meme direction), echoue sinon
for T_C in (100.0, 110.0, 120.0, 130.0):
    ok_primed = np.linalg.norm(
        phi([E_A, GAP, ("RESEED", U1, 0.0), (U1, G2, T_C), (U1, 0.0, 400.0)])[0]) > 1
    ok_bare = np.linalg.norm(
        phi([("RESEED", U1, 0.0), (U1, G2, T_C), (U1, 0.0, 400.0)])[0]) > 1
    print(f"  T_C={T_C}: amorce->{'A' if ok_primed else '.'}  "
          f"nu->{'A' if ok_bare else '.'}")
    if ok_primed and not ok_bare:
        TC = T_C
        break
E_C = [("RESEED", U1, 0.0), (U1, G2, TC), (U1, 0.0, 200.0)]
x_PC = phi([E_A, GAP] + E_C)[0]
x_CP = phi(E_C + [E_A, GAP])[0]
print(f"  P puis C : ||x_fin|| = {np.linalg.norm(x_PC):.3f} "
      f"({'APPROPRIE' if np.linalg.norm(x_PC)>1 else 'repos'})")
print(f"  C puis P : ||x_fin|| = {np.linalg.norm(x_CP):.3f} "
      f"({'APPROPRIE' if np.linalg.norm(x_CP)>1 else 'repos amorce'})")
OUT["P4"] = {"T_C": TC, "PC": float(np.linalg.norm(x_PC)),
             "CP": float(np.linalg.norm(x_CP))}

with open("output_theory/fiii0_report.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("-> output_theory/fiii0_report.json")
