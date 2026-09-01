import numpy as np, json, sys
ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
W = float(np.tanh(2.2959))
LCHI, KQ, RCHI, KCHI, K3, THK, RP = 0.2, 0.6, 0.8, 2.5, 0.5, 1.25, 0.35
m_of = lambda n: (1+n**2)/(1+3*n**2)
EPS_S, CS, G2 = 0.005, 8.0, 0.18
U1 = np.array([1., 1.])/np.sqrt(2)

def run(state, u, g, T, dual=False, c0=4.0, dt=0.01):
    x, v, th, chi, p, s, ssc = state
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
        ssc = ssc + dt*EPS_S*(np.linalg.norm(chi)-ssc)
        p = p + dt*((0.6*np.linalg.norm(x)+0.4*np.linalg.norm(v))/np.sqrt(2)
                    + KCHI*np.linalg.norm(chi) - K3*p)
        if p > THK:
            nc = np.linalg.norm(chi)
            boost = 1 + CS*float(s@chi)/(nc+1e-12)
            if dual: boost *= (1 + c0*ssc)
            th = th + max(boost, 0.)*KQ*chi
            chi = chi*(1-RCHI); p *= RP
    return (x, v, th, chi, p, s, ssc)

def fresh(): return (np.array([1e-3,1e-3])/np.sqrt(2), np.zeros(2), np.zeros(2),
                     np.zeros(2), 0.0, np.zeros(2), 0.0)

def probe_sig(state, dual=False):
    sig = {}
    for name, u_p in (("+u1", U1), ("-u1", -U1)):
        patt = ""
        for T2 in range(40, 150, 20):
            st = (1e-3*u_p, np.zeros(2)) + tuple(state[2:])   # graine appariee
            st = run(st, u_p, G2, float(T2), dual=dual)
            st = run(st, u_p, 0.0, 300.0, dual=dual)
            patt += "A" if np.linalg.norm(st[0]) > 1 else "."
        sig[name] = patt
    return sig

out = {}
# P3-duale : histoire annulee (T2e=10 du run principal) avec canal scalaire
st = fresh()
st = run(st, U1, G2, 41.9, dual=True)
st = run(st, U1, 0.0, 200.0, dual=True)
st = run(st, -U1, G2, 10.0, dual=True)
st = run(st, U1, 0.0, 200.0, dual=True)
print(f"duale: ||s_vec||={np.linalg.norm(st[5]):.5f}  s_sc={st[6]:.4f}", flush=True)
sig_cd = probe_sig(st, dual=True)
sig_vd = probe_sig(fresh(), dual=True)
print(f"R(annulee, duale)={sig_cd}", flush=True)
print(f"R(vierge, duale) ={sig_vd}  differentes: {sig_cd != sig_vd}", flush=True)
out["P3_dual"] = {"cancel": sig_cd, "virgin": sig_vd, "s_sc": st[6]}

# P4 : episode conditionnel
prime = run(run(fresh(), U1, G2, 41.9), U1, 0.0, 200.0)
TC_found = None
for T_C in (100., 110., 120., 130., 140.):
    stp = (1e-3*U1, np.zeros(2)) + tuple(prime[2:])
    stp = run(run(stp, U1, G2, T_C), U1, 0.0, 400.0)
    stb = run(run((1e-3*U1, np.zeros(2), np.zeros(2), np.zeros(2), 0.0,
                   np.zeros(2), 0.0), U1, G2, T_C), U1, 0.0, 400.0)
    okp, okb = np.linalg.norm(stp[0]) > 1, np.linalg.norm(stb[0]) > 1
    print(f"T_C={T_C}: amorce->{'A' if okp else '.'}  nu->{'A' if okb else '.'}", flush=True)
    if okp and not okb:
        TC_found = T_C; break
if TC_found:
    # P puis C
    stPC = (1e-3*U1, np.zeros(2)) + tuple(prime[2:])
    stPC = run(run(stPC, U1, G2, TC_found), U1, 0.0, 200.0)
    # C puis P
    stC = run(run((1e-3*U1, np.zeros(2), np.zeros(2), np.zeros(2), 0.0,
                   np.zeros(2), 0.0), U1, G2, TC_found), U1, 0.0, 200.0)
    stCP = run(run(stC, U1, G2, 41.9), U1, 0.0, 200.0)
    pc, cp = float(np.linalg.norm(stPC[0])), float(np.linalg.norm(stCP[0]))
    print(f"P puis C : ||x||={pc:.3f} ({'APPROPRIE' if pc>1 else 'repos'})", flush=True)
    print(f"C puis P : ||x||={cp:.3f} ({'APPROPRIE' if cp>1 else 'repos amorce'})", flush=True)
    out["P4"] = {"T_C": TC_found, "PC": pc, "CP": cp}
json.dump(out, open("output_theory/fiii0_fin.json", "w"), indent=2, default=str)
print("OK", flush=True)
