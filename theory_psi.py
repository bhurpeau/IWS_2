"""Note T-3 (partie A) -- Psi comme objet central : demo du probleme inverse.

Cible : Psi(u) = e^{-a u}/(1-u), a=2.51.
  Psi(0)=1, Psi(1^-)=inf ; ct minimum en u* = 1 - 1/a ; zeta_min = a e^{1-a}.
Factorisation canonique (parties croissante/decroissante de log Psi) :
  s*(u) = 1                                  (u <= u*)
        = ((1-u*)/(1-u)) e^{-a(u-u*)}        (u >  u*)      -> sigma^{-1} = u s*(u)
  h*(u) = e^{-a u}/(1-u)  (u<=u*), constante ensuite      -> m(theta) = h*((b/a_)theta)
Remarque : pour ce representant a memoire minimale, m_inf = Psi(u*) = zeta_min.
Verification : (i) Psi reconstruite == cible ; (ii) simulation du systeme reduit
synthetise -> bistabilite, puits au u_s predit.
"""
import numpy as np

A_ = 2.51
ALPHA, BETA, ZETA, G0, G1 = 0.45, 0.12, 0.8, 0.25, 1.25
ustar = 1 - 1 / A_
zmin_pred = A_ * np.exp(1 - A_)
print(f"cible : u* = {ustar:.4f}, zeta_min predit = {zmin_pred:.4f}")

def s_star(u):
    u = np.asarray(u, dtype=float)
    return np.where(u <= ustar, 1.0,
                    (1 - ustar) / np.maximum(1 - u, 1e-15) * np.exp(-A_ * (u - ustar)))

def h_star(u):
    v = np.minimum(np.asarray(u, dtype=float), ustar)
    return np.exp(-A_ * v) / (1 - v)

# sigma par inversion numerique de F(u) = u s*(u)
ug = np.linspace(1e-9, 1 - 1e-9, 400000)
Fg = ug * s_star(ug)
def sigma(x):
    return np.interp(x, Fg, ug)
def m_of_theta(th):
    return float(h_star((BETA / ALPHA) * abs(th)))

# (i) Psi reconstruite
u = np.linspace(1e-4, 0.99, 20000)   # on evite le voisinage de la singularite u->1 (interp)
Psi_target = np.exp(-A_ * u) / (1 - u)
Psi_built = (np.interp(u, ug, Fg) / u) * h_star(u)     # sigma^{-1}(u)/u * m((a/b)u * b/a)
err = float(np.max(np.abs(Psi_built - Psi_target) / Psi_target))
zmin_built = float(np.min(Psi_built))
print(f"(i) reconstruction : erreur relative max = {err:.2e} ; "
      f"zeta_min mesure = {zmin_built:.4f}")

# racines predites de Psi = zeta
f = Psi_target - ZETA
sgn = np.where(np.diff(np.sign(f)))[0]
roots_u = [float(u[i]) for i in sgn]
print(f"    racines predites (u) a zeta=0.8 : {[round(r,4) for r in roots_u]}")

# (ii) simulation du systeme reduit synthetise (d=1)
def simulate(x0, steps=200000, dt=0.005):
    x, v, th = x0, 0.0, 0.0
    for _ in range(steps):
        gam = G0 + G1 * abs(th)
        sx = float(sigma(abs(x))) * np.sign(x)
        dv = -gam * v - m_of_theta(th) * x + ZETA * sx
        x, v, th = x + dt * v, v + dt * dv, th + dt * (ALPHA * sx - BETA * th)
    return x, float(sigma(abs(x)))

for x0 in (0.05, 0.5, 3.0):
    xf, uf = simulate(x0)
    verdict = "extinction" if abs(xf) < 1e-3 else f"puits actif u={uf:.4f}"
    print(f"(ii) x0={x0:4.2f} -> {verdict}   (u_s predit = {roots_u[-1]:.4f})")
