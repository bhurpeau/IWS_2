"""Addendum F-II.1 : verifications finales (ordre 2 complet, seuil R2, sommation
g=0.18, monotonie de la frontiere T, controle pulse/continu a g=0.10).
Reprend run_R2 de fii1_interface.py ; resultats commentes dans NOTE_FII1."""
# Contenu : voir historique de session ; principaux resultats :
#  - ordre 2 complet : x_res = -2.1774 d + 0.3216 d^2 (quasi-annulation
#    27.40 - 52.31 + 26.14 = 1.25) -> quasi-linearite expliquee ;
#  - branche instable persistante a g=0.145 (max Re = +0.0036) ;
#  - seuil R2 contact permanent : g ~ 0.1632 (vs 0.0556 en R0) ;
#  - g=0.18 : T*_R2 = 76.2 ; frontiere T monotone (grille 60..126) ;
#  - sommation deux expositions 41.9u : D=30 somme, D=5/80/200 oublient ;
#  - amorcage : aucun (T* croit avec le delai par erosion de la graine,
#    ln(1/x0) de la loi d'appropriation) ;
#  - g=0.10 : continu 271 Kairos sans appropriation (capture resistante) ;
#    issues dependantes de la graine (bassins entrelaces 0.10-0.163).
