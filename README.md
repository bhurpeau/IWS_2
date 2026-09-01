# IWS — Programmes des annexes A–L (paquet de reproductibilité)

**Objet.** Tous les programmes ayant produit les valeurs numériques citées dans les annexes de démonstration A–L, avec leurs sorties de référence. L'arborescence du dépôt `IWS_1` est conservée à plat : les scripts s'importent mutuellement et importent `iws_core/` depuis la racine — **ne pas les déplacer dans des sous-dossiers**.

**Porte de validation.** Avant toute exécution d'un script `fiii*`, l'intégrateur est validé bit à bit contre la référence F-III.0 : le contrôle P4 doit donner ‖x_PC‖ = 3.246573235196175 (exécuter `python run_checks.py`). Tout écart invalide l'environnement (versions numpy/scipy), pas le modèle.

**Exécution type.** `python <script>.py` depuis la racine ; chaque script `fiii*` écrit son rapport JSON (et ses figures) dans `output_theory/`. Les sorties de référence livrées permettent la comparaison champ à champ.

---

## Socle commun (toutes annexes)

| fichier | rôle |
|---|---|
| `iws_core/config.py` | constantes du modèle canonique (λ, ζ, γ₀, r_P, Θ, seuils d'événements) — ce sont les valeurs citées par A et B |
| `iws_core/engine.py` | intégrateur événementiel (flot + Kairos + regraine) |
| `iws_core/inputs.py` | générateurs d'entrées et de protocoles |
| `iws_core/validation.py` | contrôles de conservation et de tolérance |
| `model.py` | assemblage du modèle de référence |
| `run_checks.py` | **porte P4 bit à bit** (‖x_PC‖ = 3.246573235196175) |
| `plotting.py` | figures communes |
| `fiii0_histoires.py`, `fiii0_fin.py` | harnais histoires + référence F-III.0 (état P4) |

## Table annexe → programmes → sorties

| annexe | contenu | programmes | sorties de référence | valeurs clés vérifiées en relecture |
|---|---|---|---|---|
| **A** | existence, unicité, non-Zénon | analytique ; constantes de `iws_core/config.py` ; porte `run_checks.py` | — | L_M(λ) = λ/√(1+λ)·(1+3√3/8) ; δ_T = (1−r_P)Θ/K_T |
| **B** | énergie, bornage | analytique ; mêmes constantes | — | optimum de saut ε²‖H‖²/(2(1+r_V)) ; préfacteur ≤ ½ |
| **C** | réduction Tome I → II | `theory_delta.py`, `theory_psi.py`, `theory_transitions.py`, `part5_transient.py`, `run_x0.py`, `run_x4b.py` | `report.json`, `summary.json` (X0/X4b) | τ_ign = 0.377964 ; racines 0.1021/2.2376/0.0717/2.2959 ; ‖h\*‖ = 3.246832 ; p_∞ = 3.896198 ; Δ_p = 0.535537 ; spectres transverses (⚠ diagonale haute : republier depuis le script, relecture C-P1) |
| **D** | résistance, appropriation | `fii01_resistance.py`, `theory_robustness.py` | rapport fii01 | g_c = 0.032724 ; pente −2.177911 ; c₂ = 0.3220 ; g_comp = 0.37566 ; (⚠ w = 0.98006 : source probable tanh(2.300), relecture D-P1/F-P1) |
| **E** | interfaces R1/R2 | `fii1_interface.py`, `fii1_addendum.py`, `dyad_open.py` | rapports fii1 | g_K = 0.0360747 ; ρ/ηw = 0.051017 ; transport R1 = 0.09108 ; g_F^qs vs g_F = 0.393 % |
| **F** | protocoles numériques Tomes I–II | `fii11_cycle.py`, `fii12_bands_specificity.py` (cités dans l'annexe), `experiments.py` (harnais) | `fii11_report.json`, `fii12_report.json` | période 2π/ω = 14.6328 ; amorçage +7.6 % ; Q_net D=14/26 ; table de spécificité scalaire/vectorielle |
| **G** | cadre 𝔅/ρ, III-7, III-8, congruence | `fiii7_admissibles.py` (paire E1, chute de ℓ), `fiii2_quotient.py` (triplet a, b, c) | `fiii7_report.json`, `fiii2_report.json` | ℓ 2→1 sous ρ_cal à budget constant |
| **H** | réduction III-9 | `fiii8_reduction.py`, `fiii5_calendrier.py` (b, δ, équation de seuil) | `fiii8_report.json` (+ png), `fiii5_report.json` (+ png) | s\*₈₀ = 0.01627 sur sonde nue ; translation cellule à cellule exacte ; îles Ω ; pointwise 10/10 vs naïf 5/10 |
| **I** | suffisance état lent III-10 | `fiii9_frontiere.py` | `fiii9_report.json` (+ png) | fermeture 12/12, résidus hors-axe nuls ; 19 points F-III.6 sur la section (résidu médian 2.8e-4) ; anti-facilitation −0.00076 prédit vs −0.0006 |
| **J** | pli et couture (Prop III-11) | `fiii10_raccord.py` (carte 21×21, 14 K / 40 C), `fiii11_dualite.py` (axes p, T₂, composition ; propensions) | `fiii10_report.json` (+ png), `fiii11_report.json` | p : 0K/11 ; T₂ : 3/3 C ; composition : 100 % K ; état : 26 %/0 % |
| **K** | plasticité (Théorème III-12) | `fiii12_plasticite.py` (extension γ, égalisation z₄ à 1e-6, détection, s\*_figé) | `fiii12_report.json` | γ₁ = 0.834 / γ₂ = 0.932 ; s\*_figé(1) = 0.01627 = III-5 retrouvée ; Σ⁽¹⁾ ≠ Σ⁽²⁾ |
| **L** | ultramétrique (Prop III-3) | `fiii2_quotient.py` (triplet fondateur ℓ/d), `fiii3_stabilite.py` (filtration par budgets) | `fiii2_report.json`, `fiii3_report.json` (+ png) | ℓ(a,b) = 2, ℓ(a,c) = ℓ(b,c) = 1 → signature ultramétrique |

## Programmes du dépôt non requis par les annexes

`fiii1_strates.py`, `fiii4_naissance.py`, `fiii6_regimes.py` (notes F-III.1/4/6 — leurs résultats entrent dans le corps des chapitres, pas dans les fiches de démonstration) ; `theory_robustness.py` sert aussi à la note T2. Inclus quand même dans le paquet : le manuscrit les citera.

## Avertissements de relecture à retracer avant publication

1. **Annexe C** — spectre transverse de la diagonale haute : max Re recalculé −0.064 vs −0.0459 dans le texte (signe et conclusion inchangés) ; republier les spectres bloc par bloc depuis `theory_psi.py`/`run_x0.py`.
2. **Annexe D** — w = 0.98006 : aucune des racines exactes ne le donne ; source probable x_A arrondi à 2.300 (tanh(2.300) = 0.98009) ; à confirmer dans `fii01_resistance.py` et à documenter.

*Graine et déterminisme : les scripts fiii\* fixent leurs graines en tête de fichier ; toute comparaison de rapport doit se faire à graine et versions identiques (numpy ≥ 1.24 utilisé pour les références).*
