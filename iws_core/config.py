"""IWS Core 0.1.1 patch -- configuration.

Regle du patch : ne jamais modifier le comportement historique par defaut.
Chaque mecanisme est selectionne independamment ; aucune bascule globale
"core_version" diffuse dans le code.

Deux configurations de reference :
  - IWSConfig.legacy_p1() : reproduction bit-a-bit du code historique (model.py).
  - IWSConfig.core01()    : noyau Core Specification 0.1.1.

AUDIT code historique vs article [P1] (constate sur model.py, commit 2bab269) :
  (A1) la pression contient un terme supplementaire kappa_rel*||H_i-(A~H)_i||
       (surprise relationnelle, kappa_rel=0.7) absent de l'article ;
  (A2) le recablage code = retirer 2 / ajouter 2 AVEC plafond de densite 0.35
       (l'article dit retirer 2 / ajouter 3, sans plafond) ;
  (A3) chaque Kairos ajoute un coup de bruit H_i += N(0, 0.1) absent de l'article ;
  (A4) l'integrateur est un Euler semi-implicite de fait (H mis a jour avec la
       NOUVELLE vitesse, par aliasing dH = self.V puis V += ... en place) ;
  (A5) gain de couplage 0.8 et epsilon 0.15 codes en dur ;
  (A6) un seul generateur aleatoire partage (graphe, etats, W, coups de Kairos).
Le mode "legacy" reproduit (A1)-(A6) exactement ; chaque ecart est parametre.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Literal

PATCH_VERSION = "iws-core-0.1.1-patch1"

TraceMode = Literal["linear", "saturated"]
RewiringMode = Literal["legacy", "paper23", "conservative", "random"]
PressureMode = Literal["legacy", "core"]
InputMode = Literal["fixed_heterogeneous", "zero", "white_noise"]
GeometryMode = Literal["anisotropic", "scalar", "off"]
GraphMode = Literal["erdos_renyi", "circulant"]
RngMode = Literal["legacy_shared", "split"]
IntegratorMode = Literal["legacy_semi_implicit", "explicit_euler"]


@dataclass(frozen=True)
class IWSConfig:
    # --- dimensions / integration ---
    n_nodes: int = 20
    dim: int = 2
    dt: float = 0.05
    steps: int = 5000
    integrator: IntegratorMode = "legacy_semi_implicit"

    # --- dynamique continue ---
    gamma0: float = 0.25
    gamma_tau: float = 1.25
    alpha_trace: float = 0.45
    beta_trace: float = 0.12
    lambda_metric: float = 2.0
    coupling_gain: float = 0.8          # (A5) code en dur dans le code historique
    kappa1: float = 0.6
    kappa2: float = 0.4
    kappa3: float = 0.5
    kappa_rel: float = 0.7              # (A1) terme hors article, mode "legacy" seulement
    theta_kairos: float = 1.25
    relax_P: float = 0.35               # r_P
    relax_V: float = 0.5                # r_V

    # --- selecteurs de mecanismes ---
    trace_mode: TraceMode = "linear"
    rewiring_mode: RewiringMode = "legacy"
    pressure_mode: PressureMode = "legacy"
    input_mode: InputMode = "fixed_heterogeneous"
    geometry_mode: GeometryMode = "anisotropic"
    use_kairos: bool = True

    # --- parametres des mecanismes ---
    input_epsilon: float = 0.15         # amplitude de u = eps*W (fixed_heterogeneous)
    input_sigma: float = 0.0            # ecart-type du bruit blanc (white_noise)
    kairos_kick_sigma: float = 0.1      # (A3) coup de bruit sur H au Kairos ; 0 pour Core
    remove_edges: int = 2               # legacy / paper23
    add_edges: int = 3                  # paper23 uniquement (legacy code = 2)
    legacy_add_edges: int = 2           # (A2) valeur effective du code historique
    requested_rewires: int = 2          # conservative / random : m demande
    max_density: float = 0.35           # (A2) plafond de densite, legacy uniquement

    # --- graphe initial ---
    graph_mode: GraphMode = "erdos_renyi"
    er_p: float = 0.15
    circulant_offsets: tuple = (1, 2)   # degre 4

    # --- aleatoire / determinisme ---
    rng_mode: RngMode = "legacy_shared"
    deterministic_ties: bool = True     # tri stable pour les ex aequo (modes Core)
    symmetry_tolerance: float = 1e-12

    # --- etats initiaux ---
    init_h_sigma: float = 0.6
    init_v_sigma: float = 0.1
    init_w_sigma: float = 0.3
    identical_initial_states: bool = False   # X4b : etats strictement identiques
    perturb_node: int = -1                   # X4b bras 2 : noeud perturbe (-1 = aucun)
    perturb_eps: float = 0.0

    # --- fabriques ---
    @staticmethod
    def legacy_p1(**overrides) -> "IWSConfig":
        """Reproduction exacte du code historique (model.py)."""
        return replace(IWSConfig(), **overrides)

    @staticmethod
    def core01(**overrides) -> "IWSConfig":
        """Noyau Core Specification 0.1.1 : trace saturee, recablage conservatif
        m/m, u=0, pression sans terme relationnel, pas de coup de bruit,
        generateurs separes, departages stables."""
        base = IWSConfig(
            trace_mode="saturated",
            rewiring_mode="conservative",
            pressure_mode="core",
            input_mode="zero",
            kairos_kick_sigma=0.0,
            rng_mode="split",
            integrator="explicit_euler",
        )
        return replace(base, **overrides)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["patch_version"] = PATCH_VERSION
        return d

    # borne theorique de trace en mode sature (P-0.5)
    def trace_bound(self) -> float:
        return self.alpha_trace * (self.dim ** 0.5) / self.beta_trace
