"""Moteur IWS parametrable (patch 0.1.1).

Garanties :
  - config legacy_p1() : reproduction BIT-A-BIT de model.IWSSimulation
    (meme ordre de consommation du generateur, meme aliasing d'integration
    semi-implicite, memes chemins de calcul) -- verifiee par run_checks.py ;
  - config core01() : trace saturee, recablage conservatif m/m (invariant
    d'aretes verifie a chaque evenement), port u, pas de coup de bruit,
    generateurs separes, departages par tri stable ;
  - journal complet : configuration, graine, version, temps de Kairos,
    aretes retirees/ajoutees, violations d'invariants, brisures de symetrie
    par departage (evenements simultanes).
"""
from __future__ import annotations

import numpy as np

from .config import IWSConfig, PATCH_VERSION
from .inputs import InputPort, build_input


def row_normalize_with_self_loops(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    A_hat = A + np.eye(n)
    row_sums = A_hat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return A_hat / row_sums


def erdos_renyi_legacy(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Meme construction (et meme consommation RNG) que le code historique."""
    A_init = (rng.random((n, n)) < p).astype(float)
    A_init = np.triu(A_init, 1)
    return A_init + A_init.T


def circulant_graph(n: int, offsets) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        for k in offsets:
            A[i, (i + k) % n] = 1.0
            A[i, (i - k) % n] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


class IWSEngine:
    def __init__(self, config: IWSConfig, seed: int,
                 input_refine: int = 1, input_coarse_steps: int | None = None):
        self.cfg = config
        self.seed = seed
        self.n = config.n_nodes
        self.d = config.dim
        self.dt = config.dt
        self.step_index = 0

        # --- generateurs ---
        if config.rng_mode == "legacy_shared":
            shared = np.random.default_rng(seed)
            self.rng_graph = self.rng_state = self.rng_input = self.rng_rewiring = shared
        else:
            ss = np.random.SeedSequence(seed).spawn(4)
            self.rng_graph = np.random.default_rng(ss[0])
            self.rng_state = np.random.default_rng(ss[1])
            self.rng_input = np.random.default_rng(ss[2])
            self.rng_rewiring = np.random.default_rng(ss[3])

        # --- graphe initial ---
        if config.graph_mode == "erdos_renyi":
            self.A = erdos_renyi_legacy(self.n, config.er_p, self.rng_graph)
        elif config.graph_mode == "circulant":
            self.A = circulant_graph(self.n, config.circulant_offsets)
        else:
            raise ValueError(config.graph_mode)

        # --- etats initiaux (ordre RNG identique au legacy : H, V, puis W) ---
        if config.identical_initial_states:
            h0 = self.rng_state.normal(0, config.init_h_sigma, self.d)
            v0 = self.rng_state.normal(0, config.init_v_sigma, self.d)
            self.H = np.tile(h0, (self.n, 1))
            self.V = np.tile(v0, (self.n, 1))
        else:
            self.H = self.rng_state.normal(0, config.init_h_sigma, (self.n, self.d))
            self.V = self.rng_state.normal(0, config.init_v_sigma, (self.n, self.d))
        self.tau = np.zeros((self.n, self.d))
        self.P = np.zeros(self.n)
        W = self.rng_input.normal(0, config.init_w_sigma, (self.n, self.d))

        # perturbation X4b bras 2 : direction fixe e1 (deterministe en eps0)
        if config.perturb_node >= 0 and config.perturb_eps != 0.0:
            e1 = np.zeros(self.d); e1[0] = 1.0
            self.H[config.perturb_node] += config.perturb_eps * e1

        # --- port d'entree ---
        self.input: InputPort = build_input(config, W, self.rng_input,
                                            refine=input_refine,
                                            coarse_steps=input_coarse_steps)
        if config.input_mode == "zero":
            assert not np.any(self.input.sample(0)), "u=0 doit etre exactement nul"

        # --- bornes / invariants ---
        self._trace_bound = max(
            float(np.max(np.linalg.norm(self.tau, axis=1))), config.trace_bound()
        )
        self._edge_count = int(self.A.sum() // 2)

        # --- journaux ---
        self.history = {
            "mean_trace": [], "mean_pressure": [], "num_kairos": [],
            "num_edges": [], "h_norm": [], "h_var": [], "homophily": [],
        }
        self.kairos_log: list[dict] = []       # step, node, removed, added, ties
        self.violations: list[dict] = []
        self.symmetry_log: list[dict] = []     # brisures conventionnelles (D-08)

    # ------------------------------------------------------------------ #
    # dynamique continue
    # ------------------------------------------------------------------ #
    def _metric_action(self, tau_i: np.ndarray, grad_i: np.ndarray) -> np.ndarray:
        mode = self.cfg.geometry_mode
        if mode == "off":
            return grad_i
        norm2 = float(tau_i @ tau_i)
        if mode == "anisotropic":
            # chemin legacy exact (inversion explicite) pour la reproduction
            outer = np.outer(tau_i, tau_i) / (1.0 + norm2)
            g_tau = np.eye(self.d) + self.cfg.lambda_metric * outer
            g_inv = np.linalg.inv(g_tau)
            return g_inv @ grad_i
        if mode == "scalar":
            # meme attenuation spectrale, appliquee isotropiquement (X5)
            factor = 1.0 / (1.0 + self.cfg.lambda_metric * norm2 / (1.0 + norm2))
            return factor * grad_i
        raise ValueError(mode)

    def step(self):
        cfg = self.cfg
        A_norm = row_normalize_with_self_loops(self.A)
        neigh = A_norm @ self.H
        grad_phi = self.H.copy()  # H a l'instant courant (avant mise a jour)

        tau_norm = np.linalg.norm(self.tau, axis=1, keepdims=True)
        gamma = cfg.gamma0 + cfg.gamma_tau * tau_norm

        geom_grad = np.zeros_like(self.H)
        for i in range(self.n):
            geom_grad[i] = self._metric_action(self.tau[i], grad_phi[i])

        relational = cfg.coupling_gain * np.tanh(neigh)
        external = self.input.sample(self.step_index)

        dV = -gamma * self.V - geom_grad + relational + external

        if cfg.integrator == "legacy_semi_implicit":
            # reproduit l'aliasing du code historique : H avance avec la
            # NOUVELLE vitesse (Euler semi-implicite de fait, cf. audit A4)
            self.V += self.dt * dV
            self.H += self.dt * self.V
        elif cfg.integrator == "explicit_euler":
            H_old_V = self.V.copy()
            self.V += self.dt * dV
            self.H += self.dt * H_old_V
        else:
            raise ValueError(cfg.integrator)

        # trace
        if cfg.trace_mode == "linear":
            written = neigh
        elif cfg.trace_mode == "saturated":
            written = np.tanh(neigh)
        else:
            raise ValueError(cfg.trace_mode)
        self.tau += self.dt * (cfg.alpha_trace * written - cfg.beta_trace * self.tau)

        # pression -- points d'evaluation identiques au legacy. Audit (A7) :
        # dans le code historique, potential_gradient retourne une REFERENCE
        # a self.H ; le terme kappa1*||grad_phi|| est donc evalue sur le H
        # deja mis a jour. On replique (et on conserve la meme convention en
        # mode core pour la comparabilite).
        if cfg.pressure_mode == "legacy":
            # ordre de sommation IDENTIQUE au code historique (bit-exactitude)
            rel_stress = np.linalg.norm(self.H - neigh, axis=1)
            dP = (cfg.kappa1 * np.linalg.norm(self.H, axis=1)
                  + cfg.kappa2 * np.linalg.norm(self.V, axis=1)
                  + cfg.kappa_rel * rel_stress
                  - cfg.kappa3 * self.P)
        else:
            dP = (cfg.kappa1 * np.linalg.norm(self.H, axis=1)
                  + cfg.kappa2 * np.linalg.norm(self.V, axis=1)
                  - cfg.kappa3 * self.P)
        self.P += self.dt * dP

        # Kairos
        kairos_nodes = np.where(self.P > cfg.theta_kairos)[0]
        if cfg.use_kairos and len(kairos_nodes) > 0:
            if len(kairos_nodes) > 1:
                # brisure conventionnelle de symetrie (D-08) : journalisee
                self.symmetry_log.append({
                    "step": self.step_index,
                    "simultaneous_nodes": kairos_nodes.tolist(),
                    "resolution": "ascending index (D-08)",
                })
            self._apply_kairos(kairos_nodes)
        else:
            kairos_nodes = np.array([], dtype=int)

        self._check_invariants()
        self._record(len(kairos_nodes))
        self.step_index += 1

    # ------------------------------------------------------------------ #
    # evenements
    # ------------------------------------------------------------------ #
    def _argsort(self, values: np.ndarray) -> np.ndarray:
        kind = "stable" if (self.cfg.deterministic_ties
                            and self.cfg.rewiring_mode != "legacy") else None
        return np.argsort(values, kind=kind) if kind else np.argsort(values)

    def _apply_kairos(self, kairos_nodes: np.ndarray):
        cfg = self.cfg
        tau_norm = np.linalg.norm(self.tau, axis=1, keepdims=True)
        tau_safe = self.tau / np.maximum(tau_norm, 1e-8)

        for i in kairos_nodes:
            i = int(i)
            scores = tau_safe @ tau_safe[i]
            scores[i] = -np.inf
            neighbors_before = np.where(self.A[i] > 0)[0]
            non_neighbors_before = np.where(self.A[i] == 0)[0]
            non_neighbors_before = non_neighbors_before[non_neighbors_before != i]
            edges_before = int(self.A.sum() // 2)
            removed, added = [], []

            if cfg.rewiring_mode == "legacy":
                if len(neighbors_before) > 0:
                    worst = neighbors_before[
                        np.argsort(scores[neighbors_before])[: cfg.remove_edges]]
                    for j in worst:
                        self.A[i, j] = self.A[j, i] = 0.0
                        removed.append(int(j))
                density = self.A.sum() / (self.n * (self.n - 1))
                if density < cfg.max_density and len(non_neighbors_before) > 0:
                    best = non_neighbors_before[
                        np.argsort(scores[non_neighbors_before])[-cfg.legacy_add_edges:]]
                    for j in best:
                        self.A[i, j] = self.A[j, i] = 1.0
                        added.append(int(j))

            elif cfg.rewiring_mode == "paper23":
                m_rm = min(cfg.remove_edges, len(neighbors_before))
                m_ad = min(cfg.add_edges, len(non_neighbors_before))
                if m_rm > 0:
                    worst = neighbors_before[
                        self._argsort(scores[neighbors_before])[:m_rm]]
                    for j in worst:
                        self.A[i, j] = self.A[j, i] = 0.0
                        removed.append(int(j))
                if m_ad > 0:
                    best = non_neighbors_before[
                        self._argsort(scores[non_neighbors_before])[-m_ad:]]
                    for j in best:
                        self.A[i, j] = self.A[j, i] = 1.0
                        added.append(int(j))

            elif cfg.rewiring_mode == "conservative":
                m = min(cfg.requested_rewires,
                        len(neighbors_before), len(non_neighbors_before))
                if m > 0:
                    worst = neighbors_before[
                        self._argsort(scores[neighbors_before])[:m]]
                    best = non_neighbors_before[
                        self._argsort(scores[non_neighbors_before])[-m:]]
                    for j in worst:
                        self.A[i, j] = self.A[j, i] = 0.0
                        removed.append(int(j))
                    for j in best:
                        self.A[i, j] = self.A[j, i] = 1.0
                        added.append(int(j))
                edges_after = int(self.A.sum() // 2)
                assert edges_after == edges_before, (
                    f"invariant |E| viole au pas {self.step_index}, noeud {i}: "
                    f"{edges_before} -> {edges_after}")

            elif cfg.rewiring_mode == "random":
                m = min(cfg.requested_rewires,
                        len(neighbors_before), len(non_neighbors_before))
                if m > 0:
                    worst = self.rng_rewiring.choice(
                        neighbors_before, size=m, replace=False)
                    best = self.rng_rewiring.choice(
                        non_neighbors_before, size=m, replace=False)
                    for j in worst:
                        self.A[i, j] = self.A[j, i] = 0.0
                        removed.append(int(j))
                    for j in best:
                        self.A[i, j] = self.A[j, i] = 1.0
                        added.append(int(j))
                edges_after = int(self.A.sum() // 2)
                assert edges_after == edges_before
            else:
                raise ValueError(cfg.rewiring_mode)

            # relaxation post-saut
            self.P[i] *= cfg.relax_P
            self.V[i] *= cfg.relax_V
            if cfg.kairos_kick_sigma > 0.0:  # (A3) legacy uniquement
                self.H[i] += self.rng_rewiring.normal(0, cfg.kairos_kick_sigma, self.d)

            self.kairos_log.append({
                "step": self.step_index, "t": self.step_index * self.dt,
                "node": i, "removed": removed, "added": added,
            })

    # ------------------------------------------------------------------ #
    # invariants, observables, sorties
    # ------------------------------------------------------------------ #
    def _check_invariants(self):
        cfg = self.cfg
        if cfg.trace_mode == "saturated":
            max_tau = float(np.max(np.linalg.norm(self.tau, axis=1)))
            if max_tau > self._trace_bound + 1e-9:
                self.violations.append({
                    "step": self.step_index, "invariant": "trace_bound",
                    "value": max_tau, "bound": self._trace_bound})
        if cfg.rewiring_mode in ("conservative", "random"):
            edges = int(self.A.sum() // 2)
            if edges != self._edge_count:
                self.violations.append({
                    "step": self.step_index, "invariant": "edge_count",
                    "value": edges, "bound": self._edge_count})
        if not np.all(self.P >= -1e-12):
            self.violations.append({"step": self.step_index,
                                    "invariant": "pressure_nonneg",
                                    "value": float(self.P.min())})

    def _record(self, num_kairos: int):
        self.history["mean_trace"].append(np.mean(np.linalg.norm(self.tau, axis=1)))
        self.history["mean_pressure"].append(np.mean(self.P))
        self.history["num_kairos"].append(num_kairos)
        self.history["num_edges"].append(self.A.sum() / 2.0)
        self.history["h_norm"].append(np.mean(np.linalg.norm(self.H, axis=1)))
        self.history["h_var"].append(np.mean(np.var(self.H, axis=0)))
        self.history["homophily"].append(self._homophily())

    def _homophily(self) -> float:
        iu = np.triu_indices(self.n, 1)
        mask = self.A[iu] > 0
        if not np.any(mask):
            return 0.0
        norms = np.linalg.norm(self.tau, axis=1)
        safe = self.tau / np.maximum(norms, 1e-12)[:, None]
        sims = (safe @ safe.T)[iu][mask]
        return float(np.mean(sims))

    def symmetry_spread(self) -> float:
        """Ecart maximal entre noeuds (test de symetrie X4b) sur (H,V,tau,P)."""
        s = 0.0
        for X in (self.H, self.V, self.tau):
            s = max(s, float(np.max(np.linalg.norm(X - X[0], axis=1))))
        s = max(s, float(np.max(np.abs(self.P - self.P[0]))))
        return s

    def run(self, steps: int | None = None, callback=None):
        steps = steps if steps is not None else self.cfg.steps
        for _ in range(steps):
            self.step()
            if callback is not None:
                callback(self)
        return self

    def result(self) -> dict:
        return {
            "patch_version": PATCH_VERSION,
            "config": self.cfg.to_dict(),
            "seed": self.seed,
            "history": {k: np.asarray(v) for k, v in self.history.items()},
            "kairos_times": [e["t"] for e in self.kairos_log],
            "kairos_log": self.kairos_log,
            "violations": self.violations,
            "symmetry_log": self.symmetry_log,
            "final_H": self.H.copy(), "final_A": self.A.copy(),
        }
