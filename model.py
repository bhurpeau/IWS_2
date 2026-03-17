from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class IWS2Parameters:
    """Parameters for Paper II: structural plasticity in the Inner World System."""

    # Initial configuration
    n_nodes: int = 10
    dim: int = 2
    dt: float = 0.05
    edge_prob: float = 0.15

    # Fast dynamics / trace / geometry
    gamma0: float = 0.25
    gamma_tau: float = 1.25
    alpha_trace: float = 0.45
    beta_trace: float = 0.12
    lambda_metric: float = 2.0

    # Internal pressure
    kappa1: float = 0.6
    kappa2: float = 0.4
    kappa3: float = 0.5
    kappa_rel: float = 0.7

    # Kairos rewiring
    use_memory_geometry: bool = True
    use_kairos: bool = True
    theta_kairos: float = 1.25
    rewire_top_k: int = 2
    cut_bottom_k: int = 2
    max_density: float = 0.35

    # Structural plasticity flags
    use_division: bool = True
    use_apoptosis: bool = True
    use_senescence: bool = True

    # Division
    theta_div: float = 1.75
    theta_safe: float = 1.0
    kappa0: float = 5.0
    delta_div: float = 1.0
    rho_min: float = 0.4
    rho_max: float = 0.6
    p_inherit: float = 0.5
    mutation_scale_H: float = 0.02
    mutation_scale_V: float = 0.02
    mutation_scale_tau: float = 0.02
    child_pressure: float = 0.0
    parent_pressure_relief: float = 0.5

    # Apoptosis by crisis
    theta_death: float = 2.0
    deg_min: int = 2

    # Apoptosis by senescence
    theta_kappa: float = 0.5

    # Safety bounds
    min_nodes: int = 2
    max_nodes: int = 150



def row_normalize_with_self_loops(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    if n == 0:
        return A.copy()
    A_hat = A + np.eye(n)
    row_sums = A_hat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return A_hat / row_sums


class IWSSimulationPaper2:
    def __init__(self, params: IWS2Parameters, seed: int = 42):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.d = params.dim
        self.dt = params.dt

        n0 = params.n_nodes
        A0 = (self.rng.random((n0, n0)) < params.edge_prob).astype(float)
        A0 = np.triu(A0, 1)
        self.A = A0 + A0.T

        self.H = self.rng.normal(0.0, 0.6, size=(n0, self.d))
        self.V = self.rng.normal(0.0, 0.1, size=(n0, self.d))
        self.tau = np.zeros((n0, self.d))
        self.P = np.zeros(n0)
        self.kappa = np.ones(n0) * params.kappa0
        self.W = self.rng.normal(0.0, 0.3, size=(n0, self.d))

        # Lineage bookkeeping
        self.uid = np.arange(n0, dtype=int)
        self.parent_uid = np.full(n0, -1, dtype=int)
        self.lineage_uid = self.uid.copy()
        self.age = np.zeros(n0, dtype=int)
        self.next_uid = n0

        self.history: Dict[str, List[float]] = {
            "population_size": [],
            "num_kairos": [],
            "num_divisions": [],
            "num_apoptosis": [],
            "mean_trace": [],
            "mean_pressure": [],
            "mean_kappa": [],
            "state_var": [],
            "num_edges": [],
        }
        self.final_snapshot = None

    @property
    def n(self) -> int:
        return self.H.shape[0]

    def potential_gradient(self, H: np.ndarray) -> np.ndarray:
        return H

    def metric_action(self, tau_i: np.ndarray, grad_i: np.ndarray) -> np.ndarray:
        if not self.params.use_memory_geometry:
            return grad_i
        norm2 = float(tau_i @ tau_i)
        outer = np.outer(tau_i, tau_i) / (1.0 + norm2)
        g_tau = np.eye(self.d) + self.params.lambda_metric * outer
        g_inv = np.linalg.inv(g_tau)
        return g_inv @ grad_i

    def step(self) -> None:
        if self.n == 0:
            return

        self.age += 1
        A_norm = row_normalize_with_self_loops(self.A)
        neigh = A_norm @ self.H
        grad_phi = self.potential_gradient(self.H)

        tau_norm = np.linalg.norm(self.tau, axis=1, keepdims=True)
        gamma = self.params.gamma0 + self.params.gamma_tau * tau_norm

        geom_grad = np.zeros_like(self.H)
        for i in range(self.n):
            geom_grad[i] = self.metric_action(self.tau[i], grad_phi[i])

        relational_term = 0.8 * np.tanh(neigh)
        external_term = 0.15 * self.W

        dV = -gamma * self.V - geom_grad + relational_term + external_term
        dH = self.V
        self.V += self.dt * dV
        self.H += self.dt * dH

        dTau = self.params.alpha_trace * neigh - self.params.beta_trace * self.tau
        self.tau += self.dt * dTau

        rel_stress = np.linalg.norm(self.H - neigh, axis=1)
        dP = (
            self.params.kappa1 * np.linalg.norm(grad_phi, axis=1)
            + self.params.kappa2 * np.linalg.norm(self.V, axis=1)
            + self.params.kappa_rel * rel_stress
            - self.params.kappa3 * self.P
        )
        self.P += self.dt * dP

        kairos_nodes = np.where(self.P > self.params.theta_kairos)[0]
        if self.params.use_kairos and len(kairos_nodes) > 0:
            self.apply_kairos(kairos_nodes)
            num_kairos = int(len(kairos_nodes))
        else:
            num_kairos = 0

        division_nodes = self.get_division_candidates()
        if self.params.use_division and len(division_nodes) > 0:
            num_divisions = self.apply_division(division_nodes)
        else:
            num_divisions = 0

        apoptosis_nodes = self.get_apoptosis_candidates()
        if self.params.use_apoptosis and len(apoptosis_nodes) > 0:
            num_apoptosis = self.apply_apoptosis(apoptosis_nodes)
        else:
            num_apoptosis = 0

        self.record_history(num_kairos, num_divisions, num_apoptosis)

    def apply_kairos(self, kairos_nodes: np.ndarray) -> None:
        tau_norm = np.linalg.norm(self.tau, axis=1, keepdims=True)
        tau_safe = self.tau / np.maximum(tau_norm, 1e-8)

        for i in kairos_nodes:
            if i >= self.n:
                continue
            scores = tau_safe @ tau_safe[i]
            scores[i] = -np.inf
            neighbors = np.where(self.A[i] > 0)[0]
            non_neighbors = np.where(self.A[i] == 0)[0]
            non_neighbors = non_neighbors[non_neighbors != i]

            if len(neighbors) > 0:
                worst = neighbors[np.argsort(scores[neighbors])[: self.params.cut_bottom_k]]
                for j in worst:
                    self.A[i, j] = 0.0
                    self.A[j, i] = 0.0

            density = self.A.sum() / max(self.n * (self.n - 1), 1)
            if density < self.params.max_density and len(non_neighbors) > 0:
                best = non_neighbors[np.argsort(scores[non_neighbors])[-self.params.rewire_top_k :]]
                for j in best:
                    self.A[i, j] = 1.0
                    self.A[j, i] = 1.0

            self.P[i] *= 0.35
            self.V[i] *= 0.5
            self.H[i] += self.rng.normal(0, 0.1, self.d)

    def get_division_candidates(self) -> np.ndarray:
        if self.n >= self.params.max_nodes:
            return np.array([], dtype=int)
        mask = (
            (np.linalg.norm(self.tau, axis=1) > self.params.theta_div)
            & (self.P < self.params.theta_safe)
            & (self.kappa > self.params.theta_kappa)
        )
        return np.where(mask)[0]

    def apply_division(self, division_nodes: np.ndarray) -> int:
        count = 0
        for i in list(division_nodes):
            if i >= self.n or self.n >= self.params.max_nodes:
                continue
            if self.kappa[i] <= self.params.theta_kappa:
                continue

            rho_H = self.rng.uniform(self.params.rho_min, self.params.rho_max)
            rho_V = self.rng.uniform(self.params.rho_min, self.params.rho_max)
            eta_H_i = self.rng.normal(0, self.params.mutation_scale_H, self.d)
            eta_H_j = self.rng.normal(0, self.params.mutation_scale_H, self.d)
            eta_V_i = self.rng.normal(0, self.params.mutation_scale_V, self.d)
            eta_V_j = self.rng.normal(0, self.params.mutation_scale_V, self.d)
            eta_tau = self.rng.normal(0, self.params.mutation_scale_tau, self.d)

            H_parent_old = self.H[i].copy()
            V_parent_old = self.V[i].copy()
            tau_parent_old = self.tau[i].copy()
            uid_parent = int(self.uid[i])
            lineage_parent = int(self.lineage_uid[i])

            self.H[i] = rho_H * H_parent_old + eta_H_i
            self.V[i] = rho_V * V_parent_old + eta_V_i
            self.kappa[i] = max(0.0, self.kappa[i] - self.params.delta_div)
            self.P[i] *= self.params.parent_pressure_relief

            H_child = (1.0 - rho_H) * H_parent_old + eta_H_j
            V_child = (1.0 - rho_V) * V_parent_old + eta_V_j
            tau_child = tau_parent_old + eta_tau
            P_child = self.params.child_pressure
            kappa_child = self.params.kappa0
            W_child = self.rng.normal(0.0, 0.3, self.d)

            self.H = np.vstack([self.H, H_child[None, :]])
            self.V = np.vstack([self.V, V_child[None, :]])
            self.tau = np.vstack([self.tau, tau_child[None, :]])
            self.P = np.append(self.P, P_child)
            self.kappa = np.append(self.kappa, kappa_child)
            self.W = np.vstack([self.W, W_child[None, :]])
            self.uid = np.append(self.uid, self.next_uid)
            self.parent_uid = np.append(self.parent_uid, uid_parent)
            self.lineage_uid = np.append(self.lineage_uid, lineage_parent)
            self.age = np.append(self.age, 0)

            n_old = self.A.shape[0]
            A_new = np.zeros((n_old + 1, n_old + 1), dtype=float)
            A_new[:n_old, :n_old] = self.A
            child_idx = n_old
            A_new[i, child_idx] = 1.0
            A_new[child_idx, i] = 1.0
            neighbors = np.where(self.A[i] > 0)[0]
            for k in neighbors:
                if self.rng.random() < self.params.p_inherit:
                    A_new[child_idx, k] = 1.0
                    A_new[k, child_idx] = 1.0
            self.A = A_new

            self.next_uid += 1
            count += 1
        return count

    def get_apoptosis_candidates(self) -> np.ndarray:
        if self.n <= self.params.min_nodes:
            return np.array([], dtype=int)
        degrees = self.A.sum(axis=1)
        crisis_mask = (self.P > self.params.theta_death) & (degrees < self.params.deg_min)
        if self.params.use_senescence:
            senescence_mask = self.kappa < self.params.theta_kappa
        else:
            senescence_mask = np.zeros(self.n, dtype=bool)
        mask = crisis_mask | senescence_mask
        return np.where(mask)[0]

    def apply_apoptosis(self, apoptosis_nodes: np.ndarray) -> int:
        if self.n <= self.params.min_nodes:
            return 0
        count = 0
        max_removable = max(0, self.n - self.params.min_nodes)
        nodes = sorted(set(int(i) for i in apoptosis_nodes if 0 <= int(i) < self.n), reverse=True)
        nodes = nodes[:max_removable]
        for i in nodes:
            self.H = np.delete(self.H, i, axis=0)
            self.V = np.delete(self.V, i, axis=0)
            self.tau = np.delete(self.tau, i, axis=0)
            self.P = np.delete(self.P, i)
            self.kappa = np.delete(self.kappa, i)
            self.W = np.delete(self.W, i, axis=0)
            self.uid = np.delete(self.uid, i)
            self.parent_uid = np.delete(self.parent_uid, i)
            self.lineage_uid = np.delete(self.lineage_uid, i)
            self.age = np.delete(self.age, i)
            self.A = np.delete(np.delete(self.A, i, axis=0), i, axis=1)
            count += 1
        return count

    def record_history(self, num_kairos: int, num_divisions: int, num_apoptosis: int) -> None:
        self.history["population_size"].append(float(self.n))
        self.history["num_kairos"].append(float(num_kairos))
        self.history["num_divisions"].append(float(num_divisions))
        self.history["num_apoptosis"].append(float(num_apoptosis))
        self.history["mean_trace"].append(float(np.mean(np.linalg.norm(self.tau, axis=1))))
        self.history["mean_pressure"].append(float(np.mean(self.P)))
        self.history["mean_kappa"].append(float(np.mean(self.kappa)))
        self.history["state_var"].append(float(np.mean(np.var(self.H, axis=0))))
        self.history["num_edges"].append(float(self.A.sum() / 2.0))

    def snapshot(self) -> Dict[str, np.ndarray]:
        return {
            "H": self.H.copy(),
            "tau": self.tau.copy(),
            "uid": self.uid.copy(),
            "lineage_uid": self.lineage_uid.copy(),
            "parent_uid": self.parent_uid.copy(),
            "age": self.age.copy(),
        }

    def run(self, steps: int = 5000) -> None:
        for _ in range(steps):
            self.step()
        self.final_snapshot = self.snapshot()
