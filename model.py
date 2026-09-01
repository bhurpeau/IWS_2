import numpy as np
from dataclasses import dataclass

@dataclass
class IWSParameters:
    """Configuration parameters for the Inner World System (IWS) simulation."""
    n_nodes: int = 20
    dim: int = 2
    dt: float = 0.05
    
    # Trace and friction parameters
    gamma0: float = 0.25
    gamma_tau: float = 1.25
    alpha_trace: float = 0.45
    beta_trace: float = 0.12
    
    # Memory-induced geometry
    lambda_metric: float = 2.0
    
    # Internal pressure parameters
    kappa1: float = 0.6
    kappa2: float = 0.4
    kappa3: float = 0.5
    kappa_rel: float = 0.7
    
    # Kairos events parameters
    theta_kairos: float = 1.25
    rewire_top_k: int = 2
    cut_bottom_k: int = 2
    max_density: float = 0.35
    
    # Regime flags
    use_memory_geometry: bool = True
    use_kairos: bool = True

def row_normalize_with_self_loops(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    A_hat = A + np.eye(n)
    row_sums = A_hat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return A_hat / row_sums

class IWSSimulation:
    def __init__(self, params: IWSParameters, seed: int = 42):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.n = params.n_nodes
        self.d = params.dim
        self.dt = params.dt

        # Initial graph: sparse symmetric adjacency
        A_init = (self.rng.random((self.n, self.n)) < 0.15).astype(float)
        A_init = np.triu(A_init, 1)
        self.A = A_init + A_init.T

        self.H = self.rng.normal(0, 0.6, (self.n, self.d))
        self.V = self.rng.normal(0, 0.1, (self.n, self.d))
        self.tau = np.zeros((self.n, self.d))
        self.P = np.zeros(self.n)
        self.W = self.rng.normal(0, 0.3, (self.n, self.d))

        self.history = {
            "mean_trace": [], "mean_pressure": [], "num_kairos": [],
            "num_edges": [], "h_norm": [], "h_var": []
        }

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

    def step(self):
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
        else:
            kairos_nodes = np.array([], dtype=int)

        self._record_history(len(kairos_nodes))

    def apply_kairos(self, kairos_nodes: np.ndarray):
        tau_norm = np.linalg.norm(self.tau, axis=1, keepdims=True)
        tau_safe = self.tau / np.maximum(tau_norm, 1e-8)

        for i in kairos_nodes:
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

            density = self.A.sum() / (self.n * (self.n - 1))
            if density < self.params.max_density and len(non_neighbors) > 0:
                best = non_neighbors[np.argsort(scores[non_neighbors])[-self.params.rewire_top_k :]]
                for j in best:
                    self.A[i, j] = 1.0
                    self.A[j, i] = 1.0

            self.P[i] *= 0.35
            self.V[i] *= 0.5
            self.H[i] += self.rng.normal(0, 0.1, self.d)

    def _record_history(self, num_kairos_events: int):
        self.history["mean_trace"].append(np.mean(np.linalg.norm(self.tau, axis=1)))
        self.history["mean_pressure"].append(np.mean(self.P))
        self.history["num_kairos"].append(num_kairos_events)
        self.history["num_edges"].append(self.A.sum() / 2.0)
        self.history["h_norm"].append(np.mean(np.linalg.norm(self.H, axis=1)))
        self.history["h_var"].append(np.mean(np.var(self.H, axis=0)))

    def run(self, steps: int = 5000):
        for _ in range(steps):
            self.step()