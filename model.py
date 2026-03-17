from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict

import numpy as np


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class IWS2Parameters:
    n_nodes: int = 10
    dim: int = 2
    dt: float = 0.05
    max_nodes: int = 300

    use_division: bool = True
    use_apoptosis: bool = True
    use_senescence: bool = True

    theta_div: float = 3.0
    theta_safe: float = 0.8
    theta_kappa: float = 0.5
    alpha_rho: float = 5.0

    theta_death: float = 1.2
    deg_min: int = 2

    kappa0: float = 10.0
    delta_div: float = 0.3
    theta_frag: float = 1.5
    theta_age: int = 200

    trace_decay: float = 0.995
    state_noise: float = 0.08
    division_noise: float = 0.05

    a_tau: float = 1.5
    a_P: float = 1.2
    a_kappa: float = 1.0
    a_R: float = 0.08
    a_age: float = 0.5
    a_frag: float = 0.8

    R0: float = 100.0
    R_regen: float = 0.5
    cost_div: float = 1.0
    gain_death: float = 0.5
    R_offset: float = 10.0

    p_inherit: float = 0.3
    velocity_decay = 0.9
    h_damping = 1.0
    tau_coupling = 0.08
    trace_decay = 0.98
    trace_write = 0.02
    state_noise = 0.02
    intrinsic_drive = 0.01
    h_clip = 10.0
    v_clip = 10.0
    tau_clip = 5.0

class Node:
    def __init__(self, idx: int, dim: int, kappa0: float):
        self.id = idx
        self.H = np.random.normal(0.0, 1.0, size=dim)
        self.V = np.random.normal(0.0, 0.2, size=dim)
        self.tau = np.zeros(dim, dtype=float)
        self.kappa = float(kappa0)
        self.age = 0
        self.neighbors: set[int] = set()
        self.lineage_uid = idx


class IWSSimulationPaper2:
    def __init__(self, params: IWS2Parameters, seed: int = 42):
        self.params = params
        self.dim = params.dim
        np.random.seed(seed)
        random.seed(seed)

        self.nodes: Dict[int, Node] = {}
        self.next_id = 0
        self.R = float(params.R0)

        self.divisions_this_step = 0
        self.apoptoses_this_step = 0

        self.history = {
            "population_size": [],
            "num_divisions": [],
            "num_apoptosis": [],
            "mean_kappa": [],
            "mean_trace": [],
            "state_var": [],
        }

        for _ in range(params.n_nodes):
            self.add_node()

        self._refresh_arrays()

    def add_node(self) -> Node:
        node = Node(self.next_id, self.dim, self.params.kappa0)
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node

    def _refresh_arrays(self) -> None:
        ordered_ids = sorted(self.nodes.keys())
        self.node_ids = ordered_ids
        if not ordered_ids:
            self.H = np.zeros((0, self.dim))
            self.V = np.zeros((0, self.dim))
            self.tau = np.zeros((0, self.dim))
            self.kappa = np.zeros(0)
            self.lineage_uid = np.zeros(0, dtype=int)
            return
        self.H = np.stack([self.nodes[i].H for i in ordered_ids], axis=0)
        self.V = np.stack([self.nodes[i].V for i in ordered_ids], axis=0)
        self.tau = np.stack([self.nodes[i].tau for i in ordered_ids], axis=0)
        self.kappa = np.array([self.nodes[i].kappa for i in ordered_ids], dtype=float)
        self.lineage_uid = np.array([self.nodes[i].lineage_uid for i in ordered_ids], dtype=int)

    def compute_pressure(self, node: Node) -> float:
        return float(np.linalg.norm(node.H) + 0.35 * np.linalg.norm(node.V))


    def update_state(self, node: Node) -> None:
        noise = np.random.normal(0.0, self.params.state_noise, size=self.dim)
    
        node.V = (
            self.params.velocity_decay * node.V
            + self.params.dt * (
                -self.params.h_damping * node.H
                + self.params.tau_coupling * node.tau
            )
            + noise
            + self.params.intrinsic_drive
        )
    
        node.H = node.H + self.params.dt * node.V
    
        node.V = np.clip(node.V, -self.params.v_clip, self.params.v_clip)
        node.H = np.clip(node.H, -self.params.h_clip, self.params.h_clip)
    
    
    def update_trace(self, node: Node) -> None:
        node.tau = (
            self.params.trace_decay * node.tau
            + self.params.trace_write * node.H
        )
        node.tau = np.clip(node.tau, -self.params.tau_clip, self.params.tau_clip)

    def division_probability(self, node: Node) -> float:
        P = self.compute_pressure(node)
        x = (
            self.params.a_tau * (np.linalg.norm(node.tau) - self.params.theta_div)
            - self.params.a_P * (P - self.params.theta_safe)
            + self.params.a_kappa * (node.kappa - self.params.theta_kappa)
            + self.params.a_R * (self.R - self.params.R_offset)
        )
        return float(sigmoid(x))

    def apoptosis_probability(self, node: Node) -> float:
        P = self.compute_pressure(node)
        x = (
            1.0 * (self.params.theta_kappa - node.kappa)
            + self.params.a_frag * (P - self.params.theta_frag)
            + self.params.a_age * ((node.age - self.params.theta_age) / max(1, self.params.theta_age))
        )
        return float(sigmoid(x))

    def divide(self, node: Node) -> None:
        if len(self.nodes) >= self.params.max_nodes:
            return

        rho_base = np.random.beta(self.params.alpha_rho, self.params.alpha_rho)
        rho_H = np.clip(rho_base + np.random.normal(0.0, 0.05), 0.01, 0.99)
        rho_V = np.clip(rho_base + np.random.normal(0.0, 0.05), 0.01, 0.99)

        H_parent = node.H.copy()
        V_parent = node.V.copy()
        tau_parent = node.tau.copy()

        child = Node(self.next_id, self.dim, self.params.kappa0)
        child.lineage_uid = node.lineage_uid

        noise_H_child = np.random.normal(0.0, self.params.division_noise, size=self.dim)
        noise_H_parent = np.random.normal(0.0, self.params.division_noise, size=self.dim)
        noise_V_child = np.random.normal(0.0, self.params.division_noise, size=self.dim)
        noise_V_parent = np.random.normal(0.0, self.params.division_noise, size=self.dim)

        child.H = (1.0 - rho_H) * H_parent + noise_H_child
        node.H = rho_H * H_parent + noise_H_parent

        child.V = (1.0 - rho_V) * V_parent + noise_V_child
        node.V = rho_V * V_parent + noise_V_parent

        child.tau = tau_parent + np.random.normal(0.0, self.params.division_noise, size=self.dim)

        child.neighbors.add(node.id)
        parent_neighbors = list(node.neighbors)
        for neigh_id in parent_neighbors:
            if neigh_id != node.id and random.random() < self.params.p_inherit and neigh_id in self.nodes:
                child.neighbors.add(neigh_id)
                self.nodes[neigh_id].neighbors.add(child.id)
        node.neighbors.add(child.id)

        node.kappa = max(0.0, node.kappa - self.params.delta_div)
        child.kappa = max(0.0, node.kappa)

        self.nodes[child.id] = child
        self.next_id += 1
        self.R -= self.params.cost_div
        self.divisions_this_step += 1

    def apoptosis(self, node_id: int) -> None:
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        for neigh_id in list(node.neighbors):
            if neigh_id in self.nodes:
                self.nodes[neigh_id].neighbors.discard(node_id)
        del self.nodes[node_id]
        self.R += self.params.gain_death
        self.apoptoses_this_step += 1

    def _record_history(self) -> None:
        self._refresh_arrays()
        n = len(self.nodes)
        self.history["population_size"].append(float(n))
        self.history["num_divisions"].append(float(self.divisions_this_step))
        self.history["num_apoptosis"].append(float(self.apoptoses_this_step))
        self.history["mean_kappa"].append(float(np.mean(self.kappa)) if n > 0 else 0.0)
        self.history["mean_trace"].append(float(np.mean(np.linalg.norm(self.tau, axis=1))) if n > 0 else 0.0)
        self.history["state_var"].append(float(np.mean(np.var(self.H, axis=0))) if n > 0 else 0.0)

    def step(self) -> None:
        self.divisions_this_step = 0
        self.apoptoses_this_step = 0

        node_ids = list(self.nodes.keys())
        for i in node_ids:
            if i not in self.nodes:
                continue
            node = self.nodes[i]
            self.update_state(node)
            self.update_trace(node)
            node.age += 1

        for i in node_ids:
            if i not in self.nodes:
                continue
            node = self.nodes[i]

            if self.params.use_division and random.random() < self.division_probability(node):
                self.divide(node)

            if i not in self.nodes:
                continue
            node = self.nodes[i]

            if self.params.use_apoptosis:
                if self.compute_pressure(node) > self.params.theta_death and len(node.neighbors) < self.params.deg_min:
                    self.apoptosis(i)
                    continue
                if self.params.use_senescence and i in self.nodes and random.random() < self.apoptosis_probability(self.nodes[i]):
                    self.apoptosis(i)

        self.R += self.params.R_regen
        self._record_history()
