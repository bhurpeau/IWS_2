"""Port d'entree u_i(t) (decision D-04).

Le moteur ne connait pas W directement : il recoit un objet d'entree.
Pour les comparaisons appariees (dt vs dt/2, bras apparies), les realisations
sont pre-generees sur la grille fine et sous-echantillonnees, afin qu'une
divergence du nombre d'appels au generateur ne desapparie pas les experiences.
"""
from __future__ import annotations

import numpy as np


class InputPort:
    """Interface : sample(step) -> tableau (n, d) pour le pas d'indice `step`."""

    name = "abstract"

    def sample(self, step: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


class ZeroInput(InputPort):
    name = "zero"

    def __init__(self, n: int, d: int):
        self._z = np.zeros((n, d))

    def sample(self, step: int) -> np.ndarray:
        return self._z  # exactement nul, jamais echantillonne


class FixedHeterogeneousInput(InputPort):
    """u_i(t) = epsilon * W_i, W fixe (configuration experimentale de [P1])."""

    name = "fixed_heterogeneous"

    def __init__(self, W: np.ndarray, epsilon: float):
        self.W = W
        self.epsilon = epsilon
        self._u = epsilon * W

    def sample(self, step: int) -> np.ndarray:
        return self._u

    def describe(self) -> dict:
        return {"name": self.name, "epsilon": self.epsilon}


class WhiteNoiseInput(InputPort):
    """Bruit blanc isotrope, constant par morceaux sur les pas de la grille
    GROSSIERE. Pre-genere pour l'appariement dt / dt/2 : la meme realisation
    physique u(t) est vue par les deux integrations.

    refine : nombre de pas fins par pas grossier (1 pour la grille grossiere).
    """

    name = "white_noise"

    def __init__(self, n: int, d: int, sigma: float, coarse_steps: int,
                 rng: np.random.Generator, refine: int = 1):
        self.sigma = sigma
        self.refine = refine
        # une valeur par pas GROSSIER, tiree une fois pour toutes
        self._values = rng.normal(0.0, sigma, size=(coarse_steps, n, d))

    def sample(self, step: int) -> np.ndarray:
        coarse = min(step // self.refine, len(self._values) - 1)
        return self._values[coarse]

    def describe(self) -> dict:
        return {"name": self.name, "sigma": self.sigma, "refine": self.refine}


def build_input(config, W: np.ndarray, rng_input: np.random.Generator,
                refine: int = 1, coarse_steps: int | None = None) -> InputPort:
    n, d = config.n_nodes, config.dim
    if config.input_mode == "zero":
        return ZeroInput(n, d)
    if config.input_mode == "fixed_heterogeneous":
        return FixedHeterogeneousInput(W, config.input_epsilon)
    if config.input_mode == "white_noise":
        cs = coarse_steps if coarse_steps is not None else config.steps
        return WhiteNoiseInput(n, d, config.input_sigma, cs, rng_input, refine)
    raise ValueError(f"Unknown input mode: {config.input_mode}")
