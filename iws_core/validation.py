"""Validation d'integrateur : dt vs dt/2 sur grille commune.

Pour chaque observable continue O_k :
    Delta_k = sup_t |O_k^dt(t) - O_k^{dt/2}(t)| / (1 + sup_t |O_k^{dt/2}(t)|)

Pour O3 (evenements discrets) : appariement des temps de Kairos tries et
ecart maximal sur les k premiers evenements communs. Une trajectoire n'est
PAS declaree convergente sur la seule proximite des moyennes finales.

Contrainte : l'appariement exige des entrees identiques sur la grille fine ;
le bruit blanc est pre-genere par pas grossier (inputs.WhiteNoiseInput) et
le coup de bruit de Kairos doit etre desactive (kick_sigma=0), sans quoi la
comparaison est declaree non appariable.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import IWSConfig
from .engine import IWSEngine

CONTINUOUS_KEYS = ("mean_trace", "mean_pressure", "h_norm", "h_var", "homophily")


def integrator_check(config: IWSConfig, seed: int, tol: float = 0.05) -> dict:
    if config.kairos_kick_sigma > 0.0 and config.use_kairos:
        return {"pairable": False,
                "reason": "kairos_kick_sigma>0 : realisations non appariables"}

    coarse = IWSEngine(config, seed).run()
    fine_cfg = replace(config, dt=config.dt / 2.0, steps=config.steps * 2)
    fine = IWSEngine(fine_cfg, seed,
                     input_refine=2, input_coarse_steps=config.steps).run()

    rc, rf = coarse.result(), fine.result()
    deltas = {}
    for k in CONTINUOUS_KEYS:
        a = rc["history"][k]
        b = rf["history"][k][1::2]  # sous-echantillonnage sur la grille grossiere
        m = min(len(a), len(b))
        deltas[k] = float(np.max(np.abs(a[:m] - b[:m])) / (1.0 + np.max(np.abs(b[:m]))))

    ta, tb = rc["kairos_times"], rf["kairos_times"]
    m = min(len(ta), len(tb))
    event_dt = float(np.max(np.abs(np.array(ta[:m]) - np.array(tb[:m])))) if m else 0.0
    count_gap = abs(len(ta) - len(tb))

    return {
        "pairable": True,
        "deltas": deltas,
        "max_delta": max(deltas.values()) if deltas else 0.0,
        "event_count": (len(ta), len(tb)),
        "event_count_gap": count_gap,
        "event_time_max_gap": event_dt,
        "converged": (max(deltas.values()) < tol if deltas else True),
        "tolerance": tol,
    }
