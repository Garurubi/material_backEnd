# -*- coding: utf-8 -*-
"""
Gaussian Mixture clustering helpers.
- choose_k_by_bic(X, k_min, k_max): scan k with BIC and return best k (shows tqdm)
- fit_predict(X, k): fit GMM and return labels
Also appends basic metrics lines for the scans and fits.
"""
from __future__ import annotations

import numpy as np

from sklearn.mixture import GaussianMixture

try:
    from tqdm import tqdm
except Exception:
    raise ImportError("tqdm is required. Install with `pip install tqdm`.")

from ..config.model_config import NB_RANDOM_STATE, NB_THRESHOLD

def choose_k_by_bic(X: np.ndarray, k_min: int, k_max: int) -> int:
    n_samples = X.shape[0]
    k_min = max(1, int(k_min))
    k_max = min(max(k_min, int(k_max), 1), n_samples)
    bbest = float("inf"); kbest = k_min
 
    for k in tqdm(range(k_min, k_max+1), desc="[GMM] BIC scan", unit="k"):
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=NB_RANDOM_STATE)
        gm.fit(X)
        bic = gm.bic(X)
        if bic < bbest:
            bbest, kbest = bic, k

    return int(kbest)

def fit_predict(X: np.ndarray, k: int, threshold: float = NB_THRESHOLD) -> np.ndarray:
    gm = GaussianMixture(n_components=int(k), covariance_type="full", random_state=NB_RANDOM_STATE)
    gm.fit(X)
    probs = gm.predict_proba(X)   # cluster probabilities for each sample

    labels = []
    for prob in probs:
        hits = np.where(prob > threshold)[0]
        if len(hits) == 0:
            # keep best cluster if none pass threshold
            hits = [int(np.argmax(prob))]
        labels.append(hits)

    return labels