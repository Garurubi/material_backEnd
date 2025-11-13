# -*- coding: utf-8 -*-
"""
UMAP dimensionality reduction with persisted coordinates and fitted model.
- ensure_umap_cached(npy_path, model_path, X_emb, *, n_components, n_neighbors, min_dist, random_state, verbose=False)
- Logs elapsed time into tokens_time_<base>.jsonl with step="umap"
"""
from __future__ import annotations

import umap
import asyncio

import numpy as np

async def ensure_umap_cached(X_emb: np.ndarray, *, 
                       n_components: int, n_neighbors: int, min_dist: float, 
                       random_state: int, verbose: bool = False) -> np.ndarray:

    # Clamp more strictly for small-N to avoid eigsh(k >= N) issue
    n_samples = int(getattr(X_emb, "shape", [0])[0] or 0)
    n_components_eff = max(2, min(int(n_components), max(2, n_samples - 2)))
    n_neighbors_eff = max(2, min(int(n_neighbors), max(2, n_samples - 1)))

    # Safe init: use 'spectral' when safe, else fallback to 'random'
    init_mode = "spectral" if (n_components_eff + 1) < n_samples else "random"

    reducer = umap.UMAP(
        n_components=n_components_eff,
        n_neighbors=n_neighbors_eff,
        min_dist=float(min_dist),
        random_state=int(random_state),
        init=init_mode,
        n_jobs=1,
        verbose=True if verbose else False,
    )

    # X = reducer.fit_transform(X_emb).astype("float32") # 비동기 지원을 안함
    X = await asyncio.to_thread(reducer.fit_transform, X_emb)
    X = X.astype("float32")

    return X
