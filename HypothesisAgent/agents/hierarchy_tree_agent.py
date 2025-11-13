
# -*- coding: utf-8 -*-
from __future__ import annotations

import re

from typing import List

# Config-driven thresholds
from ..config.model_config import (
    MIN_CLUSTERS_TO_CONTINUE,
    SIMILARITY_STOP_THRESHOLD,
    SIMILARITY_METRIC,
)

# LLM client
from ..core.llm_client import LLMClient

# Optional: similarity backends
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    np = None
    TfidfVectorizer = None

# Keep a small state to report previous level cluster counts if needed
_prev_cluster_count = {}

def _set_prev_count(level: int, count: int) -> None:
    _prev_cluster_count[int(level)] = int(count)

def _get_prev_count(level: int) -> int:
    try:
        return int(_prev_cluster_count.get(int(level), -1))
    except Exception:
        return -1

def _parse_yes_no(answer: str) -> bool:
    a = (answer or "").strip().lower()
    
    # accept strict single-word answers
    if a == "yes":
        return True
    if a == "no":
        return False
    
    # heuristic fallback
    yes = re.search(r"\b(yes|continue|proceed|go to next|generalize)\b", a) is not None
    no  = re.search(r"\b(no|stop|halt|sufficient|final|do not continue)\b", a) is not None
    
    if yes and not no:
        return True
    if no and not yes:
        return False
    
    return False  # conservative default

# -----------------------------
# Similarity computation utils
# -----------------------------
def _mean_pairwise_sim_tfidf(texts: list[str]) -> float:
    if TfidfVectorizer is None or np is None:
        return _mean_pairwise_sim_jaccard(texts)
    if len(texts) < 2:
        return 1.0
    vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
    X = vec.fit_transform([t or "" for t in texts])
    sim_mat = (X @ X.T).toarray()  # cosine sim because rows are L2-normalized
    n = sim_mat.shape[0]
    triu = sim_mat[np.triu_indices(n, k=1)]
    
    return float(triu.mean()) if triu.size else 1.0


def _mean_pairwise_sim_jaccard(texts: list[str]) -> float:
    if len(texts) < 2:
        return 1.0
    def tok(s: str) -> set[str]:
        # Simple tokenization: lowercase alphas only
        return set(w for w in (s or "").lower().split() if w.isalpha())
    
    sets = [tok(t) for t in texts]
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            inter = len(a & b); union = len(a | b) or 1
            sims.append(inter / union)
    
    return float(sum(sims) / len(sims)) if sims else 1.0


def _mean_pairwise_similarity(texts: list[str]) -> float:
    try:
        if SIMILARITY_METRIC.lower() == "jaccard":
            return _mean_pairwise_sim_jaccard(texts)
        
        return _mean_pairwise_sim_tfidf(texts)
    except Exception:
        # Fallback if any error occurs
        return _mean_pairwise_sim_jaccard(texts)


# ---------------------------------
# Main decision: should we continue
# ---------------------------------
async def judge_continue(level_tag: str, summaries: List[str]) -> bool:
    """
    Decide whether to proceed to the next hierarchy level.

    Hard rules (no LLM call):
    - If cluster count is below MIN_CLUSTERS_TO_CONTINUE -> NO
    - If mean pairwise similarity >= SIMILARITY_STOP_THRESHOLD -> NO

    Only if both rules pass, ask the LLM to decide conservatively.
    """
    # Parse numeric level for logging
    try:
        level_num = int(str(level_tag).lstrip("L"))
    except Exception:
        level_num = None

    cluster_count = len(summaries or [])

    # RULE A: count-based stop
    if cluster_count < MIN_CLUSTERS_TO_CONTINUE:
        msg = (f"[{level_tag}] judge_continue: NO (rule-count) — "
               f"clusters={cluster_count} < {MIN_CLUSTERS_TO_CONTINUE}")
        print(msg, flush=True)
        
        return False

    # RULE B: similarity-based stop
    mean_sim = _mean_pairwise_similarity(summaries)
    if mean_sim >= SIMILARITY_STOP_THRESHOLD:
        msg = (f"[{level_tag}] judge_continue: NO (rule-sim) — "
               f"mean_sim={mean_sim:.3f} >= {SIMILARITY_STOP_THRESHOLD}")
        print(msg, flush=True)
        
        return False

    # If both rules pass, consult the LLM (strict two-line output)
    client = LLMClient(agent_key="hierarchy_tree", level=level_num)
    system = "You decide whether to add exactly one higher-level parent topic. Be conservative and default to NO. Respond in English."
    bullets = "\n".join([f"- {s}" for s in summaries])
    user = (
        f"Level: {level_tag}\n"
        f"Number of summaries: {cluster_count}\n\n"
        "TASK\n"
        "Decide if adding ONE new parent topic above this set would make the list easier to understand and navigate.\n"
        "Consider EVERY summary exactly as given (do not ignore any item).\n\n"
        "DECISION RULE (default NO)\n"
        "Say YES only if BOTH are true:\n"
        "A) The items naturally partition into 2–3 clearly different topics that readers would expect to browse separately; and\n"
        "B) Introducing one parent topic (with 2–3 child groups) would materially reduce confusion or cognitive load.\n"
        "Otherwise, say NO. If there are fewer than {MIN_CLUSTERS_TO_CONTINUE} summaries, say NO. If you are unsure, say NO. Do not invent facts.\n\n"
        "ALL SUMMARIES\n"
        f"{bullets}\n\n"
        "OUTPUT (exactly two lines, nothing else)\n"
        "Line 1: YES or NO (uppercase)\n"
        "Line 2: If YES, give 2–3 short topic labels, comma-separated (≤ 12 words total). If NO, give one short reason (≤ 12 words)."
    )

    answer = await client.chat(system, user, step="llm.hierarchy_decider")

    # Print and log LLM decision
    print(f"[{level_tag}] judge_continue rationale:\n{answer}", flush=True)

    return _parse_yes_no(answer)