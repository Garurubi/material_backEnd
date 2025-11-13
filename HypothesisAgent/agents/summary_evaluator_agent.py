# -*- coding: utf-8 -*-
"""Summary evaluator agent
- Scores candidate summaries (faithfulness & answer relevancy)
- Logs tokens/time with full context via LLMClient
"""
from __future__ import annotations

import os
import pandas as pd

from typing import List, Set

from ..core.llm_client import LLMClient
from ..config.model_config import get_summary_settings

# ---- Global input budget (character-based) ----
CHAR_BUDGET = int(os.getenv("INPUT_CHAR_BUDGET", "800000"))  # conservative cap


def select_with_char_budget(items, budget=CHAR_BUDGET):
    """Greedy selection up to a total character budget; keeps order."""
    out, total = [], 0
    for s in (items or []):
        s0 = str(s or "").strip()
        L = len(s0)
        if total + L + 1 > budget:
            break
        out.append(s0)
        total += L + 1
    return out


def _word_set(s: str) -> Set[str]:
    """Tokenize into lowercase whitespace-separated tokens."""
    return set(str(s or "").lower().split())


def _overlap_ratio(summary: str, contexts: List[str]) -> float:
    """Overlap ratio of summary tokens against concatenated context tokens (|A∩B| / |B|)."""
    base = " ".join(contexts or [])
    bw = _word_set(base)
    sw = _word_set(summary)
    return (len(bw & sw) / max(len(bw), 1)) if bw else 0.0


def shrink_contexts(texts: List[str]) -> List[str]:
    """Trim each context to MAX_CHARS_EVAL_INPUT (prompt-only; summaries are never sliced)."""
    ss = get_summary_settings()
    maxc = int(getattr(ss, "MAX_CHARS_EVAL_INPUT", 6000))
    return [str(t or "")[:maxc] for t in texts]


async def evaluate_all_candidates(level: int, cid: int, cluster_id: str,
                            cands_df: pd.DataFrame, shrunk_contexts: List[str]) -> pd.DataFrame:
    """Return DataFrame with columns:
        attempt, summary, faithfulness_score, relevance_score, is_multiline, overlap_ratio, Selected.
    Policy:
      1) Any multi-line candidate (contains \n or \r) is DISQUALIFIED.
      2) Any candidate with overlap ratio > OVERLAP_CUTOFF (default 0.7) is DISQUALIFIED.
      3) Only survivors are scored by the LLM; select by max min(faithfulness, relevance).
    NOTE:
      - No post-processing to force one line (filter-only enforcement).
      - Do NOT include 'cluster_id' in returned columns.
    """
    # Guard
    if cands_df is None or cands_df.empty:
        return cands_df if cands_df is not None else pd.DataFrame(columns=[
            "attempt","summary","faithfulness_score","relevance_score","is_multiline","overlap_ratio","Selected"]
        )

    # Ensure 'attempt'
    if "attempt" not in cands_df.columns:
        cands_df = cands_df.assign(attempt=list(range(1, len(cands_df)+1)))

    # Early filtering BEFORE LLM scoring
    ss = get_summary_settings()
    cutoff = float(getattr(ss, "OVERLAP_CUTOFF", 0.7))

    # 1) Drop multi-line
    cands_df = cands_df.copy()
    cands_df["__is_multiline__"] = cands_df["summary"].map(lambda s: "\n" in str(s) or "\r" in str(s))

    # 2) Drop high-overlap (> cutoff) vs contexts
    cands_df["__overlap__"] = cands_df["summary"].map(lambda s: _overlap_ratio(str(s), shrunk_contexts))

    filtered = cands_df.loc[(~cands_df["__is_multiline__"]) & (cands_df["__overlap__"] <= cutoff)].copy()

    # Fallbacks
    if filtered.empty:
        filtered = cands_df.loc[(~cands_df["__is_multiline__"])].copy()
    if filtered.empty:
        filtered = cands_df.copy()

    # Use filtered pool only
    cands_df = filtered.drop(columns=["__is_multiline__","__overlap__"], errors="ignore")

    # LLM evaluation
    client = LLMClient(agent_key="summary_evaluator", level=level, cid=cid, cluster_id=cluster_id)
    system = "You are a careful evaluator. Respond in English with numeric scores 0.0–1.0."

    items = []
    for _, r in cands_df.iterrows():
        items.append(f"- Attempt {int(r.get('attempt', 0))}: {str(r.get('summary','')).strip()}")
    ctx_items = select_with_char_budget(shrunk_contexts or [], budget=CHAR_BUDGET)
    ctx = "\n".join([f"* {t}" for t in ctx_items])

    user = (
        "Given the context snippets and the candidate summaries, assign two scores per attempt:\n"
        "1) Faithfulness (is the summary supported by the context?)\n"
        "2) Answer relevancy (does it capture the main theme?)\n"
        "Be conservative: any exaggeration, speculation, or biased framing must reduce the scores. "
        "If a summary overemphasizes only one aspect or omits diversity in the context, lower its relevancy. "
        "Only summaries that are strictly accurate and balanced should score above 0.8.\n"
        "Return a JSON list like: [{\"attempt\":1, \"faithfulness\":0.9, \"relevancy\":0.85}, ...]\n\n"
        f"Context:\n{ctx}\n\nCandidates:\n" + "\n".join(items)
    )

    txt = await client.chat(system, user, step="llm.summary_evaluator")

    # Parse JSON safely
    import json
    try:
        parsed = json.loads(txt)
        if not isinstance(parsed, list):
            parsed = []
    except Exception:
        parsed = []

    # Scores DF
    df_scores = pd.DataFrame(parsed)
    if not isinstance(df_scores, pd.DataFrame) or df_scores.empty:
        df_scores = pd.DataFrame(columns=["attempt","faithfulness_score","relevance_score"])

    # Normalize score column names
    if "faithfulness" in df_scores.columns:
        df_scores.rename(columns={"faithfulness": "faithfulness_score"}, inplace=True)
    if "relevancy" in df_scores.columns:
        df_scores.rename(columns={"relevancy": "relevance_score"}, inplace=True)
    if "attempt" not in df_scores.columns and not df_scores.empty:
        df_scores["attempt"] = list(range(1, len(df_scores)+1))

    # Merge scores
    base_scores = df_scores[["attempt","faithfulness_score","relevance_score"]] if not df_scores.empty else pd.DataFrame(columns=["attempt","faithfulness_score","relevance_score"])
    merged = cands_df.merge(base_scores, on="attempt", how="left")

    # Diagnostics for JSON logging
    merged["is_multiline"] = merged["summary"].map(lambda s: "\n" in str(s) or "\r" in str(s))
    merged["overlap_ratio"] = merged["summary"].map(lambda s: _overlap_ratio(str(s), shrunk_contexts))

    # Select best by max of min(faithfulness, relevance)
    merged["min_pair"] = merged[["faithfulness_score","relevance_score"]].min(axis=1)
    try:
        if merged["min_pair"].notna().any():
            best_idx = merged["min_pair"].fillna(-1).idxmax()
        else:
            best_idx = merged.index[0]
    except Exception:
        best_idx = merged.index[0]

    merged["Selected"] = "No"
    merged.loc[best_idx, "Selected"] = "Yes"

    # Return selected columns only
    return merged[["attempt","summary","faithfulness_score","relevance_score","is_multiline","overlap_ratio","Selected"]]