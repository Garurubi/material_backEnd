# -*- coding: utf-8 -*-
"""Cluster summarization agent
- Generates multiple candidate summaries for one cluster
- Includes `cluster_id` only for logging context (never written to JSON outputs)
- Logs tokens/time through LLMClient with full context
"""
from __future__ import annotations

import os
import pandas as pd

from typing import List

from ..core.llm_client import LLMClient
from ..config.model_config import get_summary_settings


# ---- Global input budget (character-based) ----
CHAR_BUDGET = int(os.getenv("INPUT_CHAR_BUDGET", "800000"))  # 900k chars as conservative cap


def select_with_char_budget(items, budget=CHAR_BUDGET):
    """Greedy selection up to a total character budget.
    Keeps original order; stops when adding next would exceed (budget).
    """
    out, total = [], 0
    for s in items or []:
        s0 = str(s or "").strip()
        L = len(s0)
        if total + L + 1 > budget:
            break
        out.append(s0)
        total += L + 1
    return out


def _choose_texts_from_group(g: pd.DataFrame) -> List[str]:
    """Pick the best available text fields from a record group.
    Priority: (title + abstract) > summary > the first object-typed column.
    Keep existing ordering, do not alter text other than str() conversion.
    """
    cols_lower = {c.lower(): c for c in g.columns}
    if "title" in cols_lower and "abstract" in cols_lower:
        tcol, acol = cols_lower["title"], cols_lower["abstract"]
        return [(str(t or "") + " " + str(a or "")).strip() for t, a in zip(g[tcol], g[acol])]
    if "summary" in g.columns:
        return [str(x or "") for x in g["summary"].tolist()]
    for c in g.columns:
        if g[c].dtype == object:
            return [str(x or "") for x in g[c].tolist()]
 
    return [""]


def _shrink(texts: List[str], max_chars: int, max_items: int) -> List[str]:
    """Lightweight length guard for context; NO semantic post-processing.
    We only cap per-item characters and max_items to keep token cost reasonable.
    """
    out: List[str] = []
    for t in texts[:max_items]:
        t = str(t or "")
        out.append(t[:max_chars])
    
    return out


def _word_set(s: str) -> set[str]:
    return set(str(s or "").lower().split())


def _overlap_ratio(summary: str, contexts: List[str]) -> float:
    base = " ".join(contexts or [])
    bw = _word_set(base)
    sw = _word_set(summary)
    
    return (len(bw & sw) / max(len(bw), 1)) if bw else 0.0


async def generate_candidates(level: int, cid: int, cluster_id: str, g: pd.DataFrame) -> pd.DataFrame:
    """Generate k candidate summaries for a cluster via prompt-only constraints.
    Returns a DataFrame with columns: attempt, summary.
    Notes:
      - Absolutely no post-processing truncation; constraints are enforced via prompt only.
      - cluster_id is used only for logging context through LLMClient (not returned).
    """
    ss = get_summary_settings()
    k = int(getattr(ss, "SUMMARY_NUM_CANDIDATES", 3))
    maxc = int(getattr(ss, "SUMMARY_OUTPUT_CHAR_LIMIT", 100))
    max_retries = int(getattr(ss, "MAX_SUMMARY_RETRIES", 8))
    overlap_cutoff = float(getattr(ss, "OVERLAP_CUTOFF", 0.7))

    texts = _choose_texts_from_group(g)
    # Use evaluation limit to keep parity with evaluator's context size
    texts = _shrink(texts, int(getattr(ss, "MAX_CHARS_EVAL_INPUT", 6000)), max_items=int(getattr(ss, "MAX_ITEMS_EVAL_INPUT", 1000000)))
    texts = select_with_char_budget(texts, budget=CHAR_BUDGET)  # apply global char budget

    client = LLMClient(agent_key="summarizer", level=level, cid=cid, cluster_id=cluster_id)
    system = "You are an expert research summarizer. Respond in English."
    BULLET = "###"
    limit = maxc

    # Base prompt template (use {NEED} only; avoid str.format on arbitrary braces)
    base_user = (
        "Each candidate MUST be a SINGLE SENTENCE on ONE LINE, under {limit} characters.\n"
        "Do not emphasize only a single perspective or method; ensure the summaries reflect the variety of approaches and topics present in the cluster.\n"
        "Do not focus on isolated technical details; instead, highlight the overall research purposes or trends.\n"
        "Produce EXACTLY {NEED} candidate summaries for the cluster below.\n"
        "If ANY candidate contains a newline or spans multiple lines, that candidate is INVALID and will be DISQUALIFIED. NEVER output multi-line candidates.\n"
        "Do NOT copy text verbatim; distill the core idea. Avoid repetition.\n"
        "NO numbering, bullets, dashes, or prefixes. NO code fences. NO extra commentary.\n"
        "NO DUPLICATION with the context (do not reuse wording from the texts).\n\n"
        "STRICT OUTPUT FORMAT:\n"
        "Return exactly {NEED} blocks separated ONLY by the token '{BULLET}'.\n\n"
        "Texts:\n- " + "\n- ".join(texts)
    )

    async def _gen(need: int, retry_idx: int | None = None) -> list[str]:
        # Safe replacement: avoid .format() parsing other braces
        user = base_user.replace("{NEED}", str(need)).replace("{limit}", str(limit)).replace("{BULLET}", BULLET)
        step = "llm.summarizer" if retry_idx is None else f"llm.summarizer.retry{retry_idx}"

        out = await client.chat(system, user, step=step)
        parts = [p.strip() for p in out.split(BULLET) if p.strip()]
        
        return parts

    # Collect valid candidates up to k
    valid: list[str] = []
    seen_texts: set[str] = set()
    retries = 0

    while len(valid) < k and retries <= max_retries:
        need = k - len(valid)
        parts = await _gen(need, None if retries == 0 else retries)

        # Validate each part: one-line + low overlap + de-duplicate across attempts
        for s in parts:
            if len(valid) >= k:
                break
            
            s0 = str(s or "").strip()
            if ("\n" in s0) or ("\r" in s0):
                continue  # multi-line -> invalid
            
            if s0 in seen_texts:
                continue  # identical duplication across attempts
            
            ov = _overlap_ratio(s0, texts)
            if ov > overlap_cutoff:
                continue  # too overlapping with context
            valid.append(s0)
            seen_texts.add(s0)

        retries += 1

    # If still short, pad with best-effort last batch (no semantic edits, just duplication to reach k)
    if len(valid) < k:
        last = await _gen(k - len(valid), retries) if retries <= max_retries + 1 else []
        last = [str(x or "").splitlines()[0].strip() for x in last]  # keep first line only for padding
        while len(valid) < k and last:
            s0 = last.pop(0)
            if s0 and s0 not in seen_texts:
                valid.append(s0)
                seen_texts.add(s0)
        while len(valid) < k:
            valid.append("")  # ensure k rows

    rows = [{"attempt": i, "summary": s} for i, s in enumerate(valid[:k], start=1)]
    
    return pd.DataFrame(rows, columns=["attempt", "summary"])  # no cluster_id leakage