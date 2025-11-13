# -*- coding: utf-8 -*-
"""
I/O utilities, minimal data munging, and JSONL writers.
- aggregate_meta_from_group: collect eid/year/citation stats from a group
"""
from __future__ import annotations

import pandas as pd

from typing import Dict, List

def _choose_group_key(df: pd.DataFrame) -> str:
    for k in ["cluster_id","cid","Cluster ID","Cluster_ID","group_id","_single_group"]:
        if k in df.columns:
            return k
    df["_single_group"] = 0
    return "_single_group"


def _extract_texts_for_cluster(g: pd.DataFrame) -> list[str]:
    # Prefer 'text', fall back to any string-like columns joined
    if "text" in g.columns:
        return g["text"].astype(str).fillna("").tolist()
    if "summary" in g.columns:
        return g["summary"].astype(str).fillna("").tolist()
    str_cols = [c for c in g.columns if g[c].dtype == object]
    if str_cols:
        return g[str_cols].astype(str).fillna("").agg(" ".join, axis=1).tolist()
    return [""]


def aggregate_meta_from_group(g: pd.DataFrame) -> Dict[str, any]:
    """Collect docs list & stats from a cluster group.
       Columns are resolved in a case-insensitive way:
       - eid: ['eid']
       - year: ['publication year','year']
       - citations: ['number of citation','citations','citation','cited by']
    """
    cols = {c.lower(): c for c in g.columns}
    
    eid_col = cols.get("eid")
    year_col = cols.get("publication year") or cols.get("year")
    cit_col = (cols.get("number of citation") or cols.get("citations") or
               cols.get("citation") or cols.get("cited by"))
    
    docs: List[Dict[str, any]] = []
    years: List[int] = []
    csum = 0
    
    for _, row in g.iterrows():
        eid = row[eid_col] if eid_col in g.columns else None
        year = row[year_col] if year_col in g.columns else None
        citations = row[cit_col] if cit_col in g.columns else 0
        try:
            year = int(year) if str(year).strip() != "" else None
        except Exception:
            year = None
        
        try:
            citations = int(citations) if str(citations).strip() != "" else 0
        except Exception:
            citations = 0
        
        if year is not None:
            years.append(year)
        
        if eid is not None and str(eid).strip() != "":
            docs.append({"eid": str(eid), "year": year, "citations": citations})
        
        csum += citations
    
    years_sorted = sorted([y for y in years if y is not None])
    
    return {
        "doc_count": int(len(g)),
        "citation_sum": int(csum),
        "years_sorted": years_sorted,
        "docs": docs,
    }