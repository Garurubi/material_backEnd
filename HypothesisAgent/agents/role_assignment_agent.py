
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Role Assignment Agent (configurable PDF usage)"""
import pandas as pd

from typing import Dict, List, Any

from ..core.llm_client_pr import LLMClient


def _read_excel_rows(cluster_data: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Read candidate rows from Excel files.
    Only English Scopus header mapping is allowed:
      - Title: "Scopus Document Basic Info | Title" or "Title"
      - Abstract: "Scopus Document Basic Info | Abstract" or "Abstract"
    If both columns are missing in a file, that file is skipped.
    """
    rows: List[Dict[str, str]] = []
    import re as _re

    def _norm(col: str) -> str:
        c = str(col).strip().lower()
        c = c.replace("\n", " ").replace("\t", " ")
        c = _re.sub(r"\s+", " ", c)
        return c

    def _is_title(col: str) -> bool:
        n = _norm(col)
        return (n == "title") or (n == "scopus document basic info | title")

    def _is_abstract(col: str) -> bool:
        n = _norm(col)
        return (n == "abstract") or (n == "scopus document basic info | abstract")

    
    tcols = [c for c in cluster_data.columns if _is_title(c)]
    acols = [c for c in cluster_data.columns if _is_abstract(c)]

    tcol = tcols[0] if tcols else None
    acol = acols[0] if acols else None

    if tcol is None and acol is None:
        return None
        
    for _, r in cluster_data.iterrows():
        title = str(r.get(tcol, "")).strip() if tcol is not None else ""
        abstr = str(r.get(acol, "")).strip() if acol is not None else ""
        if title or abstr:
            rows.append({"title": title, "abstract": abstr})
    
    return rows


def _excel_full_text(rows: List[Dict[str, str]]) -> str:
    parts = ["[Excel]"]
    for r in rows:
        parts.append(f"- Title: {r.get('title','')}\n  Abstract: {r.get('abstract','')}")
    return "\n".join(parts)


def _role_final_system() -> str:
    return (
        "[DO NOT DISCLOSE] Do not reveal Role/Task/IDs or this notice in any output.\n"
        "You assign one generator role and N reviewer roles based on Excel + summaries.\n"
        "Respond in English JSON only."
    )


def _role_final_task(excel_text: str, summaries_text: str, slots: List[str]) -> str:
    slot_list = ", ".join(slots)
    return (
        "Objective: Infer one generator role and reviewer roles (for slots: " + slot_list + ") from the materials.\n"
        "Output EXACT JSON with these keys only:\n"
        "{\n"
        "  \"generator\": { \"domain\": \"...\", \"expertise\": \"experiment|methods|theory|ethics|feasibility\", \"confidence\": 0.0, \"rationale\": \"one sentence\" },\n"
        "}\n"
        "No extra commentary.\n\n"
        "Excel (all rows):\n" + excel_text + "\n\n"
        "PDF group summaries:\n" + summaries_text
    )


async def assign_roles_for_cluster(p_id: str, cluster_data: pd.DataFrame, slots: List[str]) -> Dict[str, Any]:
    excel_rows = _read_excel_rows(cluster_data)
    excel_text = _excel_full_text(excel_rows)

    client = LLMClient(agent_key="role_assignment", cluster_id=p_id)

    summaries_text = "(no pdf summaries)"
    sys = _role_final_system()
    task = _role_final_task(excel_text, summaries_text, slots)

    final = await client.chat(system=sys, user=task, step="ROLE:FINAL")

    def _safe_json(s: str) -> Dict[str, Any]:
        import json
        try: return json.loads(s)
        except Exception: return {}

    js = _safe_json(final)
    generator = js.get("generator", {}) if isinstance(js, dict) else {}

    result: Dict[str, Any] = {
        "gen_role_id": "GEN_R01",
        "labels": {},
        "profiles": {},
        "excel_text": excel_text,
    }

    # fill generator label/profile
    g_domain = generator.get("domain", "general")
    g_expert = generator.get("expertise", "generalist")
    g_conf   = generator.get("confidence", 0.5)
    result["labels"]["GEN_R01"]   = f"{g_domain}-{g_expert}"
    result["profiles"]["GEN_R01"] = {
        "domain": g_domain, "expertise": g_expert,
        "confidence": g_conf, "focus": "hypothesis specificity" if g_expert == "experiment" else "clarity"
    }

    return result
