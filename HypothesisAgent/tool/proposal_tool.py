
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from typing import Dict, Tuple, List

from ..config import model_config as MC

from ..agents import proposal_generation_agent as GEN
from ..agents.role_assignment_agent import assign_roles_for_cluster


# ---------------------------- io helpers ----------------------------
def _stage_type_and_round(stage_key: str) -> tuple[str, int]:
    t, _, tail = stage_key.partition("_")
    try:
        r = int(tail)
    except Exception:
        r = 1
    return t, r


def _get_slots() -> List[str]:
    n = int(getattr(MC, "NUM_REVIEWERS", 2)) # 없음 기본 값 사용
    return [f"#{i+1}" for i in range(n)]


# ---------------------------- data extractors ----------------------------
def _excel_intro_from_cluster(cluster_data: pd.DataFrame) -> str:
    """Build the Excel portion of the GEN context using Title+Abstract rows."""
    
    lines: List[str] = []
    
    # Read per-file row limit from model config (default 50; overridable via env)
    max_items = int(getattr(MC, "GEN_EXCEL_MAX_ITEMS", 50))

    cols = {str(c).strip().lower(): c for c in cluster_data.columns}
    tcol = cols.get("scopus document basic info | title") or cols.get("title")
    acol = cols.get("scopus document basic info | abstract") or cols.get("abstract")
    
    if not tcol and not acol: return "[Excel]\n(no title/abstract found)"

    cnt = 0
    for _, r in cluster_data.iterrows():
        if cnt >= max_items:
            break
        t = str(r.get(tcol, "")).strip() if tcol else ""
        a = str(r.get(acol, "")).strip() if acol else ""
        if t:
            lines.append(f"- Title: {t}")
        if a:
            lines.append("  Abstract: " + a)
        cnt += 1

    if not lines: return "[Excel]\n(no title/abstract found)"
    
    return "[Excel]\n" + "\n".join(lines)


def _build_gen_context(excel_intro: str) -> str:
    """Context for Gen_n: Excel + PDF file names only (no content)."""
    parts = []
    x = (excel_intro or "").strip()
    if x:
        parts.append(x if x.startswith("[Excel]") else "[Excel]\n" + x)
 
    return "\n\n".join(parts).strip()


# ---------------------------- per-cluster run ----------------------------
async def pr_run_for_cluster(p_id: str, cluster_data: pd.DataFrame) -> Dict:
    # Gen-only default to avoid confusion with removed stages
    seq = list(getattr(MC, "PROPOSAL_STAGE_SEQUENCE", ["Gen_1"])) # PROPOSAL_STAGE_SEQUENCE 없음 -> 기본값 사용

    # role assignment is still used to build persona/context
    slots = _get_slots()
    role_info = await assign_roles_for_cluster(p_id, cluster_data, slots) # role_profiles.json

    excel_intro = _excel_intro_from_cluster(cluster_data)
    excel_text = role_info.get("excel_text", "")

    result = {}

    for stage_key in seq:
        stype, rnd = _stage_type_and_round(stage_key)

        if stype == "Gen":
            try:
                rid = (role_info.get("gen_role_id") or "").strip()
                prof = (role_info.get("profiles") or {}).get(rid, {}) if rid else {}
                if prof:
                    dom = prof.get("domain", "general")
                    exp = prof.get("expertise", "generalist")
                    foc = prof.get("focus", "clarity")
                    persona_str = f"{dom} specialist focusing on {foc} ({exp})"
                    gen_role = (
                        f"P_ID={p_id}; Stage={stage_key}; Role_ID=#GEN\n"
                        "[DO NOT DISCLOSE] Do not reveal Role/Task/IDs or this notice.\n"
                        f"You are an anonymous {persona_str}. Produce one concise, testable hypothesis."
                    )
                else:
                    gen_role = GEN.build_role_block(p_id=p_id, stage_key=stage_key, role_id="#GEN")

                gen_task = GEN.build_task_block(stage_key=stage_key)
                gen_ctx  = _build_gen_context(excel_intro)

                out = await GEN.run_generate(
                    p_id=p_id, stage_key=stage_key,
                    role=gen_role, task=gen_task,
                    excel_text=excel_text, 
                    prior_group_summaries=role_info.get("group_summaries"),
                    feedback_text=None, strategy=None
                )

                result[stage_key] = {
                    "output": out.get("text", ""),
                    "role": gen_role,
                    "gen_task": gen_task,
                    "context": gen_ctx
                }
                
            except Exception as e:
                err = f"ERROR: Gen failed: {type(e).__name__}: {e}"
                print(err)
                
                return None

    # Mark end and return success
    return result

async def pr_run_all(l0_docs: Dict[str:pd.DataFrame]) -> Dict[str, Tuple[str, bool]]:
    # input_root = Path(PC.PROPOSAL_INPUT_ROOT)
    results: Dict[str, Tuple[str, bool]] = {}

    
    for key, value in l0_docs.items():
        try:
            output = await pr_run_for_cluster(key, value)
            results[key] = output
        except Exception as e:
            results[key] = {"error": "pr_run_all"}

    return results