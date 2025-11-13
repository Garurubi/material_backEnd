import pandas as pd

from typing import List, Dict

def _detect_top_level(base: List[Dict]) -> int:
    try:
        
        lvls = []
        for line in base:
            if not line: continue
            v=line.get("level")
            try:
                if v is not None: lvls.append(int(v))
            except Exception:
                pass
        if lvls: top = int(max(lvls))
    except Exception:
        pass
    return top


async def run_emerging_pipeline(vector_df: pd.DataFrame, output_jsonl: List[Dict] | None = None, top_level: int | None = None) -> Dict:
    if top_level is None:
        top_level = _detect_top_level(output_jsonl)

    try:
        from ..core.classification import run_emerging_pipeline
        arts = run_emerging_pipeline(vector_df=vector_df, hierachy_result=output_jsonl, top_level=int(top_level))
        # print("[EMERGING] Done.", flush=True)
        return arts
    except Exception as e:
        print(f"[ERROR] Emerging classification failed: {e}", flush=True)
        return {}