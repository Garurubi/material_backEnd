# -*- coding: utf-8 -*-
import os, json, math

import numpy as np
import pandas as pd

from typing import List, Optional, Dict, Tuple

from ..config.model_config import (CLASS_EMERGING_LABEL, CLASSIFICATION_METRICS, METRIC_FUNCTIONS, round_decimals_for_level, QUAD_X_THRESHOLD, QUAD_Y_THRESHOLD, CLASS_USE_FOUR_LABELS, QUADRANT_LABELS,)


# --- dynamic metric getters (single source of truth from CLASSIFICATION_METRICS) ---
def _get_metric_names():
    try:
        xn = CLASSIFICATION_METRICS.get('x_metric')
        yn = CLASSIFICATION_METRICS.get('y_metric')
    except Exception:
        xn, yn = None, None
    if not isinstance(xn, str) or not xn:
        try:
            keys = list(METRIC_FUNCTIONS.keys())
        except Exception:
            keys = []
        xn = keys[0] if keys else "x"
    if not isinstance(yn, str) or not yn:
        try:
            keys = list(METRIC_FUNCTIONS.keys())
        except Exception:
            keys = []
        yn = keys[1] if len(keys) > 1 else (keys[0] if keys else "y")
    return xn, yn


def _normalize_docs_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["eid", "year", "citations"])
    d = df.copy()
    
    # citations
    if "citations" not in d.columns:
        for cand in ["citedby_count", "citation_count", "cited_by", "n_citations"]:
            if cand in d.columns:
                d["citations"] = pd.to_numeric(d[cand], errors="coerce"); break
    else:
        d["citations"] = pd.to_numeric(d["citations"], errors="coerce")
    
    # year
    if "year" not in d.columns:
        for cand in ["pub_year", "publication_year", "year_published"]:
            if cand in d.columns:
                d["year"] = pd.to_numeric(d[cand], errors="coerce"); break
        if "year" not in d.columns and "date" in d.columns:
            s = pd.to_datetime(d["date"], errors="coerce"); d["year"] = s.dt.year
    else:
        d["year"] = pd.to_numeric(d["year"], errors="coerce")
    
    # eid
    if "eid" not in d.columns:
        for cand in ["id", "doc_id", "paper_id", "scopus_id"]:
            if cand in d.columns:
                d["eid"] = d[cand].astype(str); break
    keep = [c for c in ["eid","year","citations"] if c in d.columns]
    if "eid" not in keep: d["eid"] = np.arange(len(d)).astype(str); keep.append("eid")
    if "year" not in keep: d["year"] = np.nan; keep.append("year")
    if "citations" not in keep: d["citations"] = np.nan; keep.append("citations")
 
    return d[keep]


def _frames_docs_stats_from_rows(rows: List[dict]) -> Tuple[Dict[str, pd.DataFrame], 
                        Dict[str, pd.DataFrame], Dict[str, dict], Dict[str, str], int]:
    """
    Build per-level frames and docs map using ONLY the 'cluster_id' string as the canonical identifier.
    Fallback: if 'cluster_id' is missing, synthesize from level/cid as "L{level}_{cid}".
    """
    frames: Dict[str, pd.DataFrame] = {}
    docs_map: Dict[str, pd.DataFrame] = {}
    raw_stats: Dict[str, dict] = {}
    cluster_key_map: Dict[str, str] = {}
    top_level = -10

    for r in rows:
        cid_str = r.get("cluster_id")
        lvl = r.get("level", None)
        if not isinstance(cid_str, str) or not cid_str:
            try:
                lvl_int = int(lvl) if lvl is not None else -1
            except Exception:
                lvl_int = -1
            legacy = r.get("cid")
            cid_str = f"L{lvl_int}_{legacy}"
        try:
            lvl_int = int(lvl) if lvl is not None else int(str(cid_str).split("_")[0].replace("L",""))
        except Exception:
            try:
                lvl_int = int(str(cid_str).split("_")[0].replace("L",""))
            except Exception:
                lvl_int = -1

        top_level = max(top_level, lvl_int)
        dd = r.get("docs") or []
        docs_df = pd.DataFrame(dd) if isinstance(dd, list) else pd.DataFrame()
        docs_df = _normalize_docs_columns(docs_df)

        key = f"L{lvl_int}"
        row = {"Level": lvl_int, "ClusterID": str(cid_str)}
        
        if key not in frames:
            frames[key] = pd.DataFrame([row])
        else:
            frames[key] = pd.concat([frames[key], pd.DataFrame([row])], ignore_index=True)

        docs_map[str(cid_str)] = docs_df.copy()
        raw_stats[str(cid_str)] = r.get("stats", {}) if isinstance(r.get("stats", {}), dict) else {}
        cluster_key_map[str(cid_str)] = r.get("cluster_key")

    for k, df in list(frames.items()):
        if "ParentID" not in df.columns:
            df["ParentID"] = None
        df["Type"] = ""
        frames[k] = df

    return frames, docs_map, raw_stats, cluster_key_map, int(top_level)


def _derive_parents_by_overlap(frames: Dict[str,pd.DataFrame], 
                                docs_map: Dict[str,pd.DataFrame], top_level: int) -> Dict[str,pd.DataFrame]:
    for L in range(top_level, 0, -1):
        upper_key = f"L{L}"; lower_key = f"L{L-1}"
        
        if upper_key not in frames or lower_key not in frames: continue
        
        upper = frames[upper_key].copy(); lower = frames[lower_key].copy()
        parent_eids = {str(r["ClusterID"]): set(
            map(str, docs_map.get(str(r["ClusterID"]), pd.DataFrame()).get("eid", []))) for _,r in upper.iterrows()}
        parent_ids = list(parent_eids.keys())
        
        for i, row in lower.iterrows():
            cid = str(row["ClusterID"])
            eids = set(map(str, docs_map.get(cid, pd.DataFrame()).get("eid", [])))
            best_parent, best_overlap = None, -1
            
            for pid in parent_ids:
                ov = len(eids & parent_eids.get(pid,set()))
                if ov > best_overlap:
                    best_overlap = ov; best_parent = pid
            
            if (best_overlap <= 0) and parent_ids:
                best_parent = parent_ids[0]
            
            lower.at[i, "ParentID"] = best_parent
        
        frames[lower_key] = lower
    
    return frames


def _collect_docs_for_cluster(level_key: str, cid: str, docs_map: Dict[str,pd.DataFrame], 
                            frames: Dict[str,pd.DataFrame]) -> pd.DataFrame:
    # Prefer own docs
    d = docs_map.get(cid, pd.DataFrame())
    if isinstance(d, pd.DataFrame) and not d.empty:
        return d
    
    # Aggregate from children
    try:
        L = int(level_key.replace("L",""))
    except Exception:
        return d
    
    low_key = f"L{L-1}"
    if low_key not in frames: 
        return d
    
    low = frames[low_key]
    if low is None or low.empty:
        return d
    
    child_ids = low.loc[low["ParentID"].astype(str)==str(cid), "ClusterID"].astype(str).tolist()
    if not child_ids:
        return d
    
    parts = []
    for ch in child_ids:
        chd = docs_map.get(str(ch), pd.DataFrame())
        if isinstance(chd, pd.DataFrame) and not chd.empty:
            parts.append(chd)
    
    if not parts:
        return d
    
    agg = pd.concat(parts, ignore_index=True)
    
    return _normalize_docs_columns(agg)


def _classify_type_by_quadrant(xv: float, yv: float, x_thr: float, y_thr: float) -> str:
    """Return one of the 4 quadrant labels, falling back to Emerging/Other if disabled."""
    if (xv is None or yv is None or (isinstance(xv,float) and np.isnan(xv)) or (isinstance(yv,float) and np.isnan(yv))):
        return "Other"
    
    if CLASS_USE_FOUR_LABELS:
        if xv >= x_thr and yv >= y_thr:  # x_hi_y_hi
            return QUADRANT_LABELS.get("x_hi_y_hi")
        if xv <  x_thr and yv >= y_thr:  # x_lo_y_hi
            return QUADRANT_LABELS.get("x_lo_y_hi", CLASS_EMERGING_LABEL)
        if xv >= x_thr and yv <  y_thr:  # x_hi_y_lo
            return QUADRANT_LABELS.get("x_hi_y_lo")
        
        return QUADRANT_LABELS.get("x_lo_y_lo")
    else:
        if xv >= x_thr and yv >= y_thr:
            return CLASS_EMERGING_LABEL
        
        return "Other"


# --- metric resolver (no hardcoded names; use METRIC_FUNCTIONS only) ---
def _metric_with_fallback(name: str, docs: pd.DataFrame) -> float:
    fn = METRIC_FUNCTIONS.get(name)
    if callable(fn):
        try:
            v = fn(docs)
            return float(v) if v is not None else float("nan")
        except Exception:
            return float("nan")
    # Unknown metric key -> NaN
    return float("nan")


# ----------------------------------------------------------------------
def _compute_metrics_and_types(frames: Dict[str,pd.DataFrame], docs_map: Dict[str,pd.DataFrame], top_level: int) -> Dict[str,pd.DataFrame]:
    x_name, y_name = _get_metric_names()

    # Phase (a): compute raw metrics and rounded outputs per level
    raw_xy: Dict[str, Tuple[pd.Series, pd.Series]] = {}
    for L in range(top_level, -1, -1):
        key = f"L{L}"
        if key not in frames:
            continue
        
        df = frames[key].copy()
        xs_raw, ys_raw = [], []
        for _, r in df.iterrows():
            cid = str(r["ClusterID"])
            docs = _collect_docs_for_cluster(key, cid, docs_map, frames)
            xs_raw.append(_metric_with_fallback(x_name, docs))
            ys_raw.append(_metric_with_fallback(y_name, docs))
        
        xv = pd.to_numeric(pd.Series(xs_raw), errors="coerce")
        yv = pd.to_numeric(pd.Series(ys_raw), errors="coerce")
        raw_xy[key] = (xv, yv)

        rx_cfg = round_decimals_for_level(key, top_level, list(xv.fillna(0.0)), x_name)
        ry_cfg = round_decimals_for_level(key, top_level, list(yv.fillna(0.0)), y_name)
        rx = int(rx_cfg if rx_cfg is not None else 3)
        ry = int(ry_cfg if ry_cfg is not None else 3)

        df[x_name] = [None if pd.isna(v) else round(float(v), rx) for v in xv]
        df[y_name] = [None if pd.isna(v) else round(float(v), ry) for v in yv]

        df["Type"] = ""
        frames[key] = df

    # Phase (b): top-down thresholds (subset by kept parents from upper level)
    kept_parents: set = set()
    for L in range(top_level, -1, -1):
        key = f"L{L}"
        if key not in frames:
            continue
        df = frames[key].copy()
        xv, yv = raw_xy.get(key, (pd.Series(dtype=float), pd.Series(dtype=float)))

        if L == top_level or not kept_parents:
            subset_mask = pd.Series([True]*len(df), index=df.index)
        else:
            subset_mask = df["ParentID"].astype(str).isin(kept_parents)

        xv_masked = xv[subset_mask.values] if len(xv) == len(df) else xv
        yv_masked = yv[subset_mask.values] if len(yv) == len(df) else yv

        x_thr = QUAD_X_THRESHOLD if QUAD_X_THRESHOLD is not None else float(pd.Series(xv_masked).dropna().median() if not pd.Series(xv_masked).dropna().empty else 0.0)
        y_thr = QUAD_Y_THRESHOLD if QUAD_Y_THRESHOLD is not None else float(pd.Series(yv_masked).dropna().median() if not pd.Series(yv_masked).dropna().empty else 0.0)

        types_out = []
        for x_raw, y_raw in zip(xv, yv):
            xv_f = None if pd.isna(x_raw) else float(x_raw)
            yv_f = None if pd.isna(y_raw) else float(y_raw)
            types_out.append(_classify_type_by_quadrant(xv_f, yv_f, x_thr, y_thr))
        
        df["Type"] = types_out
        
        # --- Enforce "Other" usage rules by level ---
        if L == top_level:
            # Top level must not contain "Other"
            fallback_type = QUADRANT_LABELS.get("x_lo_y_lo", CLASS_EMERGING_LABEL)
            df["Type"] = [ (t if t != "Other" else fallback_type) for t in df["Type"] ]
        else:
            # Lower levels: only mark "Other" when parent not kept; otherwise coerce "Other" to fallback
            fallback_type = QUADRANT_LABELS.get("x_lo_y_lo", CLASS_EMERGING_LABEL)
            if kept_parents:
                coerced = []
                for i, t in enumerate(df["Type"]):
                    pid = df.iloc[i].get("ParentID")
                    pid = None if (pid is None or (isinstance(pid, float) and pd.isna(pid))) else str(pid)
                    if pid not in kept_parents:
                        coerced.append("Other")
                    else:
                        coerced.append(t if t != "Other" else fallback_type)
                df["Type"] = coerced

        frames[key] = df

        if L > 0:
            kept_parents = set(df.loc[df["Type"] == CLASS_EMERGING_LABEL, "ClusterID"].astype(str))
        else:
            kept_parents = set()

    return frames


CASECADE_CLUSTER_PRIORITY = [
    "Emerging Cluster",
    "Dominant Cluster",
    "Saturated Cluster",
]

def cascade_emerging(frames: Dict[str,pd.DataFrame], top_level: int) -> Dict[str, pd.DataFrame]:
    out = {}
    Ltop = frames.get(f"L{top_level}", pd.DataFrame())
    # keep_top = set(Ltop.loc[Ltop["Type"]==CLASS_EMERGING_LABEL, "ClusterID"].astype(str)) if not Ltop.empty else set()
    keep_top = set()
    selected_label = ""

    if not Ltop.empty:
        for t in CASECADE_CLUSTER_PRIORITY:
            subset = Ltop.loc[Ltop["Type"] == t, "ClusterID"].astype(str)
            if not subset.empty:
                keep_top = set(subset)
                selected_label = t
                break

    out[f"L{top_level}"] = Ltop[Ltop["ClusterID"].astype(str).isin(keep_top)].copy()
    
    for L in range(top_level-1, -1, -1):
        key = f"L{L}"
        df = frames.get(key, pd.DataFrame())
        if df.empty:
            out[key] = df.copy(); continue
        allowed_parents = keep_top if L==top_level-1 else set(out[f"L{L+1}"]["ClusterID"].astype(str))
        sub = df[ df["ParentID"].astype(str).isin(allowed_parents) & (df["Type"] == selected_label) ].copy()
        out[key] = sub
 
    return out


def export_l0_docs_per_cluster(src_df: pd.DataFrame, df_l0: pd.DataFrame, docs_map: dict) -> pd.DataFrame:
    """
    Save an Excel per L0 Emerging cluster, preserving ALL columns from the source Excel.
    Preference order for source Excel:
      1) Any .xlsx under /mnt/data (newest first)
      2) DOC_SOURCE_PATH (if exists)
      3) Latest .xlsx under ./data
    Filtering by 'eid' (case-insensitive) works for 1-row or 2-row headers.
    If no matching rows, save a header-only sheet (still all original columns).
    Only if no source Excel can be read at all, fallback to docs_map[cid].
    """
    def _find_eid_series(df):
        if isinstance(df.columns, pd.MultiIndex):
            for lvl in range(df.columns.nlevels):
                labels = [str(t[lvl]).strip().lower() for t in df.columns]
                if "eid" in labels:
                    return df.iloc[:, labels.index("eid")]
                
            flat = ["|".join([str(x).strip().lower() for x in t if str(x).strip()!='' and str(x)!='None']) for t in df.columns]
            for i, name in enumerate(flat):
                if name == "eid" or name.endswith("|eid") or name.startswith("eid|") or "|eid|" in name:
                    return df.iloc[:, i]
            return None
        else:
            cols = [str(c).strip().lower() for c in df.columns]
            return df.iloc[:, cols.index("eid")] if "eid" in cols else None


    l0_clusters = (
        df_l0["ClusterID"].astype(str).tolist() 
        if isinstance(df_l0, pd.DataFrame) and not df_l0.empty 
        else []
    )

    l0_out_pd = {}

    try:
        eid_series = _find_eid_series(src_df)
    except Exception:
        eid_series = None
    
    if eid_series is None: raise ValueError("No 'eid' column found in the source Excel.")

    for cid in l0_clusters:
        d = docs_map.get(cid, pd.DataFrame())
        
        if not isinstance(d, pd.DataFrame) or "eid" not in d.columns:
            continue
        
        eids = set(map(str, d["eid"].astype(str).tolist()))
        mask = eid_series.astype(str).isin(eids) if len(eids) > 0 else (eid_series.astype(str).isin([]))
        sub = src_df[mask].copy()
        
        if sub.empty:
            continue
        
        l0_out_pd[cid] = sub
    
    return l0_out_pd


def save_level_classification_jsonl(level: int, frames_full: dict,
            kept_by_level: dict, docs_map: dict, raw_stats: dict):
    """
    Dump JSONL for the given level with:
      - all clusters at that level
      - 'kept' boolean (belongs to Emerging cascade)
      - 'classification_label' overridden to 'Other' for rows whose parent is NOT kept at the immediate upper level
        (so downstream visuals like Sankey can color them consistently)
    """
    key = f"L{level}"
    x_name, y_name = _get_metric_names()
    df = frames_full.get(key, pd.DataFrame()).copy()
    outp = []

    # children from FULL frames
    child_map = {}
    low_key = f"L{level-1}"
    if low_key in frames_full and not frames_full[low_key].empty:
        low = frames_full[low_key]
        for pid, grp in low.groupby("ParentID"):
            if pid is None: 
                continue

            child_map[str(pid)] = grp["ClusterID"].astype(str).tolist()

    kept_set = set()
    if key in kept_by_level and isinstance(kept_by_level[key], pd.DataFrame) and not kept_by_level[key].empty:
        kept_set = set(kept_by_level[key]["ClusterID"].astype(str))

    # parents kept at upper level (if exists)
    upper_key = f"L{level+1}"
    kept_parents = set()
    if upper_key in kept_by_level and isinstance(kept_by_level[upper_key], pd.DataFrame) and not kept_by_level[upper_key].empty:
        kept_parents = set(kept_by_level[upper_key]["ClusterID"].astype(str))
    
    for _, r in df.iterrows():
        cid = str(r.get("ClusterID"))
        parent = r.get("ParentID")
        parent_str = (str(parent) if parent is not None else None)

        # override classification to 'Other' when lower level and parent not kept
        cls_label = r.get("Type")
        if level >= 0 and upper_key in kept_by_level:
            if parent_str is None or parent_str not in kept_parents:
                cls_label = "Other"

        d = docs_map.get(cid, pd.DataFrame())
        row = {
            "cluster_id": cid,
            "kept": (cid in kept_set),
            "classification_label": cls_label,
            "classification_label_display": cls_label,
            "metrics": {
                x_name: (None if pd.isna(r.get(x_name)) else float(r.get(x_name))) if x_name in r else None,
                y_name: (None if pd.isna(r.get(y_name)) else float(r.get(y_name))) if y_name in r else None,
            },
            "relations": {
                "parent_id": parent_str,
                "children_ids": child_map.get(cid, [])
            },
            "stats": raw_stats.get(cid, {}) if isinstance(raw_stats.get(cid, {}), dict) else {},
            "docs": d.to_dict(orient="records") if isinstance(d, pd.DataFrame) else []
        }
        outp.append(row)

    return outp


def run_emerging_pipeline(vector_df: pd.DataFrame, hierachy_result: List[Dict], top_level: Optional[int] = None) -> dict:
    artifacts = {}

    rows = hierachy_result # PATH

    frames, docs_map, raw_stats, cluster_key_map, detected_top = _frames_docs_stats_from_rows(rows)
    if top_level is None: top_level = detected_top

    frames = _derive_parents_by_overlap(frames, docs_map, top_level)
    frames = _compute_metrics_and_types(frames, docs_map, top_level)
    kept = cascade_emerging(frames, top_level=top_level)

    artifacts["l0_docs"] = export_l0_docs_per_cluster(vector_df, kept.get("L0", pd.DataFrame()), docs_map)

    for level in [top_level, top_level-1, top_level-2, top_level-3]:
        if level < 0: continue
        artifacts[f"L{level}_jsonl"] = save_level_classification_jsonl(level, frames, kept, docs_map, raw_stats)

    return artifacts