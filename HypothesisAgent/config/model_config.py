# -*- coding: utf-8 -*-
"""Model and pipeline configuration"""
from __future__ import annotations

import os
import re

import numpy as _np
import pandas as _pd

from dataclasses import dataclass
from typing import Optional, Dict, Literal, Tuple, List, Callable

# ------------------------------------------------------------------
# AGENTS: Which LLM handles each role in the pipeline
# ------------------------------------------------------------------
AgentKey = Literal[
    "summarizer",
    "summary_evaluator",
    "promotion_judge",
    "role_assignment",
    "proposal_generation",
]

@dataclass(frozen=True)
class AgentLLM:
    MODEL: str
    TEMPERATURE: Optional[float] = None
    TOP_P: Optional[float] = None
    SEED: Optional[int] = None
    MAX_TOKENS: Optional[int] = None


AGENTS: Dict[AgentKey, AgentLLM] = {
    "summarizer":        AgentLLM(MODEL=os.getenv("LLM_SUMMARIZER", "gpt-4.1-mini"), TEMPERATURE=0.5, TOP_P=0.9),
    "summary_evaluator": AgentLLM(MODEL=os.getenv("LLM_EVALUATOR", "gpt-4.1-mini"), TEMPERATURE=0.2, TOP_P=0.8),
    "hierarchy_tree":   AgentLLM(MODEL=os.getenv("LLM_JUDGE", "gpt-4.1-mini"), TEMPERATURE=0.2, TOP_P=0.8),
    "role_assignment":     AgentLLM(MODEL=os.getenv("LLM_ROLE_ASSIGN", "gpt-4.1-mini"), TEMPERATURE=0.2, TOP_P=0.8),
    "proposal_generation": AgentLLM(MODEL=os.getenv("LLM_PROPOSAL_GEN", "gpt-4.1-mini"), TEMPERATURE=0.6, TOP_P=0.95),
}


def get_agent_llm(agent_key: AgentKey) -> AgentLLM:
    if agent_key not in AGENTS:
        raise KeyError(f"Undefined agent key: {agent_key!r}")
    
    return AGENTS[agent_key]


AGENT_MODEL: Dict[str, str] = globals().get('AGENT_MODEL', {})  # keep existing if already injected elsewhere


# ------------------------------------------------------------------
# PROPOSAL SETTINGS 
# ------------------------------------------------------------------
PROPOSAL_STAGE_SEQUENCE: List[str] = [s.strip() for s in os.getenv(
    "PROPOSAL_STAGE_SEQUENCE",
    "Gen_1"
).split(",") if s.strip()]

STAGE_TYPES = {"Gen"}
_STAGE_RE = re.compile(r"^(Gen)_(\d+)$")


def parse_stage_key(stage_key: str) -> Tuple[str, int]:
    m = _STAGE_RE.match(stage_key.strip())
    if not m:
        raise ValueError(f"Invalid stage key: {stage_key!r}. Expect one of {STAGE_TYPES} with _<int> suffix.")
    
    return m.group(1), int(m.group(2))


def get_stage_sequence() -> List[str]:
    return list(PROPOSAL_STAGE_SEQUENCE)


# Convenience: fetch model-id string for a given agent (some callers prefer a plain string)
def get_model_for_agent(agent_key: AgentKey) -> str:
    m = get_agent_llm(agent_key).MODEL
    if not isinstance(m, str) or not m.strip():
        raise RuntimeError(f"Model not configured for agent {agent_key!r}. Set it in model_config or OS env.")
    
    return m


# ------------------------------------------------------------------
# EMBEDDINGS: Which model generates vector representations
# ------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def get_embedding_model(kind: str) -> str:
    # 'kind' is reserved for future use (e.g., different embedding families)
    return EMBEDDING_MODEL


# ------------------------------------------------------------------
# UMAP/GMM
# ------------------------------------------------------------------
NB_DIM: int = int(os.getenv("NB_DIM", "10"))                 # Target UMAP dimensions
NB_THRESHOLD: float = float(os.getenv("NB_THRESHOLD", "0.1"))
NB_MAX_CLUSTERS: int = int(os.getenv("NB_MAX_CLUSTERS", "100"))
NB_RANDOM_STATE: int = int(os.getenv("NB_RANDOM_STATE", "42"))
NB_N_NEIGHBORS: int = int(os.getenv("NB_N_NEIGHBORS", "15"))
NB_MIN_DIST: float = float(os.getenv("NB_MIN_DIST", "0.1"))


# ------------------------------------------------------------------
# RECURSION GATE: When to stop going deeper in the hierarchy
#  - Unchanged. The pipeline reads this to decide if recursion continues.
# ------------------------------------------------------------------
MIN_CLUSTERS_TO_CONTINUE: int = int(os.getenv("MIN_CLUSTERS_TO_CONTINUE", "8"))

SIMILARITY_STOP_THRESHOLD: float = float(os.getenv("SIMILARITY_STOP_THRESHOLD", "0.7"))
SIMILARITY_METRIC: str = os.getenv("SIMILARITY_METRIC", "tfidf_cosine")  # or "jaccard"

# ------------------------------------------------------------------
# SUMMARY / EVALUATION SETTINGS
#  - How many candidate summaries to produce and input size limits.
# ------------------------------------------------------------------
@dataclass(frozen=True)
class SummarySettings:
    SUMMARY_OUTPUT_CHAR_LIMIT: int = int(os.getenv("SUMMARY_OUTPUT_CHAR_LIMIT", "100"))
    SUMMARY_NUM_CANDIDATES:   int = int(os.getenv("SUMMARY_NUM_CANDIDATES", "10"))
    MAX_SUMMARY_RETRIES: int = int(os.getenv("MAX_SUMMARY_RETRIES", "8"))
    OVERLAP_CUTOFF: float = float(os.getenv("OVERLAP_CUTOFF", "0.7"))

    MAX_CHARS_SUMMARY_INPUT: int = int(os.getenv("MAX_CHARS_SUMMARY_INPUT", "6000"))
    MAX_CHARS_EVAL_INPUT: int = int(os.getenv("MAX_CHARS_EVAL_INPUT", "6000"))
    ENABLE_LOCAL_L0: bool = os.getenv("ENABLE_LOCAL_L0", "1") != "0"
    ENABLE_LOCAL_L1: bool = os.getenv("ENABLE_LOCAL_L1", "1") != "0"

SUMMARY_BULLET_TOKEN: str = os.getenv("SUMMARY_BULLET_TOKEN", "###")

def get_summary_settings() -> SummarySettings:
    return SummarySettings()


# ------------------------------------------------------------------
# Classification schema (columns in cluster & document tables)
# ------------------------------------------------------------------
CLASS_LEVEL_COL: str = "Level"
CLASS_ID_COL: str = "ClusterID"
CLASS_PARENT_COL: str = "ParentID"
CLASS_TYPE_COL: str = "Type"
CLASS_SCORE_COL: str | None = "Score"
CLASS_EMERGING_LABEL: str = "Emerging Cluster"
CLASS_USE_FOUR_LABELS: bool = True

QUADRANT_LABELS = {
    "x_hi_y_hi": "Dominant Cluster",
    "x_lo_y_hi": "Saturated Cluster",
    "x_hi_y_lo": "Emerging Cluster",
    "x_lo_y_lo": "Declining Cluster",
}

# COLORS = {
#     QUADRANT_LABELS["x_hi_y_hi"]: "#1f77b4",
#     CLASS_EMERGING_LABEL:         "#d62728",
#     QUADRANT_LABELS["x_lo_y_hi"]: "#2ca02c",
#     QUADRANT_LABELS["x_lo_y_lo"]: "#ff7f0e",
#     "Other":                      "#7f7f7f",
# }

DOC_CLUSTER_COL: str = "ClusterID"
DOC_ID_COL: str = "DocID"
DOC_CITATION_COL: str = "citations"
DOC_DATE_COL: str = "year"

# ------------------------------------------------------------------
# Metric functions registry (plug-in style)
# ------------------------------------------------------------------
def metric_average_citation_count(df: _pd.DataFrame) -> float:
    if df is None or df.empty or DOC_CITATION_COL not in df.columns:
        return 0.0
    x = _pd.to_numeric(df[DOC_CITATION_COL], errors="coerce").dropna()
    
    return float(x.mean()) if not x.empty else 0.0


def metric_median_citation_count(df: _pd.DataFrame) -> float:
    if df is None or df.empty or DOC_CITATION_COL not in df.columns:
        return 0.0
    
    x = _pd.to_numeric(df[DOC_CITATION_COL], errors="coerce").dropna()
    
    return float(x.median()) if not x.empty else 0.0


def metric_publication_trend_slope(df: _pd.DataFrame) -> float:
    if df is None or df.empty or DOC_DATE_COL not in df.columns or DOC_CITATION_COL not in df.columns:
        return 0.0
    
    years = _pd.to_numeric(df[DOC_DATE_COL], errors="coerce").dropna().astype(int)
    cits = _pd.to_numeric(df[DOC_CITATION_COL], errors="coerce").dropna().astype(float)
    df2 = _pd.DataFrame({"y": years, "c": cits}).dropna()
    
    if df2.shape[0] < 2:
        return 0.0
    try:
        a, b = _np.polyfit(df2["y"].astype(float).values, df2["c"].astype(float).values, 1)
        return float(a)
    except Exception:
        return 0.0


METRIC_FUNCTIONS: Dict[str, Callable[[_pd.DataFrame], float]] = {
    "average_citation_count": metric_average_citation_count,
    "publication_trend_slope": metric_publication_trend_slope,
}


# ------------------------------------------------------------------
# Metric selection (X/Y) and dynamic rounding rules
# ------------------------------------------------------------------
CLASSIFICATION_METRICS = {
    "x_metric": "publication_trend_slope",
    "y_metric": "average_citation_count",
}

METRIC_DISPLAY_NAME = {
    "average_citation_count": "Average Citation Count",
    "publication_trend_slope": "Publication Trend Slope",
}

ROUNDING_CONFIG = {
    "top_level_decimals": 2,
    "per_level_decimals": {"L3": 2, "L2": 2, "L1": 3, "L0": 3},
    "per_metric_decimals": {"average_citation_count": 2, "publication_trend_slope": 3},
    "default_decimals": 3,
    "enable_auto_heuristic": True,
}


def round_decimals_for_level(level: str, top_level: int, values, metric_name: str) -> int:
    try:
        lvl_map = ROUNDING_CONFIG.get("per_level_decimals") or {}
        if level in lvl_map:
            return int(lvl_map[level])
        
        if level == f"L{int(top_level)}":
            return int(ROUNDING_CONFIG.get("top_level_decimals", 3))
        
        pm = ROUNDING_CONFIG.get("per_metric_decimals") or {}
        if metric_name in pm:
            return int(pm[metric_name])
        
        if ROUNDING_CONFIG.get("enable_auto_heuristic", False):
            import numpy as _nph
            arr = _nph.array(list(values), dtype=float)
            if arr.size:
                m = _nph.nanmax(_nph.abs(arr))
                if m >= 100: return 1
                if m >= 10:  return 2
                if m >= 1:   return 3
                
                return 4
            
        return int(ROUNDING_CONFIG.get("default_decimals", 3))
    except Exception:
        return 3


# ------------------------------------------------------------------
# Visualization defaults — Sankey & Quadrant
# ------------------------------------------------------------------
# ---- Quadrant (4-quadrant scatter) ----
QUAD_X_THRESHOLD = None      # if None, median(x) used per-level  
QUAD_Y_THRESHOLD = None      # if None, median(y) used per-level  


# ------------------------------------------------------------------
# TASK TEMPLATES
# ------------------------------------------------------------------
TASK_TEMPLATES_BY_TYPE: Dict[str, Dict] = {
    "Gen": {
        "objective": "Generate one testable hypothesis grounded in the given materials.",
        "checklist": [
            "name variables and control groups",
            "cite data source/availability",
            "state expected outcomes (qualitative)"
        ],
        "output_spec": [
            "Hypothesis: <=3 sentences",
            "Variables/Control: 2-4 bullets"
        ]
    }
}

# Per-exact-stage prompt overrides (optional; partial override allowed)
STAGE_PROMPT_OVERRIDES: Dict[str, Dict] = {}

def get_task_template(stage_key: str) -> Dict:
    stage_type, _ = parse_stage_key(stage_key)
    base = dict(TASK_TEMPLATES_BY_TYPE.get(stage_type, {}))
    override = STAGE_PROMPT_OVERRIDES.get(stage_key, {})
    base.update(override)
    return base


# ------------------------------------------------------------------
# STAGE SEQUENCE VALIDATION
# ------------------------------------------------------------------
def validate_stage_sequence(seq: List[str]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    seen_eval_rounds = set()   # rounds that have Eval
    seen_rev_rounds = set()    # rounds that have Rev

    for i, sk in enumerate(seq):
        stype, rnd = parse_stage_key(sk)

        if stype == "Eval":
            seen_eval_rounds.add(rnd)

    return (len(errors) == 0, errors)


# ------------------------------------------------------------------
# LONG CONTEXT LIMIT HINTS & GEN BRANCHING / PDF SETTINGS
#  - These settings are read by agents/orchestrator to decide:
#    * whether to attach all PDFs once vs. group-and-summarize (miner),
#    * group size (MB) when chunking PDFs,
#    * optional final spot-attachment of small Top-K PDFs,
#    * conservative completion token budget for GEN outputs,
#    * and whether ROLE assignment may read PDFs (off by default to reduce bias/cost).
# ------------------------------------------------------------------

# Approximate context capacities (choose the larger as a conservative default)
CONTEXT_LIMIT_TOKENS: Dict[str, int] = {
    "gpt-4.1-mini": 1_000_000,
    "gpt-5-mini":   400_000,
}

def get_context_limit_for_model(name: str) -> int:
    for k, v in CONTEXT_LIMIT_TOKENS.items():
        if name.startswith(k):
            return int(v)
    # Fallback if unknown
    return int(os.getenv("DEFAULT_CONTEXT_LIMIT_TOKENS", "128000"))


DEFAULT_CONTEXT_LIMIT = max(CONTEXT_LIMIT_TOKENS.values()) if CONTEXT_LIMIT_TOKENS else 128000

# ROLE assignment: use PDFs? (default: False; Excel-only recommended to avoid bias)
ROLE_ASSIGNMENT_READ_PDFS: bool = os.getenv("ROLE_ASSIGNMENT_READ_PDFS", "0") == "1"
ROLE_GROUP_MAX_MB: int = int(os.getenv("ROLE_GROUP_MAX_MB", "30"))  # group size per ROLE PDF pass

# GEN branching strategy: "auto" | "excel_only" | "attach_all" | "miner"
GEN_STRATEGY: str = os.getenv("GEN_STRATEGY", "auto").strip().lower()
if GEN_STRATEGY not in {"auto", "excel_only", "attach_all", "miner"}:
    GEN_STRATEGY = "auto"

# Thresholds that decide attach-all vs miner
GEN_ATTACH_MAX_TOTAL_MB: int = int(os.getenv("GEN_ATTACH_MAX_TOTAL_MB", "20"))

# Upper bound on total tokens when considering attach-all (conservative)
GEN_ATTACH_MAX_TOKENS: int = int(os.getenv("GEN_ATTACH_MAX_TOKENS", str(min(180_000, DEFAULT_CONTEXT_LIMIT))))

# Group size for miner path (PDFs split into groups no larger than this many MB)
GEN_GROUP_MAX_MB: int = int(os.getenv("GEN_GROUP_MAX_MB", "30"))

# Optionally attach small Top-K PDFs in the final GEN call (0 disables)
GEN_FINAL_ATTACH_TOPK: int = int(os.getenv("GEN_FINAL_ATTACH_TOPK", "0"))

# Completion token budget for GEN outputs (does not control input side)
BUDGET_GEN_FINAL_TOKENS: int = int(os.getenv("BUDGET_GEN_FINAL_TOKENS", "32768"))


def get_gen_branching_config() -> Dict[str, int | str]:
    """Expose GEN branching thresholds to callers."""
    return {
        "strategy": GEN_STRATEGY,
        "attach_max_total_mb": GEN_ATTACH_MAX_TOTAL_MB,
        "attach_max_tokens": GEN_ATTACH_MAX_TOKENS,
        "group_max_mb": GEN_GROUP_MAX_MB,
        "final_attach_topk": GEN_FINAL_ATTACH_TOPK,
        "final_tokens_budget": BUDGET_GEN_FINAL_TOKENS,
    }


# Max number of Title+Abstract pairs to pull per Excel file for GEN context
GEN_EXCEL_MAX_ITEMS: int = int(os.getenv("GEN_EXCEL_MAX_ITEMS", "50"))
