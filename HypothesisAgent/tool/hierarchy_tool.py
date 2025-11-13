import numpy as np
import pandas as pd

from typing import List, Dict, Tuple

from ..core.io_utils import   (
	_choose_group_key,
	_extract_texts_for_cluster,
	aggregate_meta_from_group,
)


from ..config.model_config import (
	NB_DIM, NB_N_NEIGHBORS, NB_MIN_DIST, NB_RANDOM_STATE,
	NB_MAX_CLUSTERS, MIN_CLUSTERS_TO_CONTINUE,
)

from ..core.embeddings import ensure_embeddings_cached
from ..core.dim_reduction import ensure_umap_cached
from ..core.gmm_cluster import choose_k_by_bic, fit_predict
from ..agents.cluster_summarization_agent import generate_candidates
from ..agents.summary_evaluator_agent import evaluate_all_candidates, shrink_contexts
from ..agents.hierarchy_tree_agent import judge_continue, _set_prev_count, _get_prev_count


def _aggregate_from_prev_level(prev_level: int, prev_cids: list[int], total_record: List[Dict]) -> dict:
	"""Aggregate docs/stats for higher levels by merging L{prev_level} records with matching cids."""
	
	docs, years, cit_sum = [], [], 0
	if total_record:
		for line in total_record:
			if line.get("level") == prev_level and int(line.get("cid", -1)) in set(prev_cids):
				for d in line.get("docs", []) or []:
					docs.append({
						"eid": str(d.get("eid","")),
						"year": d.get("year"),
						"citations": int(d.get("citations", 0))
					})
					if d.get("year") is not None:
						years.append(d["year"])
					try:
						cit_sum += int(d.get("citations", 0))
					except Exception:
						pass
					
	years_sorted = sorted([y for y in years if y is not None])
	return {
		"doc_count": int(len(docs)),
		"citation_sum": int(cit_sum),
		"years_sorted": years_sorted,
		"docs": docs,
	}


def _ensure_gmm_global(level: int, X_umap: np.ndarray, k_max_eff: int) -> List[int]:
	k = choose_k_by_bic(X_umap, k_min=1, k_max=NB_MAX_CLUSTERS)
	labels = fit_predict(X_umap, k)

	return labels


def _ensure_gmm_local(level: int, X_umap: np.ndarray, global_labels: List[int]) -> Dict:
	local = {}
	unique_labels = set()

	for labs in global_labels:
		if isinstance(labs, (list, np.ndarray)):
			for lab in labs:
				unique_labels.add(int(lab))
		else:
			unique_labels.add(int(labs))

	for g_id in sorted(unique_labels):
		idx = [
			i for i, labs in enumerate(global_labels)
			if (isinstance(labs, (list, np.ndarray)) and g_id in labs)
			or (not isinstance(labs, (list, np.ndarray)) and g_id == labs)
		]

		if len(idx) < 2:
			local[g_id] = {"labels": [0]*len(idx), "k": 1}
			continue
		Xg = X_umap[idx]
		k = choose_k_by_bic(Xg, k_min=1, k_max=min(6, len(idx)))
		labels_g = fit_predict(Xg, k)
		local[g_id] = {"labels": labels_g, "k": k}
		# print(f"[L{level}] Local GMM (BIC) @G{g_id}: k={k}", flush=True)

	return local


def _attach_local_groups(level: int, df: pd.DataFrame, global_labels: List[int], local_lables: Dict) -> pd.DataFrame:
    n = len(df)
    local_vec = [0]*n
    pos_by_g: Dict[int, List[int]] = {}

    for i, g in enumerate(global_labels):
        if isinstance(g, (list, np.ndarray)):
            for gi in g:
                pos_by_g.setdefault(int(gi), []).append(i)
        else:
            pos_by_g.setdefault(int(g), []).append(i)

    for g, pos_list in pos_by_g.items():
        labels_g = list(local_lables.get(g, {}).get("labels", [0]*len(pos_list)))
        if len(labels_g) != len(pos_list):
            labels_g = labels_g[:len(pos_list)] + [0]*max(0, len(pos_list)-len(labels_g))
            for j, pos in enumerate(pos_list):
                if isinstance(labels_g[j], (list, np.ndarray)):
                    local_vec[pos] = [int(x) for x in labels_g[j]] if len(labels_g[j]) > 0 else [int(-1)]
                else:
                    local_vec[pos] = [int(labels_g[j])]

    pairs = []
    for g, l in zip(global_labels, local_vec):
        g_list = g if isinstance(g, (list, np.ndarray)) else [g]
        l_list = l if isinstance(l, (list, np.ndarray)) else [l]
        for gi in g_list:
            for li in l_list:
                pairs.append((int(gi), int(li)))

    uniq = {}
    seq_ids = []
    for p in pairs:
        if p not in uniq:
            uniq[p] = len(uniq)
        seq_ids.append(uniq[p])

    out = df.copy()
    out["cluster_id"] = seq_ids
    out["_cluster_pair"] = [f"G{g}-L{l}" for g, l in pairs]
    # print(f"[L{level}] Attached labels (global+local): {len(set(seq_ids))} clusters", flush=True)
    return out


async def _run_level(level: int, items_df: pd.DataFrame, total_record:List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	# 1) L0 texts
	if level == 0:
		cols_lower = {c.lower(): c for c in items_df.columns}
		tcol = cols_lower.get("title")
		acol = cols_lower.get("abstract")
		parts = []
		if tcol: parts.append(items_df[tcol].fillna(""))
		if acol: parts.append(items_df[acol].fillna(""))
		if not parts:
			raise RuntimeError("L0 requires at least one of [title, abstract] columns")
		texts = (parts[0] if len(parts)==1 else (parts[0] + "\n" + parts[1])).astype(str).tolist()
	else:
		if "summary" not in items_df.columns:
			raise RuntimeError("L>=1 requires 'summary' column from previous level carry-over")
		texts = items_df["summary"].astype(str).fillna("").tolist()

	# 2) Embedding + UMAP
	X_emb = await ensure_embeddings_cached(texts) if level > 0 else np.array(items_df["abstract_embed"].tolist(), dtype="float32")

	X_umap = await ensure_umap_cached(
		X_emb,
		n_components=NB_DIM,
		n_neighbors=NB_N_NEIGHBORS,
		min_dist=NB_MIN_DIST,
		random_state=NB_RANDOM_STATE,
		verbose=True
	)

	# 3) Global / Local GMM
	labels_g = _ensure_gmm_global(level, X_umap, NB_MAX_CLUSTERS)
	labels_l = _ensure_gmm_local(level, X_umap, labels_g)

	# 4) attach labels
	df_lab = items_df.copy()

	cluster_ids = []
	for labs in labels_g:
		if isinstance(labs, (list, np.ndarray)):
			labs = [int(x) for x in labs]
			if len(labs) == 0:
				labs = [-1]   # fallback for empty
			cluster_ids.append(labs)
		else:
			cluster_ids.append([int(labs)])

	if len(cluster_ids) != len(df_lab):
		raise ValueError(
			f"[BUG] Label count {len(cluster_ids)} does not match df_lab rows {len(df_lab)}"
		)
	
	df_lab = df_lab.copy()
	df_lab["cluster_id"] = cluster_ids
	df_lab = df_lab.explode("cluster_id", ignore_index=True)
	df_lab["cluster_id"] = df_lab["cluster_id"].astype(int)

	df_lab = _attach_local_groups(level, df_lab, labels_g, labels_l)

	key = _choose_group_key(df_lab)
	groups = list(df_lab.groupby(key))

	top1_list = []
	carry_rows = []

	for idx, (cid, g) in enumerate(groups, start=1):
		print(f"[L{level}] Cluster {idx}/{len(groups)} (cid={cid})", flush=True)
		if level > 0:
			cluster_texts = g["summary"].astype(str).fillna("").tolist()
		else:
			if ("title" in g.columns and "abstract" in g.columns):
				cluster_texts = (g["title"].fillna("").astype(str) + "\n" + g["abstract"].fillna("").astype(str)).tolist()
			else:
				cluster_texts = _extract_texts_for_cluster(g)
		shrunk = shrink_contexts(cluster_texts)

		cluster_id = f"L{level}_{int(cid)}"
		cands = await generate_candidates(level, int(cid), cluster_id, g)
		if not isinstance(cands, pd.DataFrame) or "summary" not in cands.columns:
			rows = []
			for i, t in enumerate(cluster_texts, start=1):
				rows.append({"summary": str(t).strip(), "attempt": i})
			cands = pd.DataFrame(rows)
		cands_scored = await evaluate_all_candidates( level, int(cid), cluster_id, cands, shrunk)

		if "Selected" not in cands_scored.columns:
			cands_scored["min_pair"] = cands_scored[["faithfulness_score","relevance_score"]].min(axis=1)
			best_idx = cands_scored["min_pair"].idxmax()
			cands_scored["Selected"] = "No"
			cands_scored.loc[best_idx, "Selected"] = "Yes"

		selected_row = cands_scored[cands_scored["Selected"]=="Yes"].iloc[0].to_dict()
		sel_summary = str(selected_row.get("summary","")).strip()
		sel_ov = float(selected_row.get("overlap_ratio", 0.0))
		sel_is_multi = bool(selected_row.get("is_multiline", False))
		# print(f"    → Selected attempt {int(selected_row.get('attempt',1))}/{len(cands_scored)}", flush=True)

		if 'cid' in g.columns:
			prev_cids = list(pd.to_numeric(g['cid'], errors='coerce').dropna().astype(int).unique())
		else:
			prev_cids = []

		meta = aggregate_meta_from_group(g) if level == 0 else _aggregate_from_prev_level(prev_level=level-1, prev_cids=prev_cids, total_record=total_record)

		attempts = []
		for i, row in enumerate(cands_scored.to_dict(orient="records"), start=1):
			ov = float(row.get("overlap_ratio", 0.0)) if row.get("overlap_ratio") is not None else 0.0
			attempts.append({
				"attempt": int(row.get("attempt", i)),
				"summary": str(row.get("summary","")),
				"scores": {
					"faithfulness": float(row.get("faithfulness_score",0.0)),
					"answer_relevancy": float(row.get("relevance_score",0.0))
				},
				"overlap_ratio": ov,
				"overlap_pct": round(ov*100, 2),
				"is_multiline": bool(row.get("is_multiline", False))
			})

		record = {
			"level": int(level),
			"cid": int(cid),

			"selected": {
				"summary": sel_summary,
				"attempt": int(selected_row.get("attempt",1)),
				"scores": {
					"faithfulness": float(selected_row.get("faithfulness_score",0.0)),
					"answer_relevancy": float(selected_row.get("relevance_score",0.0))
				},
				"overlap_ratio": sel_ov,
				"overlap_pct": round(sel_ov*100, 2),
				"is_multiline": sel_is_multi
			},
			"cluster_id": f"L{level}_{cid}",  # [NEW] unified UID
			"stats": {
				"doc_count": int(meta.get("doc_count", len(g))),
				"citation_sum": int(meta.get("citation_sum", 0)),
				"years_sorted": meta.get("years_sorted", [])
			},
			"attempts": attempts,
			"docs": meta.get("docs", [])
		}

		total_record.append(record)

		carry_rows.append({"summary": sel_summary, "cid": int(cid)})
		top1_list.append({"cid": cid, "summary": sel_summary, "Selected": "Yes"})

	top1_df = pd.DataFrame(top1_list)
	carry_df = pd.DataFrame(carry_rows) if carry_rows else pd.DataFrame(columns=["summary"])
	# print(f"[END] [L{level}] Summaries & Evaluation finished.", flush=True)
	return df_lab, top1_df, carry_df


async def run_hierarchy_pipeline(dosc: pd.DataFrame) -> Dict:
	total_record = []

	_, top1_df, carry_df = await _run_level(0, dosc, total_record)
	_set_prev_count(0, len(top1_df))

	if not top1_df.empty:
		_, top1_df, carry_df = await _run_level(1, top1_df.copy(), total_record)
		_set_prev_count(1, len(top1_df))
	else:
		print("[INFO] No selected summaries at L0; stopping.")
		print("[DONE] Pipeline complete.")
		return

	level = 2
	while True:
		if top1_df.empty:
			break
		prev_cnt = _get_prev_count(level - 1)
		if prev_cnt is not None and prev_cnt <= max(1, int(MIN_CLUSTERS_TO_CONTINUE) - 1):
			print(f"[STOP] L{level} blocked: prev cluster count ({prev_cnt}) < MIN_CLUSTERS_TO_CONTINUE ({MIN_CLUSTERS_TO_CONTINUE}).")
			break
		if not judge_continue(f"L{level}", top1_df["summary"].astype(str).tolist()):
			print(f"[STOP] judge_continue rejected L{level}."); break

		_, top1_df, carry_df = await _run_level(level, top1_df.copy(), total_record)
		_set_prev_count(level, len(top1_df))
		level += 1

	return total_record
	