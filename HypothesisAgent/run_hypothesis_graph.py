import os, sys
import time

# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from typing import TypedDict, Dict
from .tool import hierarchy_tool, emerging_tool, proposal_tool
from pymilvus import AsyncMilvusClient
from openai import AsyncOpenAI

# from dotenv import load_dotenv
# load_dotenv()

# 환경 변수 확인
required_env_vars = [
    "OPENAI_API_KEY",
    "MILVUS_ADDR",
    "MILVUS_TOKEN",
    "QWEN_EMBED_ADDR"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# Graph State
class HypothesisState(TypedDict):
	hypothesis_query: str
	search_docs: Dict
	hierarchy_output: Dict
	emerging_output: Dict
	proposal_output: Dict
	step_timestamp:Dict

async def get_docs(state: HypothesisState) -> HypothesisState:
	start_time = time.time()
	query = state["hypothesis_query"] if state["hypothesis_query"] else None
	docs = None

	output_fields = {
		"eid": "eid", 
		"title": "Title", 
		"abstract": "Abstract",
		"publication_year": "Publication Year", 
		"number_of_citation": "Number of Citation", 
		"abstract_embed": "abstract_embed"
	}

	EMBED_CLIENT = AsyncOpenAI(base_url=os.getenv("QWEN_EMBED_ADDR"), api_key="")
	VECTOR_CLIENT = AsyncMilvusClient(
		uri=os.getenv("MILVUS_ADDR"),
		token=os.getenv("MILVUS_TOKEN"),
		db_name=os.getenv("MILVUS_DB_NAME")
	)
	MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "hypothesis_docs_collection")
	
	# 추수 수정 필요
	resp = await EMBED_CLIENT.embeddings.create(model=os.getenv("QWEN_EMBED_MODEL"), input=query)
	query_vector = np.array([resp.data[0].embedding], dtype="float32")

	docs = await VECTOR_CLIENT.search(
		collection_name=MILVUS_COLLECTION_NAME,
		data=query_vector,
		output_fields=list(output_fields.keys()),
		limit=50
	)
	
	if not docs: return {"search_docs": None}

	docs_df = pd.DataFrame([d["entity"] for d in docs[0]])
	docs_df = docs_df[list(output_fields.keys())].rename(columns=output_fields)
	
	state["step_timestamp"]["get_docs"] = f"Node 실행 시간: {(time.time()-start_time):.4f} 초"
	
	return {"search_docs": docs_df.to_dict(orient="records")}


async def run_hierarchy(state: HypothesisState) -> HypothesisState:
	start_time = time.time()
	
	search_dict = state.get("search_docs", [])
	search_docs = pd.DataFrame.from_records(search_dict)
	hierarchy_output = await hierarchy_tool.run_hierarchy_pipeline(search_docs)

	state["step_timestamp"]["run_hierarchy"] = f"Node 실행 시간: {(time.time()-start_time):.4f} 초"

	return {"hierarchy_output": hierarchy_output}


async def run_emerging(state: HypothesisState) -> HypothesisState:
	start_time = time.time()

	search_dict = state.get("search_docs", [])
	search_docs = pd.DataFrame.from_records(search_dict)
	hierarchy_output = state.get("hierarchy_output")

	emerging_output = await emerging_tool.run_emerging_pipeline(vector_df=search_docs, output_jsonl=hierarchy_output)
	state["step_timestamp"]["run_emerging"] = f"Node 실행 시간: {(time.time()-start_time):.4f} 초"

	# dataframe 직렬화 가능하게 처리
	new_value = {}
	for col, df in emerging_output.get("l0_docs", {}).items():
		new_value[col] = df.to_dict(orient="records")
	emerging_output["l0_docs"] = new_value

	return {"emerging_output": emerging_output}


async def run_proposal(state: HypothesisState) -> HypothesisState:
	start_time = time.time()

	emerging_output = state.get("emerging_output")

	l0_docs = {}
	# dataframe으로 변환
	for col, records in emerging_output.get("l0_docs", {}).items():
		l0_docs[col] = pd.DataFrame.from_records(records)

	proposal_output = await proposal_tool.pr_run_all(l0_docs)

	state["step_timestamp"]["run_proposal"] = f"Node 실행 시간: {(time.time()-start_time):.4f} 초"

	return {"proposal_output": proposal_output}



from langgraph.graph import StateGraph, START, END

builder = StateGraph(HypothesisState)

# 노드 추가
builder.add_node("searchDB", get_docs)
builder.add_node("hierarchy", run_hierarchy)
builder.add_node("emerging", run_emerging)
builder.add_node("proposal", run_proposal)

# 엣지 추가
builder.add_edge(START, "searchDB")
builder.add_edge("searchDB", "hierarchy")
builder.add_edge("hierarchy", "emerging")
builder.add_edge("emerging", "proposal")
builder.add_edge("proposal", END)

hypothesis_workflow = builder.compile(name="hypothesis_generation_agent")