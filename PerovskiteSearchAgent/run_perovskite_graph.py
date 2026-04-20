import os
import json
import ast

from typing import Annotated, Literal, List, Dict, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_tavily import TavilySearch
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages

from .utils.prompt_set import (get_query_check_prompt, 
							   get_query_gen_prompt,)

# 환경 변수 확인
required_env_vars = [
    "OPENAI_API_KEY",
    "TAVILY_API_KEY"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")


### Database setting
try:
	base_dir = os.path.dirname(os.path.abspath(__file__))
	_db_path = "perovskite.db"
	# DB 파일 존재 확인
	if not os.path.exists(base_dir+"/"+_db_path):
		raise FileNotFoundError(f"SQLite DB 파일이 존재하지 않습니다: {base_dir+'/'+_db_path}")

	# DB 연결 시도
	db = SQLDatabase.from_uri(f"sqlite:///{base_dir+'/'+_db_path}")

except Exception as e:
	print(f"SQLite DB 필요 또는 연결 실패: {e}")
	raise 

MODEL_NAME = "gpt-4o"
MAX_RECURSION = 4

# Prompt Load
query_gen_prompt = get_query_gen_prompt()
query_check_prompt = get_query_check_prompt()


#### Define the workflow

# Define the STATE for the workflow
class SQLState(TypedDict):
	messages: Annotated[list[AnyMessage], add_messages]
	sql_result: Optional[str | List[Dict]]
	answer_flag: Optional[Literal["SQL", "Tavily"]]
	recursion_count: int


# Define the tools schema
class GenerateSqlQuery(BaseModel):
	"""
	A tool schema used to generate a syntactically correct SQL query
	based on the user's natural language question.

	This model should return:
	- `sql_query`: A valid SQL query string for the given database dialect (e.g., SQLite, MySQL, PostgreSQL)
	- `error`: An optional error message if query generation fails
	"""
	sql_query: Optional[str] = Field(
		None, 
		description="A syntactically correct SQLite query string generated based on the user's input question."
	)
	error: Optional[str] = Field(None, description="Optional error message if no answer could be generated.")

	model_config = {
        "extra": "forbid",
        "strict": True,
    }


## Define SQL Tools
@tool
def db_query_tool(query: str) -> str:
	"""
	Execute a SQL query against the database and get back the result.

	If the query is not correct, an error message will be returned.
	If an error is returned, rewrite the query, check the query, and try again.
	"""
	return_data={}
	result = db.run_no_throw(query, include_columns=True)

	if isinstance(result, str) and result.startswith("Error:"):
		return_data["messages"] = AIMessage(content=f"Here is the error: {repr(result)}\n\nPlease fix your mistakes.")
		return_data["answer_flag"] = None
	elif isinstance(result, str) and not result:
		return_data["messages"] = AIMessage(content="No records were found matching your query.")
		return_data["recursion_count"] = MAX_RECURSION + 1
	else:
		# parser_result = execute_and_convert(query, result)	
		parser_result = ast.literal_eval(result)
		return_data["messages"] = AIMessage(content=parser_result[:5])
		return_data["sql_result"] = parser_result[:10]

	return return_data

## LangChain chains
query_gen = query_gen_prompt | ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(
    [GenerateSqlQuery], tool_choice="GenerateSqlQuery")

query_check = query_check_prompt | ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(
	[db_query_tool], tool_choice="db_query_tool")


## SQL Node
def get_db_node(state: SQLState):
	"""
	Load database metadata into the state.

	This node retrieves database schema and table information
	using SQLDatabase tools and saves them to the shared state.
	"""
	db_dialect = db.dialect	
	content = "DB Dialect: "+db_dialect + "\n" + db.get_table_info()

	return {"messages": AIMessage(content=content), "recursion_count": state.get("recursion_count", 0)}


def query_gen_node(state: SQLState):
	"""
	Generate a SQL query and update the graph state.

    This node produce a syntactically and semantically valid SQL query based on the current state.
    The resulting query and potential error messages are parsed into a 
    standardized format for downstream nodes (e.g., `query_execute_node`).
	"""
	update_state = {"recursion_count": state["recursion_count"] + 1}
	
	result = query_gen.invoke(state)

	if result.tool_calls:
		tool_call = result.tool_calls[0]

		sql_query = tool_call["args"].get("sql_query")
		error = tool_call["args"].get("error")
		
		update_state["messages"] = AIMessage(
        		content=json.dumps({"sql_query": sql_query, "error": error}, ensure_ascii=False)
    		)
		update_state["answer_flag"] = "SQL" if sql_query else state.get("answer_flag", None)
	
	return update_state


def query_execute_node(state: SQLState):
	"""
	Execute the generated SQL query and update the graph state with the results.

    This node takes the last generated SQL query from the model output,
    executes it against the connected database using `db_query_tool`,
    and stores the execution result in the state.  
    If the generated SQL is invalid or missing, an appropriate error message
    is added to the state instead.
	"""
	update_state={}
	
	try:
		_obj = GenerateSqlQuery.model_validate_json(state["messages"][-1].content)
		
		if _obj.sql_query:
			update_state = db_query_tool.invoke(_obj.sql_query)
		else:
			update_state["messages"] = AIMessage(content=_obj.error)
			update_state["answer_flag"] = None

	except Exception as e:
		update_state["messages"] = AIMessage(content=f"Please rewrite your query and try again. [{e}]")
		update_state["answer_flag"] = None
	
	return update_state
	

# tavily Node
tavily_search_tool = TavilySearch(max_results=5)
tavily_model = ChatOpenAI(model=MODEL_NAME, temperature=0)


def tavily_search_node(state: SQLState):
	"""Node that triggers Tavily Search when the graph hits the recursion limit"""
	query_message = state["messages"][0].content
	
	try:
		results = tavily_search_tool.invoke({"query": query_message})
		if not isinstance(results, str):
			results = str(results)
		
		summary_prompt = (
            f"Summarize the following search results as a final answer to the user in Korean language:\n\n{results}"
        )

		summary = tavily_model.invoke(summary_prompt)
		return {
			"messages": [AIMessage(content=f"Answer: {summary.content}")],
			"sql_result": summary,
			"answer_flag": "Tavily"
		}

	except Exception as e:
		return {
			"messages": [AIMessage(content=f"Error: Tavily search failed. {e}")],
			"answer_flag": None
		}


# Define a conditional Edge Function
def should_gen_continue(state:SQLState) -> Literal["query_execute", "tavily_search"]:
	"""
    Determine the next node to execute based on the SQL generation result.

    This function inspects the latest message in the given SQLState and decides 
    whether to proceed with SQL query execution or fall back to a Tavily search 
    when the generated query is deemed irrelevant to the user's question.
	"""
	last_msg = state["messages"][-1].content
	
	try:
		_obj = GenerateSqlQuery.model_validate_json(last_msg)
		if _obj.error.startswith("The question is unrelated"):
			return "tavily_search"
		elif _obj:
			return "query_execute"
		else:
			return "tavily_search"
	except Exception:
		return "query_execute"


def should_execute_continue(state: SQLState) -> Literal[END, "query_gen", "tavily_search"]: # type: ignore
	"""
	Determine whether to terminate execution, retry SQL generation, or fall back to Tavily search.

    This edge function inspects the current SQLState after query execution
    to decide the next step in the graph flow.
	"""
	recursion_count = state.get("recursion_count")

	if recursion_count > MAX_RECURSION:
		return "tavily_search"
	elif state.get("sql_result"):
		return END
	else:
		return "query_gen"


## Specify the edges between the nodes
workflow = StateGraph(SQLState)

workflow.add_node("get_db_info", get_db_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("query_execute", query_execute_node)

workflow.add_node("tavily_search", tavily_search_node)


workflow.add_edge(START, "get_db_info")
workflow.add_edge("get_db_info", "query_gen")
workflow.add_conditional_edges("query_gen", should_gen_continue)
workflow.add_conditional_edges("query_execute", should_execute_continue)
workflow.add_edge("tavily_search", END)


perovskite_workflow = workflow.compile()