from .find_DB import  create_mongo_query
from mcp import  StdioServerParameters 
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os
import json
import sys
from ..state import collectState 
from langchain_core.tools import tool
import ast
from langchain.chat_models import init_chat_model


BASE_DIR =  os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.abspath(os.path.join(BASE_DIR,".."))
# load_dotenv()

model = init_chat_model(model=os.getenv("DATA_COLLECT_MODEL"))

server_params = StdioServerParameters(
    command="python",
    args=[os.path.join(SERVER_DIR, "material_server.py")],
)

PY = sys.executable

client = MultiServerMCPClient(
    {
        "catalysis":{
            "command": PY,
            "args": [os.path.join(SERVER_DIR, "catalysis_server.py")],
            "transport": "stdio"
        },

        "material":{
            "command": PY,
            "args": [os.path.join(SERVER_DIR, "material_server.py")],
            "transport": "stdio",
            "env": {"MP_API_KEY": os.getenv("MP_API_KEY")}
        },
    }
)

async def agent_node (state:collectState) -> collectState :
    # 입력으로 들어온 state 로 부터 사용자 질의를 꺼내고 그것으로 선택
    tools = await client.get_tools()
    tools +=  [create_mongo_query]

    agent = create_agent(model=model, tools=tools ,system_prompt="Return only a valid JSON object following this schema. Start with { and end with }. Do not include any explanations or markdown formatting")
    
    
    # 입력으로 들어온 메세지를 살짝 추가
    if not state["agent_query"]:
        return {}
    user_query = state["agent_query"] 

    answer_list = {}
    # agent 결과 도출
    result_list = []
    async for result  in agent.astream({"messages":user_query }):
        result_list.append(result)
        if "tools" in result : 
            tool_name = result["tools"]["messages"][-1].name
            if tool_name == 'search_from_mongoDB' : 
                try : 
                    search_result = json.loads(result["tools"]["messages"][-1].content) 
                    answer_list[tool_name]=search_result
                except Exception as e :
                    print("실패")
            elif tool_name == 'search_element_materials' : 
                answer_list[tool_name]= result["tools"]["messages"][-1].content
            elif tool_name == 'search_reactions' : 
                answer_list[tool_name]=result["tools"]["messages"][-1].content
    
    state["response"] = answer_list
    return state