from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4.1")

server_params = StdioServerParameters(
    command="python",
    args=["./material_server.py"],
)

client = MultiServerMCPClient(
    {
        "catalysis":{
            "command": "python",
            "args": ["catalysis_server.py"],
            "transport": "stdio"
        },

        "material":{
            "command": "python",
            "args": ["material_server.py"],
            "transport": "stdio"
        },
    }
)


async def run():
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    # question = "H2가 반응하고 OH가 생성되는 반응 사례를 3개만 찾아줘"
    question = "Si와 O가 포함된 안정한 물질 중에서 밴드갭이 1~3 eV인 걸 찾아줘"
    # question = "Li–Fe–O 계에서 안정(materials.is_stable=true) 후보 중에서 energy_above_hull, formation_energy, is_stable 지표를 비교표로 보여주고, mp-13과 mp-149의 탄성상수/유전 텐서/자성 데이터를 요약해"
    # Agent 실행
    answer = await agent.ainvoke({"messages": question})

    print("\n=== 최종 답변 ===")
    print(answer)
        


import asyncio

asyncio.run(run())
