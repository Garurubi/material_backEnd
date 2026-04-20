from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from .perovskite_predict import element_fingerprint, element_net
import os
import asyncio
import sys

TOOLS = {
    "element_fingerprint": element_fingerprint,
    "element_net": element_net
}

BASE_DIR =  os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

model = init_chat_model(model=os.getenv("MATERIAL_PREDICT_MODEL"))

client = MultiServerMCPClient(
    {
        "material_predict": {
            "command": sys.executable,
            # Make sure to update to the full absolute path to your math_server.py file
            "args": [os.path.join(BASE_DIR, "perovskite_predict.py")],
            "transport": "stdio",
        },
    }
)

async def perovskite_predict_workflow(message: str):
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    # response = await agent.ainvoke({"messages": messages})
    tool_calls = []
    async for event in agent.astream({"messages": message}, stream_mode="values"):
        if event["messages"][-1].type == "ai":
            tool_calls = event["messages"][-1].tool_calls
            break

    # 함수 호출
    predict_result = []
    for call in tool_calls:
        function_name = call["name"]
        param_formula = call["args"]["formula"]

        predict_value = TOOLS[function_name](param_formula)
        predict_result.append({"property": "bandgap", "formula": param_formula, "predict_value": predict_value, "unit": "eV"})

    return predict_result

if __name__ == "__main__":
    asyncio.run(perovskite_predict_workflow("CsSnI3 bandgap 예측해줘."))