from dotenv import load_dotenv
load_dotenv()

from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage
from DataCollectAgent import build_data_collect_graph
from HypothesisAgent import hypothesis_workflow
from DebateAgent import debate_graph
import os
from langchain.chat_models import init_chat_model
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

langfuse = Langfuse(
    secret_key="sk-lf-9eed493c-d257-47a0-9e4e-2b30ecf0a2a9",
    public_key="pk-lf-d1d7529a-e609-4685-b751-c34f1823048d",
    host="http://192.168.2.134:3000",
)
langfuse_handler = CallbackHandler()

model = init_chat_model(model=os.getenv("SUPERVISOR_MODEL"), temperature=0.0)

state = {
    "messages": [HumanMessage(content="산소를 만드는 반응에서는 어떤 금속 단원자가 가장 효과적일까?")],
}

config = {"callbacks": [langfuse_handler]}

# 데이터 수집 agent 호출
data_collect_graph = build_data_collect_graph()

supervisor_workflow = create_supervisor(
    [data_collect_graph, hypothesis_workflow, debate_graph],
    model=model
)

supervisor = supervisor_workflow.compile()
result = supervisor.invoke(state, config=config)
print(result)