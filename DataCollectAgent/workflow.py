from langgraph.graph import StateGraph, END
from .nodes.create_react_agent import agent_node
from .nodes.create_agent_query import create_query
from .state import collectState


def build_data_collect_graph () :
    graph = StateGraph(collectState) 
    graph.add_node("create_query",create_query)
    graph.add_node("agent_node",agent_node)
    graph.set_entry_point("create_query")
    graph.set_finish_point("agent_node")
    graph.add_edge("create_query","agent_node")
    return graph.compile(name="data_collect_agent")


async def main() :
    
    ex_state : collectState = {
        # "requirements":"수소를 만드는 데 가장 적합한 단원자 금속은 무엇일까?" 
        "requirements":"산소를 만드는 반응에서는 어떤 금속 단원자가 가장 효과적일까?" 
    }
    # ex_state : collectState = {
    #     "requirements":"Si와 O가 포함된 안정한 물질 중에서 밴드갭이 1~3 eV인 걸 찾아줘" 
    # }
    ex_graph = build_data_collect_graph()
    final_state = await ex_graph.ainvoke(ex_state) 
    # print("최종 결과\n",final_state)

import asyncio
if __name__ == "__main__" :
    asyncio.run(main())


