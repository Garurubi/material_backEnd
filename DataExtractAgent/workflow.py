from langgraph.graph import StateGraph
from .state  import extractState 
from .nodes.data_extractor  import data_extract
from .nodes.template_manager import select_templete 


def build_data_extract_graph():
    graph = StateGraph(extractState)
    graph.add_node("select_templete",select_templete)
    graph.add_node("data_extract",data_extract)
    graph.add_node("handle_failure",handle_failure)
    graph.add_conditional_edges("select_templete", 
                                path_map={"success":"data_extract" , "failure" : "handle_failure"},
                                path= lambda state : state["status"])
    graph.set_entry_point("select_templete")
    graph.set_finish_point("data_extract")
    graph.set_finish_point("handle_failure")
    return graph.compile()

import os
def handle_failure(state: extractState) -> extractState:
      print(f"templete 부분에서 문제가 발생했습니다 : .{state['error_message']}")
      return state




# 확인 코드
def main(): 
    input_path="DataExtractAgent/data"
    pdf_texts = []
    templetes= [] 
    for filename in os.listdir(input_path) :
          if filename.endswith(".txt") :
                file_path = os.path.join(input_path,filename)
                with open(file_path,"r",encoding="utf-8") as f:
                      pdf_texts.append(f.read())
                      templetes.append("electroChemical")

    
    # 예시 state
    ex_state:extractState = {
        "pdf_text" :pdf_texts,
        "requested_templetes" : templetes
    }
    ex_graph = build_data_extract_graph()

    final_state = ex_graph.invoke(ex_state) 
 

if __name__ == "__main__":
    main()