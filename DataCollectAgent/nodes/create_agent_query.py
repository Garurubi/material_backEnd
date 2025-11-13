
from typing import TypedDict, Optional 
from openai import OpenAI
from langchain_openai import ChatOpenAI
import os
from ..state import collectState 
from ..prompts import catalysis_prompt ,material_prompt,mongoDB_prompt
from langchain.chat_models import init_chat_model

model = init_chat_model(model=os.getenv("DATA_COLLECT_CREATE_QUERY"))

create_agent_query_prompt = """
[ROLE]
You are a "query rewriter."  
Given a user's question and a set of tool descriptions (tool_docs), rewrite the question into a **natural-language instruction (one or two sentences)** that explicitly includes relevant argument keys and return field names from **any of the tools**.  
Do NOT output JSON, tables, bullet lists, or code — only fluent plain text.

[INPUTS]
<user_query>
{user_query}
</user_query>

<tool_docs>
{tools_block}  # Each tool includes its name, short description, arg keys, and return fields.
</tool_docs>


[WHAT TO DO]
1. Analyze the user query and identify all relevant information needs.  
2. Combine argument keys and return fields from **multiple tools wherever possible**, not just one.
   - Look for complementary data across tools (e.g., use material properties together with catalyst performance or reaction energetics).   
   - If information in the user query could map to multiple tools, include *all relevant keys* rather than choosing one.  
   - Keep key names exactly as defined in the tool docs.  
   - If a value for an argument is present in the user query, include it naturally in the rewritten text.  
   - If a value is missing or unclear, omit that argument rather than inventing one.  
3. Write a **coherent, single natural-language instruction** that reflects the *combined reasoning* of multiple tools — as if giving a high-level command to a multi-tool react agent.  
4. The rewritten instruction should sound natural but must clearly expose the specific field names (arg/return keys) to guide retrieval and computation.

[RULES]
- Use exact key names from the tool docs (keep lowercase, underscores, etc.).  
- It’s encouraged to **mix and combine keys from multiple tools** when conceptually related to the user query.   
- Do NOT invent new parameter or field names.  
- No JSON, quotes, or enumeration — only a short, continuous natural-language query.  
- The rewritten text should be comprehensive enough that a react_agent can coordinate multiple tools without further clarification.   

[OUTPUT FORMAT]
Return **one or two natural-language sentences** only — no preamble, no formatting.

[EXAMPLES]
**Example 1:**  
"Using elements=[Li, Fe, O] and is_stable=true from the material tool, compare energy_above_hull, formation_energy_per_atom, and is_stable,  
and also use the catalysis and catalyst data to relate reactionEnergy, activationEnergy, and metrics.overpotential for similar systems."
"""



def build_prompt(user_query: str, tool_docs: list[str]) -> str:
    tools_block = "\n\n---\n\n".join(tool_docs) 
    return create_agent_query_prompt.format(user_query=user_query,tools_block=tools_block)
 
def create_query (state :collectState )-> collectState :
    user_query = state["requirements"]

    prompt = build_prompt(user_query=user_query, tool_docs=[catalysis_prompt,material_prompt,mongoDB_prompt])
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = model.invoke(messages)
    content = response.content.strip()  
    return {"agent_query":content}


if __name__ == "__main__" :
    ex_state : collectState = {
        "requirements":"수소를 만드는 데 가장 적합한 단원자 금속은 무엇일까?"
    }
    create_query(ex_state)
