from DataCollectAgent import build_data_collect_graph
from jinja2 import Template

a = """{% if search_results %}
<SEARCH_RESULTS>
{% for k, v in search_results.items() %}
  {% if k == "search_reactions" %}
    <REACTION_RESULTS>{{ v }}</REACTION_RESULTS>
  {% elif k == "search_from_mongoDB" %}
    {% if v.db_result %}
        <MONGO_DB_RESULTS>{{ v.db_result }}</MONGO_DB_RESULTS>
    {% endif %}
  {% endif %}
{% endfor %}
</SEARCH_RESULTS>
{% endif %}
"""

async def test_agent():
    data_collect_graph = build_data_collect_graph()
    data_collect_state = await data_collect_graph.ainvoke(
            {"requirements": "수소 발생 반응(HER)에서 수소 생산에 가장 적합한 단원자 금속은 무엇일까?"},
            config={"run_name": "data_collect_agent"}
        )
    
    tmpl = Template(a)
    abc = tmpl.render(search_results=data_collect_state.get("response"))
    print(abc)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
