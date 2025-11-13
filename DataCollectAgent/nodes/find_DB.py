
from pymongo import MongoClient
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
from bson import json_util
from langchain.tools import tool

 
from ..schemas.sacs.electro_chemical import catalyst
from ..state import collectState
from ..prompts import db_query_output_format,db_query_generate_format , new_db_query_generate_format

class MongoQuery(BaseModel):
    """MongoDB query structure"""
    query: dict
    explanation: str
#-------------------------------- 사전 설정 -----------------------------------

# load_dotenv()
client = MongoClient("mongodb://mongodb:dnpdlqmdnpdj@192.168.2.131:27017/")  # 기본 포트 27017
db = client["extraction_db"]
collection = db["all_extracted_data"]


llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# 최대 반환 document 수
MAX_SEARCH_DOC = 10

# 반환시 보지 않을 필드
PROJECTION ={
    "$project":{"catalyst.extracted_data.experiments.synthesisProcess" : 0}
}


## 몽고 DB에서 들어온 입력을 기반으로 쿼리를 생성하는 함수
## 단원자 촉매 관련 데이터 검색
@tool("search_from_mongoDB",return_direct=True)  
def create_mongo_query (query : str )->dict : 
    """
    Search the internal MongoDB of extracted SAC (single-atom catalyst) literature based on a single user query.
    The query may implicitly contain information about reaction type, active metals, supports, synthesis parameters, or electrochemical performance metrics.
    The tool will analyze the query to decide which database collections, fields, and filters to apply automatically.
    ONLY use this tool when the user's query involves literature-based SAC data (e.g., catalyst composition, synthesis conditions, performance metrics).
    Do NOT use for computed material properties or general Q&A.

    Input — user_query (str).

    Output — JSON string containing:
    - "db_result": a list of matched literature records with metadata and extracted fields.
    - "count": the number of matched documents returned in this call (document-level count, not array length).
    """
    





    # 1. DB에 날릴 쿼리 작성
    # 1-1. 스키마 + 프롬프트 만들기
    parser = JsonOutputParser(pydantic_object=MongoQuery)
    schema_dict = catalyst.model_json_schema()
    schema_str = json.dumps(schema_dict)  # dict → str
    schema_json = schema_str.replace("{", "{{").replace("}", "}}") 
    

    # 1-2. llm을 통해 쿼리 생성
    human_template = "Create a MongoDB query for: {user_request}"
    system_template = build_prompt( db_query_output_format)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    chain = prompt | llm | parser 
    result = chain.invoke({"schema_json": schema_json,"user_request": query})


    # 2. DB에 쿼리 날리기(오류 체크)
    try : 
        if result["query"]["filter"] =={} or type(result["query"]["filter"])!=list:
            raise Exception
        result["query_str"] = json.dumps(result["query"],indent=2, ensure_ascii=False)
    except Exception as e :
        result["db_result"] = None
        result["error"] = str(e)
        return result

    # find 사용하는 코드
    # filter_dict = result["query"]["filter"]
    # results = collection.find(filter_dict,projection = PROJECTION).limit(MAX_SEARCH_DOC).to_list(length=None)

    result["query"]["filter"]=[stage for stage in result["query"]["filter"]if "$project" not in result["query"]["filter"]]
    result["query"]["filter"] +=[PROJECTION] 
    query_result = collection.aggregate(result["query"]["filter"]).to_list(length=None)
    query_result =query_result[:MAX_SEARCH_DOC]
    if len(query_result) == 0 :
        result["error"] = "no documents are found for the generated query."
        return result 
    

    json_str = json.dumps(query_result , default=json_util.default, ensure_ascii=False,indent=2) 
    
    # list 형태로 기록하기
    result["db_result"] = json_str 
    # 3. 결과 리턴하기
    result["count"] = len(query_result) 


    return result 

def build_prompt( output_format):
    prompt_first = f"""
    {new_db_query_generate_format}

    <schema>
    """
    prompt_second= "{schema_json}"
    prompt_third =f"""
    </schema>

    <output_format>
    {output_format}
    </output_format>

    """
    prompt= prompt_first+prompt_second+prompt_third
    return prompt




if __name__ == "__main__" :
    query = "수소를 만드는데 가장 적합한 단원자 금속은 무엇인가?"
    query ="어떤 금속 단원자가 CO₂를 가장 잘 줄일 수 있을까?"
    # query = "single-atom catalysts active metals various monatomic metals high electrochemical performance HER reaction compare metrics: overpotential, tafel slope, faradaic efficiency"
    create_mongo_query.invoke(query)

