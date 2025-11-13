 
from typing import TypedDict, Literal, Optional,Union,List
from .schemas.sacs.electro_chemical import extracted_data 
 
# 추후 추가될 templete 대비
Schema = Union[  extracted_data]


class collectState(TypedDict):
    # 수집 요구 사항
    requirements : str

    # react_agent 에 보낼 출력
    agent_query : str

    # 사용하는 데이터셋 종류
    #requested_templetes : str

    # prompt에 들어갈 schema 
    # templete_prompt_list : Optional[List[str]]

    # 출력 , 후보데이터 셋
    response: Optional[List[str]]


    # 상태관리
    status:str  = "success"
    error_message:Optional[str]

 