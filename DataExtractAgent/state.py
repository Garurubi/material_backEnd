 
from typing import TypedDict, Literal, Optional,Union,List
from .schemas.sacs.electro_chemical import FinalOutput 

# 추후 추가될 templete 대비
Response = Union[  FinalOutput]


class extractState(TypedDict):
    # 총괄 에이전트의 입력 ( pdf->text  , schema 종류 )
    pdf_text: List[str]
    requested_templetes : List[str]

    # prompt에 들어갈 schema 
    templete_prompt_list : Optional[List[str]]

    # 최종 출력 json
    response: Optional[List[Response]]

    # 상태관리
    status:str="success"
    error_message:Optional[str]

