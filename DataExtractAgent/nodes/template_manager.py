
from ..state import extractState
from ..schemas.sacs.electro_chemical  import *
import inspect
def select_templete(state: extractState) -> extractState:
    # 템플릿 선정
    if not state["requested_templetes"] :
        print("templete이 존재하지 않습니다.")
        return {"status":"failure" , "error_message":"주신 templete이 존재하지 않습니다."}
    
    schema_prompt_list = []

    input_templete_name_list = state["requested_templetes"]
    for idx, templete_name in enumerate(input_templete_name_list) : 
        if not templete_name :
            print(f"{idx+1} , templete을 찾지 못했습니다. ")
            schema_prompt_list.append(other_templete_schema)
            continue
            # return {"status":"failure","error_message":"지식베이스에서 templete을 찾지 못했습니다. "}

        if templete_name == "electroChemical":
            schema_prompt_list.append(get_pydantic_models_as_string_for_prompt_electroChemical())
    
            

    return {"templete_prompt_list":schema_prompt_list , "status":"success"}


other_templete_schema = """
not supported schema
"""

def get_pydantic_models_as_string_for_prompt_electroChemical() -> str:
        """프롬프트에 넣을 스키마 코드를 안정된 순서로 직렬화"""
        parts = []
        parts.append(inspect.getsource(OperationTypeEnum))
        parts.append(inspect.getsource(ActiveMetal))
        parts.append(inspect.getsource(CatalystComposition))
        parts.append(inspect.getsource(Catalyst))
        parts.append(inspect.getsource(Precursor))
        parts.append(inspect.getsource(HeatingValue))
        parts.append(inspect.getsource(HeatingConditions))
        parts.append(inspect.getsource(MixingConditions))
        parts.append(inspect.getsource(HeatingStep))
        parts.append(inspect.getsource(MixingStep))
        parts.append(inspect.getsource(OtherStep))
        parts.append("SynthesisStep = Union[HeatingStep, MixingStep, OtherStep]\n")
        parts.append(inspect.getsource(SynthesisProcess))
        parts.append(inspect.getsource(Electrolyte))
        parts.append(inspect.getsource(PerformanceConditions))
        parts.append(inspect.getsource(OverpotentialMetric))
        parts.append(inspect.getsource(OnsetPotentialMetric))
        parts.append(inspect.getsource(TafelSlopeMetric))
        parts.append(inspect.getsource(TOFMetric))
        parts.append(inspect.getsource(MassActivityMetric))
        parts.append(inspect.getsource(StabilityMetric))
        parts.append(inspect.getsource(SelectivityMetric))
        parts.append(inspect.getsource(PerformanceMetrics))
        parts.append(inspect.getsource(ElectrochemicalPerformance))
        parts.append(inspect.getsource(Experiment))
        parts.append(inspect.getsource(FinalOutput))
        return "\n".join(parts)